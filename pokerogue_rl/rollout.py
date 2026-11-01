"""Asynchronous PPO rollout collection.

Each environment runs its own `steps`-long segment in its own thread, with inference on a CPU copy
of the policy, so environments never wait for each other mid-rollout. (In lockstep, the batch waits
for its slowest env every step: heuristic-played reward screens etc. make step times very uneven,
and that wait cost half of all env time.) Small-batch CPU inference is also ~3x faster than GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from pokerogue_rl import features as F
from pokerogue_rl.env import BattleEnv, VecBattleEnv
from pokerogue_rl.policy import BattlePolicy, to_tensors

CPU = torch.device("cpu")


@dataclass
class Rollout:
    """Arrays are [T, N, ...]: T steps per env, N envs."""

    feats: dict[str, np.ndarray]
    actions: np.ndarray
    logps: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    next_values: np.ndarray  # [N] value of each env's observation after the last step
    episodes: list[dict] = field(default_factory=list)
    invalid: int = 0
    sim_errors: int = 0


def _act(policy: BattlePolicy, obs: dict) -> tuple[dict, int, float, float]:
    enc = F.encode(obs)
    with torch.no_grad():
        logits, value = policy(to_tensors(F.stack([enc]), CPU))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
    return enc, int(action), float(dist.log_prob(action)), float(value)


def _run_env(env: BattleEnv, policy: BattlePolicy, steps: int) -> dict:
    out = {k: [] for k in ("feats", "actions", "logps", "values", "rewards", "dones")}
    episodes, invalid, sim_errors = [], 0, 0
    obs = env.obs
    for _ in range(steps):
        enc, action, logp, value = _act(policy, obs)
        obs, reward, done, info = env.step(action)
        for k, v in zip(out, (enc, action, logp, value, reward, done)):
            out[k].append(v)
        invalid += info["invalid"]
        sim_errors += "sim_error" in info
        if "episode" in info:
            episodes.append(info["episode"])
    with torch.no_grad():
        _, next_value = policy(to_tensors(F.stack([F.encode(obs)]), CPU))
    return {**out, "next_value": float(next_value), "episodes": episodes, "invalid": invalid,
            "sim_errors": sim_errors}


def collect(envs: VecBattleEnv, policy: BattlePolicy, steps: int) -> Rollout:
    """Collect `steps` policy steps from every env in parallel. `policy` must live on the CPU."""
    parts = list(envs.pool.map(lambda e: _run_env(e, policy, steps), envs.envs))
    feats = {k: np.stack([F.stack(p["feats"])[k] for p in parts], axis=1) for k in parts[0]["feats"][0]}
    col = lambda key, dtype: np.array([p[key] for p in parts], dtype=dtype).T  # noqa: E731  [T, N]
    return Rollout(
        feats=feats,
        actions=col("actions", np.int64),
        logps=col("logps", np.float32),
        values=col("values", np.float32),
        rewards=col("rewards", np.float32),
        dones=col("dones", np.float32),
        next_values=np.array([p["next_value"] for p in parts], np.float32),
        episodes=[e for p in parts for e in p["episodes"]],
        invalid=sum(p["invalid"] for p in parts),
        sim_errors=sum(p["sim_errors"] for p in parts),
    )
