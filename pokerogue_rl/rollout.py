"""Asynchronous PPO rollout collection.

Each environment runs its own `steps`-long segment in its own thread, with inference on a CPU copy
of the policy, so environments never wait for each other mid-rollout. (In lockstep, the batch waits
for its slowest env every step: heuristic-played reward screens etc. make step times very uneven,
and that wait cost half of all env time.) Small-batch CPU inference is also ~3x faster than GPU.

Inference is batched across envs (`_Batcher`): each env thread hands its observation to one inference
thread, which runs everything waiting through the network at once. With one forward per env per
step, 20 env threads queued on the GIL behind Python overhead (tensor building, module calls,
distribution checks) and one node collected ~650 steps/s with most simulator cores idle. Actions
come from the same distributions either way; only the batching differs.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
import torch

from pokerogue_rl import features as F
from pokerogue_rl.env import BattleEnv, VecBattleEnv
from pokerogue_rl.policy import AgentPolicy, encode_step, to_tensors

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


def _infer(policy: AgentPolicy, encs: list[dict]) -> list[tuple[int, float, float]]:
    """(action, log prob, value) for each encoded observation, in one forward pass."""
    with torch.no_grad():
        logits, value = policy(to_tensors(F.stack(encs), CPU))
        # The logits are built masked; skipping argument validation saves ~8% of rollout time.
        dist = torch.distributions.Categorical(logits=logits, validate_args=False)
        action = dist.sample()
        logp = dist.log_prob(action)
    return list(zip(action.tolist(), logp.tolist(), value.tolist()))


class _Batcher:
    """One inference thread serving every env thread: whatever observations arrived while the last
    batch ran go through the network together, so no env waits for a fixed batch to fill."""

    def __init__(self, policy: AgentPolicy):
        self.policy = policy
        self.cv = threading.Condition()
        self.pending: list[tuple[dict, list]] = []  # (encoding, [event, result])
        self.closed = False
        self.error: BaseException | None = None
        self.batch_sizes: list[int] = []
        self.thread = threading.Thread(target=self._loop, name="rollout-inference", daemon=True)
        self.thread.start()

    def act(self, enc: dict) -> tuple[int, float, float]:
        slot = [threading.Event(), None]
        with self.cv:
            if self.error is not None:
                raise RuntimeError("rollout inference thread failed") from self.error
            self.pending.append((enc, slot))
            self.cv.notify()
        slot[0].wait()
        if slot[1] is None:
            raise RuntimeError("rollout inference thread failed") from self.error
        return slot[1]

    def _loop(self) -> None:
        while True:
            with self.cv:
                while not self.pending and not self.closed:
                    self.cv.wait()
                if not self.pending:
                    return
                batch, self.pending = self.pending, []
            try:
                results = _infer(self.policy, [enc for enc, _ in batch])
            except BaseException as e:  # wake the waiting env threads instead of hanging them
                with self.cv:
                    self.error = e
                    batch += self.pending
                    self.pending = []
                for _, slot in batch:
                    slot[0].set()
                return
            self.batch_sizes.append(len(batch))
            for (_, slot), result in zip(batch, results):
                slot[1] = result
                slot[0].set()

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify()
        self.thread.join()


def _run_env(env: BattleEnv, policy: AgentPolicy, steps: int, batcher: _Batcher | None = None) -> dict:
    out = {k: [] for k in ("feats", "actions", "logps", "values", "rewards", "dones")}
    episodes, invalid, sim_errors = [], 0, 0
    obs = env.obs
    for _ in range(steps):
        enc = encode_step(obs)
        action, logp, value = batcher.act(enc) if batcher else _infer(policy, [enc])[0]
        obs, reward, done, info = env.step(action)
        for k, v in zip(out, (enc, action, logp, value, reward, done)):
            out[k].append(v)
        invalid += info["invalid"]
        sim_errors += "sim_error" in info
        if "episode" in info:
            episodes.append(info["episode"])
    with torch.no_grad():
        _, next_value = policy(to_tensors(F.stack([encode_step(obs)]), CPU))
    return {**out, "next_value": float(next_value), "episodes": episodes, "invalid": invalid,
            "sim_errors": sim_errors}


def collect(envs: VecBattleEnv, policy: AgentPolicy, steps: int, batched: bool = True) -> Rollout:
    """Collect `steps` policy steps from every env in parallel. `policy` must live on the CPU.
    `batched=False`: each env thread runs its own forward passes (the pre-batching behaviour)."""
    batcher = _Batcher(policy) if batched else None
    try:
        parts = list(envs.pool.map(lambda e: _run_env(e, policy, steps, batcher), envs.envs))
    finally:
        if batcher:
            batcher.close()
    stacked = [F.stack(p["feats"]) for p in parts]  # once per env (this used to re-stack per feature key)
    feats = {k: np.stack([st[k] for st in stacked], axis=1) for k in stacked[0]}
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
