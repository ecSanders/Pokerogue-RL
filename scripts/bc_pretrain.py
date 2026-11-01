"""Behaviour-clone the heuristic's battle or run decisions to warm-start PPO.

Collects (observation, heuristic action, discounted return) for one decision group, then trains
that half of AgentPolicy: the policy head with cross-entropy, the value head on the returns.

Usage:
  python scripts/bc_pretrain.py --steps 200000 --name bc
  python scripts/train_ppo.py --name ppo-bc --init runs/bc/policy.pt
  python scripts/bc_pretrain.py --kinds run --steps 100000 --name bc-run
  python scripts/train_ppo.py --name run-v1 --run-policy --init runs/ppo-v5/policy.pt --run-init runs/bc-run/policy.pt
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pokerogue_rl import features as F  # noqa: E402
from pokerogue_rl import run_features as RF  # noqa: E402
from pokerogue_rl.env import BattleEnv, RewardConfig  # noqa: E402
from pokerogue_rl.policy import AgentPolicy  # noqa: E402
from pokerogue_rl.sim_client import launch_sims  # noqa: E402

GROUPS = {"battle": (F.BATTLE_KINDS, F.encode), "run": (RF.RUN_KINDS, RF.encode)}


def collect(env: BattleEnv, steps: int, gamma: float, encode) -> tuple[list[dict], list[int], list[float]]:
    """Play the heuristic for `steps` policy decisions; returns are computed per finished episode."""
    feats, actions, returns = [], [], []
    episode_feats, episode_actions, episode_rewards = [], [], []
    obs = env.reset()
    while len(feats) < steps:
        action = env.helper.act(obs)
        episode_feats.append(encode(obs))
        episode_actions.append(action)
        obs, reward, done, _ = env.step(action)
        episode_rewards.append(reward)
        if done:
            g = 0.0
            ep_returns = []
            for r in reversed(episode_rewards):
                g = r + gamma * g
                ep_returns.append(g)
            feats += episode_feats
            actions += episode_actions
            returns += ep_returns[::-1]
            episode_feats, episode_actions, episode_rewards = [], [], []
    return feats, actions, returns


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="bc")
    p.add_argument("--kinds", choices=GROUPS, default="battle", help="which decisions to clone")
    p.add_argument("--steps", type=int, default=200_000, help="decisions to collect")
    p.add_argument("--num-envs", type=int, default=12)
    p.add_argument("--round-steps", type=int, default=2000,
                   help="decisions per env before its simulator is restarted")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    out = ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    kinds, encode = GROUPS[args.kinds]
    parts = []
    # Collect in rounds on fresh simulators: each leaks ~3 MB per episode.
    rounds = max(1, -(-args.steps // (args.num_envs * args.round_steps)))
    for r in range(rounds):
        group = launch_sims(args.num_envs, request_timeout=120)
        envs = [BattleEnv(i, seed_prefix=f"{args.name}-r{r}", reward=RewardConfig(gamma=args.gamma), sim=sim,
                          policy_kinds=kinds) for i, sim in enumerate(group.clients)]
        per_env = args.steps // (args.num_envs * rounds)
        with ThreadPoolExecutor(args.num_envs) as pool:
            parts += list(pool.map(lambda e: collect(e, per_env, args.gamma, encode), envs))
        for e in envs:
            e.close()
        group.close()
        print(f"round {r + 1}/{rounds}: {sum(len(p[0]) for p in parts)} decisions", flush=True)
    feats = [f for part in parts for f in part[0]]
    data = {k: torch.as_tensor(np.stack([f[k] for f in feats]), device=device) for k in feats[0]}
    actions = torch.as_tensor([a for part in parts for a in part[1]], device=device)
    returns = torch.as_tensor([g for part in parts for g in part[2]], device=device, dtype=torch.float32)
    print(f"collected {len(feats)} {args.kinds} decisions in {time.time() - t0:.0f}s", flush=True)

    agent = AgentPolicy().to(device)
    policy = getattr(agent, args.kinds)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    n = len(feats)
    split = int(n * 0.9)  # hold out the last 10% (later episodes of each env) for validation
    for epoch in range(args.epochs):
        perm = torch.randperm(split, device=device)
        for i in range(0, split, args.batch):
            idx = perm[i:i + args.batch]
            logits, value = policy({k: v[idx] for k, v in data.items()})
            loss = torch.nn.functional.cross_entropy(logits, actions[idx]) \
                + 0.5 * (value - returns[idx]).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits, value = policy({k: v[split:] for k, v in data.items()})
            acc = (logits.argmax(-1) == actions[split:]).float().mean().item()
            v_err = (value - returns[split:]).pow(2).mean().item()
        print(f"epoch {epoch + 1}: val action accuracy {acc:.3f}, val value MSE {v_err:.3f}", flush=True)

    torch.save({"policy": agent.state_dict(), "args": vars(args)}, out / "policy.pt")
    (out / "bc.json").write_text(json.dumps({"decisions": n, "val_accuracy": acc, "val_value_mse": v_err}))
    print(f"saved {out / 'policy.pt'}")


if __name__ == "__main__":
    main()
