"""Train the battle policy with PPO.

Usage:
  python scripts/train_ppo.py --name ppo-v1 --total-steps 2000000
  python scripts/train_ppo.py --name ppo-bc --init runs/bc/policy.pt   # warm start
  python scripts/train_ppo.py --name run-v1 --run-policy --init runs/ppo-v5/policy.pt \
      --run-init runs/bc-run/policy.pt      # the policy also plays rewards, item targets, moves, releases

Writes runs/<name>/log.jsonl (one line per update) and checkpoints runs/<name>/policy.pt.
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pokerogue_rl.env import RewardConfig, VecBattleEnv  # noqa: E402
from pokerogue_rl.features import BATTLE_KINDS  # noqa: E402
from pokerogue_rl.policy import AgentPolicy  # noqa: E402
from pokerogue_rl.run_features import RUN_KINDS  # noqa: E402
from pokerogue_rl.rollout import collect  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="ppo")
    p.add_argument("--num-envs", type=int, default=12)
    p.add_argument("--rollout", type=int, default=128, help="steps per env per update")
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--max-wave", type=int, default=500)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--init", help="checkpoint to start from (e.g. behaviour-cloned policy)")
    p.add_argument("--run-init", help="checkpoint whose run-policy half replaces the --init one (e.g. from "
                                     "bc_pretrain.py --kinds run)")
    p.add_argument("--run-policy", action="store_true",
                   help="the policy also makes run decisions (rewards, item targets, moves, releases)")
    p.add_argument("--resume", action="store_true",
                   help="continue runs/<name>/policy.pt: weights, optimizer, update count and LR schedule")
    p.add_argument("--recycle-every", type=int, default=100,
                   help="updates between simulator restarts (bounds a slow per-episode memory leak)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-every", type=int, default=10, help="updates between checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)  # leave CPU cores to the simulators; rollout threads infer in parallel
    device = torch.device(args.device)
    out = ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    policy = AgentPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    first_update, global_step = 1, 0
    if args.resume:
        ckpt = policy.load_checkpoint(out / "policy.pt", map_location=device)
        optimizer.load_state_dict(ckpt["optimizer"])
        first_update, global_step = ckpt["update"] + 1, ckpt["step"]
    elif args.init:
        policy.load_checkpoint(args.init, map_location=device)
    if args.run_init and not args.resume:
        run_init = AgentPolicy()
        run_init.load_checkpoint(args.run_init)
        policy.run.load_state_dict(run_init.run.state_dict())

    # Rollouts run per-env threads with small-batch inference on a CPU copy (faster than GPU there).
    actor = AgentPolicy()
    actor.load_state_dict(policy.state_dict())

    envs = VecBattleEnv(args.num_envs, seed_prefix=args.name, max_wave=args.max_wave,
                        reward=RewardConfig(gamma=args.gamma),
                        policy_kinds=BATTLE_KINDS + (RUN_KINDS if args.run_policy else ()))
    envs.reset()
    n, t_len = args.num_envs, args.rollout
    batch_size = n * t_len
    num_updates = args.total_steps // batch_size
    recent = collections.deque(maxlen=100)  # finished episodes
    start = time.time()

    for update in range(first_update, num_updates + 1):
        # Linear learning-rate decay.
        for g in optimizer.param_groups:
            g["lr"] = args.lr * (1 - (update - 1) / num_updates)

        rollout_start = time.time()
        rollout = collect(envs, actor, t_len)
        recent.extend(rollout.episodes)
        invalid, sim_errors = rollout.invalid, rollout.sim_errors
        global_step += batch_size
        rollout_time = time.time() - rollout_start

        # GAE. Episodes end inside the rollout (done=1 cuts bootstrapping); bootstrap the tail.
        to_dev = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
        values_t, rewards_t, dones_t = to_dev(rollout.values), to_dev(rollout.rewards), to_dev(rollout.dones)
        next_value = to_dev(rollout.next_values)
        advantages = torch.zeros_like(rewards_t)
        last = torch.zeros(n, device=device)
        for t in reversed(range(t_len)):
            nv = next_value if t == t_len - 1 else values_t[t + 1]
            nonterminal = 1.0 - dones_t[t]
            delta = rewards_t[t] + args.gamma * nv * nonterminal - values_t[t]
            last = delta + args.gamma * args.gae_lambda * nonterminal * last
            advantages[t] = last
        returns = advantages + values_t

        flat = {k: to_dev(v.reshape(batch_size, *v.shape[2:])) for k, v in rollout.feats.items()}
        b_actions = to_dev(rollout.actions).flatten()
        run_frac = flat["is_run"].float().mean().item()
        b_logps, b_adv, b_ret = to_dev(rollout.logps).flatten(), advantages.flatten(), returns.flatten()

        stats = collections.defaultdict(list)
        mb_size = batch_size // args.minibatches
        for _ in range(args.epochs):
            perm = torch.randperm(batch_size, device=device)
            for i in range(0, batch_size, mb_size):
                idx = perm[i:i + mb_size]
                logits, value = policy({k: v[idx] for k, v in flat.items()})
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(b_actions[idx])
                ratio = (new_logp - b_logps[idx]).exp()
                adv = b_adv[idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                v_loss = 0.5 * (value - b_ret[idx]).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                stats["pg_loss"].append(pg_loss.item())
                stats["v_loss"].append(v_loss.item())
                stats["entropy"].append(entropy.item())
                is_run = flat["is_run"][idx]
                if is_run.any():
                    stats["run_entropy"].append(dist.entropy()[is_run].mean().item())
                stats["clipfrac"].append(((ratio - 1).abs() > args.clip).float().mean().item())

        actor.load_state_dict({k: v.cpu() for k, v in policy.state_dict().items()})

        waves = [e["wave"] for e in recent]
        row = {
            "update": update, "step": global_step, "time": round(time.time() - start),
            "sps": round(batch_size / rollout_time),
            "episodes": len(recent),
            "wave_mean": round(float(np.mean(waves)), 2) if waves else None,
            "wave_median": float(np.median(waves)) if waves else None,
            "return_mean": round(float(np.mean([e["return"] for e in recent])), 3) if recent else None,
            "invalid": invalid,
            "sim_errors": sim_errors,
            "run_frac": round(run_frac, 3),
            **{k: round(float(np.mean(v)), 4) for k, v in stats.items()},
        }
        with (out / "log.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

        if update % args.save_every == 0 or update == num_updates:
            torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "update": update,
                        "step": global_step, "args": vars(args)}, out / "policy.pt")
        if update % args.recycle_every == 0 and update < num_updates:
            envs.recycle()

    envs.close()


if __name__ == "__main__":
    main()
