"""Profile where PPO rollout time goes: encoding, inference, and environment stepping.

Usage: python scripts/profile_rollout.py --num-envs 4 --steps 200 [--device cpu|cuda]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pokerogue_rl import features as F  # noqa: E402
from pokerogue_rl.env import BattleEnv, VecBattleEnv  # noqa: E402
from pokerogue_rl.policy import AgentPolicy, BattlePolicy, to_tensors  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=200, help="lockstep iterations")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--init", default=str(ROOT / "runs" / "bc" / "policy.pt"))
    p.add_argument("--async", dest="use_async", action="store_true",
                   help="time pokerogue_rl.rollout.collect (per-env threads, CPU inference) instead")
    args = p.parse_args()
    if args.use_async:
        profile_async(args)
        return
    torch.set_num_threads(2)
    device = torch.device(args.device)
    policy = BattlePolicy().to(device)
    policy.load_state_dict(torch.load(args.init, map_location=device)["policy"])

    # Time each env's own step so we can see how long the batch waits for its slowest env.
    per_env: list[float] = []
    real_step = BattleEnv.step

    def timed_step(self, action):
        t = time.perf_counter()
        out = real_step(self, action)
        per_env.append(time.perf_counter() - t)
        return out

    BattleEnv.step = timed_step

    envs = VecBattleEnv(args.num_envs, seed_prefix="profile")
    obs = envs.reset()
    t_encode = t_infer = t_env = 0.0
    lockstep_max, lockstep_mean = [], []
    start = time.perf_counter()
    for _ in range(args.steps):
        t0 = time.perf_counter()
        enc = F.stack([F.encode(o) for o in obs])
        t1 = time.perf_counter()
        with torch.no_grad():
            logits, _ = policy(to_tensors(enc, device))
            action = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        t2 = time.perf_counter()
        per_env.clear()
        obs, *_ = envs.step(action)
        t3 = time.perf_counter()
        t_encode += t1 - t0
        t_infer += t2 - t1
        t_env += t3 - t2
        lockstep_max.append(max(per_env))
        lockstep_mean.append(float(np.mean(per_env)))
    total = time.perf_counter() - start
    envs.close()

    steps = args.steps * args.num_envs
    print(f"{steps} policy steps in {total:.1f}s = {steps / total:.0f} steps/s ({args.num_envs} envs, {device})")
    for name, t in [("encode", t_encode), ("inference", t_infer), ("env step (wall)", t_env)]:
        print(f"  {name:16s} {t:6.1f}s  {100 * t / total:4.0f}%  {1000 * t / args.steps:6.2f} ms/iteration")
    print(f"  per-env step: mean {1000 * np.mean(lockstep_mean):.1f} ms, slowest-in-batch {1000 * np.mean(lockstep_max):.1f} ms"
          f" -> lockstep waiting costs {100 * (1 - np.mean(lockstep_mean) / np.mean(lockstep_max)):.0f}% of env time")


def profile_async(args) -> None:
    from pokerogue_rl.rollout import collect

    torch.set_num_threads(1)
    policy = AgentPolicy()
    policy.load_checkpoint(args.init)
    envs = VecBattleEnv(args.num_envs, seed_prefix="profile")
    envs.reset()
    start = time.perf_counter()
    rollout = collect(envs, policy, args.steps)
    total = time.perf_counter() - start
    envs.close()
    steps = args.steps * args.num_envs
    print(f"{steps} policy steps in {total:.1f}s = {steps / total:.0f} steps/s "
          f"({args.num_envs} envs, async, cpu); {len(rollout.episodes)} episodes finished")


if __name__ == "__main__":
    main()
