"""Smoke test: play random legal actions in the simulator and report throughput.

Usage:
  python scripts/smoke_sim.py                 # launch a simulator process
  python scripts/smoke_sim.py --port 5555     # connect to an already-running one
"""

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerogue_rl.sim_client import SimClient  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help="connect to a running simulator instead of launching one")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-wave", type=int, default=30)
    parser.add_argument("--seed", default="smoke")
    args = parser.parse_args()

    sim = SimClient(args.port) if args.port else SimClient.launch()
    rng = random.Random(0)
    total_steps = 0
    total_time = 0.0
    auto_inputs: Counter = Counter()

    with sim:
        for ep in range(args.episodes):
            t0 = time.perf_counter()
            resp = sim.reset(seed=f"{args.seed}-{ep}", max_wave=args.max_wave)
            reset_time = time.perf_counter() - t0
            steps = invalid = 0
            t0 = time.perf_counter()
            while not resp["done"]:
                legal = [i for i, ok in enumerate(resp["obs"]["legal"]) if ok]
                resp = sim.step(rng.choice(legal))
                steps += 1
                invalid += resp["info"].get("invalidAction", False)
            elapsed = time.perf_counter() - t0
            total_steps += steps
            total_time += elapsed
            auto_inputs.update(resp["info"]["autoInputs"])
            print(f"episode {ep}: reached wave {resp['obs']['wave']} ({resp['info']['reason']}), "
                  f"{steps} steps in {elapsed:.1f}s ({steps / max(elapsed, 1e-9):.1f} steps/s), "
                  f"reset {reset_time:.2f}s, invalid actions {invalid}")

    print(f"\noverall: {total_steps} steps, {total_steps / max(total_time, 1e-9):.1f} steps/s")
    print("scripted inputs:", dict(auto_inputs.most_common()))


if __name__ == "__main__":
    main()
