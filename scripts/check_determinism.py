"""Check that the simulator is reproducible: same seed + same actions => same trajectory.

Usage: python scripts/check_determinism.py [--port 5555] [--seeds 3] [--max-wave 30]
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerogue_rl.sim_client import SimClient  # noqa: E402


def play(sim: SimClient, seed: str, max_wave: int) -> list[dict]:
    rng = random.Random(seed)
    resp = sim.reset(seed=seed, max_wave=max_wave)
    trace = [resp["obs"]]
    while not resp["done"]:
        resp = sim.step(rng.choice([i for i, ok in enumerate(resp["obs"]["legal"]) if ok]))
        trace.append(resp["obs"])
    return trace


def first_divergence(a: list[dict], b: list[dict]) -> str | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            keys = [k for k in x if x[k] != y.get(k)]
            detail = "; ".join(f"{k}: {json.dumps(x[k])[:200]} vs {json.dumps(y[k])[:200]}" for k in keys)
            return f"step {i} (wave {x['wave']}, turn {x['turn']}): {detail}"
    if len(a) != len(b):
        return f"lengths differ ({len(a)} vs {len(b)} steps)"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help="connect to a running simulator instead of launching one")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-wave", type=int, default=30)
    args = parser.parse_args()

    sim = SimClient(args.port) if args.port else SimClient.launch()
    failures = 0
    with sim:
        for i in range(args.seeds):
            seed = f"determinism-{i}"
            a, b = play(sim, seed, args.max_wave), play(sim, seed, args.max_wave)
            diff = first_divergence(a, b)
            status = "OK" if diff is None else f"DIVERGED at {diff}"
            print(f"seed {seed}: {len(a)} steps, wave {a[-1]['wave']} -> {status}")
            failures += diff is not None
    print("PASS" if failures == 0 else f"FAIL ({failures}/{args.seeds} seeds diverged)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
