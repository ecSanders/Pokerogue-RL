"""Evaluate an agent over many seeded endless runs using parallel simulator processes.

Usage:
  python scripts/eval_agent.py --agent heuristic --episodes 40 --workers 8 --max-wave 200
"""

import argparse
import queue
import statistics
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerogue_rl.agents import AGENTS  # noqa: E402
from pokerogue_rl.sim_client import SimClient, launch_sims  # noqa: E402


def worker(worker_id: int, sim: SimClient, args, jobs: queue.Queue, results: list, lock: threading.Lock) -> None:
    agent = AGENTS[args.agent]()
    while True:
        try:
            ep = jobs.get_nowait()
        except queue.Empty:
            return
        resp = sim.reset(seed=f"{args.seed}-{ep}", max_wave=args.max_wave)
        agent.reset()
        steps, kinds = 0, Counter()
        t0 = time.perf_counter()
        while not resp["done"]:
            kinds[resp["obs"]["kind"]] += 1
            resp = sim.step(agent.act(resp["obs"]))
            steps += 1
        elapsed = time.perf_counter() - t0
        with lock:
            results.append({
                "episode": ep, "wave": resp["obs"]["wave"], "reason": resp["info"]["reason"],
                "steps": steps, "time": elapsed, "kinds": kinds, "auto": resp["info"]["autoInputs"],
            })
            print(f"[w{worker_id}] episode {ep}: wave {resp['obs']['wave']} ({resp['info']['reason']}), "
                  f"{steps} steps, {steps / max(elapsed, 1e-9):.0f} steps/s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=sorted(AGENTS), default="heuristic")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-wave", type=int, default=200)
    parser.add_argument("--seed", default="eval")
    args = parser.parse_args()

    jobs: queue.Queue = queue.Queue()
    for ep in range(args.episodes):
        jobs.put(ep)
    results: list = []
    lock = threading.Lock()

    t0 = time.perf_counter()
    group = launch_sims(args.workers)
    threads = [threading.Thread(target=worker, args=(i, sim, args, jobs, results, lock))
               for i, sim in enumerate(group.clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    group.close()
    wall = time.perf_counter() - t0

    waves = sorted(r["wave"] for r in results)
    steps = sum(r["steps"] for r in results)
    kinds = sum((r["kinds"] for r in results), Counter())
    auto = sum((Counter(r["auto"]) for r in results), Counter())
    print(f"\nagent={args.agent} episodes={len(results)} workers={args.workers} max_wave={args.max_wave}")
    print(f"waves: mean {statistics.mean(waves):.1f}, median {statistics.median(waves)}, "
          f"min {waves[0]}, max {waves[-1]}, reached max_wave {sum(r['reason'] == 'max_wave' for r in results)}")
    print(f"throughput: {steps} steps in {wall:.1f}s wall = {steps / wall:.0f} steps/s "
          f"(includes process startup)")
    print("decisions:", dict(kinds.most_common()))
    print("scripted inputs:", dict(auto.most_common()))


if __name__ == "__main__":
    main()
