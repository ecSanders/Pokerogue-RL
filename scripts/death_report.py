"""Play seeded runs and report how they end: what the fatal wave was, and the team's state going in.

Usage:
  python scripts/death_report.py --agent heuristic --episodes 96 --workers 12
Writes one JSON record per run to runs/deaths.jsonl and prints a summary.
"""

import argparse
import json
import queue
import statistics
import sys
import threading
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pokerogue_rl.agents import AGENTS  # noqa: E402
from pokerogue_rl.sim_client import SimClient, launch_sims  # noqa: E402


def team_summary(party: list[dict]) -> dict:
    alive = [p for p in party if not p["fainted"]]
    return {
        "size": len(party),
        "alive": len(alive),
        "hp": round(sum(p["hp"] for p in alive) / max(sum(p["maxHp"] for p in alive), 1), 3),
        "levels": [p["level"] for p in party],
        "bst": [p["bst"] for p in party],
        "shiny": sum(p["shiny"] for p in party),
        "pokerus": sum(p["pokerus"] for p in party),
    }


def enemy_summary(enemies: list[dict]) -> list[dict]:
    return [{"species": e["species"], "level": e["level"], "bst": e["bst"], "boss": e["boss"],
             "hp": round(e["hp"] / e["maxHp"], 3) if e["maxHp"] else 0} for e in enemies]


def play(sim: SimClient, agent, seed: str, max_wave: int) -> dict:
    resp = sim.reset(seed=seed, max_wave=max_wave)
    agent.reset()
    wave_start: dict = {}
    level_gap: dict[int, float] = {}  # wave -> enemy max level - team max level
    last_command = None
    turns_in_wave = 0
    while not resp["done"]:
        obs = resp["obs"]
        if obs["wave"] not in wave_start:
            wave_start = {obs["wave"]: team_summary(obs["party"])}
            turns_in_wave = 0
        if obs["kind"] == "command":
            last_command = obs
            turns_in_wave += 1
            if obs["wave"] not in level_gap and obs["enemies"]:
                level_gap[obs["wave"]] = (max(e["level"] for e in obs["enemies"])
                                          - max(p["level"] for p in obs["party"]))
        resp = sim.step(agent.act(obs))

    final = resp["obs"]
    lc = last_command or final
    return {
        "seed": seed,
        "wave": final["wave"],
        "reason": resp["info"]["reason"],
        "battle_type": lc.get("battleType"),
        "trainer": lc.get("trainer"),
        "double": lc.get("double"),
        "enemies": enemy_summary(lc.get("enemies", [])),
        "team_at_wave_start": wave_start.get(final["wave"]),
        "turns_in_final_wave": turns_in_wave,
        "level_gap": level_gap,
        "money": final["money"],
        "auto_inputs": resp["info"]["autoInputs"],
    }


def worker(sim: SimClient, args, jobs: queue.Queue, records: list, lock: threading.Lock) -> None:
    agent = AGENTS[args.agent]()
    while True:
        try:
            ep = jobs.get_nowait()
        except queue.Empty:
            return
        rec = play(sim, agent, f"{args.seed}-{ep}", args.max_wave)
        with lock:
            records.append(rec)
            print(f"run {ep}: died wave {rec['wave']} ({rec['battle_type']}"
                  f"{' ' + rec['trainer'] if rec['trainer'] else ''})", flush=True)


def pct(n: int, total: int) -> str:
    return f"{n}/{total} ({100 * n / total:.0f}%)"


def summarize(records: list[dict]) -> None:
    deaths = [r for r in records if r["reason"] == "game_over"]
    n = len(deaths)
    print(f"\n=== {len(records)} runs, {n} ended in game over ===")
    if not n:
        return
    waves = sorted(r["wave"] for r in deaths)
    print(f"death wave: mean {statistics.mean(waves):.1f}, median {statistics.median(waves)}, "
          f"range {waves[0]}-{waves[-1]}")
    buckets = Counter(10 * (w // 10) for w in waves)
    print("  by decade:", ", ".join(f"{b}-{b + 9}: {buckets[b]}" for b in sorted(buckets)))
    ends_in_0 = sum(w % 10 == 0 for w in waves)
    print(f"  on waves ending in 0 (boss/milestone waves): {pct(ends_in_0, n)}")

    print("\nfatal battle:")
    types = Counter(r["battle_type"] for r in deaths)
    print("  type:", ", ".join(f"{t}: {pct(c, n)}" for t, c in types.most_common()))
    print(f"  double battle: {pct(sum(bool(r['double']) for r in deaths), n)}")
    print(f"  vs a boss: {pct(sum(any(e['boss'] for e in r['enemies']) for r in deaths), n)}")
    trainers = Counter(r["trainer"] for r in deaths if r["trainer"])
    if trainers:
        print("  trainers:", ", ".join(f"{t} x{c}" for t, c in trainers.most_common(8)))
    enemy_hp = [min(e["hp"] for e in r["enemies"]) for r in deaths if r["enemies"]]
    print(f"  lowest enemy HP left when we lost: median {statistics.median(enemy_hp):.2f}, "
          f"untouched (>0.9) in {pct(sum(h > 0.9 for h in enemy_hp), len(enemy_hp))}")
    turns = [r["turns_in_final_wave"] for r in deaths]
    print(f"  our turns in fatal wave: median {statistics.median(turns)}, mean {statistics.mean(turns):.1f}")

    print("\nteam entering the fatal wave:")
    starts = [r["team_at_wave_start"] for r in deaths if r["team_at_wave_start"]]
    print(f"  party size: {dict(sorted(Counter(s['size'] for s in starts).items()))}")
    print(f"  alive: {dict(sorted(Counter(s['alive'] for s in starts).items()))}")
    print(f"  HP of alive members: median {statistics.median(s['hp'] for s in starts):.2f}, "
          f"below half in {pct(sum(s['hp'] < 0.5 for s in starts), len(starts))}")
    print(f"  top level: median {statistics.median(max(s['levels']) for s in starts)}, "
          f"level spread (max-min) median {statistics.median(max(s['levels']) - min(s['levels']) for s in starts)}")
    print(f"  Pokerus carriers: {dict(sorted(Counter(s['pokerus'] for s in starts).items()))}")
    print(f"  shinies: {dict(sorted(Counter(s['shiny'] for s in starts).items()))}")

    print("\nlevel gap (enemy max level - our max level) by wave decade, all runs:")
    gaps = defaultdict(list)
    for r in records:
        for wave, gap in r["level_gap"].items():
            gaps[10 * (int(wave) // 10)].append(gap)
    for b in sorted(gaps):
        print(f"  waves {b}-{b + 9}: median {statistics.median(gaps[b]):+.0f} (n={len(gaps[b])})")
    fatal_gaps = [r["level_gap"].get(str(r["wave"]), r["level_gap"].get(r["wave"])) for r in deaths]
    fatal_gaps = [g for g in fatal_gaps if g is not None]
    if fatal_gaps:
        print(f"  at the fatal wave: median {statistics.median(fatal_gaps):+.0f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=sorted(AGENTS), default="heuristic")
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-wave", type=int, default=500)
    parser.add_argument("--seed", default="death")
    parser.add_argument("--out", default=str(ROOT / "runs" / "deaths.jsonl"))
    args = parser.parse_args()

    jobs: queue.Queue = queue.Queue()
    for ep in range(args.episodes):
        jobs.put(ep)
    records: list = []
    lock = threading.Lock()
    group = launch_sims(args.workers, request_timeout=120)
    threads = [threading.Thread(target=worker, args=(sim, args, jobs, records, lock)) for sim in group.clients]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    group.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for rec in sorted(records, key=lambda r: r["seed"]):
            f.write(json.dumps(rec) + "\n")
    summarize(records)
    print(f"\nper-run records: {out}")


if __name__ == "__main__":
    main()
