"""Check simulator snapshots: loading a wave-start snapshot and replaying the same actions must give the
same trajectory as the run it was taken from. Also checks that a re-seeded load plays.

Usage: python scripts/check_snapshot.py [--port 5555] [--seeds 3] [--waves 3,8,15] [--steps 300]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from check_determinism import first_divergence  # noqa: E402

from pokerogue_rl.agents.heuristic import HeuristicAgent  # noqa: E402
from pokerogue_rl.env import BattleEnv, RewardConfig  # noqa: E402
from pokerogue_rl.sim_client import SimClient  # noqa: E402


def play_from(sim: SimClient, resp: dict, agent: HeuristicAgent, steps: int) -> list[dict]:
    trace = [resp["obs"]]
    while not resp["done"] and len(trace) <= steps:
        resp = sim.step(agent.act(resp["obs"]))
        trace.append(resp["obs"])
    return trace


def replay_from(sim: SimClient, resp: dict, actions: list[int]) -> list[dict]:
    trace = [resp["obs"]]
    for action in actions:
        if resp["done"]:
            break
        resp = sim.step(action)
        trace.append(resp["obs"])
    return trace


def env_check(sim: SimClient, seed: str, wave: int, steps: int) -> bool:
    """BattleEnv.snapshot/reset_from: replaying from a snapshot gives the same observations, rewards
    (including the EXP streak and faint terms, which depend on env state) and strategy measures."""
    reward = RewardConfig(exp_coef=1.0, exp_carry_bonus=1.0, faint_cost=0.5, carry_coef=0.1, pokerus_coef=0.3,
                          double_coef=0.3)
    env = BattleEnv(0, seed_prefix=seed, sim=sim, reward=reward)
    agent = HeuristicAgent()
    agent.reset()
    obs, snap = env.reset(), None
    while snap is None:
        if obs["wave"] >= wave:
            snap = env.snapshot()
        if snap is None:
            obs, _, done, _ = env.step(agent.act(obs))
            if done:
                print(f"env {seed}: run ended before a clean wave-start snapshot at wave >= {wave}")
                return True
    original, actions, end = [(obs, 0.0)], [], None
    while len(actions) < steps:
        actions.append(agent.act(obs))
        obs, r, done, info = env.step(actions[-1])
        if done:
            original.append((None, r))
            end = info.get("episode")
            break
        original.append((obs, r))

    obs = env.reset_from(snap)
    replay, end2 = [(obs, 0.0)], None
    for a in actions:
        obs, r, done, info = env.step(a)
        if done:
            replay.append((None, r))
            end2 = info.get("episode")
            break
        replay.append((obs, r))
    problems = []
    if [o for o, _ in original] != [o for o, _ in replay]:
        problems.append("observations differ: " + str(first_divergence(
            [o for o, _ in original if o], [o for o, _ in replay if o])))
    bad = [i for i, ((_, r1), (_, r2)) in enumerate(zip(original, replay)) if abs(r1 - r2) > 1e-9]
    if bad or len(original) != len(replay):
        problems.append(f"rewards differ at steps {bad[:5]}")
    skip = {"return", "steps", "start_wave"}  # the replayed episode began at the snapshot
    if end and end2 and {k: v for k, v in end.items() if k not in skip} != {k: v for k, v in end2.items() if k not in skip}:
        problems.append(f"episode summaries differ: {end} vs {end2}")
    if end2 and end2["start_wave"] != snap["wave"]:
        problems.append(f"start_wave {end2['start_wave']} != {snap['wave']}")

    # Re-seeded: same snapshot wave, different run afterwards.
    obs = env.reset_from(snap, seed=f"{seed}-reseeded")
    same_start = obs == snap["obs"]
    while obs["wave"] <= snap["wave"] + 1:
        obs, _, done, _ = env.step(agent.act(obs))
        if done:
            break
    print(f"env {seed} wave {snap['wave']}: {len(actions)} steps -> "
          f"{'OK' if not problems else 'FAILED: ' + '; '.join(problems)}"
          f" (re-seeded load starts {'identically' if same_start else 'DIFFERENTLY'})")
    return not problems and same_start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, help="connect to a running simulator instead of launching one")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--waves", default="3,8,15", help="snapshot at the start of these waves")
    parser.add_argument("--steps", type=int, default=300, help="steps to compare after each snapshot")
    parser.add_argument("--env-seeds", type=int, default=2, help="BattleEnv-level checks (rewards, env state)")
    args = parser.parse_args()
    waves = sorted(int(w) for w in args.waves.split(","))

    sim = SimClient(args.port) if args.port else SimClient.launch()
    checked = failures = 0
    with sim:
        for i in range(args.seeds):
            seed = f"snapshot-{i}"
            agent = HeuristicAgent()
            agent.reset()
            # The original run: heuristic actions, a snapshot on arriving at each target wave (at trace
            # index k), continuing until --steps past the last snapshot or the end of the run.
            resp = sim.reset(seed=seed)
            trace, actions, snaps = [resp["obs"]], [], []
            while not resp["done"] and (len(snaps) < len(waves) or len(actions) < snaps[-1][1] + args.steps):
                obs = resp["obs"]
                if len(snaps) < len(waves) and obs["wave"] >= waves[len(snaps)]:
                    snap = sim.snapshot()
                    assert snap["wave"] == obs["wave"], f"snapshot wave {snap['wave']} != obs wave {obs['wave']}"
                    snaps.append((snap, len(trace) - 1))
                actions.append(agent.act(obs))
                resp = sim.step(actions[-1])
                trace.append(resp["obs"])
            if len(snaps) < len(waves):
                print(f"seed {seed}: run ended at wave {trace[-1]['wave']}; "
                      f"{len(snaps)} of the snapshot waves reached")

            # Load each snapshot and replay the original actions from it: the trajectory must match.
            for snap, k in snaps:
                original = trace[k:k + args.steps + 1]
                replay = replay_from(sim, sim.load(snap["session"]), actions[k:k + args.steps])
                diff = first_divergence(original, replay)
                checked += 1
                failures += diff is not None
                print(f"seed {seed} wave {snap['wave']} ({len(snap['session']) // 1024} KB): {len(original) - 1} steps "
                      f"to wave {original[-1]['wave']} -> {'OK' if diff is None else 'DIVERGED at ' + diff}")

            if snaps:
                snap, _ = snaps[-1]
                agent.reset()
                reseeded = play_from(sim, sim.load(snap["session"], seed=f"{seed}-reseeded"), agent, 10_000)
                print(f"seed {seed} wave {snap['wave']} re-seeded: played to wave {reseeded[-1]['wave']}")

        for i in range(args.env_seeds):
            checked += 1
            failures += not env_check(sim, f"snapshot-env-{i}", waves[len(waves) // 2], args.steps)

    print("PASS" if checked and not failures else f"FAIL ({failures}/{checked} snapshot checks failed)")
    sys.exit(0 if checked and not failures else 1)


if __name__ == "__main__":
    main()
