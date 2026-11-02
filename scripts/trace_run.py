"""Replay one seeded run with an agent and print every decision from a given wave on.

Usage:
  POLICY_CHECKPOINT=runs/run-v2/policy.pt python scripts/trace_run.py --agent policy --seed death-0 --from-wave 48
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pokerogue_rl.agents import AGENTS  # noqa: E402
from pokerogue_rl.sim_client import SimClient  # noqa: E402


def mon(p: dict) -> str:
    flags = "".join(c for c, on in (("B", p["boss"]), ("*", p["shiny"]), ("x", p["fainted"])) if on)
    return f"{p['species']}{flags} L{p['level']} bst{p['bst']} {p['hp']}/{p['maxHp']} st{p['status']}"


def moves(p: dict) -> str:
    out = []
    for i, m in enumerate(p["moves"]):
        dmg = ",".join(f"{d:.2f}" for d in (m.get("dmg") or []) if d is not None)
        out.append(f"{i}:{m['id']}/t{m['type']}/p{m['power']}/pp{m['pp']}" + (f"/dmg[{dmg}]" if dmg else ""))
    return " ".join(out)


def describe(obs: dict, action: int) -> str:
    kind = obs["kind"]
    head = f"w{obs['wave']} t{obs['turn']} {kind:13s} -> {action}"
    if kind == "command":
        actor = obs["party"][obs["actor"]]
        enemies = " | ".join(mon(e) for e in obs["enemies"])
        legal = [i for i, ok in enumerate(obs["legal"]) if ok]
        return f"{head}  actor {mon(actor)}  [{moves(actor)}]  vs {enemies}  legal {legal}"
    if kind == "faint_switch":
        return f"{head}  options " + " | ".join(
            f"{i}:{mon(p)}" for i, p in enumerate(obs["party"]) if obs["legal"][i])
    if kind == "reward":
        opts = [f"{i}:{o['id']}" + (f"(${o['cost']})" if o["cost"] > 0 else "") + ("" if o["legal"] else "!")
                for i, o in enumerate(obs["options"])]
        return f"{head}  money {obs['money']}  " + " ".join(opts)
    if kind == "reward_target":
        return f"{head}  item {obs['item']['id']}"
    if kind == "learn_move":
        return f"{head}  slot {obs['learn']['slot']} learns {obs['learn']['move']['id']}"
    if kind == "release":
        return f"{head}  caught {mon(obs['caught'])}"
    return head


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="policy", choices=AGENTS)
    p.add_argument("--seed", required=True)
    p.add_argument("--from-wave", type=int, default=1)
    p.add_argument("--max-wave", type=int, default=500)
    args = p.parse_args()

    agent = AGENTS[args.agent]()
    sim = SimClient.launch(request_timeout=120)
    try:
        resp = sim.reset(seed=args.seed, max_wave=args.max_wave)
        agent.reset()
        team_shown = None
        while not resp["done"]:
            obs = resp["obs"]
            action = agent.act(obs)
            if obs["wave"] >= args.from_wave:
                if team_shown != obs["wave"]:
                    team_shown = obs["wave"]
                    print(f"--- wave {obs['wave']} money {obs['money']} balls {obs['balls']}")
                    for i, q in enumerate(obs["party"]):
                        print(f"    {i}: {mon(q)}  [{moves(q)}]")
                print(describe(obs, action))
            resp = sim.step(action)
        print(f"=== {resp['info'].get('reason')} at wave {resp['obs']['wave']}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
