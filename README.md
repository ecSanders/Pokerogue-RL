# Pokerogue RL

Reinforcement learning agent for PokeRogue endless mode. Target: wave 1000.

The game runs headlessly inside PokeRogue's own Vitest environment (`sim/sim-server.test.ts`),
and Python talks to it over a localhost JSON socket (`pokerogue_rl/sim_client.py`).

## Setup

1. **Node.js ≥ 24.9 and pnpm 10.34.5.** Without root, install the Node tarball from nodejs.org
   into `~/.local`, then `npm install -g --prefix ~/.local pnpm@10.34.5`.
2. **Python environment with PyTorch.** On an AMD GPU: `scripts/setup_rocm.sh`.
   Elsewhere, create `.venv` and install the PyTorch build for your hardware.
3. **Simulator:** `scripts/setup_sim.sh`. This checks out the pinned PokeRogue commit, its English
   locales, the battle animation data (a sparse checkout, not the multi-GB assets repo), and
   JavaScript dependencies.

Check it works:

```bash
.venv/bin/python scripts/smoke_sim.py          # random agent, reports steps/s
.venv/bin/python scripts/check_determinism.py  # same seed + actions => same game
```

Evaluate an agent over seeded endless runs with parallel simulators, or see how its runs end:

```bash
.venv/bin/python scripts/eval_agent.py --agent heuristic --episodes 40 --workers 8 --max-wave 200
.venv/bin/python scripts/death_report.py --agent heuristic --episodes 96 --workers 12
```

## Training the battle policy

The learned policy makes battle decisions (`command`, `target`, `faint_switch`); the heuristic
plays everything else. Warm-start from the heuristic with behaviour cloning, then train with PPO:

```bash
.venv/bin/python scripts/bc_pretrain.py --steps 150000 --name bc
.venv/bin/python scripts/train_ppo.py --name ppo-bc --init runs/bc/policy.pt --total-steps 2000000
POLICY_CHECKPOINT=runs/ppo-bc/policy.pt .venv/bin/python scripts/death_report.py --agent policy
```

Training logs one JSON line per update to `runs/<name>/log.jsonl`. The reward is +1 per wave
cleared, plus potential-based team-health shaping and a small bonus for damage dealt (see
`pokerogue_rl/env.py`).

Training on a Slurm cluster (sweeps, more seeds) is covered in [`hpc/README.md`](hpc/README.md).

## Decisions

Each observation has a `kind` and a `legal` mask over that kind's actions:

| kind | actions |
|---|---|
| `command` | 0–3 use move slot, 4+k switch to party slot k, 10+b throw ball type b |
| `target` | doubles, single-target move: battler index 0–1 our side, 2–3 enemies |
| `faint_switch` | party slot to send in after a faint |
| `reward` | index into `obs.options`: free rewards, shop items, reroll, skip |
| `reward_target` | party slot that receives the chosen item |
| `learn_move` | 0–3 replace that slot with `obs.learn.move`, 4 don't learn |
| `release` | party full after a catch: 0–5 release that slot, 6 let the catch go |

Biome choice and other prompts are scripted defaults (counted in `info.autoInputs`).
Mystery encounters are disabled.

## Layout

| Path | Contents |
|---|---|
| `sim/` | Simulator server (TypeScript). Copied into `pokerogue/test/rl/` by `pokerogue_rl.install_sim`, because Vite won't load test files from outside `pokerogue/`. Edit it here, not in the copy. |
| `pokerogue_rl/` | Python package: simulator client, environments, agents |
| `scripts/` | Setup, smoke tests, benchmarks |
| `pokerogue/` | PokeRogue, pinned as a submodule |
| `docs/` | Notes, e.g. [`pokerogue-mechanics.md`](docs/pokerogue-mechanics.md): endless-mode rules (level cap, bosses, EXP) read from the source |

## Running a simulator by hand

```bash
.venv/bin/python -m pokerogue_rl.install_sim
cd pokerogue && RL_PORT=5555 pnpm exec vitest run test/rl/sim-server.test.ts --silent
```

Then connect with `SimClient(5555)` or `scripts/smoke_sim.py --port 5555`.
