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

## Layout

| Path | Contents |
|---|---|
| `sim/` | Simulator server (TypeScript). Copied into `pokerogue/test/rl/` by `pokerogue_rl.install_sim`, because Vite won't load test files from outside `pokerogue/`. Edit it here, not in the copy. |
| `pokerogue_rl/` | Python package: simulator client, environments, agents |
| `scripts/` | Setup, smoke tests, benchmarks |
| `pokerogue/` | PokeRogue, pinned as a submodule |

## Running a simulator by hand

```bash
.venv/bin/python -m pokerogue_rl.install_sim
cd pokerogue && RL_PORT=5555 pnpm exec vitest run test/rl/sim-server.test.ts --silent
```

Then connect with `SimClient(5555)` or `scripts/smoke_sim.py --port 5555`.
