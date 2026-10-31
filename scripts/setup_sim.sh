#!/usr/bin/env bash
# Prepare the headless PokeRogue simulator in a fresh clone of this repo.
#
# Usage: scripts/setup_sim.sh
#
# Needs Node >= 24.9 and pnpm 10.34.5 on PATH (see README). Safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

step "Checking out pinned PokeRogue commit"
git submodule update --init pokerogue
cd pokerogue

step "Fetching English locales"
git submodule update --init --depth 1 locales

# The game only needs battle animation data headlessly; the full assets repo is several GB.
step "Fetching battle animations (sparse checkout of assets)"
ASSETS_COMMIT="$(git ls-tree HEAD assets | awk '{print $3}')"
if [[ ! -f assets/battle-anims/tackle.json ]]; then
    rmdir assets 2>/dev/null || true
    git clone --filter=blob:none --no-checkout --depth 1 --quiet \
        https://github.com/pagefaultgames/pokerogue-assets.git assets
fi
git -C assets fetch --depth 1 --filter=blob:none --quiet origin "$ASSETS_COMMIT"
git -C assets sparse-checkout set battle-anims
git -C assets checkout --quiet "$ASSETS_COMMIT"

# --ignore-scripts skips PokeRogue's postinstall (git hooks + git config) and dependency build scripts.
step "Installing JavaScript dependencies"
pnpm install --frozen-lockfile --ignore-scripts

step "Installing simulator into the checkout"
grep -qxF "test/rl/" "$(git rev-parse --git-dir)/info/exclude" || echo "test/rl/" >> "$(git rev-parse --git-dir)/info/exclude"
cd "$ROOT"
python3 -m pokerogue_rl.install_sim

step "Done"
echo "Smoke test: .venv/bin/python scripts/smoke_sim.py"
