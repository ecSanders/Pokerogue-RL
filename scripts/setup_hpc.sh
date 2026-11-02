#!/usr/bin/env bash
# Set up this repo on a Slurm cluster with NVIDIA GPUs, without root.
#
# Usage: scripts/setup_hpc.sh
#
# Installs Node and pnpm into ~/.local when the cluster has no usable Node, creates .venv with the
# CUDA PyTorch wheel, then runs scripts/setup_sim.sh. Safe to re-run. Loads the modules listed in
# hpc/config.sh first, if that file exists.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NODE_VERSION="24.9.0"
PNPM_VERSION="10.34.5"
PREFIX="$HOME/.local"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

if [[ -f hpc/config.sh ]]; then
    # shellcheck source=/dev/null
    source hpc/config.sh
    if [[ -n "${HPC_MODULES:-}" ]]; then
        step "Loading modules: $HPC_MODULES"
        # shellcheck disable=SC2086
        module load $HPC_MODULES
    fi
fi
export PATH="$PREFIX/bin:$PATH"

# --- Node + pnpm --------------------------------------------------------------
node_ok() {
    command -v node >/dev/null || return 1
    local v; v="$(node --version | tr -d v)"
    [[ "$(printf '%s\n%s\n' "$NODE_VERSION" "$v" | sort -V | head -1)" == "$NODE_VERSION" ]]
}
if node_ok; then
    step "Node $(node --version) found"
else
    step "Installing Node $NODE_VERSION into $PREFIX"
    case "$(uname -m)" in
        x86_64) arch=x64 ;;
        aarch64) arch=arm64 ;;
        *) die "unsupported CPU architecture $(uname -m)" ;;
    esac
    tmp="$(mktemp -d)"
    curl -fsSL "https://nodejs.org/dist/v$NODE_VERSION/node-v$NODE_VERSION-linux-$arch.tar.xz" \
        | tar -xJ -C "$tmp" --strip-components=1
    mkdir -p "$PREFIX"
    cp -r "$tmp"/{bin,include,lib,share} "$PREFIX"/
    rm -rf "$tmp"
    node_ok || die "Node install failed: $(command -v node) is $(node --version)"
fi
if [[ "$(pnpm --version 2>/dev/null || true)" != "$PNPM_VERSION" ]]; then
    step "Installing pnpm $PNPM_VERSION into $PREFIX"
    npm install -g --prefix "$PREFIX" "pnpm@$PNPM_VERSION"
fi

# --- Python + PyTorch (CUDA) ----------------------------------------------------
step "Creating .venv"
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || die "need Python >= 3.10, found $(python3 --version); add a python module to HPC_MODULES"
[[ -x .venv/bin/python ]] || python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
# The default Linux wheel on PyPI is built against CUDA and bundles its runtime.
.venv/bin/pip install --quiet torch numpy

# --- Simulator -------------------------------------------------------------------
PATH="$ROOT/.venv/bin:$PATH" scripts/setup_sim.sh

step "Checking PyTorch"
.venv/bin/python - <<'EOF'
import torch
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU visible here (normal on a login node); jobs get one through SLURM_GRES.")
EOF

step "Done"
echo "Test a simulator: .venv/bin/python scripts/smoke_sim.py"
echo "Submit:           hpc/submit.sh NAME [train_ppo.py args]   (see hpc/README.md)"
