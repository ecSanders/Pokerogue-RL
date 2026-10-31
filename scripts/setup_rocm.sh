#!/usr/bin/env bash
# Set up ROCm + PyTorch for training on the discrete AMD GPU (Ubuntu 26.04).
#
# Usage: scripts/setup_rocm.sh [--full] [--override]
#   --full      also install the full `rocm` metapackage (dev tools + tests, several GB)
#   --override  set HSA_OVERRIDE_GFX_VERSION=11.0.0 (only if PyTorch fails to use the GPU)
#
# Uses Ubuntu's own ROCm packages (universe) and the matching PyTorch rocm7.1 wheel.
# The PyTorch wheel bundles its own ROCm runtime, so only the tools are needed from apt.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
TORCH_INDEX="https://download.pytorch.org/whl/rocm7.1"

FULL=0
OVERRIDE=0
for arg in "$@"; do
    case "$arg" in
        --full) FULL=1 ;;
        --override) OVERRIDE=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33mWARNING: %s\033[0m\n' "$*"; }

# --- 0. Sanity checks -------------------------------------------------------
step "Checking system"
. /etc/os-release
if [[ "$VERSION_ID" != "26.04" ]]; then
    warn "Written for Ubuntu 26.04, found $VERSION_ID. Package versions may differ."
fi
# Capture output first: `cmd | grep -q` under pipefail fails spuriously when
# grep exits early and cmd gets SIGPIPE.
LSPCI="$(lspci)"
if ! grep -qiE 'vga.*(Navi 3|RX 7)' <<< "$LSPCI"; then
    warn "No RDNA3 discrete GPU found in lspci; continuing anyway."
fi
[[ -e /dev/kfd ]] || { echo "/dev/kfd missing: amdgpu kernel driver not loaded." >&2; exit 1; }

# Ask for the sudo password once, up front.
if ! sudo -n true 2>/dev/null; then
    if [[ ! -t 0 ]]; then
        echo "sudo needs a password but there's no terminal to ask for it." >&2
        echo "Run this script from a regular terminal window." >&2
        exit 1
    fi
    sudo -v
fi

# --- 1. ROCm tools from Ubuntu's repos -------------------------------------
step "Installing ROCm tools (sudo)"
PKGS=(rocminfo rocm-smi python3-venv python3-pip)
(( FULL )) && PKGS+=(rocm)
sudo apt-get update
sudo apt-get install -y "${PKGS[@]}"

# --- 2. Device permissions --------------------------------------------------
# A local desktop login already gets access via logind ACLs, but SSH/tmux/cron
# sessions need group membership.
step "Adding $USER to render and video groups"
GROUPS_NOW="$(id -nG "$USER")"
if grep -qw render <<< "$GROUPS_NOW" && grep -qw video <<< "$GROUPS_NOW"; then
    echo "Already a member."
    NEED_RELOGIN=0
else
    sudo usermod -aG render,video "$USER"
    NEED_RELOGIN=1
fi

# --- 3. Pick the discrete GPU ------------------------------------------------
# HIP numbers GPU agents in rocminfo order. Skip the Raphael iGPU (gfx1036),
# which PyTorch doesn't support and which can break device selection.
step "Locating discrete GPU"
mapfile -t GFX < <(rocminfo | awk '/^ *Name: *gfx/ {print $2}')
echo "GPU agents: ${GFX[*]:-none}"
DGPU_INDEX=""
for i in "${!GFX[@]}"; do
    if [[ "${GFX[$i]}" == gfx110[0-2] ]]; then DGPU_INDEX=$i; DGPU_ARCH=${GFX[$i]}; break; fi
done
if [[ -z "$DGPU_INDEX" ]]; then
    echo "Could not find an RDNA3 (gfx110x) GPU in rocminfo output." >&2
    exit 1
fi
echo "Using HIP device $DGPU_INDEX ($DGPU_ARCH)"

# --- 4. Python venv + PyTorch ------------------------------------------------
step "Creating venv at $VENV_DIR"
python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install torch --index-url "$TORCH_INDEX"
pip install numpy

# Export GPU selection whenever the venv is activated.
ACTIVATE="$VENV_DIR/bin/activate"
sed -i '/# >>> rocm-env >>>/,/# <<< rocm-env <<</d' "$ACTIVATE"
{
    echo "# >>> rocm-env >>>"
    echo "export HIP_VISIBLE_DEVICES=$DGPU_INDEX"
    (( OVERRIDE )) && echo "export HSA_OVERRIDE_GFX_VERSION=11.0.0"
    echo "# <<< rocm-env <<<"
} >> "$ACTIVATE"
. "$ACTIVATE"

# --- 5. Verify -----------------------------------------------------------------
step "Verifying PyTorch sees the GPU"
python "$PROJECT_DIR/scripts/check_gpu.py"

step "Done"
echo "Activate with: source .venv/bin/activate"
if (( NEED_RELOGIN )); then
    echo "Log out and back in so the render/video group change applies to SSH and tmux sessions."
fi
