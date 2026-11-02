#!/usr/bin/env bash
# Submit PPO training to Slurm with the resources in hpc/config.sh.
#
# Usage:
#   hpc/submit.sh NAME [train_ppo.py args...]   one run, written to runs/NAME
#   hpc/submit.sh --sweep FILE                  one array task per line of FILE: "NAME args..."
#
# Extra sbatch flags go in SBATCH_EXTRA, e.g. SBATCH_EXTRA="--time=4:00:00" hpc/submit.sh ...
# Jobs resume from runs/NAME/policy.pt when it exists, so resubmitting a name continues that run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f hpc/config.sh ]] || { echo "Missing hpc/config.sh: cp hpc/config.example.sh hpc/config.sh" >&2; exit 1; }
# shellcheck source=/dev/null
source hpc/config.sh
mkdir -p logs/slurm

flags=(--cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" --requeue --signal=B:USR1@300
       --open-mode=append)  # a requeued job keeps its earlier log
[[ -n "${SLURM_ACCOUNT:-}" ]] && flags+=(--account="$SLURM_ACCOUNT")
[[ -n "${SLURM_PARTITION:-}" ]] && flags+=(--partition="$SLURM_PARTITION")
[[ -n "${SLURM_QOS:-}" ]] && flags+=(--qos="$SLURM_QOS")
[[ -n "${SLURM_GRES:-}" ]] && flags+=(--gres="$SLURM_GRES")
# shellcheck disable=SC2206
flags+=(${SBATCH_EXTRA:-})

if [[ "${1:-}" == "--sweep" ]]; then
    file="${2:?usage: hpc/submit.sh --sweep FILE}"
    # Freeze the sweep: later edits to FILE don't change tasks that have not started yet.
    frozen="logs/slurm/sweep-$(basename "$file" .txt)-$(date +%Y%m%d-%H%M%S).txt"
    grep -vE '^[[:space:]]*(#|$)' "$file" > "$frozen"
    n="$(wc -l < "$frozen")"
    (( n > 0 )) || { echo "no runs in $file" >&2; exit 1; }
    echo "Sweep of $n runs from $file (frozen copy: $frozen)"
    sbatch "${flags[@]}" --job-name="sweep-$(basename "$file" .txt)" --array="0-$((n - 1))" \
        --output="logs/slurm/%x-%A_%a.out" --export=ALL,SWEEP_FILE="$ROOT/$frozen" hpc/job.sh
else
    name="${1:?usage: hpc/submit.sh NAME [train_ppo.py args...]}"
    shift
    sbatch "${flags[@]}" --job-name="$name" --output="logs/slurm/%x-%j.out" hpc/job.sh "$name" "$@"
fi
