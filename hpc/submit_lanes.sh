#!/usr/bin/env bash
# Submit a plan of runs as "lanes": each lane is a chain of single-node jobs, each starting when the
# previous one in its lane ends (afterany, so one failure doesn't stall the lane). With L lanes, at most
# L nodes are in use at once, which keeps within a per-user node limit without Slurm QOS.
#
# Usage:
#   hpc/submit_lanes.sh PLAN                      submit (prints the job ids)
#   DRY_RUN=1 hpc/submit_lanes.sh PLAN            show what would be submitted
#   LANE_AFTER="A=1406_2 B=1406_3" hpc/submit_lanes.sh PLAN   first job of each lane waits for these jobs
#
# PLAN lines: "LANE NAME train_ppo.py-args..."; blank lines and # comments are ignored. Lines of a lane run
# in file order. Resources come from hpc/config.sh plus SBATCH_EXTRA (e.g. --cpus-per-task, --exclude).
# Each run writes runs/NAME and resumes from runs/NAME/policy.pt if it exists, like hpc/submit.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
plan="${1:?usage: hpc/submit_lanes.sh PLAN}"
mkdir -p logs/slurm
frozen="logs/slurm/lanes-$(basename "$plan" .txt)-$(date +%Y%m%d-%H%M%S).txt"
grep -vE '^[[:space:]]*(#|$)' "$plan" > "$frozen"
echo "Plan $plan (frozen copy: $frozen)"

declare -A last
for pair in ${LANE_AFTER:-}; do
    last["${pair%%=*}"]="${pair#*=}"
done

while read -r lane name args; do
    dep=""
    [[ -n "${last[$lane]:-}" ]] && dep="--dependency=afterany:${last[$lane]}"
    if [[ -n "${DRY_RUN:-}" ]]; then
        echo "lane $lane: $name ${dep:-(no dependency)} :: $args"
        last[$lane]="<$name>"
        continue
    fi
    # shellcheck disable=SC2086
    out="$(SBATCH_EXTRA="${SBATCH_EXTRA:-} $dep" hpc/submit.sh "$name" $args)"
    id="$(grep -oE '[0-9]+$' <<< "$out")"
    echo "lane $lane: $name -> job $id ${dep:+($dep)}"
    last[$lane]="$id"
done < "$frozen"
