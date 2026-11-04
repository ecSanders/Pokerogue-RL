#!/usr/bin/env bash
# Slurm job body: one PPO run. Submit with hpc/submit.sh, not sbatch directly.
#
# Arguments are "NAME [train_ppo.py args...]", or come from line SLURM_ARRAY_TASK_ID of SWEEP_FILE.
# Resumes from runs/NAME/policy.pt if it exists. Five minutes before the time limit (or on
# preemption) Slurm sends USR1; the job then stops training and requeues itself to continue.
# With several nodes (submit.sh --nodes N) srun starts one trainer per node and they train one model.
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?run through hpc/submit.sh}"
cd "$ROOT"
# shellcheck source=/dev/null
source hpc/config.sh
if [[ -n "${HPC_MODULES:-}" ]]; then
    # shellcheck disable=SC2086
    module load $HPC_MODULES
fi
export PATH="$HOME/.local/bin:$ROOT/.venv/bin:$PATH"

if [[ -n "${SWEEP_FILE:-}" ]]; then
    line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SWEEP_FILE")"
    # shellcheck disable=SC2086
    set -- $line
fi
name="${1:?no run name}"
shift
args=("$@")

# Default the env count to the job's cores (4 left for the trainer), at most 32 per run.
if [[ " ${args[*]} " != *" --num-envs "* ]]; then
    envs=$(( ${SLURM_CPUS_PER_TASK:-16} - 4 ))
    (( envs > 32 )) && envs=32
    args+=(--num-envs "$envs")
fi
if [[ -f "runs/$name/policy.pt" ]]; then
    echo "Resuming runs/$name"
    args+=(--resume)
fi

nodes="${SLURM_NNODES:-1}"
echo "job ${SLURM_JOB_ID} on $(hostname) ($nodes node(s)): train_ppo.py --name $name ${args[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"
launch=(python)
if (( nodes > 1 )); then
    # Rendezvous on the first node; the port is derived from the job id so jobs sharing a node differ.
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
    MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 20000 ))
    export MASTER_ADDR MASTER_PORT
    echo "distributed: $nodes ranks, rendezvous $MASTER_ADDR:$MASTER_PORT"
    # Recent Slurm doesn't pass the job's --cpus-per-task on to srun; --kill-on-bad-exit stops the
    # other ranks (stuck in an all-reduce) if one fails.
    launch=(srun --ntasks="$nodes" --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}"
            --kill-on-bad-exit=1 python)
fi

requeue() {
    echo "Time limit or preemption: stopping and requeueing"
    kill -TERM "$train_pid" 2>/dev/null || true
    wait "$train_pid" || true
    scontrol requeue "$SLURM_JOB_ID"  # an array task has its own job id
    exit 0
}
trap requeue USR1

"${launch[@]}" scripts/train_ppo.py --name "$name" "${args[@]}" &
train_pid=$!
wait "$train_pid"
