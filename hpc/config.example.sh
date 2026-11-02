# Cluster settings for hpc/submit.sh and hpc/job.sh.
# Copy to hpc/config.sh (git-ignored) and fill in: cp hpc/config.example.sh hpc/config.sh

SLURM_ACCOUNT=""          # --account; leave empty if the cluster has no accounts
SLURM_PARTITION=""        # --partition; empty = cluster default
SLURM_QOS=""              # --qos; empty = default
SLURM_GRES="gpu:1"        # GPU per job; "" for CPU-only (the GPU only runs the PPO update)

# Per-job resources. These match the local machine (12 envs on 16 cores), so sweep results
# transfer back home unchanged. More envs means a bigger PPO batch, which changes the
# hyperparameters a sweep is measuring.
CPUS=16                   # simulators need about one core each, plus a few for the trainer
MEM="32G"                 # ~1.5-2 GB per simulator; 12 envs peaked near 20 GB locally
TIME="24:00:00"           # wall time per job; preempted or timed-out runs requeue and resume

# Environment modules loaded before setup and in every job, space-separated. Leave empty if
# the cluster has no `module` command or setup_hpc.sh installs everything into ~/.local.
HPC_MODULES=""            # e.g. "python/3.12 cuda/12.4"
