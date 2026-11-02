# Running on the HPC (Slurm, NVIDIA)

Everything here is optional; training works the same locally. The cluster runs many local-sized
jobs side by side (sweeps, seeds), and the checkpoints it writes load at home unchanged.

## One-time setup (login node)

```bash
git clone https://github.com/ecSanders/Pokerogue-RL.git && cd Pokerogue-RL
cp hpc/config.example.sh hpc/config.sh    # fill in account / partition / modules
scripts/setup_hpc.sh                      # Node + pnpm in ~/.local if needed, .venv with CUDA torch, simulator
.venv/bin/python scripts/smoke_sim.py     # one simulator works
```

`runs/` is not in git. Copy the checkpoints a sweep starts from, e.g. from home:

```bash
rsync -a runs/run-exp-v2 runs/bc-run HPC:Pokerogue-RL/runs/
```

If home has a small quota, make `runs` a symlink to scratch space before training.

## Submitting

```bash
hpc/submit.sh v3-test --init runs/run-exp-v2/policy.pt --run-policy --total-steps 200000   # one run
hpc/submit.sh --sweep hpc/sweeps/v3-ent-lr.txt                                           # array job
squeue --me
tail -f logs/slurm/v3-test-*.out
```

- Arguments after the name go straight to `scripts/train_ppo.py`.
- Each sweep line is `NAME args...`. `submit.sh` freezes a copy in `logs/slurm/`, so editing
  the sweep file after submitting doesn't affect queued tasks.
- `--num-envs` defaults to the job's CPUs minus 4, capped at 32. The default 16 CPUs gives 12
  envs, the same as at home. Changing it changes the PPO batch size.
- Resources come from `hpc/config.sh`. Override them for one submission with
  `SBATCH_EXTRA="--time=2:00:00" hpc/submit.sh ...`.

## Time limits and preemption

Jobs run with `--requeue`. Five minutes before the time limit, Slurm signals the job, which
stops training and requeues itself. When the job starts again it finds `runs/NAME/policy.pt` and
passes `--resume`, so it continues from that checkpoint. Checkpoints are written every 10
updates (about 25 s) by write-then-rename, so a kill loses at most that much work. Resubmitting
a finished or cancelled name also resumes it. Use a new name to start fresh.

## Bringing results home

```bash
rsync -a HPC:Pokerogue-RL/runs/v3-* runs/
```

Each run directory has `config.json`, `log.jsonl` and `policy.pt`. Evaluate them locally with
`scripts/eval_agent.py` / `scripts/death_report.py` as usual.
