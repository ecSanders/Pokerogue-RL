"""Train the battle policy with PPO.

Usage:
  python scripts/train_ppo.py --name ppo-v1 --total-steps 2000000
  python scripts/train_ppo.py --name ppo-bc --init runs/bc/policy.pt   # warm start
  python scripts/train_ppo.py --name run-v1 --run-policy --init runs/ppo-v5/policy.pt \
      --run-init runs/bc-run/policy.pt      # the policy also plays rewards, item targets, moves, releases

Writes runs/<name>/log.jsonl (one line per update) and checkpoints runs/<name>/policy.pt.

Multi-node (one model trained by several nodes, see pokerogue_rl/distributed.py and hpc/README.md):
launch one process per node with srun or torchrun; each runs --num-envs simulators and the ranks
average gradients. An update then covers world_size * num_envs * rollout steps.
"""

import argparse
import collections
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pokerogue_rl.distributed import Dist  # noqa: E402
from pokerogue_rl.env import RewardConfig, VecBattleEnv  # noqa: E402
from pokerogue_rl.install_sim import install_sim  # noqa: E402
from pokerogue_rl.features import BATTLE_KINDS  # noqa: E402
from pokerogue_rl.policy import AgentPolicy  # noqa: E402
from pokerogue_rl.run_features import RUN_KINDS, item_bucket  # noqa: E402
from pokerogue_rl.strategy import LURES, gate_clears  # noqa: E402
from pokerogue_rl.rollout import collect  # noqa: E402
from pokerogue_rl.snapshots import SnapshotBank, parse_waves  # noqa: E402
from pokerogue_rl.tracking import Tracker  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="ppo")
    p.add_argument("--num-envs", type=int, default=12)
    p.add_argument("--rollout", type=int, default=128, help="steps per env per update")
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--max-wave", type=int, default=500)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--lr-schedule", choices=["linear", "constant"], default="linear",
                   help="linear: decay to 0 over --total-steps; constant: keep --lr")
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--wave-reward", type=float, default=1.0, help="reward per wave cleared")
    p.add_argument("--team-coef", type=float, default=1.0, help="potential-based team-health shaping")
    p.add_argument("--damage-coef", type=float, default=0.2, help="reward per enemy HP fraction removed")
    p.add_argument("--exp-coef", type=float, default=0.0,
                   help="reward per level gained (with progress to the next) by a Pokemon kept on the team")
    p.add_argument("--exp-carry-bonus", type=float, default=0.0,
                   help="EXP streak bonus: the step's largest gainer earns x(1 + this * min(streak - 1, 10) / 10), "
                        "streak = consecutive EXP steps it was the largest gainer")
    p.add_argument("--faint-cost", type=float, default=0.0,
                   help="cost per party member fainting, times the number of team members fainted counting it "
                        "(everyone still standing faints at game over)")
    p.add_argument("--faint-gameover", type=float, default=None,
                   help="flat cost for losing the run, replacing --faint-cost's game-over wipe (which costs more "
                        "the bigger the team); mid-run faints still escalate")
    p.add_argument("--alive-coef", type=float, default=0.0,
                   help="reward per wave cleared, times the party members not fainted / 6")
    p.add_argument("--level-coef", type=float, default=0.0,
                   help="reward shaping per party level gained (see env.py)")
    p.add_argument("--carry-coef", type=float, default=0.0,
                   help="reward shaping per level gained by the highest-level member")
    p.add_argument("--boss-coef", type=float, default=0.0,
                   help="reward per level the carry is above the expected boss level on arriving at an "
                        "uncatchable boss wave (every 50th; clipped to -10..+5 levels)")
    p.add_argument("--ball-cost", type=float, default=0.0,
                   help="reward cost per ball thrown at a non-boss with a full party (catches earn no EXP)")
    p.add_argument("--double-coef", type=float, default=0.0,
                   help="reward per double battle won (more EXP for the carry)")
    p.add_argument("--lure-bonus", type=float, default=0.0,
                   help="reward for picking a lure, which makes double battles more common")
    p.add_argument("--lure-prior", type=float, default=None,
                   help="set the run policy's logit offset for lures to this at the start (the BC-derived "
                        "policies give lures ~1e-7 probability, so --lure-bonus is never discovered)")
    p.add_argument("--pokerus-coef", type=float, default=0.0,
                   help="reward shaping per party member gaining Pokerus (lost on release)")
    p.add_argument("--double-envs", type=int, default=0,
                   help="envs that play with --double-chance, for double-battle practice")
    p.add_argument("--double-chance", type=float, default=2.0,
                   help="1-in-N wild double roll on normal waves for --double-envs (real game: 8)")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--init", help="checkpoint to start from (e.g. behaviour-cloned policy)")
    p.add_argument("--run-init", help="checkpoint whose run-policy half replaces the --init one (e.g. from "
                                     "bc_pretrain.py --kinds run)")
    p.add_argument("--run-policy", action="store_true",
                   help="the policy also makes run decisions (rewards, item targets, moves, releases)")
    p.add_argument("--resume", action="store_true",
                   help="continue runs/<name>/policy.pt: weights, optimizer, update count and LR schedule")
    p.add_argument("--recycle-every", type=int, default=100,
                   help="updates between simulator restarts (bounds a slow per-episode memory leak)")
    p.add_argument("--recycle-max-wait", type=int, default=50,
                   help="updates to wait for in-progress episodes to finish before a restart cuts them")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-every", type=int, default=10, help="updates between checkpoints")
    p.add_argument("--snapshot-waves", help="start-wave curriculum (see pokerogue_rl/snapshots.py): bank wave-start "
                   "snapshots from this range of waves, e.g. 20-45, and start some episodes from them")
    p.add_argument("--snapshot-frac", type=float, default=0.5,
                   help="fraction of new episodes started from a banked snapshot (once the bank has any)")
    p.add_argument("--snapshot-collect", type=float, default=0.25,
                   help="chance each wave start in --snapshot-waves is offered to the bank")
    p.add_argument("--snapshot-capacity", type=int, default=64, help="snapshots kept per wave")
    p.add_argument("--snapshot-seed-bank", nargs="*", default=[],
                   help="saved banks (runs/NAME/snapshots*.pkl) to fill this run's bank from at the start")
    p.add_argument("--clearml", action="store_true",
                   help="track the run in ClearML (needs ~/clearml.conf): args, log charts, console, checkpoints")
    p.add_argument("--clearml-upload-every", type=int, default=200,
                   help="updates between checkpoint uploads to ClearML (0: only the final one)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    def on_sigterm(signum, frame):
        print(f"received SIGTERM (pid {os.getpid()}, parent {os.getppid()}); stopping", flush=True)
        sys.exit(143)  # run the cleanup in `finally`
    signal.signal(signal.SIGTERM, on_sigterm)
    torch.set_num_threads(1)  # leave CPU cores to the simulators; rollout threads infer in parallel
    dd = Dist()
    device = dd.device(args.device)
    dd.init(device)
    out = ROOT / "runs" / args.name
    if dd.main:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps({**vars(args), "world_size": dd.world}, indent=2))
        install_sim()  # once, before the other ranks' simulators read the shared checkout
    dd.barrier()

    policy = AgentPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    first_update, global_step = 1, 0
    if args.resume:
        ckpt = policy.load_checkpoint(out / "policy.pt", map_location=device)
        optimizer.load_state_dict(ckpt["optimizer"])
        first_update, global_step = ckpt["update"] + 1, ckpt["step"]
        # Drop log rows written after the checkpoint; those updates are redone.
        log = out / "log.jsonl"
        if dd.main and log.exists():
            rows = [r for r in log.read_text().splitlines() if json.loads(r)["update"] <= ckpt["update"]]
            log.write_text("".join(r + "\n" for r in rows))
    elif args.init:
        policy.load_checkpoint(args.init, map_location=device)
    if args.run_init and not args.resume:
        run_init = AgentPolicy()
        run_init.load_checkpoint(args.run_init)
        policy.run.load_state_dict(run_init.run.state_dict())
    if args.lure_prior is not None and not args.resume:
        with torch.no_grad():
            for item in LURES:
                policy.run.item_bias.weight[item_bucket(item)] = args.lure_prior
    dd.broadcast_params(policy)

    # Rank 0 only: the other ranks' episodes reach it through dd.gather. Started after the model and
    # optimizer: Task.init hooks every import and starts a monitor process and a git scan, and rank 0 of
    # exp-coef-v1 (job 1384) hung for 10+ minutes in the Adam constructor's import of torch._dynamo /
    # sympy under those hooks, holding the other rank at the weight broadcast.
    tracker = Tracker(args.clearml and dd.main, args.name, out, {**vars(args), "world_size": dd.world},
                      resume=args.resume, upload_every=args.clearml_upload_every)

    # Rollouts run per-env threads with small-batch inference on a CPU copy (faster than GPU there).
    actor = AgentPolicy()
    actor.load_state_dict(policy.state_dict())

    # Start-wave curriculum: one bank per rank, shared by its envs, saved with the run for --resume.
    bank, bank_file = None, out / (f"snapshots-r{dd.rank}.pkl" if dd.enabled else "snapshots.pkl")
    if args.snapshot_waves:
        bank = SnapshotBank(parse_waves(args.snapshot_waves), capacity=args.snapshot_capacity, seed=dd.rank)
        for path in ([bank_file] if args.resume and bank_file.exists() else []) + [Path(p) for p in args.snapshot_seed_bank]:
            print(f"{'rank ' + str(dd.rank) + ': ' if dd.enabled else ''}snapshot bank: "
                  f"{bank.load(path)} snapshots from {path}", flush=True)
    envs = VecBattleEnv(args.num_envs, bank=bank, snapshot_frac=args.snapshot_frac,
                        snapshot_collect=args.snapshot_collect, double_envs=args.double_envs, double_chance=args.double_chance,
                        seed_prefix=f"{args.name}-r{dd.rank}" if dd.enabled else args.name, max_wave=args.max_wave,
                        reward=RewardConfig(gamma=args.gamma, wave_reward=args.wave_reward, team_coef=args.team_coef,
                                            damage_coef=args.damage_coef, exp_coef=args.exp_coef,
                                            exp_carry_bonus=args.exp_carry_bonus, faint_cost=args.faint_cost,
                                            level_coef=args.level_coef, carry_coef=args.carry_coef,
                                            pokerus_coef=args.pokerus_coef, boss_coef=args.boss_coef,
                                            ball_cost=args.ball_cost, double_coef=args.double_coef,
                                            lure_bonus=args.lure_bonus, faint_gameover=args.faint_gameover,
                                            alive_coef=args.alive_coef),
                        policy_kinds=BATTLE_KINDS + (RUN_KINDS if args.run_policy else ()))
    envs.reset()
    n, t_len = args.num_envs, args.rollout
    batch_size = n * t_len  # this rank's
    num_updates = args.total_steps // (batch_size * dd.world)
    recent = collections.deque(maxlen=100)  # finished episodes that started at wave 1
    recent_snap = collections.deque(maxlen=100)  # finished episodes that started from a banked snapshot
    start = time.time()

    try:
        for update in range(first_update, num_updates + 1):
            for g in optimizer.param_groups:
                g["lr"] = args.lr * (1 - (update - 1) / num_updates) if args.lr_schedule == "linear" else args.lr

            rollout_start = time.time()
            rollout = collect(envs, actor, t_len)
            rollout_time = time.time() - rollout_start
            parts = dd.gather((rollout.episodes, rollout.invalid, rollout.sim_errors, rollout_time))
            for eps, _, _, _ in parts:
                for e in eps:
                    (recent_snap if e.get("start_wave", 1) > 1 else recent).append(e)
            bank_sizes = dd.gather((len(bank), bank.waves_held()) if bank is not None else (0, 0))
            invalid, sim_errors = sum(p[1] for p in parts), sum(p[2] for p in parts)
            rollout_time = max(p[3] for p in parts)
            global_step += batch_size * dd.world

            # GAE. Episodes end inside the rollout (done=1 cuts bootstrapping); bootstrap the tail.
            to_dev = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
            values_t, rewards_t, dones_t = to_dev(rollout.values), to_dev(rollout.rewards), to_dev(rollout.dones)
            next_value = to_dev(rollout.next_values)
            advantages = torch.zeros_like(rewards_t)
            last = torch.zeros(n, device=device)
            for t in reversed(range(t_len)):
                nv = next_value if t == t_len - 1 else values_t[t + 1]
                nonterminal = 1.0 - dones_t[t]
                delta = rewards_t[t] + args.gamma * nv * nonterminal - values_t[t]
                last = delta + args.gamma * args.gae_lambda * nonterminal * last
                advantages[t] = last
            returns = advantages + values_t

            flat = {k: to_dev(v.reshape(batch_size, *v.shape[2:])) for k, v in rollout.feats.items()}
            b_actions = to_dev(rollout.actions).flatten()
            run_frac = flat["is_run"].float().mean().item()
            b_logps, b_adv, b_ret = to_dev(rollout.logps).flatten(), advantages.flatten(), returns.flatten()

            stats = collections.defaultdict(list)
            params = list(policy.parameters())
            mb_size = batch_size // args.minibatches
            for _ in range(args.epochs):
                perm = torch.randperm(batch_size, device=device)
                for i in range(0, batch_size, mb_size):
                    idx = perm[i:i + mb_size]
                    logits, value = policy({k: v[idx] for k, v in flat.items()})
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(b_actions[idx])
                    ratio = (new_logp - b_logps[idx]).exp()
                    adv = b_adv[idx]
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                    pg_loss = torch.max(-adv * ratio, -adv * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                    v_loss = 0.5 * (value - b_ret[idx]).pow(2).mean()
                    entropy = dist.entropy().mean()
                    loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    dd.average_grads(params)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                    optimizer.step()
                    stats["pg_loss"].append(pg_loss.item())
                    stats["v_loss"].append(v_loss.item())
                    stats["entropy"].append(entropy.item())
                    is_run = flat["is_run"][idx]
                    if is_run.any():
                        stats["run_entropy"].append(dist.entropy()[is_run].mean().item())
                    stats["clipfrac"].append(((ratio - 1).abs() > args.clip).float().mean().item())

            actor.load_state_dict({k: v.cpu() for k, v in policy.state_dict().items()})

            waves = [e["wave"] for e in recent]
            row = {
                "update": update, "step": global_step, "time": round(time.time() - start),
                "sps": round(batch_size * dd.world / rollout_time), "lr": optimizer.param_groups[0]["lr"],
                "episodes": len(recent),
                "wave_mean": round(float(np.mean(waves)), 2) if waves else None,
                "wave_median": float(np.median(waves)) if waves else None,
                "return_mean": round(float(np.mean([e["return"] for e in recent])), 3) if recent else None,
                # Strategy measures over recent episodes (see pokerogue_rl/strategy.py).
                **{k: round(float(np.mean([e[k] for e in recent])), 3) if recent else None
                   for k in ("double_frac", "pickup_frac", "lures", "ball_frac", "carry_gap", "pokerus")},
                "boss_gap": round(float(np.mean(g)), 2) if (g := [e["boss_gap"] for e in recent
                                                                  if e.get("boss_gap") is not None]) else None,
                # Clear rate at each uncatchable-boss gate (every 50th wave) among recent episodes that
                # reached it, and how many reached it: mean wave hides that one boss decides the run.
                **{k: v for g, (c, n) in gate_clears([(e["wave"], e["reason"]) for e in recent]).items()
                   for k, v in ((f"gate{g}", round(c / n, 3)), (f"gate{g}_n", n))},
                # Curriculum: the bank (summed over ranks) and episodes started from it. The stats above
                # cover only episodes from wave 1, so they stay comparable with runs without a curriculum.
                **({"bank": sum(b[0] for b in bank_sizes), "bank_waves": max(b[1] for b in bank_sizes),
                    "snap_episodes": len(recent_snap),
                    "snap_waves_gained": round(float(np.mean([e["wave"] - e["start_wave"] for e in recent_snap])), 2)
                    if recent_snap else None,
                    **{k: v for g, (c, n) in gate_clears([(e["wave"], e["reason"], e["start_wave"])
                                                           for e in recent_snap]).items()
                       for k, v in ((f"snap_gate{g}", round(c / n, 3)), (f"snap_gate{g}_n", n))}}
                   if bank is not None else {}),
                "invalid": invalid,
                "sim_errors": sim_errors,
                "run_frac": round(run_frac, 3),
                **{k: round(float(np.mean(v)), 4) for k, v in stats.items()},
            }
            if dd.main:
                with (out / "log.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
                tracker.log(row, update)

            if dd.main and (update % args.save_every == 0 or update == num_updates):
                # Write then rename, so a job killed mid-save (e.g. Slurm preemption) keeps the last good checkpoint.
                torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "update": update,
                            "step": global_step, "args": vars(args)}, out / "policy.pt.tmp")
                (out / "policy.pt.tmp").replace(out / "policy.pt")
            if bank is not None and update % (10 * args.save_every) == 0:
                bank.save(bank_file)  # every rank its own; tens of MB, so less often than the checkpoint
                tracker.checkpoint(out / "policy.pt", update, final=update == num_updates)
            if update % args.recycle_every == 0 and update < num_updates and not envs.draining:
                envs.start_drain()
                drain_start = update
            if envs.draining and update < num_updates and (envs.drained or update - drain_start >= args.recycle_max_wait):
                cut = envs.recycle()
                print(f"{'rank ' + str(dd.rank) + ': ' if dd.enabled else ''}recycled simulators after waiting {update - drain_start} updates"
                      + (f"; cut {cut} unfinished episodes" if cut else ""), flush=True)
    finally:
        # Also on Ctrl-C or SIGTERM (Slurm requeue), or the simulator processes outlive the trainer.
        envs.close()
        if bank is not None:
            bank.save(bank_file)
        dd.close()
        tracker.close()


if __name__ == "__main__":
    main()
