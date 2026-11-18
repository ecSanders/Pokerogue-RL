# ClearML metrics

What each scalar in a ClearML task (project `Pokerogue-RL`) means. Every scalar is a key of the
per-update row that `scripts/train_ppo.py` writes to `runs/<name>/log.jsonl`; `pokerogue_rl/tracking.py`
forwards the numeric keys to ClearML. `scripts/clearml_upload.py` replays an existing `log.jsonl`
the same way for runs trained without `--clearml`.

New to Pokémon or to reinforcement learning? Read the two primers below first. They explain the
terms the metric tables use. If you know both, skip to [How the reporting works](#how-the-reporting-works).

## Primer: Pokémon and PokeRogue

**Pokémon battles.** You own a *party* of up to 6 creatures (Pokémon). Each has a *level*, hit points
(HP), up to 4 *moves* (attacks or other effects), an *ability* (a passive trait), and can hold one or more
*items*. Battles are turn-based: each turn every Pokémon on the field picks an action at the same time
(use a move, switch to another party member, use an item, or throw a *ball* to catch a wild Pokémon). A
Pokémon whose HP reaches 0 *faints* and can't fight until healed. Moves and Pokémon have *types*
(Fire, Water, …) that make some attacks much stronger or weaker against some targets. Winning a battle
gives *EXP* (experience points) to the Pokémon that took part; enough EXP raises its level, and higher
levels mean higher stats. Most battles are 1 vs 1; a *double battle* is 2 vs 2.

**PokeRogue** is a free fan-made browser game that turns Pokémon into a roguelike: one run, no saves
to fall back on, and losing ends it. We play its **endless mode**:

- A run is a sequence of **waves**. Each wave is one battle. Losing every party member in a battle
  is **game over**, and the run's score is the wave it ended on. The project's goal is wave 1000.
- After each wave you **pick a reward** (an item, healing, EXP boosts, …) and can buy items in a shop.
  Waves come in 5-wave *biomes* (areas); the party is fully healed every 10 waves.
- **Enemies get stronger every wave**, faster than linearly (enemy level ≈ `1 + w/2 + (w/25)²` at wave
  `w`). There is also a **level cap** near the current enemy level: a Pokémon at the cap earns no
  battle EXP, so the team can't simply out-grind the curve.
- **Bosses** come every 10th wave: a stronger enemy with several health bars (*shields*). Most can be
  caught once their shields break. Every **50th wave** is the End biome, where the boss **can't be
  caught** and has to be beaten. These are the **gates** in the metrics, and they are where most
  runs die. Wave 250 is Eternatus, an especially hard boss.
- **Catching** wild Pokémon adds them to the party (or replaces a member). It costs a turn and a ball,
  and gives no EXP.

Terms the strategy metrics use:

| Term | Meaning |
|---|---|
| **Carry** | One Pokémon that gets most of the items and EXP so it stays at or above boss level. Measured by `carry_gap`, the carry's level minus the party average. |
| **Pickup** | An ability that sometimes finds a free item after a battle. |
| **Pokérus** | A rare, harmless "virus" that gives ×1.5 EXP and spreads to party neighbours. The ideal is all 6 members having it. |
| **Lure** | A reward item that makes double battles more likely. Doubles give more EXP per wave. |

[`pokerogue-mechanics.md`](pokerogue-mechanics.md) has the exact rules, with source references.

## Primer: reinforcement learning

**Reinforcement learning (RL)** trains a program by trial and error rather than from examples of
correct answers. The vocabulary, mapped onto this project:

| Term | Meaning here |
|---|---|
| **Agent / policy** | The neural network that plays. Given what it sees, it outputs a probability for each possible action, and an action is sampled from those probabilities. |
| **Environment** | The game. We run real PokeRogue code headlessly in a simulator (`sim/`) and talk to it from Python. |
| **Observation** | What the policy sees at a decision: party, enemy, moves, HP, wave, offered rewards, and so on. |
| **Action** | One decision: which move to use, whom to target, which reward to take, and so on. The learned policy makes battle decisions, plus run decisions such as reward picks when `--run-policy` is on. A hand-written heuristic plays the rest. |
| **Step** | One decision by the policy. Training lengths (`--total-steps`) count these. |
| **Episode** | One full run, from its starting wave to game over (or a `--max-wave` cap). |
| **Reward** | A number the environment returns after each step, saying how good that was. We design it, for example +1 per wave cleared, a bonus for EXP, or a cost for fainting. The reward settings are in each task's args. The agent learns to maximise total reward, so a badly designed reward teaches the wrong habits. |
| **Return** | The sum of rewards over an episode (or, for learning, from some step onward, with later rewards discounted by `gamma` per step). |
| **Value function** | A second output of the network, the *critic*, that predicts the return from the current state. It doesn't play. It tells the learner whether an outcome was better or worse than expected (the *advantage*). |
| **Entropy** | How spread out the policy's action probabilities are. High means it still explores; near zero means it always does the same thing. |
| **Curriculum** | Making training easier to learn from by controlling what it practises. Here, some episodes start from saved mid-run *snapshots* just before where the policy usually dies, instead of from wave 1. |

**PPO** (Proximal Policy Optimization) is the training algorithm. It repeats one cycle, called an
**update**:

1. **Collect a rollout.** Run the current policy in many games in parallel (`--num-envs`) for a fixed
   number of steps each (`--rollout`), recording observations, actions, rewards and the critic's
   predictions.
2. **Learn from it.** Compute each action's advantage, then take several passes (*epochs*) of
   gradient descent over shuffled chunks (*minibatches*) of that data. Actions that turned out better
   than expected become more likely, worse ones less likely. A **clip** limits how far one update can
   move the policy, because large jumps tend to wreck it.
3. Throw the data away and repeat with the improved policy.

Every metric below is logged once per update. A run is typically thousands of updates long, and
episodes are long (dozens to hundreds of waves), so one episode spans many updates.

**What "learning" looks like on the charts:** `wave_mean` and gate clear rates trend up, `entropy`
falls slowly (not to zero), `clipfrac` stays modest, and `invalid` and `sim_errors` stay near zero.
The loss values themselves don't need to go down. In RL the data changes as the policy changes, so a
flat or even rising loss is normal; judge a run by its outcome metrics.

## How the reporting works

- **x-axis is the PPO update number** (absolute, also across Slurm requeues and `--resume`: the
  tracker subtracts ClearML's initial iteration offset). One update = `num_envs × rollout × world_size`
  env steps. `update` and `step` themselves are not reported; `step` is in `log.jsonl`.
- **Subsampled.** Only every `--clearml-report-every` update is sent (default 10) because of the
  ClearML free-tier metrics quota. `log.jsonl` has every update.
- **Missing points.** Keys whose value is `None` (e.g. `wave_mean` before any episode finished) or that
  are absent from a row (curriculum keys without `--snapshot-waves`, `gateN` before any episode reached
  wave N) are skipped, so a chart can start late or have gaps.
- **Rank 0 only.** In multi-GPU runs the row is written by the main rank. Episode stats and bank
  sizes are gathered from all ranks; PPO loss stats and `run_frac` are rank 0's own minibatches
  (gradients are averaged, so they are representative).

### Chart grouping

| Chart | Series |
|---|---|
| `waves` | `wave_mean`, `wave_median` |
| `loss` | `pg_loss`, `v_loss` |
| `entropy` | `entropy`, `run_entropy` |
| `gate clear rate` | `wave 50`, `wave 100`, … (from `gate50`, `gate100`, …) |
| `gate reached (episodes)` | `wave 50`, `wave 100`, … (from `gate50_n`, …) |
| *one chart per key* | everything else |

Note: `snap_gateN` / `snap_gateN_n` don't match the grouping rule (it checks the `gate` prefix), so
each of those gets its own chart.

## Episode window

Most outcome metrics are averages over **`recent`: the last 100 finished episodes that started at
wave 1**. Episodes started from a curriculum snapshot go to a separate window (`recent_snap`, also 100)
and only feed the `snap_*` metrics, so the main numbers stay comparable with runs without a
curriculum. Because it is a sliding window of episodes, not of updates, these metrics lag the policy
and are noisy when episodes are long and few finish per update.

## Throughput and schedule

| Key | Meaning |
|---|---|
| `time` | Wall seconds since **this process** started. Resets on a requeue/resume, so it is not total training time. |
| `sps` | Env steps per second of the rollout phase: `batch × world_size / rollout_time` (slowest rank). Excludes the PPO update, so it overstates end-to-end throughput. |
| `lr` | Learning rate for this update (`--lr-schedule linear` anneals it to 0 over `total_steps`). |

## Outcome (episodes from wave 1)

| Key | Meaning |
|---|---|
| `episodes` | Episodes in the window (caps at 100). Below 100 the averages are over fewer runs. |
| `wave_mean` | Mean final wave. The headline progress metric. |
| `wave_median` | Median final wave; less pulled by a few long streaks. |
| `return_mean` | Mean undiscounted episode return in raw reward units (before `--reward-norm`). Only comparable between runs with the same reward config. |
| `gateN` | Clear rate at gate wave N (every 50th wave: the uncatchable End-biome boss): cleared / reached. A run *reached* N if it got past N or lost on N; one cut off on N for another reason counts for neither; a run capped by `--max-wave` at N counts as cleared. See `strategy.gate_clears`. |
| `gateN_n` | Episodes in the window that reached gate N; read `gateN` together with this (a rate over 2 episodes means little). |

## Strategy measures

Per-episode summaries from `StrategyTracker` (`pokerogue_rl/strategy.py`), averaged over the window.
They track the endless-run habits we want the policy to pick up.

| Key | Per-episode value | Want |
|---|---|---|
| `double_frac` | Share of waves that were double battles. | higher (more EXP) |
| `pickup_frac` | Share of waves with a Pickup user in the party. | higher |
| `lures` | Lures taken from reward screens (Lure / Super / Max). | higher |
| `ball_frac` | Share of battle commands that threw a ball. | lower |
| `carry_gap` | At the end: highest party level minus mean party level. | positive (a carry) |
| `pokerus` | Party members with Pokérus at the end (0–6). | 6 |
| `boss_gap` | Carry level minus expected boss level at the last gate wave reached (on arrival). Averaged only over episodes that reached wave 50. | ≥ 0 |

## Curriculum (only with `--snapshot-waves`)

| Key | Meaning |
|---|---|
| `bank` | Snapshots held, summed over ranks. |
| `bank_waves` | Distinct waves held (max over ranks). |
| `bank_deepest` | Deepest wave held (max over ranks). |
| `death_p50`, `death_p90` | Median / 90th percentile of the waves where recent wave-1 episodes lost (last 1000 deaths, rank 0's bank). Present only once there are deaths. In deaths mode this is where starts are aimed. |
| `snap_episodes` | Episodes in the snapshot window (caps at 100). |
| `snap_waves_gained` | Mean waves gained by snapshot episodes (final wave − start wave). |
| `snap_gateN`, `snap_gateN_n` | Like `gateN`, for snapshot episodes; an episode only counts for gates after its start wave. Shows whether practice at a wall is working. |

## Health

| Key | Meaning |
|---|---|
| `invalid` | Actions the simulator rejected as invalid this update (all ranks). Should be ~0; a rise points at an action-mask bug. |
| `sim_errors` | Episodes cut short this update because a simulator crashed or hung and was replaced (all ranks). Occasional ones are fine; a steady rate needs a look at the console log. |
| `run_frac` | Share of this update's samples that were run-level decisions (`RUN_KINDS`: reward picks and their targets, move learning, releases) rather than battle commands. |
| `reward_std` | Running std of the discounted return that rewards are divided by (only with `--reward-norm`). Grows as the policy reaches later waves if rewards scale with the wave. |

## PPO optimisation

Means over every minibatch of every epoch in the update.

| Key | Meaning |
|---|---|
| `pg_loss` | Clipped policy-gradient loss (advantages normalised per minibatch, so its scale is ~fixed). |
| `v_loss` | Value loss, `0.5 × (V − return)²`, on the (normalised, if `--reward-norm`) reward scale. |
| `entropy` | Mean policy entropy over all decisions. A fast collapse means premature determinism. |
| `run_entropy` | Entropy over run-level decisions only (missing in updates whose minibatches had none). |
| `clipfrac` | Share of samples whose probability ratio left `[1 − clip, 1 + clip]`. Persistently high (≳0.3) means updates are too large. |

## Also in the task

- **Configuration → args**: all `train_ppo.py` arguments plus `world_size`.
- **Console**: the same per-update line printed to stdout.
- **Models**: `policy.pt` uploaded every `--clearml-upload-every` updates (default 200) and at the end.
