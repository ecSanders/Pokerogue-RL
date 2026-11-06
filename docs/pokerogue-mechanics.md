# PokeRogue endless mode: mechanics that matter for the agent

Read from the pinned PokeRogue source (submodule commit `7c0881255af5`) on 2026-11-06. Paths are relative
to `pokerogue/src/`; line numbers are for that commit. Community strategy at the end is from player guides
and is lower confidence. Our simulator's departures from the real game are listed in
[What our simulator changes](#what-our-simulator-changes).

## Run structure

| Rule | Detail | Source |
|---|---|---|
| Boss waves | Every 10th wave is a boss | `game-mode.ts:313` `isBoss` |
| End biome | Every 50th wave is the End biome; its boss can't be caught and balls are blocked there | `phases/select-biome-phase.ts:102`, `phases/command-phase.ts:373` |
| Eternatus | Every 250th wave; Eternamax every 1000th | `game-mode.ts:296` `isWaveFinal`, `isEndlessMinorBoss`, `isEndlessMajorBoss` |
| Biomes | A new biome every 5 waves (`hasShortBiomes`); the next one follows biome links, or is chosen with a Map | `phases/select-biome-phase.ts` |
| Trainers | None in endless (`hasTrainers` is off) | `game-mode.ts:478`, `battle-scene.ts` `getNewBattleProps` |
| Healing | Full party heal entering every x1 wave (11, 21, ...); Coin Case interest is paid then too | `phases/select-biome-phase.ts:109` |
| Shop | Open every wave; money per wave grows with the wave | `game-mode.ts:382`, `battle-scene.ts:2383` |
| After each 50th-wave boss | A voucher, and permanent enemy buffs (damage boost, damage reduction, status chance, endure, ...); `ceil(wave / 250)` buffs each time | `phases/victory-phase.ts:116`, `phases/add-enemy-buff-modifier-phase.ts` |
| Double battles | Wild: 1 in 8 (1 in 32 on x0 waves); each lure or double-chance ability divides by 4; 50th-wave bosses are always single | `battle-scene.ts:1253`, `:1504` |

Damage builds up only within a 10-wave stretch, so the team is most worn down just before each boss.

## Enemy scaling

`battle.ts:134` `getLevelForWave`, with `w` = wave:

- Wild: `round(1 + w/2 + (w/25)^2)` plus a small positive random offset
- Boss: `floor(1.2 * (1 + w/2 + (w/25)^2))`, randomly ±`floor(w/10)` (except on 250-multiples, which round up to a multiple of 25)

| Wave | Wild level | Boss level | EXP level cap |
|---|---|---|---|
| 10 | ~6 | 7 ±1 | 10 |
| 20 | ~12 | 13 ±2 | 16 |
| 30 | ~17 | 20 ±3 | 24 |
| 40 | ~24 | 28 ±4 | 32 |
| **50** | ~30 | **36 ±5** | **38** |
| 100 | ~67 | 80 ±10 | 84 |
| 250 | ~226 | 275 | 274 |
| 500 | ~651 | 781 ±50 | 784 |

- **Boss health bars (shields)**, `battle-scene.ts:1959`: 2, plus 1 if the species' base stat total is ≥ 670, plus 1 at level ≥ 100, plus 1 per 250 waves. After wave 250, ordinary waves can also roll a boss (2% more per 50 waves, up to 30%).
- **Enemy held items**: `ceil(w/10)` rolls per enemy, each succeeding 1 in 12 (wild) or 1 in 4 (boss); bosses always get at least half of the rolls.

## EXP

- **Level cap**, `battle-scene.ts:2298` `getMaxExpLevel`: `ceil(1.2 * (1 + W/2 + (W/25)^2) / 2) * 2 + 2`, where `W` is the wave rounded
  up to the next multiple of 10. **A Pokemon at or above the cap gains no battle EXP** (`applyPartyExp`,
  `battle-scene.ts:3316`). At wave 50 the cap is 38 and the boss is level 31-41.
- **Rare Candy and Rarer Candy ignore the cap** (`modifier/modifier.ts:2260` `PokemonLevelIncrementModifier`). They
  are the only way past it.
- **Who gets EXP**: only Pokemon that took part, split evenly between them, so switching in splits the EXP.
  EXP Share gives each non-participant 20% per stack (`modifier/modifier.ts:2553`). EXP Balance moves EXP
  toward members below the median level.
- **Multipliers**: Pokerus x1.5 (spreads to party neighbours, `battle-scene.ts:2003`); Lucky Egg +40% and
  Golden Egg +100% for the holder; EXP Charm +25%, Super +60%, Golden +100% for the whole team
  (`modifier/modifier-type.ts:1974`).
- EXP needed per level grows roughly with level squared, while EXP per battle grows with the enemy's level.
  Measured with exp-v1's policy: levels gained per wave stay around 0.7-1.4 all run, while raw EXP per wave
  grows from ~90 (waves 1-9) to ~1800 (waves 31-45). See `hpc/sweeps/exp-raw.txt`.

## What our simulator changes

- Starters are Charmander, Squirtle and Bulbasaur, with 31 IVs and a Hardy nature; the first starts with Pokerus
  (`sim/sim-server.ts` `reset`).
- Fusion (DNA Splicers) and Memory Mushroom rewards are never legal, since their menus aren't modelled.
- Mystery encounters are disabled; biome choice and other prompts are scripted defaults.
- Battle randomness is the real game's (the test harness's forced max rolls are undone), and runs are
  deterministic given the seed and actions.

## What this means for training

1. **Levels alone can't win the 50th-wave boss.** EXP can put the carry at most about 2 levels above an average boss,
   and it can be 3 below a strong one. Winning depends on held items, candies, type matchups, moves and a
   second attacker; level-based rewards only see levels.
2. **A capped Pokemon earns nothing.** Once the carry reaches the cap, the EXP reward for it drops to zero and the
   EXP streak breaks. Under raw EXP, EXP Share and the bench become the main source of reward until the cap
   rises at the next x1 wave.
3. **The agent can't see the cap.** Candidate features: the current cap, and each Pokemon's levels below it.
   Boss readiness (`--boss-coef`, `strategy.boss_level`) could also be measured against the cap rather than
   the expected boss level.
4. **Candies are valuable.** They're the only way past the cap, and any EXP reward pays for them.
5. **Catching costs EXP.** A wave won by catching gives no EXP. exp-all-v1's carry was about level 18 when it
   died around wave 30, against a cap of 24 (death report on exp-all-v1, 2026-11-06).
6. **Pacing matters around bosses.** Healing comes only every 10 waves, and the waves before each x0 boss decide how
   healthy the team arrives.

## Player strategy (lower confidence)

From 2024 r/pokerogue guides:
- Invest in one carry early, then build mid- and late-game carries as its damage falls off.
- Stack HP and healing items. The popular "Blissey mech" and "Burster" builds rely on fusions, which our
  simulator doesn't allow.
- Bring support Pokemon for specific threats, such as ability-negating moves.
- Strong legendary species outperform others.

Sources:
- [r/pokerogue "Endless guide" (May 2024)](https://lr.in.psf.lt/r/pokerogue/comments/1cu69nw/endless_guide/l4zu3tp/?context=3)
- [r/pokerogue "The Ultimate Endless Mode Guide" (May 2024)](https://redlib.snake-cloud.ts.net/r/pokerogue/comments/1d3g11a/the_ultimate_endless_mode_guide)
- [r/pokerogue FAQ and comprehensive guide (May 2024)](https://pol1.lr.ggtyler.dev/r/pokerogue/comments/1czask1/faq_and_comprehensive_guide/l5l1f2d/?context=3)
- [PokéRogue wiki: biomes](https://wiki.pokerogue.net/biomes:biomes)
