"""Inputs added for long endless runs (2026-11-06): the EXP level cap, held and team items, enemy buffs,
and where the next bosses are. Shared by the battle and run policies under "x_" keys.

The policies feed these through zero-initialised projections (ExtraEncoder + extra_proj in policy.py),
so checkpoints trained before they existed load and behave exactly as before until training uses them.
See docs/pokerogue-mechanics.md for why the level cap matters.
Layout:
  x_glob   float [GLOBAL_DIM]        level cap, waves to next boss / 50th-wave gate, End biome, enemy buffs
  x_mon    float [6, MON_DIM]        per party slot: levels below the cap, held item stacks
  x_held   int64 [6, HELD]           held item buckets per party slot (0 = none)
  x_held_n float [6, HELD]           their stacks
  x_team   int64 [TEAM]              team-wide item buckets (charms, EXP Share, lures, ...)
  x_team_n float [TEAM]              their stacks
"""

from __future__ import annotations

import math

import numpy as np

from pokerogue_rl import features as F
from pokerogue_rl.run_features import item_bucket

HELD = 8
TEAM = 16
END_BIOME = 50  # BiomeId.END (pokerogue/src/enums/biome-id.ts)
ENEMY_BUFFS = ("ENEMY_DAMAGE_BOOSTER", "ENEMY_DAMAGE_REDUCTION", "ENEMY_HEAL", "ENEMY_ATTACK_POISON_CHANCE",
               "ENEMY_ATTACK_PARALYZE_CHANCE", "ENEMY_ATTACK_BURN_CHANCE", "ENEMY_STATUS_EFFECT_HEAL_CHANCE",
               "ENEMY_ENDURE_CHANCE", "ENEMY_FUSED_CHANCE")
GLOBAL_DIM = 6 + len(ENEMY_BUFFS)
MON_DIM = 4


def encode(obs: dict) -> dict[str, np.ndarray]:
    wave = obs.get("wave", 0)
    cap = obs.get("levelCap") or 0
    buffs = np.zeros(len(ENEMY_BUFFS), np.float32)
    for b in obs.get("enemyBuffs") or []:
        if b["id"] in ENEMY_BUFFS:
            buffs[ENEMY_BUFFS.index(b["id"])] += b["stacks"] / 5
    total_buffs = sum(b["stacks"] for b in obs.get("enemyBuffs") or [])
    glob = np.concatenate([
        np.array([
            math.log1p(cap) / 8,
            (10 - wave % 10) % 10 / 10,  # waves until the next boss (0 = boss now)
            (50 - wave % 50) % 50 / 50,  # waves until the next End-biome boss
            float(obs.get("biome") == END_BIOME),
            math.log1p(total_buffs) / 4,
            float(cap > 0),
        ], np.float32),
        buffs,
    ])

    mon = np.zeros((F.PARTY_SLOTS, MON_DIM), np.float32)
    held = np.zeros((F.PARTY_SLOTS, HELD), np.int64)
    held_n = np.zeros((F.PARTY_SLOTS, HELD), np.float32)
    team = np.zeros(TEAM, np.int64)
    team_n = np.zeros(TEAM, np.float32)
    counts = [0] * F.PARTY_SLOTS
    t = 0
    for it in obs.get("items") or []:
        slot = it.get("slot", -1)
        if 0 <= slot < F.PARTY_SLOTS:
            if counts[slot] < HELD:
                held[slot, counts[slot]] = item_bucket(it["id"])
                held_n[slot, counts[slot]] = math.log1p(it["stacks"])
            counts[slot] += 1
            mon[slot, 2] += it["stacks"]
        elif slot < 0 and t < TEAM:
            team[t] = item_bucket(it["id"])
            team_n[t] = math.log1p(it["stacks"])
            t += 1
    for s, p in enumerate(obs.get("party", [])[:F.PARTY_SLOTS]):
        below = (cap - p["level"]) / cap if cap else 0.0
        mon[s, 0] = max(-1.0, min(1.0, below))  # 0 at the cap; negative above it (candies)
        mon[s, 1] = float(cap > 0 and p["level"] >= cap)  # gains no battle EXP
        mon[s, 2] = math.log1p(mon[s, 2]) / 3
        mon[s, 3] = 1.0
    return {"x_glob": glob, "x_mon": mon, "x_held": held, "x_held_n": held_n, "x_team": team, "x_team_n": team_n}
