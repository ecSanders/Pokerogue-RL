"""Encode simulator observations into fixed-size arrays for the battle policy.

Layout (all float32 unless noted):
  species  int64 [8]          6 party slots, then 2 enemy slots (0 = empty)
  ability  int64 [8]          ability id per slot (e.g. Pickup)
  moves    int64 [8, 4]       move ids per slot (0 = empty)
  mon      float [8, MON_DIM] per-Pokemon numeric features
  move     float [8, 4, MOVE_DIM]
  glob     float [GLOBAL_DIM] wave, decision kind, actor, balls, target move, ...
  mask     bool  [NUM_ACTIONS]
"""

from __future__ import annotations

import math

import numpy as np

PARTY_SLOTS = 6
ENEMY_SLOTS = 2
SLOTS = PARTY_SLOTS + ENEMY_SLOTS
MOVES = 4
NUM_SPECIES = 9000  # SpeciesId values go up to ~8901 (regional forms use high ranges)
NUM_MOVES = 1024
NUM_ABILITIES = 512  # AbilityId has ~320 values
NUM_TYPES = 20  # NORMAL..FAIRY plus STELLAR
NUM_STATUS = 8
NUM_STAT_STAGES = 7
NUM_BALLS = 5

# Battle decisions share one action head. command: 0-3 moves, 4-9 switch, 10-14 balls;
# target: 0-3 battler index; faint_switch: 0-5 party slot.
BATTLE_KINDS = ("command", "target", "faint_switch")
NUM_ACTIONS = 15

MON_DIM = 1 + 1 + 1 + 1 + 6 + NUM_STAT_STAGES + NUM_STATUS + NUM_TYPES + 7
MOVE_DIM = NUM_TYPES + 3 + 5 + 2 * ENEMY_SLOTS * 2
GLOBAL_DIM = len(BATTLE_KINDS) + 2 + 6 + NUM_BALLS * 2 + MOVE_DIM + MOVES


def ability_index(p: dict) -> int:
    return min(max(p.get("ability", 0), 0), NUM_ABILITIES - 1)


def _one_hot(index: int, size: int) -> np.ndarray:
    v = np.zeros(size, np.float32)
    if 0 <= index < size:
        v[index] = 1.0
    return v


def _types(types: list[int]) -> np.ndarray:
    v = np.zeros(NUM_TYPES, np.float32)
    for t in types:
        if 0 <= t < NUM_TYPES:
            v[t] = 1.0
    return v


def _move_features(m: dict, pp_known: bool = True) -> np.ndarray:
    eff = (m.get("eff") or []) + [None] * ENEMY_SLOTS
    dmg = (m.get("dmg") or []) + [None] * ENEMY_SLOTS
    vs_enemy = []
    for i in range(ENEMY_SLOTS):
        e, d = eff[i], dmg[i]
        vs_enemy += [
            math.log2(e) / 2 if e else -2.0 if e == 0 else 0.0,  # effectiveness, log scale
            1.0 if e is not None else 0.0,
            min(d, 2.0) if d is not None else 0.0,  # simulated damage / enemy HP
            1.0 if d is not None else 0.0,
        ]
    max_pp = m.get("maxPp") or 0
    return np.concatenate([
        _types([m["type"]]),
        _one_hot(m["category"], 3),
        np.array([
            m["power"] / 150 if m["power"] > 0 else 0.0,
            1.0 if m["accuracy"] < 0 else m["accuracy"] / 100,
            m["priority"] / 5,
            m["pp"] / max_pp if pp_known and max_pp else 1.0,
            1.0 if not pp_known or m.get("pp", 1) > 0 else 0.0,
        ], np.float32),
        np.array(vs_enemy, np.float32),
    ])


def _mon_features(p: dict) -> np.ndarray:
    stages = (p.get("statStages") or [])[:NUM_STAT_STAGES]
    stages = list(stages) + [0] * (NUM_STAT_STAGES - len(stages))
    return np.concatenate([
        np.array([
            p["level"] / 100,
            p["hp"] / p["maxHp"] if p["maxHp"] else 0.0,
            math.log1p(p["maxHp"]) / 7,
            p["bst"] / 700,
        ], np.float32),
        np.array([math.log1p(s) / 7 for s in p["stats"]], np.float32),
        np.array(stages, np.float32) / 6,
        _one_hot(p["status"], NUM_STATUS),
        _types(p["types"]),
        np.array([1.0, p["onField"], p["fainted"], p["boss"], p["shiny"], p["pokerus"], p["luck"] / 3],
                 np.float32),
    ])


def encode(obs: dict) -> dict[str, np.ndarray]:
    species = np.zeros(SLOTS, np.int64)
    ability = np.zeros(SLOTS, np.int64)
    moves = np.zeros((SLOTS, MOVES), np.int64)
    mon = np.zeros((SLOTS, MON_DIM), np.float32)
    move = np.zeros((SLOTS, MOVES, MOVE_DIM), np.float32)

    mons = obs["party"][:PARTY_SLOTS] + [None] * (PARTY_SLOTS - len(obs["party"][:PARTY_SLOTS]))
    mons += obs["enemies"][:ENEMY_SLOTS] + [None] * (ENEMY_SLOTS - len(obs["enemies"][:ENEMY_SLOTS]))
    for s, p in enumerate(mons):
        if p is None:
            continue
        species[s] = min(p["species"], NUM_SPECIES - 1)
        ability[s] = ability_index(p)
        mon[s] = _mon_features(p)
        for j, m in enumerate(p["moves"][:MOVES]):
            moves[s, j] = min(m["id"], NUM_MOVES - 1)
            move[s, j] = _move_features(m, pp_known=s < PARTY_SLOTS)

    kind = obs["kind"]
    actor = obs.get("actor", -1)
    target_move = np.zeros(MOVE_DIM, np.float32)
    target_slot = np.zeros(MOVES, np.float32)
    if kind == "target":
        slot = obs["move"]["slot"]
        actor_moves = obs["party"][actor]["moves"]
        if slot is not None and slot < len(actor_moves):
            target_move = _move_features(actor_moves[slot])
            target_slot[slot] = 1.0
    wave = obs["wave"]
    glob = np.concatenate([
        _one_hot(BATTLE_KINDS.index(kind), len(BATTLE_KINDS)),
        _one_hot(actor, 2),
        np.array([
            wave / 200,
            math.log1p(wave) / 7,
            (wave % 10) / 10,  # bosses every 10th wave
            min(obs["turn"], 30) / 30,
            float(obs["double"]),
            math.log1p(obs["money"]) / 10,
        ], np.float32),
        np.array([math.log1p(b) / 3 for b in obs["balls"]], np.float32),
        np.array(obs.get("catchChance") or [0.0] * NUM_BALLS, np.float32),
        target_move,
        target_slot,
    ])

    mask = np.zeros(NUM_ACTIONS, bool)
    legal = obs["legal"][:NUM_ACTIONS]
    mask[: len(legal)] = legal
    return {"species": species, "ability": ability, "moves": moves, "mon": mon, "move": move, "glob": glob, "mask": mask}


BALL_ACTION = 10  # command actions 10-14 throw Poke/Great/Ultra/Rogue/Master Ball
MASTER_BALL_ACTION = 14


def restrict_boss_balls(obs: dict) -> dict:
    """No ordinary balls against a boss: once its shields break it is catchable in principle, but the
    odds are low and every miss is a turn the boss attacks for free (v4 threw on ~82% of its chances).
    A Master Ball never misses, so it stays legal."""
    if obs["kind"] != "command" or not any(e["boss"] and not e["fainted"] for e in obs["enemies"]):
        return obs
    legal = [ok and not BALL_ACTION <= i < MASTER_BALL_ACTION for i, ok in enumerate(obs["legal"])]
    return {**obs, "legal": legal} if any(legal) else obs


def stack(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {k: np.stack([b[k] for b in batch]) for k in batch[0]}
