"""Encode run-management decisions (rewards, item targets, move learning, releases) for the run policy.

Each decision is a set of candidate actions scored against the run state, so option lists of any
length share one network:
  reward         candidates = obs.options (rewards, shop items, reroll, skip)
  reward_target  candidates = party slots, with the chosen item as context
  learn_move     candidates = the learner's 4 move slots + "don't learn", the new move as context
  release        candidates = party slots + "let it go", the caught Pokemon as context

Layout:
  species     int64 [7]                   6 party slots, then the caught Pokemon (0 = empty)
  moves       int64 [7, 4]
  mon         float [7, MON_DIM]
  move        float [7, 4, MOVE_DIM]
  glob        float [GLOBAL_DIM]          decision kind, wave, money, balls, luck, party state
  ctx_item    int64 []                    item being applied (reward_target)
  ctx_move_id int64 []                    move being learned (learn_move)
  ctx_move    float [MOVE_DIM]
  cand_item   int64 [MAX_CANDS]           item id bucket of a reward option (0 = none)
  cand_move   int64 [MAX_CANDS]           move id of a TM option or move slot (0 = none)
  cand_slot   int64 [MAX_CANDS]           party slot + 1 a candidate refers to, 7 = caught, 0 = none
  cand        float [MAX_CANDS, CAND_DIM]
  mask        bool  [MAX_CANDS]
"""

from __future__ import annotations

import math
import zlib

import numpy as np

from pokerogue_rl import features as F

RUN_KINDS = ("reward", "reward_target", "learn_move", "release")
MAX_CANDS = 32  # reward screens: up to 14 shop items + rewards (3 plus extras) + reroll + skip
MON_SLOTS = F.PARTY_SLOTS + 1  # party + caught Pokemon
ITEM_BUCKETS = 1024
NUM_TIERS = 6  # COMMON..MASTER, LUXURY
CAND_KINDS = ("reward", "shop", "reroll", "skip", "party", "move_slot", "decline")

GLOBAL_DIM = len(RUN_KINDS) + 6 + F.NUM_BALLS + 4
CAND_DIM = len(CAND_KINDS) + NUM_TIERS + 1 + 4 + F.MOVE_DIM + 2


def item_bucket(item_id: str | None) -> int:
    """Stable id -> embedding row; 0 is reserved for "no item"."""
    if not item_id:
        return 0
    return 1 + zlib.crc32(item_id.encode()) % (ITEM_BUCKETS - 1)


def _hp(p: dict) -> float:
    return p["hp"] / p["maxHp"] if p["maxHp"] else 0.0


def encode(obs: dict) -> dict[str, np.ndarray]:
    kind = obs["kind"]
    party = obs["party"][: F.PARTY_SLOTS]
    species = np.zeros(MON_SLOTS, np.int64)
    moves = np.zeros((MON_SLOTS, F.MOVES), np.int64)
    mon = np.zeros((MON_SLOTS, F.MON_DIM), np.float32)
    move = np.zeros((MON_SLOTS, F.MOVES, F.MOVE_DIM), np.float32)
    mons = list(party) + [None] * (F.PARTY_SLOTS - len(party)) + [obs.get("caught")]
    for s, p in enumerate(mons):
        if p is None:
            continue
        species[s] = min(p["species"], F.NUM_SPECIES - 1)
        mon[s] = F._mon_features(p)
        for j, m in enumerate(p["moves"][: F.MOVES]):
            moves[s, j] = min(m["id"], F.NUM_MOVES - 1)
            move[s, j] = F._move_features(m)

    wave, money = obs["wave"], obs["money"]
    fainted = sum(p["fainted"] for p in party)
    alive_hp = [_hp(p) for p in party if not p["fainted"]]
    glob = np.concatenate([
        F._one_hot(RUN_KINDS.index(kind), len(RUN_KINDS)),
        np.array([
            wave / 200,
            math.log1p(wave) / 7,
            (wave % 10) / 10,
            float(wave % 10 == 9),  # boss wave next
            math.log1p(money) / 10,
            obs.get("luck", 0) / 14,
        ], np.float32),
        np.array([math.log1p(b) / 3 for b in obs["balls"]], np.float32),
        np.array([
            len(party) / 6,
            fainted / 6,
            min(alive_hp, default=0.0),
            float(np.mean(alive_hp)) if alive_hp else 0.0,
        ], np.float32),
    ])

    ctx_item = ctx_move_id = 0
    ctx_move = np.zeros(F.MOVE_DIM, np.float32)
    cand_item = np.zeros(MAX_CANDS, np.int64)
    cand_move = np.zeros(MAX_CANDS, np.int64)
    cand_slot = np.zeros(MAX_CANDS, np.int64)
    cand = np.zeros((MAX_CANDS, CAND_DIM), np.float32)
    legal = list(obs["legal"])
    if len(legal) > MAX_CANDS:
        raise ValueError(f"{kind} has {len(legal)} options; raise MAX_CANDS")

    def put(i: int, ckind: str, tier: int | None = None, cost: float = 0.0, needs_target: bool = False,
            mv: dict | None = None, extra: tuple[float, float] = (0.0, 0.0)) -> None:
        tier_v = F._one_hot(tier if tier is not None else -1, NUM_TIERS)
        cost_v = np.array([
            float(tier is None),
            min(cost / money, 2.0) if cost > 0 and money > 0 else (2.0 if cost > 0 else 0.0),
            math.log1p(max(cost, 0)) / 10,
            float(cost > 0),
            float(needs_target),
        ], np.float32)
        cand[i] = np.concatenate([
            F._one_hot(CAND_KINDS.index(ckind), len(CAND_KINDS)), tier_v, cost_v,
            F._move_features(mv) if mv else np.zeros(F.MOVE_DIM, np.float32),
            np.array(extra, np.float32),
        ])

    if kind == "reward":
        for i, o in enumerate(obs["options"]):
            cand_item[i] = item_bucket(o["id"] if o["kind"] in ("reward", "shop") else None)
            tm = o.get("tmMove")
            if tm:
                cand_move[i] = min(tm["id"], F.NUM_MOVES - 1)
                tm = {**tm, "pp": 1, "maxPp": 1}
            put(i, o["kind"], o.get("tier"), max(o["cost"], 0), o.get("needsTarget", False), tm)
    elif kind == "reward_target":
        ctx_item = item_bucket(obs["item"]["id"])
        for i in range(len(party)):
            cand_slot[i] = i + 1
            put(i, "party")
    elif kind == "learn_move":
        learner = party[obs["learn"]["slot"]]
        new = {**obs["learn"]["move"], "pp": 1, "maxPp": 1}
        ctx_move_id = min(new["id"], F.NUM_MOVES - 1)
        ctx_move = F._move_features(new)
        for j, m in enumerate(learner["moves"][: F.MOVES]):
            cand_move[j] = min(m["id"], F.NUM_MOVES - 1)
            cand_slot[j] = obs["learn"]["slot"] + 1
            stab = float(m["type"] in learner["types"])
            put(j, "move_slot", mv=m, extra=(stab, float(new["type"] in learner["types"])))
        put(4, "decline")
    elif kind == "release":
        for i in range(len(party)):
            cand_slot[i] = i + 1
            put(i, "party")
        cand_slot[6] = MON_SLOTS  # letting it go: compare against the caught Pokemon
        put(6, "decline")
    else:
        raise ValueError(f"not a run decision: {kind}")

    mask = np.zeros(MAX_CANDS, bool)
    mask[: len(legal)] = legal
    return {"species": species, "moves": moves, "mon": mon, "move": move, "glob": glob,
            "ctx_item": np.int64(ctx_item), "ctx_move_id": np.int64(ctx_move_id), "ctx_move": ctx_move,
            "cand_item": cand_item, "cand_move": cand_move, "cand_slot": cand_slot, "cand": cand,
            "mask": mask}


class TmMemory:
    """Declining a TM's move returns to the reward screen; mark that TM illegal for the rest of the
    wave so the policy can't loop on it. Feed every (obs, action) through `note`, and filter reward
    observations with `filter` before encoding them."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.declined: set[tuple[int, int]] = set()
        self.pending: int | None = None

    def filter(self, obs: dict) -> dict:
        if obs["kind"] != "reward" or not self.declined:
            return obs
        options = [
            {**o, "legal": False} if o.get("tmMove") and (obs["wave"], o["tmMove"]["id"]) in self.declined else o
            for o in obs["options"]
        ]
        return {**obs, "options": options, "legal": [o["legal"] for o in options]}

    def note(self, obs: dict, action: int) -> None:
        if obs["kind"] == "reward":
            tm = obs["options"][action].get("tmMove")
            self.pending = tm["id"] if tm else None
        elif obs["kind"] == "learn_move":
            if action == 4 and self.pending == obs["learn"]["move"]["id"]:
                self.declined.add((obs["wave"], self.pending))
            self.pending = None
