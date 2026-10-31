"""Scripted baseline agent.

Battle: catch wild Pokemon that would strengthen the team (weakening them first if needed); else
take a likely KO, otherwise the move with the highest expected damage; switch out only when badly
outmatched and a healthy teammate does much better. Rewards: score items against the
team's current state. Learn move: replace the weakest move if the new one is stronger.
"""

from __future__ import annotations

STATUS_CATEGORY = 2
STATUS_MOVE_VALUE = 10.0
BALL_ACTION = 10  # command actions 10-14 throw Poke/Great/Ultra/Rogue/Master Ball
MASTER_BALL = 4
MASTER_BALL_BST = 580  # save the Master Ball for legendary-tier catches

HEALS = {  # fraction of max HP restored (approximate; Full Restore also cures status)
    "POTION": 0.2, "SUPER_POTION": 0.4, "HYPER_POTION": 0.6, "MAX_POTION": 1.0, "FULL_RESTORE": 1.0,
}
REVIVES = {"REVIVE", "MAX_REVIVE"}
TEAM_REVIVES = {"SACRED_ASH"}
PP_ITEMS = {"ETHER", "MAX_ETHER", "ELIXIR", "MAX_ELIXIR"}

# Base desirability of items that don't depend on the team's current HP.
ITEM_VALUE = {
    # Levels and EXP
    "RARER_CANDY": 1.0, "RARE_CANDY": 0.8, "EXP_SHARE": 0.75, "EXP_CHARM": 0.7, "SUPER_EXP_CHARM": 0.8,
    "GOLDEN_EXP_CHARM": 0.9, "LUCKY_EGG": 0.6, "GOLDEN_EGG": 0.75, "EXP_BALANCE": 0.5,
    # Permanent stat and battle items
    "BASE_STAT_BOOSTER": 0.6, "ATTACK_TYPE_BOOSTER": 0.55, "LEFTOVERS": 0.65, "SHELL_BELL": 0.6,
    "FOCUS_BAND": 0.5, "KINGS_ROCK": 0.45, "QUICK_CLAW": 0.4, "GRIP_CLAW": 0.35, "WIDE_LENS": 0.45,
    "MULTI_LENS": 0.6, "SOOTHE_BELL": 0.2, "SCOPE_LENS": 0.45, "LEEK": 0.3, "TOXIC_ORB": 0.1,
    "FLAME_ORB": 0.1, "MYSTICAL_ROCK": 0.3, "BATON": 0.2, "REVIVER_SEED": 0.5, "WHITE_HERB": 0.3,
    "BERRY": 0.35, "BERRY_POUCH": 0.4, "MINT": 0.3, "EVOLUTION_ITEM": 0.7, "RARE_EVOLUTION_ITEM": 0.75,
    "FORM_CHANGE_ITEM": 0.4, "TERA_SHARD": 0.2, "IV_SCANNER": 0.05,
    # Moves
    "TM_COMMON": 0.35, "TM_GREAT": 0.45, "TM_ULTRA": 0.55, "PP_UP": 0.2, "PP_MAX": 0.25,
    # Economy and one-wave buffs
    "AMULET_COIN": 0.5, "NUGGET": 0.3, "BIG_NUGGET": 0.45, "RELIC_GOLD": 0.55, "GOLDEN_PUNCH": 0.3,
    "TEMP_STAT_STAGE_BOOSTER": 0.25, "DIRE_HIT": 0.2, "MEMORY_MUSHROOM": 0.0,
    # Balls build the team; lures add double battles, where catching isn't possible.
    "POKEBALL": 0.15, "GREAT_BALL": 0.25, "ULTRA_BALL": 0.35, "ROGUE_BALL": 0.4, "MASTER_BALL": 0.7,
    "LURE": -0.1, "SUPER_LURE": -0.1, "MAX_LURE": -0.1, "VOUCHER": 0.0, "VOUCHER_PLUS": 0.0,
    "VOUCHER_PREMIUM": 0.0, "MAP": 0.05,
}
DEFAULT_ITEM_VALUE = 0.3
SKIP_THRESHOLD = 0.05


def hp_ratio(p: dict) -> float:
    return p["hp"] / p["maxHp"] if p["maxHp"] else 0.0


def accuracy(move: dict) -> float:
    return 1.0 if move["accuracy"] < 0 else move["accuracy"] / 100


def best_damage(p: dict) -> float:
    """Highest expected damage fraction this Pokemon's moves deal to any enemy on field."""
    best = 0.0
    for m in p["moves"]:
        if m["pp"] <= 0:
            continue
        dmg = max((d for d in m.get("dmg", []) if d is not None), default=0.0)
        best = max(best, min(dmg, 1.0) * accuracy(m))
    return best


def move_value(move: dict, user_types: list[int]) -> float:
    if move["category"] == STATUS_CATEGORY or move["power"] <= 0:
        return STATUS_MOVE_VALUE
    stab = 1.5 if move["type"] in user_types else 1.0
    return move["power"] * accuracy(move) * stab


def learn_gain(move: dict, p: dict) -> float:
    """How much better p's moveset gets by learning `move` (<= 0 means it would decline)."""
    if any(m["id"] == move["id"] for m in p["moves"]):
        return 0.0
    new = move_value(move, p["types"])
    if len(p["moves"]) < 4:
        return new
    return new - min(move_value(m, p["types"]) for m in p["moves"])


class HeuristicAgent:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Call at the start of each episode."""
        # Declining a TM returns to the reward screen; remember declines so we don't pick it again.
        self.declined_tms: set[tuple[int, int]] = set()
        self.pending_tm: dict | None = None

    def act(self, obs: dict) -> int:
        return getattr(self, f"_{obs['kind']}")(obs)

    # Battle

    def _catch(self, obs: dict) -> int | str | None:
        """A ball action, "weaken" to lower the target's HP first, or None to battle normally."""
        legal, party, chance = obs["legal"], obs["party"], obs["catchChance"]
        balls = [b for b in range(5) if legal[BALL_ACTION + b]]
        if not balls:
            return None
        enemy = next(e for e in obs["enemies"] if e["hp"] > 0)
        if len(party) >= 4 and enemy["bst"] < min(p["bst"] for p in party):
            return None
        if MASTER_BALL in balls and enemy["bst"] >= MASTER_BALL_BST:
            return BALL_ACTION + MASTER_BALL
        normal = sorted(b for b in balls if b != MASTER_BALL)
        if not normal:
            return None
        for b in normal:  # cheapest ball that's likely to work
            if chance[b] >= 0.5:
                return BALL_ACTION + b
        if hp_ratio(enemy) > 0.35:
            return "weaken"
        best = max(normal, key=lambda b: chance[b])
        return BALL_ACTION + best if chance[best] >= 0.15 else None

    def _command(self, obs: dict) -> int:
        legal = obs["legal"]
        party = obs["party"]
        actor = party[obs["actor"]]

        catch = self._catch(obs)
        if isinstance(catch, int):
            return catch
        weaken = catch == "weaken"

        best_move, best_score = -1, -1.0
        for i, m in enumerate(actor["moves"]):
            if i >= 4 or not legal[i]:
                continue
            dmg = max((d for d in m.get("dmg", []) if d is not None), default=0.0)
            if m["category"] == STATUS_CATEGORY or dmg <= 0:
                score = 0.01
            elif weaken:
                score = dmg if dmg < 0.85 else -0.5  # chip damage without knocking it out
            elif dmg >= 1.0:
                score = 1.0 + accuracy(m) + 0.1 * max(m["priority"], 0)  # likely KO
            else:
                score = dmg * accuracy(m)
            if score > best_score:
                best_move, best_score = i, score

        current = best_damage(actor)
        for slot in range(len(party)):
            if not legal[4 + slot]:
                continue
            bench = party[slot]
            if current < 0.15 and hp_ratio(bench) > 0.5 and best_damage(bench) > max(0.4, 2.5 * current):
                return 4 + slot

        if best_move >= 0:
            return best_move
        return legal.index(True)

    # Rewards

    def _reward(self, obs: dict) -> int:
        party = obs["party"]
        missing = max((1 - hp_ratio(p) for p in party if not p["fainted"]), default=0.0)
        fainted = sum(p["fainted"] for p in party)
        low_pp = any(m["pp"] / m["maxPp"] < 0.25 for p in party for m in p["moves"] if m["maxPp"])

        best, best_score = None, SKIP_THRESHOLD
        for i, opt in enumerate(obs["options"]):
            if not opt["legal"] or opt["kind"] in ("skip", "reroll"):
                continue
            item = opt["id"]
            if item in HEALS:
                score = min(HEALS[item], missing) * 1.2 if missing > 0.3 else 0.0
            elif item in REVIVES:
                score = 1.1 if fainted else 0.0
            elif item in TEAM_REVIVES:
                score = 1.2 if fainted >= 2 else 0.0
            elif item in PP_ITEMS:
                score = 0.5 if low_pp else 0.0
            elif item == "FULL_HEAL":
                score = 0.4 if any(p["status"] and not p["fainted"] for p in party) else 0.0
            elif opt["tmMove"] is not None:
                tm = opt["tmMove"]
                useful = (obs["wave"], tm["id"]) not in self.declined_tms and any(
                    learn_gain(tm, p) > 0 for p in party)
                score = ITEM_VALUE.get(item, DEFAULT_ITEM_VALUE) if useful else 0.0
            else:
                score = ITEM_VALUE.get(item, DEFAULT_ITEM_VALUE)
            if opt["kind"] == "shop":
                # Only buy what's needed now, and keep a reserve so free rewards still get picked.
                score = score - 0.2 if score > 0.5 else 0.0
            if score > best_score:
                best, best_score = i, score

        if best is None:
            return next(i for i, o in enumerate(obs["options"]) if o["kind"] == "skip")
        self.pending_tm = obs["options"][best]["tmMove"]
        return best

    def _reward_target(self, obs: dict) -> int:
        party = obs["party"]
        legal = [i for i, ok in enumerate(obs["legal"]) if ok]
        item = obs["item"]["id"]
        if self.pending_tm is not None:
            return max(legal, key=lambda i: learn_gain(self.pending_tm, party[i]))
        if item in HEALS:
            return max(legal, key=lambda i: party[i]["maxHp"] - party[i]["hp"])
        if item in REVIVES:
            return max(legal, key=lambda i: party[i]["level"])
        if item in PP_ITEMS or item in ("PP_UP", "PP_MAX"):
            return min(legal, key=lambda i: min(m["pp"] / m["maxPp"] for m in party[i]["moves"] if m["maxPp"]))
        # Growth items and held items go to the strongest Pokemon.
        return max(legal, key=lambda i: (party[i]["level"], -i))

    # Moves

    def _learn_move(self, obs: dict) -> int:
        learner = obs["party"][obs["learn"]["slot"]]
        new_value = move_value(obs["learn"]["move"], learner["types"])
        values = [move_value(m, learner["types"]) for m in learner["moves"]]
        worst = min(range(len(values)), key=lambda i: values[i])
        action = worst if new_value > values[worst] else 4
        move_id = obs["learn"]["move"]["id"]
        if action == 4 and self.pending_tm is not None and self.pending_tm["id"] == move_id:
            self.declined_tms.add((obs["wave"], move_id))
        self.pending_tm = None
        return action
