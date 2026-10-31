"""Scripted baseline agent.

Strategy, from the death report and player experience:
- One "carry" (highest level, Pokerus breaks ties) takes the growth items and leads every wave.
- Fainted members are the main cause of runs ending: revives are top priority, and money (only
  from items in endless, which has no trainers) is saved for them.
- Shiny Pokemon raise party luck, which upgrades reward tiers: always catch them.
- Double battles give more EXP per wave but block catching; whether to seek them with lures is
  configurable (`lures="seek"` or `"avoid"`). In doubles, aim at the enemy we can KO or hurt most.
- Prepare for boss waves (every 10th) with heals and stat boosters on the wave before.

Battle: take a likely KO, otherwise the move with the highest expected damage; switch out only
when badly outmatched and a healthy teammate does much better.
"""

from __future__ import annotations

STATUS_CATEGORY = 2
STATUS_MOVE_VALUE = 10.0
SWITCH_ACTION = 4  # command actions 4-9 switch to party slot 0-5
BALL_ACTION = 10  # command actions 10-14 throw Poke/Great/Ultra/Rogue/Master Ball
MASTER_BALL = 4
MASTER_BALL_BST = 580  # save the Master Ball for legendary-tier catches (or shinies)
RELEASE_DECLINE = 6
ENEMY_TARGET = 2  # target decision: battler indices 0-1 are our side, 2-3 the enemies

HEALS = {  # fraction of max HP restored (approximate; Full Restore also cures status)
    "POTION": 0.2, "SUPER_POTION": 0.4, "HYPER_POTION": 0.6, "MAX_POTION": 1.0, "FULL_RESTORE": 1.0,
}
REVIVES = {"REVIVE", "MAX_REVIVE"}
TEAM_REVIVES = {"SACRED_ASH"}
PP_ITEMS = {"ETHER", "MAX_ETHER", "ELIXIR", "MAX_ELIXIR"}
BOSS_PREP = {"TEMP_STAT_STAGE_BOOSTER", "DIRE_HIT"}
# Items that make one Pokemon stronger: they go to the carry.
GROWTH_ITEMS = {
    "RARER_CANDY", "RARE_CANDY", "BASE_STAT_BOOSTER", "ATTACK_TYPE_BOOSTER", "LEFTOVERS", "SHELL_BELL",
    "FOCUS_BAND", "KINGS_ROCK", "QUICK_CLAW", "GRIP_CLAW", "WIDE_LENS", "MULTI_LENS", "SCOPE_LENS",
    "LEEK", "REVIVER_SEED", "WHITE_HERB", "BATON", "LUCKY_EGG", "GOLDEN_EGG", "SOOTHE_BELL", "MINT",
    "BERRY", "MYSTICAL_ROCK", "EVOLUTION_ITEM", "RARE_EVOLUTION_ITEM", "FORM_CHANGE_ITEM", "TERA_SHARD",
}

# Base desirability of items that don't depend on the team's current state.
ITEM_VALUE = {
    # Levels and EXP
    "RARER_CANDY": 1.0, "RARE_CANDY": 0.8, "EXP_SHARE": 0.6, "EXP_CHARM": 0.7, "SUPER_EXP_CHARM": 0.8,
    "GOLDEN_EXP_CHARM": 0.9, "LUCKY_EGG": 0.7, "GOLDEN_EGG": 0.8, "EXP_BALANCE": 0.2,
    # Permanent stat and battle items
    "BASE_STAT_BOOSTER": 0.6, "ATTACK_TYPE_BOOSTER": 0.55, "LEFTOVERS": 0.65, "SHELL_BELL": 0.6,
    "FOCUS_BAND": 0.5, "KINGS_ROCK": 0.45, "QUICK_CLAW": 0.4, "GRIP_CLAW": 0.35, "WIDE_LENS": 0.45,
    "MULTI_LENS": 0.6, "SOOTHE_BELL": 0.2, "SCOPE_LENS": 0.45, "LEEK": 0.3, "TOXIC_ORB": 0.1,
    "FLAME_ORB": 0.1, "MYSTICAL_ROCK": 0.3, "BATON": 0.2, "REVIVER_SEED": 0.6, "WHITE_HERB": 0.3,
    "BERRY": 0.35, "BERRY_POUCH": 0.4, "MINT": 0.3, "EVOLUTION_ITEM": 0.7, "RARE_EVOLUTION_ITEM": 0.75,
    "FORM_CHANGE_ITEM": 0.4, "TERA_SHARD": 0.2, "IV_SCANNER": 0.05,
    # Moves
    "TM_COMMON": 0.35, "TM_GREAT": 0.45, "TM_ULTRA": 0.55, "PP_UP": 0.2, "PP_MAX": 0.25,
    # Money: endless has no trainers, so items are the only income, and it buys revives.
    "AMULET_COIN": 0.65, "NUGGET": 0.45, "BIG_NUGGET": 0.6, "RELIC_GOLD": 0.7, "GOLDEN_PUNCH": 0.3,
    "TEMP_STAT_STAGE_BOOSTER": 0.25, "DIRE_HIT": 0.2, "MEMORY_MUSHROOM": 0.0,
    # Balls build the team. Lures: see LURE_VALUES.
    "POKEBALL": 0.15, "GREAT_BALL": 0.25, "ULTRA_BALL": 0.35, "ROGUE_BALL": 0.4, "MASTER_BALL": 0.7,
    "VOUCHER": 0.0, "VOUCHER_PLUS": 0.0,
    "VOUCHER_PREMIUM": 0.0, "MAP": 0.05,
}
# Lures make double battles more common: more EXP per wave, but no catching while two enemies are up.
LURE_VALUES = {
    "avoid": {"LURE": -0.1, "SUPER_LURE": -0.1, "MAX_LURE": -0.1},
    "seek": {"LURE": 0.35, "SUPER_LURE": 0.4, "MAX_LURE": 0.45},
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


def member_value(p: dict) -> float:
    """Rough worth of a team member: base stats scaled by level."""
    return p["bst"] * p["level"]


class HeuristicAgent:
    def __init__(self, lures: str = "avoid") -> None:
        self.item_value = {**ITEM_VALUE, **LURE_VALUES[lures]}
        self.reset()

    def reset(self) -> None:
        """Call at the start of each episode."""
        # Declining a TM returns to the reward screen; remember declines so we don't pick it again.
        self.declined_tms: set[tuple[int, int]] = set()
        self.pending_tm: dict | None = None
        self.carry_species: int | None = None
        self.carry_wave = -1  # last wave the carry was switched in at turn start

    def act(self, obs: dict) -> int:
        self._update_carry(obs["party"])
        return getattr(self, f"_{obs['kind']}")(obs)

    # Carry

    def _update_carry(self, party: list[dict]) -> None:
        """Keep the same carry while it's in the party; otherwise pick the strongest member."""
        if any(p["species"] == self.carry_species for p in party):
            return
        best = max(party, key=lambda p: (p["level"], p["pokerus"], p["bst"]))
        self.carry_species = best["species"]

    def _carry_slot(self, party: list[dict]) -> int | None:
        return next((i for i, p in enumerate(party) if p["species"] == self.carry_species), None)

    def _is_protected(self, p: dict) -> bool:
        """Members never released: the carry, shinies (luck) and Pokerus carriers (EXP, spreads)."""
        return p["species"] == self.carry_species or p["shiny"] or p["pokerus"]

    # Battle

    def _catch(self, obs: dict) -> int | str | None:
        """A ball action, "weaken" to lower the target's HP first, or None to battle normally."""
        legal, party, chance = obs["legal"], obs["party"], obs["catchChance"]
        balls = [b for b in range(5) if legal[BALL_ACTION + b]]
        if not balls:
            return None
        enemy = next(e for e in obs["enemies"] if e["hp"] > 0)
        if not enemy["shiny"]:
            if len(party) < 4:
                pass  # any catch fills the team out
            elif len(party) < 6:
                if enemy["bst"] < min(p["bst"] for p in party):
                    return None
            else:
                # Full party: only worth it if it beats the weakest member we'd release.
                releasable = [p for p in party if not self._is_protected(p)]
                if not releasable or member_value(enemy) < 1.2 * min(member_value(p) for p in releasable):
                    return None
        normal = sorted(b for b in balls if b != MASTER_BALL)
        best_normal = max(normal, key=lambda b: chance[b]) if normal else None
        if MASTER_BALL in balls and (
            enemy["bst"] >= MASTER_BALL_BST
            or (enemy["shiny"] and (best_normal is None or chance[best_normal] < 0.5))
        ):
            return BALL_ACTION + MASTER_BALL
        if best_normal is None:
            return None
        for b in normal:  # cheapest ball that's likely to work
            if chance[b] >= 0.5:
                return BALL_ACTION + b
        if hp_ratio(enemy) > 0.35:
            return "weaken"
        return BALL_ACTION + best_normal if chance[best_normal] >= 0.15 else None

    def _command(self, obs: dict) -> int:
        legal = obs["legal"]
        party = obs["party"]
        actor = party[obs["actor"]]

        catch = self._catch(obs)
        if isinstance(catch, int):
            return catch
        weaken = catch == "weaken"

        # Bring the carry back in at the start of a wave so it takes the EXP.
        carry = self._carry_slot(party)
        if (obs["turn"] <= 1 and self.carry_wave != obs["wave"] and carry is not None
                and legal[SWITCH_ACTION + carry] and hp_ratio(party[carry]) > 0.5):
            self.carry_wave = obs["wave"]
            return SWITCH_ACTION + carry

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

        # Bosses take many hits: be quicker to swap in a better attacker.
        vs_boss = any(e["boss"] for e in obs["enemies"] if e["hp"] > 0)
        current = best_damage(actor)
        threshold = 0.25 if vs_boss else 0.15
        for slot in range(len(party)):
            if not legal[SWITCH_ACTION + slot]:
                continue
            bench = party[slot]
            if current < threshold and hp_ratio(bench) > 0.5 and best_damage(bench) > max(0.4, 2.5 * current):
                return SWITCH_ACTION + slot

        if best_move >= 0:
            return best_move
        return legal.index(True)

    def _target(self, obs: dict) -> int:
        """Doubles: finish off an enemy we can KO (the strongest such), else hit the one we hurt most."""
        legal = [t for t, ok in enumerate(obs["legal"]) if ok]
        enemy_targets = [t for t in legal if t >= ENEMY_TARGET]
        if not enemy_targets:
            return legal[0]  # only allies are valid for this move
        actor = obs["party"][obs["actor"]]
        slot = obs["move"]["slot"]
        dmg = actor["moves"][slot]["dmg"] if slot is not None and slot < len(actor["moves"]) else []

        def score(t: int) -> tuple:
            i = t - ENEMY_TARGET
            d = dmg[i] if i < len(dmg) and dmg[i] is not None else 0.0
            return (d >= 1.0, obs["enemies"][i]["bst"] if d >= 1.0 else d)

        return max(enemy_targets, key=score)

    def _faint_switch(self, obs: dict) -> int:
        party = obs["party"]
        legal = [i for i, ok in enumerate(obs["legal"]) if ok]
        carry = self._carry_slot(party)
        if carry in legal:
            return carry
        return max(legal, key=lambda i: (best_damage(party[i]), hp_ratio(party[i])))

    def _release(self, obs: dict) -> int:
        party, caught = obs["party"], obs["caught"]
        releasable = [i for i, p in enumerate(party) if obs["legal"][i] and not self._is_protected(p)]
        if not releasable:
            return RELEASE_DECLINE
        worst = min(releasable, key=lambda i: member_value(party[i]))
        if caught["shiny"] or member_value(caught) > member_value(party[worst]):
            return worst
        return RELEASE_DECLINE

    # Rewards

    def _reward(self, obs: dict) -> int:
        party = obs["party"]
        missing = max((1 - hp_ratio(p) for p in party if not p["fainted"]), default=0.0)
        fainted = sum(p["fainted"] for p in party)
        carry = self._carry_slot(party)
        carry_fainted = carry is not None and party[carry]["fainted"]
        low_pp = any(m["pp"] / m["maxPp"] < 0.25 for p in party for m in p["moves"] if m["maxPp"])
        boss_next = obs["wave"] % 10 == 9

        best, best_score = None, SKIP_THRESHOLD
        for i, opt in enumerate(obs["options"]):
            if not opt["legal"] or opt["kind"] in ("skip", "reroll"):
                continue
            item = opt["id"]
            if item in HEALS:
                heal_needed = 0.1 if boss_next else 0.3
                score = min(HEALS[item], missing) * (1.5 if boss_next else 1.2) if missing > heal_needed else 0.0
            elif item in REVIVES:
                score = (1.6 if carry_fainted else 1.3) if fainted else 0.0
            elif item in TEAM_REVIVES:
                score = 1.5 if fainted >= 2 else 0.0
            elif item in PP_ITEMS:
                score = 0.5 if low_pp else 0.0
            elif item == "FULL_HEAL":
                score = 0.4 if any(p["status"] and not p["fainted"] for p in party) else 0.0
            elif item in BOSS_PREP and boss_next:
                score = 0.6
            elif opt["tmMove"] is not None:
                tm = opt["tmMove"]
                useful = (obs["wave"], tm["id"]) not in self.declined_tms and any(
                    learn_gain(tm, p) > 0 for p in party)
                score = self.item_value.get(item, DEFAULT_ITEM_VALUE) if useful else 0.0
            else:
                score = self.item_value.get(item, DEFAULT_ITEM_VALUE)
            if opt["kind"] == "shop":
                # Money is scarce: buy only revives for fainted members, or urgent heals.
                if item in REVIVES and fainted:
                    score += 0.1  # beats a free reward: the free one can still be taken after
                elif score <= 0.5:
                    score = 0.0
                else:
                    score -= 0.2
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
        carry = self._carry_slot(party)
        if self.pending_tm is not None:
            return max(legal, key=lambda i: learn_gain(self.pending_tm, party[i]))
        if item in HEALS:
            return max(legal, key=lambda i: party[i]["maxHp"] - party[i]["hp"])
        if item in REVIVES:
            return carry if carry in legal else max(legal, key=lambda i: party[i]["level"])
        if item in PP_ITEMS or item in ("PP_UP", "PP_MAX"):
            return min(legal, key=lambda i: min(m["pp"] / m["maxPp"] for m in party[i]["moves"] if m["maxPp"]))
        if item in GROWTH_ITEMS and carry in legal:
            return carry
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
