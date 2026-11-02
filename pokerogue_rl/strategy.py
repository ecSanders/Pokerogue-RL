"""Per-run measures of the strategies that lead to long endless streaks (from player experience):

- a carry: one Pokemon takes the items and EXP (carry level well above the team average)
- a Pickup user in the team
- Pokerus kept until the whole team has it
- double battles sought with lures, for more EXP
- a carry at or above the level of the boss every 10th wave (most runs die to these bosses)

StrategyTracker watches every decision of a run (`note`) and summarises it at the end (`summary`).
"""

from __future__ import annotations

import math

PICKUP = 53  # AbilityId.PICKUP
LURES = {"LURE", "SUPER_LURE", "MAX_LURE"}


def pokerus_count(party: list[dict]) -> int:
    return sum(bool(p["pokerus"]) for p in party)


def carry_level(party: list[dict]) -> int:
    return max((p["level"] for p in party), default=0)


def boss_level(wave: int) -> int:
    """Expected level of the endless-mode boss on `wave` (a multiple of 10), as Battle.getLevelForWave
    computes it before a random offset of up to +-(wave // 10); every 250th wave rounds up to 25s."""
    level = math.floor((1 + wave / 2 + (wave / 25) ** 2) * 1.2)
    return math.ceil(level / 25) * 25 if wave % 250 == 0 else level


def boss_gap(wave: int, party: list[dict]) -> int:
    """Carry level minus the expected boss level of `wave`."""
    return carry_level(party) - boss_level(wave)


class StrategyTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.waves: set[int] = set()
        self.double_waves: set[int] = set()
        self.pickup_waves: set[int] = set()
        self.lures = 0
        self.balls_thrown = 0
        self.commands = 0
        self.full_pokerus_wave: int | None = None
        self.last_party: list[dict] = []
        self.boss_gaps: dict[int, int] = {}  # boss wave -> carry level minus boss level, on arrival

    def note(self, obs: dict, action: int) -> None:
        wave, party = obs["wave"], obs["party"]
        if party:
            self.last_party = party
        if obs["kind"] == "command":
            self.waves.add(wave)
            self.commands += 1
            self.balls_thrown += action >= 10
            if obs["double"]:
                self.double_waves.add(wave)
            if any(p.get("ability") == PICKUP for p in party):
                self.pickup_waves.add(wave)
            if wave % 10 == 0 and wave not in self.boss_gaps:
                self.boss_gaps[wave] = boss_gap(wave, party)
            if self.full_pokerus_wave is None and len(party) == 6 and pokerus_count(party) == 6:
                self.full_pokerus_wave = wave
        elif obs["kind"] == "reward" and 0 <= action < len(obs["options"]):
            self.lures += obs["options"][action]["id"] in LURES

    def summary(self, final_obs: dict | None = None) -> dict:
        party = (final_obs or {}).get("party") or self.last_party
        levels = [p["level"] for p in party]
        n = max(len(self.waves), 1)
        return {
            "double_frac": round(len(self.double_waves) / n, 3),
            "pickup_frac": round(len(self.pickup_waves) / n, 3),
            "lures": self.lures,
            "ball_frac": round(self.balls_thrown / max(self.commands, 1), 3),
            "carry_level": carry_level(party),
            "carry_gap": round(carry_level(party) - sum(levels) / len(levels), 1) if levels else 0.0,
            "pokerus": pokerus_count(party),
            "full_pokerus_wave": self.full_pokerus_wave,
            # At the last boss wave reached, usually the one that ended the run (the early bosses are
            # easy, so a mean over all of them hides the shortfall). None if the run reached none.
            "boss_gap": self.boss_gaps[max(self.boss_gaps)] if self.boss_gaps else None,
        }
