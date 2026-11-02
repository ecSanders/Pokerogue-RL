"""Per-run measures of the strategies that lead to long endless streaks (from player experience):

- a carry: one Pokemon takes the items and EXP (carry level well above the team average)
- a Pickup user in the team
- Pokerus kept until the whole team has it
- double battles sought with lures, for more EXP

StrategyTracker watches every decision of a run (`note`) and summarises it at the end (`summary`).
"""

from __future__ import annotations

PICKUP = 53  # AbilityId.PICKUP
LURES = {"LURE", "SUPER_LURE", "MAX_LURE"}


def pokerus_count(party: list[dict]) -> int:
    return sum(bool(p["pokerus"]) for p in party)


def carry_level(party: list[dict]) -> int:
    return max((p["level"] for p in party), default=0)


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
        }
