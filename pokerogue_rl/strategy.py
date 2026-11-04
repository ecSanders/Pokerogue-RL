"""Per-run measures of the strategies that lead to long endless streaks (from player experience):

- a carry: one Pokemon takes the items and EXP (carry level well above the team average)
- a Pickup user in the team
- Pokerus kept until the whole team has it
- double battles sought with lures, for more EXP
- a carry at or above the level of the boss every 50th wave: there the End biome's Paradox boss can't
  be caught, so it must be beaten (other bosses are mostly caught once their shields break)

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


UNCATCHABLE_BOSS_EVERY = 50  # endless: every 50th wave is the End biome, where the boss can't be caught


def gate_clears(runs: list[tuple], every: int = UNCATCHABLE_BOSS_EVERY) -> dict[int, tuple[int, int]]:
    """Per gate wave (every 50th: the uncatchable boss), (runs that cleared it, runs that reached it).

    `runs` holds each run's final (wave, reason), or (wave, reason, start wave) for runs that began
    from a snapshot; a run only counts for gates after its start. A run reached gate g if it got past g
    or lost on g; it cleared g if it got past it. A run cut off on g for another reason
    (decision_limit) counts for neither. On "max_wave" the final wave is max_wave + 1, so a run capped
    at a gate cleared it.
    """
    out: dict[int, tuple[int, int]] = {}
    runs = [(r[0], r[1], r[2] if len(r) > 2 else 1) for r in runs]
    top = max((w for w, _, _ in runs), default=0)
    for g in range(every, top + 1, every):
        cleared = sum(w > g and s < g for w, _, s in runs)
        lost = sum(w == g and r == "game_over" and s < g for w, r, s in runs)
        if cleared + lost:
            out[g] = (cleared, cleared + lost)
    return out


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
        self.boss_gaps: dict[int, int] = {}  # uncatchable boss wave -> carry minus boss level, on arrival

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
            if wave % UNCATCHABLE_BOSS_EVERY == 0 and wave not in self.boss_gaps:
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
            # At the last uncatchable boss reached (wave 50, 100, ...). None if the run reached none.
            "boss_gap": self.boss_gaps[max(self.boss_gaps)] if self.boss_gaps else None,
        }
