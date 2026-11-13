"""Snapshot bank and start-wave curriculum.

Training runs pass through waves the policy rarely gets to practise: whatever decides the wave-50
boss happens many waves before it, and a run reaches it at most once. The bank keeps wave-start
snapshots (BattleEnv.snapshot) from a range of waves, collected from training's own episodes as they
pass through it, and a fraction of new episodes start from one of them (BattleEnv.reset_from), re-seeded
so they are new games from that point rather than replays.

One bank is shared by the envs of a trainer process (they step in threads). It keeps up to `capacity`
snapshots per wave, a uniform sample of those offered (reservoir sampling), so later, stronger
policies keep refreshing it. Sampling picks a wave uniformly, then a snapshot of that wave, so waves
seen less often are practised as often as common ones.

Frontier mode (an open range like "20-", with `frontier` and `max_waves`): the bank takes every wave
from the first one up, drops the shallowest waves once it holds `max_waves` different waves, and
samples starts among the `frontier` deepest waves it holds, so practice follows the furthest point
the policy reaches (wave 50, then 100, 150, ...).

Deaths mode (sampling="deaths", 2026-11-09): frontier mode let snapshot-started episodes feed the bank,
so the frontier ratcheted to waves 500-750 by chaining segments (each gaining ~2 waves) while episodes
from wave 1 still died around 50-250, and its runs did no better than their no-curriculum controls.
In deaths mode the env feeds the bank only from episodes that started at wave 1, records where those
episodes die (`note_death`), and each start picks a recent death wave d and starts at the deepest
snapshot at or before d - lead. Starts land just before where the policy really dies, in proportion
to how often it dies there (boss walls like Eternatus at 250 get the most practice).

Focus (focus=250, focus_frac=0.25, 2026-11-13): few episodes from wave 1 even reach Eternatus (~1-2%),
so deaths there are rare and so is practice. A `focus_frac` share of starts instead begins 1 to
`focus_lead` waves before the focus wave (the deepest held snapshot at or below that), so a wall the
policy rarely reaches still gets steady practice.
"""

from __future__ import annotations

import collections
import pickle
import random
import threading
from bisect import bisect_right
from pathlib import Path


def parse_waves(spec: str) -> tuple[int, int | None]:
    """"20-45" -> (20, 45); "30" -> (30, 30); "20-" -> (20, None), open-ended."""
    lo, dash, hi = spec.partition("-")
    lo = int(lo)
    hi = None if dash and not hi else int(hi or lo)
    if not 1 < lo or (hi is not None and hi < lo):
        raise ValueError(f"bad wave range {spec!r}: need 1 < first <= last")
    return lo, hi


class SnapshotBank:
    def __init__(self, waves: tuple[int, int | None], capacity: int = 64, seed: int = 0,
                 max_waves: int | None = None, frontier: int = 0, sampling: str = "uniform", lead: int = 5,
                 deaths_window: int = 1000, focus: int | None = None, focus_frac: float = 0.0,
                 focus_lead: int = 10):
        if sampling not in ("uniform", "deaths"):
            raise ValueError(f"unknown sampling {sampling!r}")
        self.lo, self.hi = waves
        self.capacity = capacity
        self.max_waves = max_waves
        self.frontier = frontier
        self.sampling = sampling
        self.lead = lead
        self.deaths: collections.deque[int] = collections.deque(maxlen=deaths_window)
        self.focus, self.focus_frac, self.focus_lead = focus, focus_frac, focus_lead
        self.by_wave: dict[int, list[dict]] = {}
        self.offered: dict[int, int] = {}  # snapshots offered per wave, for reservoir sampling
        self.rng = random.Random(seed)
        self.lock = threading.Lock()

    def wants(self, wave: int) -> bool:
        return self.lo <= wave and (self.hi is None or wave <= self.hi)

    def add(self, snap: dict) -> None:
        """Offer a snapshot (from BattleEnv.snapshot). Kept with probability capacity / offered so far."""
        wave = snap["wave"]
        if not self.wants(wave):
            return
        entry = {"wave": wave, "session": snap["session"], "state": snap["state"]}  # obs isn't needed to load
        with self.lock:
            kept = self.by_wave.setdefault(wave, [])
            self.offered[wave] = self.offered.get(wave, 0) + 1
            if len(kept) < self.capacity:
                kept.append(entry)
            else:
                i = self.rng.randrange(self.offered[wave])
                if i < self.capacity:
                    kept[i] = entry
            if self.max_waves and len(self.by_wave) > self.max_waves:
                for w in sorted(self.by_wave)[: len(self.by_wave) - self.max_waves]:
                    del self.by_wave[w]  # keep the deepest waves (frontier mode)

    def note_death(self, wave: int) -> None:
        """Record the wave an episode from wave 1 was lost on (deaths mode)."""
        with self.lock:
            self.deaths.append(wave)

    def death_quantiles(self, qs=(0.5, 0.9)) -> list[int] | None:
        with self.lock:
            d = sorted(self.deaths)
        return [d[min(len(d) - 1, int(q * len(d)))] for q in qs] if d else None

    def sample(self, rng: random.Random) -> dict | None:
        """A snapshot to start from, or None for a fresh start (deaths mode: no snapshot before the
        chosen death wave yet)."""
        with self.lock:
            waves = sorted(w for w, kept in self.by_wave.items() if kept)
            if not waves:
                return None
            if self.focus and rng.random() < self.focus_frac:
                i = bisect_right(waves, self.focus - rng.randint(1, self.focus_lead))
                if i:
                    return rng.choice(self.by_wave[waves[i - 1]])
            if self.sampling == "deaths":
                if not self.deaths:
                    return None
                target = rng.choice(self.deaths) - self.lead
                i = bisect_right(waves, target)
                return rng.choice(self.by_wave[waves[i - 1]]) if i else None
            if self.frontier:
                waves = waves[-self.frontier:]
            return rng.choice(self.by_wave[rng.choice(waves)])

    def __len__(self) -> int:
        with self.lock:
            return sum(len(kept) for kept in self.by_wave.values())

    def deepest(self) -> int:
        with self.lock:
            return max((w for w, kept in self.by_wave.items() if kept), default=0)

    def waves_held(self) -> int:
        with self.lock:
            return sum(bool(kept) for kept in self.by_wave.values())

    def save(self, path: Path) -> None:
        """Write then rename, so a job killed mid-save keeps the previous bank."""
        with self.lock:
            data = {"by_wave": self.by_wave, "offered": self.offered, "deaths": list(self.deaths)}
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    def load(self, path: Path) -> int:
        """Add the snapshots in a saved bank (this run's, on resume, or another run's to seed it) that
        fall in this bank's wave range. Returns how many were added."""
        with path.open("rb") as f:
            data = pickle.load(f)
        added = 0
        with self.lock:
            for wave, kept in data["by_wave"].items():
                if not self.wants(wave):
                    continue
                mine = self.by_wave.setdefault(wave, [])
                room = self.capacity - len(mine)
                mine.extend(kept[:room])
                added += min(room, len(kept))
                self.offered[wave] = self.offered.get(wave, 0) + data["offered"].get(wave, len(kept))
            if self.max_waves and len(self.by_wave) > self.max_waves:
                for w in sorted(self.by_wave)[: len(self.by_wave) - self.max_waves]:
                    del self.by_wave[w]
            self.deaths.extend(data.get("deaths", []))
        return added
