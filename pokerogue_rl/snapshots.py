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
"""

from __future__ import annotations

import pickle
import random
import threading
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
                 max_waves: int | None = None, frontier: int = 0):
        self.lo, self.hi = waves
        self.capacity = capacity
        self.max_waves = max_waves
        self.frontier = frontier
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

    def sample(self, rng: random.Random) -> dict | None:
        with self.lock:
            waves = sorted(w for w, kept in self.by_wave.items() if kept)
            if not waves:
                return None
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
            data = {"by_wave": self.by_wave, "offered": self.offered}
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
        return added
