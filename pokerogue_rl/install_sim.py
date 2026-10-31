"""Copy the simulator sources from sim/ into the PokeRogue checkout.

Vite only loads test files that live inside pokerogue/ (symlinks out of it fail), so sim/ is
the source of truth and pokerogue/test/rl/ holds a generated copy. SimClient.launch() runs this
automatically; run it by hand before starting a server yourself:

    python -m pokerogue_rl.install_sim
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM_SRC = ROOT / "sim"
SIM_DEST = ROOT / "pokerogue" / "test" / "rl"


def install_sim() -> list[Path]:
    """Copy changed sim/*.ts files into the PokeRogue checkout; return the files copied."""
    SIM_DEST.mkdir(exist_ok=True)
    copied = []
    for src in SIM_SRC.glob("*.ts"):
        dest = SIM_DEST / src.name
        if not dest.exists() or dest.read_bytes() != src.read_bytes():
            shutil.copyfile(src, dest)
            copied.append(dest)
    return copied


if __name__ == "__main__":
    copied = install_sim()
    print(f"copied {len(copied)} file(s) to {SIM_DEST}" if copied else "simulator already up to date")
