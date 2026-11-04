"""Upload finished runs to ClearML: config.json as arguments, log.jsonl as charts, policy.pt as a model.

For runs trained without --clearml (e.g. at home). Needs ~/clearml.conf (`clearml-init`).

Usage:
  python scripts/clearml_upload.py runs/run-exp-v2 runs/bc-run
  python scripts/clearml_upload.py runs/v3-*            # skips runs that already have a ClearML task
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pokerogue_rl.tracking import Tracker  # noqa: E402


def upload(run: Path, force: bool) -> None:
    if (run / "clearml_task_id").exists() and not force:
        print(f"{run}: already tracked (task {(run / 'clearml_task_id').read_text().strip()}); --force to upload again")
        return
    config = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    tracker = Tracker(True, run.name, run, config)
    rows = [json.loads(line) for line in (run / "log.jsonl").read_text().splitlines()] \
        if (run / "log.jsonl").exists() else []
    for row in rows:
        tracker.log(row, row["update"])
    checkpoints = sorted(run.glob("*.pt"))
    for ckpt in checkpoints:
        tracker.checkpoint(ckpt, rows[-1]["update"] if rows else 0, final=True)
    tracker.close()
    print(f"{run}: {len(rows)} log rows, {len(checkpoints)} checkpoint(s) -> task {tracker.task.id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--force", action="store_true", help="upload runs that already have a ClearML task")
    args = parser.parse_args()
    for run in args.runs:
        upload(run, args.force)


if __name__ == "__main__":
    main()
