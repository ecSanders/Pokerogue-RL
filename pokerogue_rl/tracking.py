"""Optional experiment tracking in ClearML (train_ppo.py --clearml): arguments, the per-update log as
charts, console output, and the checkpoint as a model.

Credentials come from ~/clearml.conf (`clearml-init`). Without --clearml nothing here imports ClearML.
A run keeps one ClearML task across Slurm requeues and resumes: its id is stored in
runs/<name>/clearml_task_id and continued on --resume.
"""

from __future__ import annotations

from pathlib import Path

PROJECT = "Pokerogue-RL"

# Log keys drawn together in one chart; every other numeric key gets a chart of its own.
CHARTS = {
    "wave_mean": "waves", "wave_median": "waves",
    "pg_loss": "loss", "v_loss": "loss",
    "entropy": "entropy", "run_entropy": "entropy",
}


def chart(key: str) -> tuple[str, str]:
    """(chart title, series) for a log key."""
    if key.startswith("gate") and key[4:].isdigit():
        return "gate clear rate", f"wave {key[4:]}"
    if key.startswith("gate") and key.endswith("_n"):
        return "gate reached (episodes)", f"wave {key[4:-2]}"
    return CHARTS.get(key, key), key


class Tracker:
    """No-op unless `enabled`. Call `log` once per update and `checkpoint` after writing policy.pt."""

    def __init__(self, enabled: bool, name: str, out: Path, config: dict, resume: bool = False,
                 upload_every: int = 0):
        self.task = None
        self.models = {}  # checkpoint file stem -> OutputModel
        self.upload_every = upload_every
        self.offset = 0
        if not enabled:
            return
        from clearml import Task

        id_file = out / "clearml_task_id"
        previous = id_file.read_text().strip() if resume and id_file.exists() else None
        self.task = Task.init(
            project_name=PROJECT,
            task_name=name,
            continue_last_task=previous or False,
            reuse_last_task_id=False,  # a new run with an old name gets its own task
            # Checkpoints are uploaded explicitly (`checkpoint`), not on every torch.save of policy.pt.tmp.
            auto_connect_frameworks={"pytorch": False},
            auto_connect_arg_parser=False,
            output_uri=True,  # upload checkpoints to the ClearML files server, not just record their path
        )
        # A continued task adds its last iteration to every reported one; ours are absolute update numbers.
        self.offset = self.task.get_initial_iteration()
        self.task.connect(config, name="args")
        id_file.write_text(self.task.id + "\n")
        self.logger = self.task.get_logger()

    def log(self, row: dict, iteration: int) -> None:
        if not self.task:
            return
        for key, value in row.items():
            if key in ("update", "step") or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            title, series = chart(key)
            self.logger.report_scalar(title, series, value=value, iteration=iteration - self.offset)

    def checkpoint(self, path: Path, update: int, final: bool = False) -> None:
        """Upload the checkpoint every `upload_every` updates (0: only the final one) and at the end."""
        if not self.task or not (final or (self.upload_every and update % self.upload_every == 0)):
            return
        if path.stem not in self.models:
            from clearml import OutputModel
            self.models[path.stem] = OutputModel(task=self.task, name=path.stem, framework="PyTorch")
        self.models[path.stem].update_weights(weights_filename=str(path), iteration=update, auto_delete_file=False,
                                              async_enable=False)  # a background upload can be cut off at exit

    def close(self) -> None:
        if self.task:
            self.task.flush(wait_for_uploads=True)
            self.task.close()
