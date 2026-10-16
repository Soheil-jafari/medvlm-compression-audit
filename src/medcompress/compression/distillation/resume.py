"""Resume state for a training run that stops (a Kaggle session that ends, an error).

Every ``checkpoint_every`` optimizer steps, and after the last one, ``train_lora`` saves its
state in the run's resume folder: the trainable (adapter) weights, the optimizer and
gradient-scaler state, the random-number states, and the progress (step, position in the
example order, losses so far, seconds so far). A run started again with the same settings
continues from the last saved step instead of from the start; with other settings or other
examples it stops (:class:`ResumeError`) instead of mixing two runs.

The folder holds ``current/`` (the last complete state). A new state is written to ``next/``
and only then swapped in, so a stop in the middle of a save leaves the previous state usable.
This module has no torch dependency: the tensors are saved and loaded by callbacks.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PROGRESS_FILE = "progress.json"
DEFAULT_CHECKPOINT_EVERY = 50
"""Optimizer steps between two saves (at 1 x 16 and side 672 about 11 minutes of training,
an estimate from RC-S0b's 0.85 s per example); the last step is always saved."""


class ResumeError(ValueError):
    """The saved state belongs to another run (other settings or examples)."""


@dataclass
class Progress:
    """How far a training run got; saved next to the tensors."""

    fingerprint: str
    step: int = 0
    cursor: int = 0
    examples_seen: int = 0
    seconds: float = 0.0
    losses: list[float] = field(default_factory=list)
    ce_losses: list[float] = field(default_factory=list)
    kd_losses: list[float] = field(default_factory=list)


def run_fingerprint(
    settings: Mapping[str, Any], seed: int, example_ids: Sequence[str], model: str | None = None
) -> str:
    """SHA-256 over what makes two training runs the same run.

    ``settings`` are the training settings without the checkpoint ones; the example ids are
    taken in the order the run visits them.
    """
    ids = hashlib.sha256("\n".join(example_ids).encode("utf-8")).hexdigest()
    text = json.dumps(
        {"settings": settings, "seed": seed, "examples": ids, "model": model},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def settings_for_fingerprint(cfg: Any) -> dict[str, Any]:
    """The training settings of a ``TrainConfig`` that define the run (checkpointing excluded)."""
    data = asdict(cfg)
    data["extra"] = {
        k: v for k, v in dict(data.get("extra") or {}).items() if k != "checkpoint_every"
    }
    return data


class ResumeStore:
    """The resume folder of one training run."""

    def __init__(self, folder: Path, fingerprint: str, *, every: int = DEFAULT_CHECKPOINT_EVERY):
        if every < 1:
            raise ValueError("checkpoint_every must be at least 1")
        self.folder = Path(folder)
        self.fingerprint = fingerprint
        self.every = int(every)

    @property
    def current(self) -> Path:
        return self.folder / "current"

    def _readable(self) -> Path | None:
        for name in ("current", "old"):  # "old" only exists if a stop came during the swap
            candidate = self.folder / name
            if (candidate / PROGRESS_FILE).is_file():
                return candidate
        return None

    def load(self, load_state: Callable[[Path], None]) -> Progress | None:
        """The saved progress (after ``load_state(folder)`` restored the tensors), or ``None``."""
        source = self._readable()
        if source is None:
            return None
        progress = Progress(**json.loads((source / PROGRESS_FILE).read_text(encoding="utf-8")))
        if progress.fingerprint != self.fingerprint:
            raise ResumeError(
                f"the resume state in {self.folder} belongs to another run (its settings or "
                "examples differ); remove that folder to start this run from the beginning"
            )
        load_state(source)
        return progress

    def due(self, steps_done: int, total: int) -> bool:
        """Save after every ``every`` steps and after the last step."""
        return steps_done == total or steps_done % self.every == 0

    def save(self, progress: Progress, save_state: Callable[[Path], None]) -> None:
        """Write the tensors (``save_state(folder)``) and the progress, then swap them in."""
        if progress.fingerprint != self.fingerprint:
            raise ResumeError("progress of another run")
        nxt, old = self.folder / "next", self.folder / "old"
        shutil.rmtree(nxt, ignore_errors=True)
        nxt.mkdir(parents=True)
        save_state(nxt)
        (nxt / PROGRESS_FILE).write_text(json.dumps(asdict(progress)), encoding="utf-8")
        shutil.rmtree(old, ignore_errors=True)
        if self.current.exists():
            self.current.rename(old)
        nxt.rename(self.current)
        shutil.rmtree(old, ignore_errors=True)

    def clear(self) -> None:
        """Remove the folder once the run's final output is saved."""
        shutil.rmtree(self.folder, ignore_errors=True)
