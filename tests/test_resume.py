"""Resume state of a training run: saved every N steps, swapped in whole, refused for another run.

Torch-free: a pure-Python loop stands in for ``train_lora`` (``tests/test_training.py`` runs
the same check on the tiny model when torch is installed).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medcompress.compression.distillation.resume import (
    PROGRESS_FILE,
    Progress,
    ResumeError,
    ResumeStore,
    run_fingerprint,
)

STEPS = 7


def _train(
    folder: Path | None, *, stop_at: int | None = None, lr: float = 0.5
) -> tuple[list, Progress]:
    """A deterministic loop with state (``weights``), saved and restored like ``train_lora``."""
    fingerprint = run_fingerprint({"lr": lr, "steps": STEPS}, 0, ["a", "b", "c"])
    weights = [1.0, -2.0]
    progress = Progress(fingerprint)
    store = None
    if folder is not None:
        store = ResumeStore(folder, fingerprint, every=3)

        def load(source: Path) -> None:
            weights[:] = json.loads((source / "w.json").read_text())

        progress = store.load(load) or progress
    for step in range(progress.step, STEPS):
        if step == stop_at:
            raise RuntimeError("session ended")
        weights[:] = [w - lr * (w - step) for w in weights]
        progress.losses.append(sum(abs(w) for w in weights))
        progress.step, progress.cursor = step + 1, progress.cursor + 2
        if store is not None and store.due(progress.step, STEPS):
            store.save(progress, lambda target: (target / "w.json").write_text(json.dumps(weights)))
    return weights, progress


def test_a_stopped_run_continues_from_the_last_saved_step(tmp_path: Path) -> None:
    reference, ref_progress = _train(None)
    folder = tmp_path / "resume"
    with pytest.raises(RuntimeError):
        _train(folder, stop_at=5)  # saved after step 3, stopped in step 6
    saved = json.loads((folder / "current" / PROGRESS_FILE).read_text())
    assert saved["step"] == 3 and len(saved["losses"]) == 3
    weights, progress = _train(folder)
    assert weights == reference and progress.losses == ref_progress.losses
    assert progress.cursor == ref_progress.cursor
    assert json.loads((folder / "current" / PROGRESS_FILE).read_text())["step"] == STEPS
    again, _ = _train(folder)  # the final state is saved: nothing is trained twice
    assert again == reference


def test_another_run_is_refused_instead_of_mixed(tmp_path: Path) -> None:
    folder = tmp_path / "resume"
    with pytest.raises(RuntimeError):
        _train(folder, stop_at=4)
    with pytest.raises(ResumeError, match="belongs to another run"):
        _train(folder, lr=0.25)


def test_a_stop_during_a_save_keeps_the_previous_state(tmp_path: Path) -> None:
    store = ResumeStore(tmp_path, "f", every=1)
    store.save(Progress("f", step=1), lambda t: (t / "w.json").write_text("[1]"))

    def broken(target: Path) -> None:
        (target / "w.json").write_text("[2")
        raise OSError("disk full")

    with pytest.raises(OSError):
        store.save(Progress("f", step=2), broken)
    seen = []
    loaded = store.load(lambda source: seen.append((source / "w.json").read_text()))
    assert loaded is not None and loaded.step == 1 and seen == ["[1]"]
    (tmp_path / "current").rename(tmp_path / "old")  # a stop between the two renames
    assert store.load(lambda source: None).step == 1
    store.clear()
    assert not tmp_path.exists() and store.load(lambda source: None) is None
    with pytest.raises(ValueError, match="at least 1"):
        ResumeStore(tmp_path, "f", every=0)
