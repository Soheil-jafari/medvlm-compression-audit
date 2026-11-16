"""``scripts/check_nonfinite_steps.py``: the training loop's batch order, rebuilt (2026-11-16)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_nonfinite_steps.py"


def _module():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("check_nonfinite_steps", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_steps_follow_the_loop_and_wrap_around_the_pool() -> None:
    m = _module()
    order = [3, 0, 4, 1, 2]
    positions = m.step_positions(5, 2, 3, order)
    assert positions == [[3, 0], [4, 1], [2, 3]]  # the third step wraps to the start
    images = ["a", "b", "bad", "c", "d"]
    assert m.flagged_steps(images, positions, {"bad"}) == {3: [2]}
    assert m.flagged_steps(["bad", "b", "c", "d", "e"], positions, {"bad"}) == {1: [0]}
    flagged = {3: [2], 30: [1], 60: [0], 61: [4]}
    assert m.cumulative(flagged, 70) == [(25, 1), (50, 2), (70, 4)]


def test_the_order_is_the_training_loops() -> None:
    """The script must visit the pool in the order ``train_lora`` uses (same function, seed)."""
    from medcompress.compression.distillation.offline_kd import select_subset

    ids = [f"s{i}" for i in range(40)]
    assert select_subset(ids, None, 0) == select_subset(list(ids), None, 0)
    assert select_subset(ids, None, 0) != select_subset(ids, None, 1)


def test_the_script_runs_end_to_end_on_a_fake_split(
    tmp_path: Path,
    monkeypatch,
    capsys,  # noqa: ANN001
) -> None:
    """``main`` with the real E5 recovery settings, a fake training split and a scan file."""
    from medcompress.compression.pruning import stage
    from medcompress.data.base import VQASample

    train = [VQASample(f"s{i:03d}", "d", f"images/{i % 40}.jpg", "q", "a") for i in range(100)]
    monkeypatch.setattr(stage, "load_train_samples", lambda config: train)
    scan = tmp_path / "vision_scan_student.csv"
    scan.write_text("split,sample_id,first_non_finite\ntrain,s007,block 31\nval,x,block 31\n"
                    "train,unknown,block 31\n")  # fmt: skip
    m = _module()
    assert m.main(["--exp", "E5", "--variant", "magnitude", "--budget", "B2", "--seed", "0",
                   "--scan", str(scan)]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "157 steps of 16" in out and "1 listed training images (1 not in the split)" in out
    assert "3 pool rows show one of them" in out  # images/7.jpg: rows 7, 47, 87
    assert "cumulative, as the log prints it: step 25:" in out and "step 157:" in out
