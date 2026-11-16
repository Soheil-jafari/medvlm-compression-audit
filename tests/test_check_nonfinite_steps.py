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
