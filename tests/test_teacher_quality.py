"""Rule 3, quality part: E2 waits until E0 is at least E1 seed 0 + 0.02 (or was extended)."""

from __future__ import annotations

from pathlib import Path

import pytest

from medcompress.config import ConfigError
from medcompress.models.teacher_training import require_teacher_quality, teacher_quality
from medcompress.utils.io import append_results


def _write(root: Path, run: str, **values: float) -> None:
    rows = [{"experiment": run, "budget": "", "seed": 0, "metric": k, "value": v}
            for k, v in values.items()]  # fmt: skip
    append_results(rows, root / run / "nobudget" / "seed0" / "metrics.csv")


def test_the_quality_rule_reads_the_committed_validation_rows(tmp_path: Path) -> None:
    assert teacher_quality(tmp_path, 0.02)[0] is None
    _write(tmp_path, "E0", macro_balanced_accuracy=0.80, teacher_extension_examples=0.0)
    passed, why = teacher_quality(tmp_path, 0.02)
    assert passed is None and "E1 seed 0" in why
    _write(tmp_path, "E1", macro_balanced_accuracy=0.77)
    assert teacher_quality(tmp_path, 0.02)[0] is True  # 0.80 >= 0.79


def test_a_failing_teacher_stops_e2_unless_it_was_extended(tmp_path: Path) -> None:
    _write(tmp_path, "E0", macro_balanced_accuracy=0.80, teacher_extension_examples=0.0)
    _write(tmp_path, "E1", macro_balanced_accuracy=0.79)
    config = {"paths": {"results_root": str(tmp_path)}}
    with pytest.raises(ConfigError, match="rule 3 fails: E0 0.8000 < E1 seed 0 0.7900"):
        require_teacher_quality(config)
    extended = tmp_path / "extended"
    _write(extended, "E0", macro_balanced_accuracy=0.80, teacher_extension_examples=19000.0)
    _write(extended, "E1", macro_balanced_accuracy=0.79)
    require_teacher_quality({"paths": {"results_root": str(extended)}})  # accepted either way


def test_e2_waits_while_e1_seed0_is_missing(tmp_path: Path) -> None:
    _write(tmp_path, "E0", macro_balanced_accuracy=0.80)
    with pytest.raises(ConfigError, match="cannot be read yet"):
        require_teacher_quality({"paths": {"results_root": str(tmp_path)}})
