"""Test images not finite in fp16 are dropped for every arm (second amendment of 2026-11-13)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from medcompress.config import ConfigError
from medcompress.data import nonfinite


def _rows() -> list[SimpleNamespace]:
    rows = []
    for image in ("a", "b", "c"):
        for k in range(3):  # three QA rows per image
            rows.append(SimpleNamespace(sample_id=f"{image}:{k}", image_path=f"img/{image}.png"))
    return rows


def _write(root: Path, lines: list[str]) -> None:
    path = nonfinite.drop_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "split,sample_id,encoder,first_non_finite\n" + "".join(f"{line}\n" for line in lines)
    )


def test_a_listed_image_loses_every_row_of_its_split() -> None:
    kept, images, rows = nonfinite.apply_drop(_rows(), {"b:0"})
    assert (images, rows) == (1, 3) and {r.image_path for r in kept} == {"img/a.png", "img/c.png"}
    with pytest.raises(ValueError, match="not in this split"):
        nonfinite.apply_drop(_rows(), {"z:0"})


def test_test_splits_need_the_committed_list_and_report_the_counts(tmp_path: Path) -> None:
    config = {"paths": {"results_root": str(tmp_path)}, "nonfinite_drop": {"required": True}}
    samples = _rows()
    assert nonfinite.drop_for_split(config, "val", samples) == (samples, {})  # val untouched
    with pytest.raises(ConfigError, match="no committed list"):
        nonfinite.drop_for_split(config, "test", samples)
    _write(tmp_path, ["test,a:1,student,block 31", "test_unfiltered,c:2,student,block 31"])
    kept, counts = nonfinite.drop_for_split(config, "test", samples)
    assert len(kept) == 6 and counts == {
        "nonfinite_images_dropped": 1.0, "nonfinite_rows_dropped": 3.0,
        "eval_images_before_drop": 3.0, "eval_images_after_drop": 2.0, "eval_rows_after_drop": 6.0,
    }  # fmt: skip
    assert nonfinite.drop_for_split({"paths": config["paths"]}, "test", samples) == (samples, {})


def test_the_list_is_the_union_of_both_encoders_on_the_test_splits() -> None:
    student = [{"split": "val", "sample_id": "v:0", "first_non_finite": "block 31"},
               {"split": "test", "sample_id": "t:0", "first_non_finite": "block 31"},
               {"split": "test_unfiltered", "sample_id": "t:0", "first_non_finite": "block 31"},
               {"split": "train", "sample_id": "r:0", "first_non_finite": "block 31"}]  # fmt: skip
    teacher = [{"split": "test", "sample_id": "t:9", "first_non_finite": "block 2"},
               {"split": "test", "sample_id": "t:0", "first_non_finite": "block 30"}]  # fmt: skip
    rows = nonfinite.drop_rows_from_scans(student, teacher)
    assert rows == [
        ("test", "t:0", "student+teacher", "block 31"),
        ("test", "t:9", "teacher", "block 2"),
        ("test_unfiltered", "t:0", "student", "block 31"),
    ]  # validation and training images are not dropped by this rule


def test_the_real_configs_require_the_list() -> None:
    from medcompress.config import load_experiment

    for exp in ("E0", "E2", "E7", "E9"):
        assert nonfinite.drop_required(load_experiment(exp)[0]), exp
