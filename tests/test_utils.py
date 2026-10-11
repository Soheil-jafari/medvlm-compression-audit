"""Small utilities: results CSV writer, seeding, logging, hardware description."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest

from medcompress.compression.recovery import recovery_fingerprint
from medcompress.utils.hardware import describe_hardware
from medcompress.utils.io import RESULTS_COLUMNS, append_results
from medcompress.utils.logging import get_logger
from medcompress.utils.seed import set_seed


def test_append_results_schema(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    rows = [
        {
            "experiment": "E7",
            "budget": "B2",
            "seed": 0,
            "metric": "small_finding_recall",
            "value": 0.5,
            "ci_low": 0.4,
            "ci_high": 0.6,
        },
        {"experiment": "E3:w4", "seed": 0, "metric": "ece", "value": 0.1},
    ]
    assert append_results(rows, path) == 2
    assert append_results(rows[:1], path) == 1
    with path.open() as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == RESULTS_COLUMNS
        records = list(reader)
    assert len(records) == 3
    assert records[1]["budget"] == ""


def test_append_results_rejects_bad_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        append_results(
            [{"experiment": "E7", "seed": 0, "metric": "m", "value": 1, "x": 1}], tmp_path / "a.csv"
        )
    with pytest.raises(ValueError, match="Missing"):
        append_results([{"experiment": "E7", "metric": "m"}], tmp_path / "b.csv")


def test_set_seed_is_reproducible() -> None:
    set_seed(123)
    first = random.random()
    set_seed(123)
    assert random.random() == first


def test_logger_has_single_handler() -> None:
    logger = get_logger("medcompress.test")
    get_logger("medcompress.test")
    assert len(logger.handlers) == 1


def test_describe_hardware() -> None:
    info = describe_hardware().to_dict()
    assert info["python"]
    assert isinstance(info["gpus"], list)


def test_recovery_fingerprint_is_order_independent() -> None:
    assert recovery_fingerprint({"a": 1, "b": 2}) == recovery_fingerprint({"b": 2, "a": 1})
    assert recovery_fingerprint({"a": 1}) != recovery_fingerprint({"a": 2})
