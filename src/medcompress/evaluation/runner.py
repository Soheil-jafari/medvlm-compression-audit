"""Runs the full evaluation suite for one run and writes results rows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medcompress.config import RunContext

EVALUATION_SUITES: tuple[str, ...] = (
    "clinical",
    "small_findings",
    "rare_findings",
    "image_reliance",
    "calibration",
    "hallucination",
    "language",
)


def run_evaluation(config: Mapping[str, Any], run: RunContext | None) -> None:
    """Stage entry point for ``scripts/06_evaluate.py``.

    Loads the model for ``run`` (checkpoint under paths.checkpoint_root), runs
    every suite enabled in ``evaluation.suites`` on the frozen test split (and
    the deduplicated external test set if configured), and appends rows to
    ``<run_dir>/metrics.csv`` in the results schema.

    Used by: all experiments (E0-E9, A1-A6).
    """
    raise NotImplementedError("TODO: load model, run enabled suites, write metrics.csv.")
