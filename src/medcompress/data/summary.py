"""Values read from the committed data summary (``results/data/summary.csv``, RC-D1).

Some settings are fixed by a rule on the committed counts instead of being typed by hand
(pre-registration amendment of 2026-10-31, before any model result). The generation limit of
D04 is one: a config holds the rule's name, ``d04_rule``, and :func:`resolve_max_new_tokens`
computes ``ceil(1.5 x p99 answer words) + 8`` from the file each time a run reads it. So the
number always follows the committed counts, and nobody copies it by hand.

Only aggregate rows (experiment ``DATA``) are read; the file holds no per-image row.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SUMMARY_FILE = Path("results") / "data" / "summary.csv"
"""The committed RC-D1 summary, relative to the repository root."""
DATA_EXPERIMENT = "DATA"
D04_RULE = "d04_rule"
"""Config value that stands for D04's generation limit, computed from the summary."""
D04_FACTOR = 1.5
D04_MARGIN = 8
P99_WORDS = "answer_words:p99"
"""Training answers' 99th percentile length in words (``prepare.py``)."""
ABNORMALITY_RULE = "abnormalities"


def summary_path() -> Path:
    """The committed summary of the repository this code runs from."""
    from medcompress.config import repo_root

    return repo_root() / SUMMARY_FILE


def read_summary(path: str | Path | None = None) -> dict[str, float]:
    """``{metric: value}`` of the ``DATA`` rows of the summary.

    Raises:
        ConfigError: the file is missing (RC-D1's summary must be committed).
    """
    from medcompress.config import ConfigError

    path = Path(path) if path is not None else summary_path()
    if not path.is_file():
        raise ConfigError(f"{path} is missing: the RC-D1 summary must be committed")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["metric"]: float(row["value"])
            for row in csv.DictReader(handle)
            if row.get("experiment") == DATA_EXPERIMENT and row.get("value") not in (None, "")
        }


def d04_max_new_tokens(values: Mapping[str, float]) -> int:
    """D04's rule: ``ceil(1.5 x p99 answer words) + 8`` (``answer_words:p99`` of the summary)."""
    from medcompress.config import ConfigError

    if P99_WORDS not in values:
        raise ConfigError(f"the summary has no {P99_WORDS} row (RC-D1 writes it)")
    p99 = float(values[P99_WORDS])
    if not math.isfinite(p99) or p99 <= 0:
        raise ConfigError(f"{P99_WORDS} must be a positive number, got {p99}")
    return math.ceil(D04_FACTOR * p99) + D04_MARGIN


def resolve_max_new_tokens(value: Any, path: str | Path | None = None) -> Any:
    """The generation limit a config value stands for.

    ``d04_rule`` becomes :func:`d04_max_new_tokens` of the committed summary (``path`` for
    tests); any other value is returned unchanged, so an open ``TODO`` or ``None`` stays open
    for the caller's own check.
    """
    if isinstance(value, str) and value.strip() == D04_RULE:
        return d04_max_new_tokens(read_summary(path))
    return value


def answer_images(
    values: Mapping[str, float], answer_value: str, split: str, rule: str = ABNORMALITY_RULE
) -> int:
    """Images of ``split`` whose ``rule`` answer holds ``answer_value`` (0 if no row)."""
    return int(values.get(f"answer_count:{rule}:{answer_value}:{split}", 0.0))


def training_prevalence(
    values: Mapping[str, float], answer_value: str, rule: str = ABNORMALITY_RULE
) -> float:
    """Training images with the answer value / training images (the prevalence of D02)."""
    from medcompress.config import ConfigError

    images = values.get("images:train")
    if not images:
        raise ConfigError("the summary has no images:train row")
    return answer_images(values, answer_value, "train", rule) / float(images)
