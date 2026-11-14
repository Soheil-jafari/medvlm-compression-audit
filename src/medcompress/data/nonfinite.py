"""Test images whose fp16 vision tower output is not finite are dropped (pre-registration
section 10, second amendment of 2026-11-13, Soheil and Amir; before any test evaluation).

Any test image (the filtered and the unfiltered internal test, and later PolypGen) whose fp16
vision tower output is non-finite for the student **or** the teacher is dropped from that test for
every arm before any test evaluation; the list is committed and its count is reported next to the
test sizes. The list comes from RC-CAL0's scan (``scripts/scan_vision_finite.py``), converted by
``scripts/make_drop_list.py`` into ``results/data/nonfinite_drop.csv`` (rows ``split, sample_id,
encoder, first_non_finite``: one representative sample id per image). A listed image loses
**every** QA row of its split. The evaluation of a test split refuses to run without the list
(``nonfinite_drop.required`` in ``configs/base.yaml``) and records how many images and rows it
dropped.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from medcompress.config import ConfigError

if TYPE_CHECKING:
    from medcompress.data.base import VQASample

DROP_FILE = Path("data") / "nonfinite_drop.csv"
DROPPED_SPLITS: tuple[str, ...] = ("test", "test_unfiltered")
"""The splits the second amendment drops from (PolypGen joins when it is built)."""


def drop_path(results_root: Path | str) -> Path:
    """``results/data/nonfinite_drop.csv``."""
    return Path(results_root) / DROP_FILE


def read_drop(results_root: Path | str, split: str) -> set[str] | None:
    """The listed sample ids of ``split`` (None when no list is committed)."""
    path = drop_path(results_root)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        return {r["sample_id"] for r in csv.DictReader(handle) if r["split"] == split}


def apply_drop(samples: Sequence[VQASample], listed: set[str]) -> tuple[list[VQASample], int, int]:
    """``(kept rows, dropped images, dropped rows)``: every row of a listed image goes.

    Raises:
        ValueError: a listed id is not a row of this split (the list belongs to another split).
    """
    by_id = {s.sample_id: s for s in samples}
    missing = sorted(listed - set(by_id))
    if missing:
        raise ValueError(f"listed ids not in this split: {missing[:3]}")
    images = {by_id[i].image_path for i in listed}
    kept = [s for s in samples if s.image_path not in images]
    return kept, len(images), len(samples) - len(kept)


def drop_required(config: Mapping[str, Any]) -> bool:
    """True when test evaluations must apply the committed list (every real config)."""
    return bool((config.get("nonfinite_drop") or {}).get("required"))


def drop_for_split(
    config: Mapping[str, Any], split: str, samples: Sequence[VQASample]
) -> tuple[list[VQASample], dict[str, float]]:
    """The rows of ``split`` that every arm is evaluated on, and the counts to report.

    Raises:
        ConfigError: a test split without the committed list (when the config requires it).
    """
    if split not in DROPPED_SPLITS or not drop_required(config):
        return list(samples), {}
    listed = read_drop(config["paths"]["results_root"], split)
    if listed is None:
        where = drop_path(config["paths"]["results_root"])
        raise ConfigError(
            f"no committed list of non-finite test images ({where}): the test "
            "evaluations wait for it (second amendment of 2026-11-13, RC-CAL0)"
        )
    images_before = len({s.image_path for s in samples})
    kept, images, rows = apply_drop(samples, listed)
    counts = {
        "nonfinite_images_dropped": float(images),
        "nonfinite_rows_dropped": float(rows),
        "eval_images_before_drop": float(images_before),
        "eval_images_after_drop": float(images_before - images),
        "eval_rows_after_drop": float(len(kept)),
    }
    print(
        f"{split}: dropped {images} non-finite image(s) ({rows} rows) for every arm: "
        f"{images_before} -> {images_before - images} images, {len(samples)} -> {len(kept)} rows"
    )
    return kept, counts


def drop_rows_from_scans(
    student: Sequence[Mapping[str, str]], teacher: Sequence[Mapping[str, str]]
) -> list[tuple[str, str, str, str]]:
    """The committed list: the union of both encoders' non-finite images on the test splits
    (``split, sample_id, encoder, first_non_finite``; an image both encoders flag once, as
    ``student+teacher``)."""
    found: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for encoder, rows in (("student", student), ("teacher", teacher)):
        for row in rows:
            if row["split"] in DROPPED_SPLITS:
                found.setdefault((row["split"], row["sample_id"]), []).append(
                    (encoder, row.get("first_non_finite", ""))
                )
    out = []
    for (split, sample_id), hits in sorted(found.items()):
        encoders = "+".join(e for e, _ in hits)
        out.append((split, sample_id, encoders, hits[0][1]))
    return out
