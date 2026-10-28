"""The rule sweep of D02 amendment 5 (2026-10-28; post hoc, after RC-D1 run 4 failed its
deciding check).

Run 4 joined images by the rule chosen on the both-raters truth (full:0.93). Its final contact
sheet still showed many same-scene pairs across splits. Amendment 5 takes the truth
"`same_scene` for either rater". The rule chosen on it by amendment 3's criterion is read from
the committed labels file (``labels:either_rater:rule:*``), not written here. Then a short
sweep of rules ``full:T OR crop >= 0.80``, fixed before any new sheet is seen.

For each rule, highest T first (:func:`rule_sweep`):
1. the groups over all pairs of images, their largest group and its fraction, and the joined
   pairs;
2. if the largest group stays within the chaining guard (10 %), the split exactly as
   ``prepare`` builds it (it calls ``prepare``), with the val/test confirmed duplicates (0 by
   construction) and the three bands of the cross-split contact sheet;
3. a rule over the guard is marked and gets no split and no sheet.

The images are read once, the embeddings come from RC-D1-embed's output, and the crop pass
runs once at the lowest crop level of the sweep. These are the inputs ``prepare`` would compute
for each rule, so every split is the one ``prepare --rule`` gives (run 4b).

The team chooses on the sheets: the highest T within the guard whose three bands show no
same-scene pair. Only aggregate counts go to the repository
(``results/data/near_duplicate_sweep.csv``); the sheets show images and stay private.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medcompress.data.candidates import (
    JoinRule,
    component_groups,
    crop_pass,
    joined_pairs_all,
    largest_group,
)
from medcompress.data.dedup import CHAIN_MIN_IMAGES
from medcompress.data.embeddings import EmbeddingSettings, load_embeddings
from medcompress.data.prepare import (
    PreparedData,
    _features,
    _row,
    labels_values,
    near_duplicate_settings,
    prepare,
)

SWEEP_FILE = Path("data") / "near_duplicate_sweep.csv"
"""The sweep's aggregate counts; ``prepare --rule`` accepts only a rule listed in it within the
guard."""
SWEEP_THRESHOLDS: tuple[float, ...] = (0.92, 0.90, 0.88, 0.86, 0.84)
SWEEP_CROP = 0.80
SWEPT_METRICS: tuple[str, ...] = (
    "groups:total",
    "groups:multi_image",
    "images:train",
    "images:val",
    "images:test",
    "val_test_confirmed_duplicates_of_train",
    "val_test_close_dhash_to_train",
    "strict_test:images",
    "strict_test:image_fraction_of_test",
    "cross_split_top_cosine:max",
    "cross_split_top_cosine:min",
    "cross_split_top:max_score",
    "cross_split_top:min_score",
    "cross_split_closest_dhash:min_bits",
    "cross_split_closest_dhash:max_bits",
)
"""What each rule within the guard reports from its split (``prepare``'s rows)."""


class SweepError(ValueError):
    """The sweep's inputs do not fit amendment 5, or a rule is not one the sweep allows."""


@dataclass(frozen=True)
class SweepResult:
    """One rule of the sweep. ``prepared`` is ``None`` for a rule over the guard."""

    rule: JoinRule
    joined_pairs: int
    largest_group: int
    largest_fraction: float
    within_guard: bool
    prepared: PreparedData | None


def sweep_rules(
    thresholds: Sequence[float], crop: float | None = SWEEP_CROP, view: str = "full"
) -> list[JoinRule]:
    """``view:T OR crop >= crop`` for every T, the highest T (the least joining) first."""
    levels = sorted({round(float(t), 4) for t in thresholds}, reverse=True)
    if not levels:
        raise SweepError("the sweep needs at least one threshold")
    return [JoinRule(view, t, None if crop is None else round(float(crop), 4)) for t in levels]


def either_rater_rule(results_root: Path) -> JoinRule:
    """The rule chosen on the truth "`same_scene` for either rater" by amendment 3's criterion,
    as RC-D1-label's scoring wrote it (``labels:either_rater:rule:*``, D02 amendment 4)."""
    values = labels_values(results_root)
    try:
        return JoinRule.from_code(
            values["labels:either_rater:rule:view_code"],
            values["labels:either_rater:rule:threshold"],
            values["labels:either_rater:rule:crop_threshold"],
        )
    except KeyError as exc:
        raise SweepError(
            f"{Path(results_root) / 'data' / 'near_duplicate_labels.csv'} holds no "
            f"either-rater rule ({exc}): score the labels with --no-third-member (D02 amendment 4)"
        ) from exc


def rule_sweep(
    data_root: Path,
    dataset_config: Mapping[str, Any],
    embed_dir: Path,
    rules: Sequence[JoinRule],
    *,
    split_seed: int,
    close_bits: int,
    mask_threshold: int,
    log: Callable[[str], None] = print,
) -> list[SweepResult]:
    """Every rule of ``rules`` in order (module docstring)."""
    data_root, embed_dir = Path(data_root), Path(embed_dir)
    if any(r.view == "crop" for r in rules):
        raise SweepError("the sweep takes cosine rules only (full, centre or max)")
    settings = near_duplicate_settings(dataset_config, close_bits=close_bits, confirm_threshold=1.0)
    features = _features(data_root, dataset_config, settings.crop)
    _, images, _, signatures = features
    log(f"read {len(images)} images")
    embeddings, _ = load_embeddings(
        embed_dir / "embeddings.npz", images, EmbeddingSettings.from_config(dataset_config)
    )
    levels = [r.crop_threshold for r in rules if r.crop_threshold is not None]
    crop_above = None
    if levels:
        _, _, crop_above = crop_pass(
            [signatures[p] for p in images], settings.crop, k=0, floor=min(levels)
        )
        log(f"crop pass: {len(crop_above.a)} pairs at or above {min(levels):.2f}")
    results = []
    for rule in rules:
        ja, jb = joined_pairs_all(rule, embeddings, crop_above)
        largest = largest_group(component_groups(images, ja, jb))
        over = largest >= CHAIN_MIN_IMAGES and largest > settings.max_group_fraction * len(images)
        prepared = None
        if not over:  # the split exactly as prepare builds it
            prepared = prepare(
                data_root, dataset_config, split_seed=split_seed, close_bits=close_bits,
                mask_threshold=mask_threshold, rule=rule, embeddings=embeddings,
                features=features, crop_pairs=crop_above,
            )  # fmt: skip
        results.append(
            SweepResult(rule, len(ja), largest, largest / len(images), not over, prepared)
        )
    return results


def sweep_rows(
    results: Sequence[SweepResult],
    *,
    label_rule: JoinRule,
    split_seed: int,
    close_bits: int,
    mask_threshold: int,
    max_group_fraction: float,
) -> list[dict[str, Any]]:
    """The aggregate rows of ``results/data/near_duplicate_sweep.csv``."""
    rows = [_row(f"sweep:label_rule:{k}", v) for k, v in label_rule.code().items()]
    rows += [
        _row("sweep:rules", len(results)),
        _row("sweep:split_seed", split_seed),
        _row("sweep:close_bits", close_bits),
        _row("sweep:mask_threshold", mask_threshold),
        _row("sweep:max_group_fraction", max_group_fraction),
    ]
    for r in results:
        name = f"sweep:{r.rule.label}"
        rows += [_row(f"{name}:{k}", v) for k, v in r.rule.code().items()]
        rows += [
            _row(f"{name}:joined_pairs", r.joined_pairs),
            _row(f"{name}:largest_group", r.largest_group),
            _row(f"{name}:largest_fraction", r.largest_fraction),
            _row(f"{name}:within_guard", int(r.within_guard)),
        ]
        if r.prepared is not None:
            values = {row["metric"]: row["value"] for row in r.prepared.rows}
            rows += [_row(f"{name}:{m}", values[m]) for m in SWEPT_METRICS if m in values]
            rows += [
                _row(f"{name}:sheet_band_pairs:cosine", len(r.prepared.top_cosine_cross_split)),
                _row(f"{name}:sheet_band_pairs:crop", len(r.prepared.top_cross_split)),
                _row(f"{name}:sheet_band_pairs:dhash", len(r.prepared.closest_dhash_cross_split)),
            ]
    return rows


def swept_rule(results_root: Path, label: str) -> JoinRule:
    """The rule ``label`` (e.g. ``full:0.88:or_crop:0.80``) if the committed sweep file lists
    it within the guard; else :class:`SweepError` (D02 amendment 5: run 4b takes a swept rule)."""
    try:
        rule = JoinRule.from_label(label)
    except ValueError as exc:
        raise SweepError(str(exc)) from exc
    path = Path(results_root) / SWEEP_FILE
    if not path.is_file():
        raise SweepError(f"{path} is missing: run rule-sweep and commit its counts first")
    with path.open(newline="", encoding="utf-8") as handle:
        values = {r["metric"]: float(r["value"]) for r in csv.DictReader(handle) if r["value"]}
    flag = values.get(f"sweep:{rule.label}:within_guard")
    if flag is None:
        swept = sorted(
            m.split(":", 1)[1].rsplit(":within_guard", 1)[0]
            for m in values
            if m.endswith(":within_guard")
        )
        raise SweepError(f"--rule {rule.label} is not in the sweep ({path}: {', '.join(swept)})")
    if flag < 1:
        raise SweepError(f"--rule {rule.label} is over the chaining guard in the sweep ({path})")
    return rule


__all__ = [
    "SWEEP_CROP",
    "SWEEP_FILE",
    "SWEEP_THRESHOLDS",
    "SweepError",
    "SweepResult",
    "either_rater_rule",
    "rule_sweep",
    "swept_rule",
    "sweep_rows",
    "sweep_rules",
]
