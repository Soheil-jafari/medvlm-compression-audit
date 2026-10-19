"""Data preparation for Kvasir-VQA-x1 + Kvasir-SEG (run card RC-D1).

Runs on a team member's machine against their own copy of the data. It writes
splits, manifests and mask links **outside** the repository, and only aggregate counts to
``results/data/summary.csv`` (experiment ``DATA``). No per-image row ever reaches the
repository.

Steps (D02 amendment of 2026-10-18: the official Kvasir-VQA-x1 split is by question, not by
image, so it is not used for training or evaluation):
1. read every question of both official splits of Kvasir-VQA-x1 (``data/kvasir.py``);
2. fingerprint every image (dHash) and compute its crop signature, then group near-duplicates
   in two stages (D02 amendment 2 of 2026-10-18, final values 2026-10-19,
   ``dedup.two_stage_groups``): every pair within ``candidate_bits`` is joined when, and only
   when, its crop correlation reaches ``min_correlation`` (no automatic join at 0-4 bits);
   union-find; the run STOPS if the largest group holds more than ``max_group_fraction`` of the
   images (chaining guard);
3. link Kvasir-SEG masks to the images by fingerprint;
4. assign whole groups to train / validation / test (70 / 15 / 15 % of the images, the
   pre-registered seed), stratified by the answer values of the finding question and by
   the linked masks (``splits.make_image_disjoint_splits``);
5. check that no validation or test image has a joined pair with a training image (none, by
   construction); count the dHash distances and the crop correlations to the training split,
   and the pairs within ``close_bits`` that are not joined (data-card numbers, not filters);
   keep, for the final contact sheet, the most similar pairs across splits by the crop and by
   the dHash, and the closest not-joined pairs across splits;
6. count QA rows, images, groups, masked images and positives per split, prevalence and
   lesion-size thresholds on the training split, answer lengths, answer values per split,
   and the overlap of the official split (for the data card and the appendix item).

``calibrate`` is the step before (RC-D1 section 3a): it prints the crop correlation over all
candidate pairs and what each threshold would give, so ``min_correlation`` is chosen on the
data before any split is made.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from medcompress.config import ConfigError
from medcompress.data.base import VQASample
from medcompress.data.dedup import (
    ChainingError,
    CropSettings,
    NearDuplicateSettings,
    candidate_pairs,
    check_chaining,
    crop_similarities,
    cross_similarity_matrix,
    distances_to,
    image_features,
    joined_pairs,
    min_distances,
    two_stage_groups,
    union_groups,
)
from medcompress.data.kvasir import (
    KvasirVQAx1,
    link_masks,
    load_kvasir_seg,
    parse_original,
)
from medcompress.data.manifests import build_manifest, write_manifest
from medcompress.data.masks import lesion_area_fraction, load_mask, size_strata_thresholds
from medcompress.data.splits import make_image_disjoint_splits, split_fingerprint

DATA_EXPERIMENT = "DATA"


@dataclass(frozen=True)
class PreparedData:
    """Everything the preparation step decided, kept outside the repository."""

    splits: Mapping[str, list[VQASample]]
    excluded: tuple[str, ...]
    mask_links: Mapping[str, str]
    thresholds: tuple[float, ...]
    rows: tuple[Mapping[str, Any], ...]
    split_sha256: str
    near_duplicates: Mapping[str, float] = field(default_factory=dict)
    top_cross_split: tuple[CrossPair, ...] = ()
    closest_dhash_cross_split: tuple[CrossPair, ...] = ()
    close_not_joined: tuple[CrossPair, ...] = ()


@dataclass(frozen=True)
class CrossPair:
    """One of the most similar pairs across splits (final contact sheet; private)."""

    first: str
    first_split: str
    second: str
    second_split: str
    score: float
    bits: int


def _slug(text: str) -> str:
    tokens = re.findall(r"[a-z]+", text.lower())
    return max(tokens, key=len) if tokens else "rule"


def _row(metric: str, value: float) -> dict[str, Any]:
    return {
        "experiment": DATA_EXPERIMENT,
        "budget": "",
        "seed": 0,
        "metric": metric,
        "value": float(value),
    }


def answer_value_counts(
    raw_rows: Mapping[str, Sequence[Mapping[str, Any]]], rules_cfg: Any
) -> Counter[str]:
    """Count answer values of each label question over unique images (aggregate only).

    Keys are ``<rule name>:<answer value>``; the rule name is the rule's ``name``
    in the config, else ``abnormalities`` for the default question.
    """
    cfg_rules = list((rules_cfg or {}).get("rules") or []) if isinstance(rules_cfg, Mapping) else []
    wanted = [
        (str(r.get("name") or _slug(r["question_contains"])), str(r["question_contains"]).lower())
        for r in cfg_rules
    ] or [("abnormalities", "are there any abnormalities in the image")]
    seen: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    for rows in raw_rows.values():
        for row in rows:
            for q, a in parse_original(row.get("original")):
                for name, question in wanted:
                    if question in q.lower() and (str(row["img_id"]), name) not in seen:
                        seen.add((str(row["img_id"]), name))
                        for value in a.split(";"):
                            if value.strip():
                                counts[f"{name}:{value.strip().lower()}"] += 1
    return counts


DEFAULT_SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
TAIL_MAX_BITS = 17
"""The distances above the grouping threshold are counted up to this one (RC-D1 run 1 found
the official test-only images at up to 17 bits from a training image)."""


SCORE_LOW, SCORE_STEP, SCORE_BINS = 0.0, 0.05, 20
"""Crop correlations are counted in bins of 0.05 from 0.00 (``0.95`` holds 0.95 to 1.00)."""
TOP_CROSS_SPLIT = 24
"""How many of the most similar cross-split pairs the final contact sheet shows (by the crop
correlation, and again by the dHash distance)."""
CLOSE_NOT_JOINED_SHOWN = 12
"""How many of the pairs within ``close_bits`` that are not joined, across splits, the final
contact sheet shows (the highest crop correlation first)."""
GROUP_SIZE_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5", 5, 5), ("6_to_10", 6, 10),
    ("11_to_50", 11, 50), ("51_to_100", 51, 100), ("101_or_more", 101, 10**9),
)  # fmt: skip


def crop_settings(dataset_config: Mapping[str, Any]) -> CropSettings:
    """The crop of the finer measure, from ``near_duplicates`` of the dataset config."""
    cfg = dataset_config.get("near_duplicates") or {}
    base = CropSettings()
    return CropSettings(
        crop_fraction=float(cfg.get("crop_fraction", base.crop_fraction)),
        side=int(cfg.get("crop_side", base.side)),
        highpass_sigma=float(cfg.get("crop_highpass_sigma", base.highpass_sigma)),
        max_shift=int(cfg.get("crop_max_shift", base.max_shift)),
    )


def near_duplicate_settings(
    dataset_config: Mapping[str, Any],
    *,
    close_bits: int,
    confirm_threshold: float | None = None,
) -> NearDuplicateSettings:
    """Both stages of the grouping (D02 amendment 2). ``close_bits`` (``--close-bits``) is the
    band of very close dHash pairs reported separately (they are joined only by the crop, like
    every candidate); ``confirm_threshold`` overrides ``min_correlation``.

    Raises :class:`ConfigError` while ``min_correlation`` is open (``TODO``): the calibration
    step comes first.
    """
    cfg = dataset_config.get("near_duplicates") or {}
    threshold = confirm_threshold if confirm_threshold is not None else cfg.get("min_correlation")
    if threshold is None or isinstance(threshold, str):
        raise ConfigError(
            "near_duplicates.min_correlation (configs/data/kvasir_vqa_x1.yaml) is still open: "
            "run `prepare_data.py calibrate` first (run card RC-D1, section 3a), then fill it "
            "in or give --confirm-threshold"
        )
    base = NearDuplicateSettings(min_correlation=float(threshold))
    return NearDuplicateSettings(
        min_correlation=float(threshold),
        close_bits=int(close_bits),
        candidate_bits=int(cfg.get("candidate_bits", base.candidate_bits)),
        max_group_fraction=float(cfg.get("max_group_fraction", base.max_group_fraction)),
        crop=crop_settings(dataset_config),
    )


def score_bin(value: float) -> str:
    """The label of the 0.05-wide bin of one crop correlation (``below_0.00`` under 0)."""
    if value < SCORE_LOW:
        return f"below_{SCORE_LOW:.2f}"
    k = min(int(np.floor((value - SCORE_LOW) / SCORE_STEP + 1e-9)), SCORE_BINS - 1)
    return f"{SCORE_LOW + k * SCORE_STEP:.2f}"


def score_histogram(values: Sequence[float]) -> dict[str, int]:
    """Counts per bin of :func:`score_bin`, every bin listed (low to high)."""
    counts = {f"below_{SCORE_LOW:.2f}": 0}
    counts.update({f"{SCORE_LOW + k * SCORE_STEP:.2f}": 0 for k in range(SCORE_BINS)})
    for value in values:
        counts[score_bin(float(value))] += 1
    return counts


def group_size_rows(sizes: Mapping[str, int], n_images: int) -> list[dict[str, Any]]:
    """How many groups of each size, the largest group, and its share of the images."""
    largest = max(sizes.values(), default=0)
    multi = [g for g, n in sizes.items() if n > 1]
    rows = [
        _row("groups:total", len(sizes)),
        _row("groups:multi_image", len(multi)),
        _row("groups:largest", largest),
        _row("groups:largest_fraction", largest / max(1, n_images)),
        _row("images_in_multi_image_groups", sum(sizes[g] for g in multi)),
    ]
    for label, low, high in GROUP_SIZE_BUCKETS:
        count = sum(1 for n in sizes.values() if low <= n <= high)
        rows.append(_row(f"groups:size:{label}", count))
    return rows


def split_ratios(dataset_config: Mapping[str, Any]) -> dict[str, float]:
    """``split_ratios`` of the dataset config (fractions of images), 70 / 15 / 15 by default."""
    configured = dataset_config.get("split_ratios") or DEFAULT_SPLIT_RATIOS
    return {str(k): float(v) for k, v in configured.items()}


def image_answer_values(
    raw_rows: Mapping[str, Sequence[Mapping[str, Any]]], rules_cfg: Any
) -> dict[str, frozenset[str]]:
    """``{img_id: {"<rule>:<value>", ...}}``: the answer values of each label question.

    Values listed as ``none_values`` of the rule (for example "none") are left out: they mark
    the absence of a finding. These values are the strata of the re-split. They do not depend
    on the label mapping, which is decided after RC-D1.
    """
    cfg_rules = list((rules_cfg or {}).get("rules") or []) if isinstance(rules_cfg, Mapping) else []
    wanted = [
        (
            str(r.get("name") or _slug(r["question_contains"])),
            str(r["question_contains"]).lower(),
            {str(v).lower() for v in r.get("none_values") or ("none", "no")},
        )
        for r in cfg_rules
    ] or [("abnormalities", "are there any abnormalities in the image", {"none", "no"})]
    out: dict[str, set[str]] = {}
    for rows in raw_rows.values():
        for row in rows:
            for q, a in parse_original(row.get("original")):
                for name, question, none_values in wanted:
                    if question in q.lower():
                        values = {v.strip().lower() for v in a.split(";") if v.strip()}
                        out.setdefault(str(row["img_id"]), set()).update(
                            f"{name}:{v}" for v in values - none_values
                        )
    return {img: frozenset(values) for img, values in out.items()}


def _histogram(distances: Sequence[int], low: int, high: int) -> dict[str, int]:
    counts = {str(d): 0 for d in range(low, high + 1)}
    counts[f"{high + 1}_or_more"] = 0
    for d in distances:
        if d < low:
            continue
        key = str(d) if d <= high else f"{high + 1}_or_more"
        counts[key] += 1
    return counts


def official_split_rows(
    raw: Mapping[str, Sequence[Mapping[str, Any]]],
    fingerprints: Mapping[str, str],
    image_of: Mapping[str, str],
) -> list[dict[str, Any]]:
    """How the official split overlaps by image (data card; appendix item of D02/D04).

    ``image_of`` maps an img_id to its image path, ``fingerprints`` an image path to its dHash.
    """
    train_ids = {str(r["img_id"]) for r in raw.get("train", ())}
    test_ids = {str(r["img_id"]) for r in raw.get("test", ())}
    both, test_only = train_ids & test_ids, test_ids - train_ids
    rows = [
        _row("official:images:train", len(train_ids)),
        _row("official:images:test", len(test_ids)),
        _row("official:images:both", len(both)),
        _row("official:images:test_only", len(test_only)),
        _row("official:images:all", len(train_ids | test_ids)),
        _row(
            "official:test_rows:shared_images",
            sum(1 for r in raw.get("test", ()) if str(r["img_id"]) in train_ids),
        ),
        _row(
            "official:test_rows:test_only_images",
            sum(1 for r in raw.get("test", ()) if str(r["img_id"]) in test_only),
        ),
    ]
    train_fp = {image_of[i]: fingerprints[image_of[i]] for i in sorted(train_ids) if i in image_of}
    test_fp = {image_of[i]: fingerprints[image_of[i]] for i in sorted(test_only) if i in image_of}
    closest = min_distances(train_fp, test_fp)
    for key, value in _histogram(list(closest.values()), 0, TAIL_MAX_BITS).items():
        rows.append(_row(f"official:test_only_min_distance_to_train:{key}", value))
    return rows


def _features(
    data_root: Path, dataset_config: Mapping[str, Any], crop: CropSettings
) -> tuple[list[VQASample], list[str], dict[str, str], dict[str, np.ndarray]]:
    """Every question, the images, and each image's dHash and crop signature (read once)."""
    samples = KvasirVQAx1(dataset_config, data_root=data_root).load()
    images = sorted({s.image_path for s in samples})
    fingerprints: dict[str, str] = {}
    signatures: dict[str, np.ndarray] = {}
    for path in images:
        fingerprints[path], signatures[path] = image_features(str(data_root / path), crop)
    return samples, images, fingerprints, signatures


def _cross_split(
    split_images: Mapping[str, Sequence[str]],
    fingerprints: Mapping[str, str],
    signatures: Mapping[str, np.ndarray],
    settings: NearDuplicateSettings,
) -> tuple[list[dict[str, Any]], tuple[CrossPair, ...], tuple[CrossPair, ...]]:
    """Crop correlations and dHash distances across splits, after the split.

    Returns the rows, the most similar cross-split pairs by the crop correlation, and the
    closest by the dHash distance (each :data:`TOP_CROSS_SPLIT`, for the final contact sheet).
    For each validation and test image: its highest correlation with any training image (a
    histogram), and whether a training image *beyond* the candidate distance reaches the
    threshold (the candidate stage would have missed it; a data-card number, not a filter).
    """
    rows: list[dict[str, Any]] = []
    by_score: list[CrossPair] = []
    by_bits: list[CrossPair] = []
    names = list(split_images)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            a, b = list(split_images[first]), list(split_images[second])
            if not a or not b:
                continue
            matrix = cross_similarity_matrix(
                [signatures[p] for p in a], [signatures[p] for p in b], settings.crop
            )
            flat = matrix.ravel()
            k = min(TOP_CROSS_SPLIT, flat.size)
            for index in np.argpartition(-flat, k - 1)[:k]:
                r, c = divmod(int(index), len(b))
                bits = int(distances_to([fingerprints[a[r]]], fingerprints[b[c]])[0])
                by_score.append(CrossPair(a[r], first, b[c], second, float(flat[index]), bits))
            first_fp = [fingerprints[p] for p in a]
            best, beyond = [], 0
            for col, path in enumerate(b):
                distances = distances_to(first_fp, fingerprints[path])
                scores = matrix[:, col]
                near = np.argsort(distances, kind="stable")[: min(TOP_CROSS_SPLIT, len(a))]
                by_bits += [
                    CrossPair(a[r], first, path, second, float(scores[r]), int(distances[r]))
                    for r in near
                ]
                best.append(float(scores.max()))
                far = distances > settings.candidate_bits
                if far.any() and float(scores[far].max()) >= settings.min_correlation:
                    beyond += 1
            if first != "train":
                continue
            for key, value in score_histogram(best).items():
                rows.append(_row(f"max_score_to_train:{second}:{key}", value))
            rows.append(_row(f"similar_to_train_beyond_candidates:{second}", beyond))
    by_score.sort(key=lambda t: (-t.score, t.first, t.second))
    by_bits.sort(key=lambda t: (t.bits, -t.score, t.first, t.second))
    top, closest = tuple(by_score[:TOP_CROSS_SPLIT]), tuple(by_bits[:TOP_CROSS_SPLIT])
    if top:
        rows.append(_row("cross_split_top:max_score", top[0].score))
        rows.append(_row("cross_split_top:min_score", top[-1].score))
        rows.append(
            _row("cross_split_top:within_candidate_bits",
                 sum(1 for t in top if t.bits <= settings.candidate_bits))
        )  # fmt: skip
    if closest:
        rows.append(_row("cross_split_closest_dhash:min_bits", closest[0].bits))
        rows.append(_row("cross_split_closest_dhash:max_bits", closest[-1].bits))
        rows.append(_row("cross_split_closest_dhash:max_score", max(t.score for t in closest)))
    return rows, top, closest


def prepare(
    data_root: Path,
    dataset_config: Mapping[str, Any],
    *,
    split_seed: int,
    close_bits: int,
    mask_threshold: int,
    confirm_threshold: float | None = None,
) -> PreparedData:
    """Run steps 1-6 and return the decisions plus the aggregate result rows.

    Raises :class:`ConfigError` while the confirmation threshold is open, and
    :class:`~medcompress.data.dedup.ChainingError` when the chaining guard stops the run.
    """
    data_root = Path(data_root)
    settings = near_duplicate_settings(
        dataset_config, close_bits=close_bits, confirm_threshold=confirm_threshold
    )
    samples, images, fingerprints, signatures = _features(data_root, dataset_config, settings.crop)
    grouping = two_stage_groups(fingerprints, signatures, settings)
    image_group = grouping.groups

    seg = load_kvasir_seg(data_root)
    links = link_masks(
        images, seg, data_root=data_root, threshold=mask_threshold, fingerprints=fingerprints
    )
    raw = {split: _raw_rows(data_root, split) for split in ("train", "test")}
    values = image_answer_values(raw, dataset_config.get("label_mapping"))
    image_of = {str(s.metadata["img_id"]): s.image_path for s in samples}
    findings_of = {s.image_path: s.findings for s in samples}
    labels: dict[str, set[str]] = {}
    for img_id, path in image_of.items():
        group_labels = labels.setdefault(image_group[path], set())
        group_labels.update(f"answer:{v}" for v in values.get(img_id, ()))
        group_labels.update(f"finding:{f}" for f, on in findings_of[path].items() if on)
        if path in links.links:
            group_labels.add("masked")
    splits = make_image_disjoint_splits(
        samples,
        image_group,
        {g: frozenset(v) for g, v in labels.items()},
        split_ratios(dataset_config),
        seed=split_seed,
    )

    split_images = {
        name: sorted({s.image_path for s in members}) for name, members in splits.items()
    }
    split_of = {p: name for name, paths in split_images.items() for p in paths}
    train_fp = {p: fingerprints[p] for p in split_images["train"]}
    closest = {
        name: min_distances(train_fp, {p: fingerprints[p] for p in split_images[name]})
        for name in split_images
        if name != "train"
    }
    close = sum(1 for d in closest.values() for v in d.values() if v <= close_bits)
    crossing: dict[str, set[str]] = {name: set() for name in closest}
    close_apart: list[CrossPair] = []  # within close_bits, not joined, in different splits
    for (a, b, d), keep, score in zip(
        grouping.candidates, grouping.joined, grouping.scores, strict=True
    ):
        if split_of[a] == split_of[b]:
            continue
        if keep:
            for path, other in ((a, b), (b, a)):
                if split_of[path] != "train" and split_of[other] == "train":
                    crossing[split_of[path]].add(path)
        elif d <= close_bits:
            close_apart.append(CrossPair(a, split_of[a], b, split_of[b], score, d))
    confirmed_leaks = sum(len(v) for v in crossing.values())
    if confirmed_leaks:  # impossible by construction: joined pairs share a group
        raise AssertionError(
            f"{confirmed_leaks} held-out images have a confirmed duplicate in the training split"
        )
    close_apart.sort(key=lambda t: (-t.score, t.first, t.second))

    areas = {
        img: lesion_area_fraction(load_mask(str(data_root / m))) for img, m in links.links.items()
    }
    train_areas = [areas[p] for p in split_images["train"] if p in areas]
    thresholds = tuple(size_strata_thresholds(train_areas)) if len(train_areas) >= 4 else ()
    splits = {
        name: [
            replace(s, mask_path=links.links.get(s.image_path), lesion_area=areas.get(s.image_path))
            for s in members
        ]
        for name, members in splits.items()
    }

    rows: list[dict[str, Any]] = [_row("images:all", len(images))]
    rows += group_size_rows(grouping.sizes, len(images))
    rows += [_row(f"near_duplicates:{k}", v) for k, v in settings.rows().items()]
    joined = sum(grouping.joined)
    close_pairs = [
        keep for (_, _, d), keep in zip(grouping.candidates, grouping.joined, strict=True)
        if d <= close_bits
    ]  # fmt: skip
    rows += [
        _row("near_duplicates:candidate_pairs", len(grouping.candidates)),
        _row("near_duplicates:joined_pairs", joined),
        _row("near_duplicates:candidate_pairs_not_joined", len(grouping.candidates) - joined),
        _row("near_duplicates:close_pairs", len(close_pairs)),
        _row("near_duplicates:close_pairs_joined", sum(close_pairs)),
        _row("near_duplicates:close_pairs_not_joined", len(close_pairs) - sum(close_pairs)),
        _row("near_duplicates:close_pairs_not_joined:cross_split", len(close_apart)),
    ]
    for name, members in splits.items():
        in_split = set(split_images[name])
        rows.append(_row(f"qa_rows:{name}", len(members)))
        rows.append(_row(f"images:{name}", len(in_split)))
        rows.append(_row(f"image_fraction:{name}", len(in_split) / max(1, len(images))))
        rows.append(_row(f"groups:{name}", len({image_group[p] for p in in_split})))
        rows.append(_row(f"masked_images:{name}", len(in_split & set(links.links))))
        by_image: dict[str, Mapping[str, bool]] = {s.image_path: s.findings for s in members}
        for finding in sorted({f for labels_ in by_image.values() for f in labels_}):
            positives = sum(1 for labels_ in by_image.values() if labels_.get(finding))
            rows.append(_row(f"positives:{finding}:{name}", positives))
            if name == "train" and by_image:
                rows.append(_row(f"prevalence:{finding}", positives / len(by_image)))
        per_value: Counter[str] = Counter(
            v for p in in_split for v in values.get(str(Path(p).stem), ())
        )
        for value, count in sorted(per_value.items()):
            rows.append(_row(f"answer_count:{value}:{name}", count))
    rows.append(_row("val_test_confirmed_duplicates_of_train", confirmed_leaks))
    for name, found in crossing.items():
        rows.append(_row(f"confirmed_duplicates_of_train:{name}", len(found)))
    rows.append(_row("val_test_close_dhash_to_train", close))  # not joined: the crop differs
    for name, distances in closest.items():
        hist = _histogram(list(distances.values()), 0, TAIL_MAX_BITS)
        for key, count in hist.items():
            rows.append(_row(f"min_distance_to_train:{name}:{key}", count))
    cross_rows, top, nearest = _cross_split(split_images, fingerprints, signatures, settings)
    rows += cross_rows
    rows.append(_row("masks:linked", len(links.links)))
    rows.append(_row("masks:ambiguous", len(links.ambiguous)))
    rows.append(_row("masks:unmatched", links.unmatched_masks))
    for k, cut in enumerate(thresholds, start=1):
        rows.append(_row(f"lesion_area_threshold:q{k}", cut))
    # answer lengths in words, on the training split (they set the generation limit, D04)
    words = sorted(len(str(s.answer).split()) for s in splits["train"])
    if words:
        for pct in (50, 90, 99):
            rows.append(
                _row(f"answer_words:p{pct}", words[min(len(words) - 1, len(words) * pct // 100)])
            )
        rows.append(_row("answer_words:max", words[-1]))
    for key, count in sorted(answer_value_counts(raw, dataset_config.get("label_mapping")).items()):
        rows.append(_row(f"answer_count:{key}", count))
    rows += official_split_rows(raw, fingerprints, image_of)
    return PreparedData(
        splits=splits,
        excluded=(),
        mask_links=dict(links.links),
        thresholds=thresholds,
        rows=tuple(rows),
        split_sha256=split_fingerprint(splits),
        near_duplicates=settings.rows(),
        top_cross_split=top,
        closest_dhash_cross_split=nearest,
        close_not_joined=tuple(close_apart[:CLOSE_NOT_JOINED_SHOWN]),
    )


CALIBRATION_GRID: tuple[float, ...] = tuple(round(0.30 + 0.05 * k, 2) for k in range(14))
"""The thresholds the calibration step reports (0.30 to 0.95)."""
RANDOM_CONTROLS = 2000
"""Random image pairs beyond the candidate distance, scored as controls (seeded): what images
the dHash already calls different score."""
CONTROL_TOLERANCE = 0.001
"""The provisional proposal lets at most this share of the controls score at or above it (2 of
2,000): a control can itself be a moved frame that the dHash missed."""
SHEET_PAIRS = {"below": 8, "above": 8, "controls": 4, "close": 4}


@dataclass(frozen=True)
class Calibration:
    """What ``calibrate`` measured: aggregate rows, the printout, and the contact-sheet pairs."""

    rows: tuple[Mapping[str, Any], ...]
    lines: tuple[str, ...]
    threshold: float | None
    bands: tuple[tuple[str, tuple[tuple[str, str, str], ...]], ...]
    crop_fraction: float


def _bands(close: int, candidate_bits: int) -> list[tuple[int, int]]:
    cuts = [(0, close), (close + 1, min(8, candidate_bits)), (max(close + 1, 9), candidate_bits)]
    return [(lo, hi) for lo, hi in cuts if lo <= hi]


def _control_pairs(
    images: Sequence[str], count: int, seed: int, skip: set[tuple[str, str]]
) -> list[tuple[str, str]]:
    """``count`` distinct random pairs of different images that are not in ``skip`` (the
    candidates); all such pairs if there are fewer. ``images`` is sorted."""
    n = len(images)
    if n * (n - 1) // 2 - len(skip) <= count:
        every = ((images[i], images[j]) for i in range(n) for j in range(i + 1, n))
        return [pair for pair in every if pair not in skip]
    rng = np.random.default_rng(seed)
    chosen: dict[tuple[str, str], None] = {}
    while len(chosen) < count:
        for i, j in rng.integers(0, n, size=(2 * count, 2)):
            pair = (images[min(i, j)], images[max(i, j)])
            if i != j and pair not in skip:
                chosen.setdefault(pair, None)
                if len(chosen) == count:
                    break
    return list(chosen)


def calibrate(
    data_root: Path,
    dataset_config: Mapping[str, Any],
    *,
    close_bits: int,
    seed: int,
    threshold: float | None = None,
    controls: int = RANDOM_CONTROLS,
) -> Calibration:
    """The calibration step of D02 amendment 2: choose ``min_correlation`` on the data.

    Scores every candidate pair (within ``candidate_bits``) and ``controls`` random pairs with
    the crop measure; counts them per bin and per dHash band; and, for each threshold of
    :data:`CALIBRATION_GRID`, what the grouping would give (joined pairs, groups, the largest
    group, the chaining guard, the controls at or above it). The controls are random pairs
    beyond ``candidate_bits``. ``threshold`` is the value the contact sheet is drawn at; without
    it, the provisional proposal is used: the smallest grid value that passes the guard and that
    at most :data:`CONTROL_TOLERANCE` of the controls reach. The proposal is a starting point to
    check by eye on the sheet, not a decision.

    Since 2026-10-19 every candidate is joined only by the crop, also within ``close_bits``;
    the table counts the pairs within ``close_bits`` that a threshold leaves apart.
    """
    data_root = Path(data_root)
    crop = crop_settings(dataset_config)
    cfg = dataset_config.get("near_duplicates") or {}
    base = NearDuplicateSettings(min_correlation=1.0)
    candidate_bits = int(cfg.get("candidate_bits", base.candidate_bits))
    max_fraction = float(cfg.get("max_group_fraction", base.max_group_fraction))
    _, images, fingerprints, signatures = _features(data_root, dataset_config, crop)
    candidates = candidate_pairs(fingerprints, threshold=candidate_bits)
    scores = crop_similarities(signatures, [(a, b) for a, b, _ in candidates], crop)
    pairs = _control_pairs(images, controls, seed, {(a, b) for a, b, _ in candidates})
    control_scores = crop_similarities(signatures, pairs, crop)
    control_bits = [int(distances_to([fingerprints[a]], fingerprints[b])[0]) for a, b in pairs]
    rows: list[dict[str, Any]] = [_row("calibration:images", len(images))]
    settings = NearDuplicateSettings(
        min_correlation=1.0, close_bits=close_bits, candidate_bits=candidate_bits,
        max_group_fraction=max_fraction, crop=crop,
    )  # fmt: skip
    rows += [
        _row(f"calibration:{k}", v) for k, v in settings.rows().items() if k != "min_correlation"
    ]
    by_bits = Counter(d for _, _, d in candidates)
    for d in range(candidate_bits + 1):
        rows.append(_row(f"calibration:candidate_pairs:bits:{d}", by_bits.get(d, 0)))
    bands = _bands(close_bits, candidate_bits)
    columns: dict[str, dict[str, int]] = {}
    for lo, hi in bands:
        chosen = [sc for (_, _, d), sc in zip(candidates, scores, strict=True) if lo <= d <= hi]
        columns[f"bits_{lo}_{hi}"] = score_histogram(chosen)
    columns["random_controls"] = score_histogram(control_scores)
    for band, hist in columns.items():
        for key, value in hist.items():
            rows.append(_row(f"calibration:score:{band}:{key}", value))
    rows.append(_row("calibration:random_controls", len(pairs)))
    quantiles = {}
    if len(control_scores):
        quantiles = {
            "p50": float(np.quantile(control_scores, 0.50)),
            "p99": float(np.quantile(control_scores, 0.99)),
            "max": float(control_scores.max()),
        }
    for key, quantile in quantiles.items():
        rows.append(_row(f"calibration:random_controls:{key}", quantile))

    grid = sorted(
        set(CALIBRATION_GRID) | ({round(threshold, 4)} if threshold is not None else set())
    )
    table: list[tuple[float, int, int, int, int, int, bool, int]] = []
    for t in grid:
        trial = replace(settings, min_correlation=t)
        joined = joined_pairs(candidates, scores, trial)
        groups = union_groups(
            images, ((a, b) for (a, b, _), keep in zip(candidates, joined, strict=True) if keep)
        )
        sizes = Counter(groups.values())
        largest = max(sizes.values(), default=0)
        try:
            check_chaining(groups, max_fraction)
            guard = True
        except ChainingError:
            guard = False
        apart = sum(1 for (_, _, d), k in zip(candidates, joined, strict=True)
                    if not k and d <= close_bits)  # fmt: skip
        at_or_above = int(np.sum(control_scores >= t))
        table.append((t, sum(joined), apart, len(sizes), sum(1 for n in sizes.values() if n > 1),
                      largest, guard, at_or_above))  # fmt: skip
        prefix = f"calibration:threshold:{t:.2f}"
        rows += [
            _row(f"{prefix}:joined_pairs", sum(joined)),
            _row(f"{prefix}:close_pairs_not_joined", apart),
            _row(f"{prefix}:groups", len(sizes)),
            _row(f"{prefix}:multi_image_groups", sum(1 for n in sizes.values() if n > 1)),
            _row(f"{prefix}:largest_group", largest),
            _row(f"{prefix}:largest_fraction", largest / max(1, len(images))),
            _row(f"{prefix}:guard_ok", int(guard)),
            _row(f"{prefix}:random_controls_at_or_above", at_or_above),
        ]
    allowed = int(CONTROL_TOLERANCE * len(pairs))
    proposal = next((t for t, *_, guard, above in table if guard and above <= allowed), None)
    chosen_t = threshold if threshold is not None else proposal
    if proposal is not None:
        rows.append(_row("calibration:provisional_proposal", proposal))

    lines = [
        f"Near-duplicate calibration (D02 amendment 2): {len(images)} images",
        f"settings: candidates <= {candidate_bits} bits, each joined only by the crop (also "
        f"within {close_bits} bits); crop: the central {crop.crop_fraction:.0%} of width and "
        f"height, grey, {crop.side} x {crop.side} (area average), Gaussian of "
        f"{crop.highpass_sigma:g} px subtracted, shifts up to {crop.max_shift} px; measure: the "
        "highest Pearson correlation over the shifts, both ways round",
        f"candidate pairs: {len(candidates)}; by dHash distance: "
        + ", ".join(f"{d}: {by_bits.get(d, 0)}" for d in range(candidate_bits + 1)),
        "crop correlation (pairs per bin; a bin holds its value up to the next):",
        "  bin        " + "".join(f"{name:>16}" for name in columns),
    ]
    for key in reversed(list(columns["random_controls"])):
        lines.append(f"  {key:<11}" + "".join(f"{columns[c][key]:>16}" for c in columns))
    if quantiles:
        lines.append(
            f"random controls: {len(pairs)} random pairs beyond {candidate_bits} bits; "
            f"p50 {quantiles['p50']:.3f}, p99 {quantiles['p99']:.3f}, max {quantiles['max']:.3f}"
        )
    lines.append(
        f"what each threshold gives (joined = candidate pairs at or above it; close apart = "
        f"pairs within {close_bits} bits left apart; guard: largest group <= {max_fraction:.0%} "
        "of the images):"
    )
    lines.append("  threshold     joined  close apart   groups  multi-image  largest (share)  "
                 "guard  controls at or above")  # fmt: skip
    for t, joined_n, apart, groups_n, multi_n, largest, guard, above in table:
        lines.append(
            f"  {t:>9.2f}  {joined_n:>9}  {apart:>11}  {groups_n:>7}  {multi_n:>11}  "
            f"{largest:>7} ({largest / max(1, len(images)):>5.1%})  "
            f"{'ok' if guard else 'STOP':>5}  {above:>19}"
        )
    if threshold is not None:
        lines.append(f"contact sheet drawn at the given threshold {threshold:.2f}")
    elif proposal is not None:
        lines.append(
            f"provisional proposal: {proposal:.2f} (the smallest threshold that passes the guard "
            f"and that at most {allowed} of the {len(pairs)} controls reach); check it on the "
            "contact sheet before it is decided"
        )
    else:
        lines.append(
            f"no threshold of the grid passes the guard with at most {allowed} controls at or "
            "above it: report"
        )

    sheet: list[tuple[str, tuple[tuple[str, str, str], ...]]] = []
    if chosen_t is not None:
        decided = sorted(
            (float(sc), a, b, d) for (a, b, d), sc in zip(candidates, scores, strict=True)
        )
        under = [x for x in decided if x[0] < chosen_t][::-1][: SHEET_PAIRS["below"]]
        over = [x for x in decided if x[0] >= chosen_t][: SHEET_PAIRS["above"]]
        close_apart = [x for x in decided if x[0] < chosen_t and x[3] <= close_bits][::-1]
        rng = np.random.default_rng(seed + 1)
        picks = sorted(rng.choice(len(pairs), size=min(SHEET_PAIRS["controls"], len(pairs)),
                                  replace=False).tolist()) if pairs else []  # fmt: skip
        shown = [(float(control_scores[k]), *pairs[k], control_bits[k]) for k in picks]

        def cells(items: Sequence[tuple[float, str, str, int]]) -> tuple[tuple[str, str, str], ...]:
            return tuple(
                (str(data_root / a), str(data_root / b), f"r={sc:.3f}  {d} bits")
                for sc, a, b, d in items
            )

        sheet = [
            (f"Just below {chosen_t:.2f}: NOT joined (the highest-scoring candidate pairs, "
             f"0-{candidate_bits} bits, under it)", cells(under)),
            (f"Just above {chosen_t:.2f}: JOINED (the lowest-scoring candidate pairs, "
             f"0-{candidate_bits} bits, at or above it)", cells(over)),
            (f"Random controls ({len(shown)} of the {len(pairs)} random pairs beyond "
             f"{candidate_bits} bits)", cells(shown)),
            (f"Within {close_bits} bits but NOT joined ({len(close_apart)} pairs): the highest "
             "crop correlation first", cells(close_apart[: SHEET_PAIRS["close"]])),
        ]  # fmt: skip
    return Calibration(tuple(rows), tuple(lines), chosen_t, tuple(sheet), crop.crop_fraction)


def _raw_rows(data_root: Path, split: str) -> list[dict[str, Any]]:
    path = Path(data_root) / "kvasir_vqa_x1" / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_private_outputs(prepared: PreparedData, out_dir: Path, data_root: Path) -> dict[str, str]:
    """Write splits, exclusions, mask links, the grouping settings and manifests to ``out_dir``
    (outside the repo).

    Returns the hashes to copy into the run card.
    """
    from medcompress.config import repo_root

    out_dir = Path(out_dir).resolve()
    if out_dir.is_relative_to(repo_root().resolve()):
        raise ValueError(f"Private outputs must be written outside the repository: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    assignment = {s.sample_id: name for name, members in prepared.splits.items() for s in members}
    (out_dir / "splits.json").write_text(json.dumps(assignment, sort_keys=True, indent=0))
    (out_dir / "excluded_images.json").write_text(json.dumps(list(prepared.excluded), indent=0))
    (out_dir / "mask_links.json").write_text(
        json.dumps(dict(prepared.mask_links), sort_keys=True, indent=0)
    )
    if prepared.near_duplicates:  # the settings of the grouping (no image names)
        (out_dir / "near_duplicates.json").write_text(
            json.dumps(dict(prepared.near_duplicates), sort_keys=True, indent=1)
        )
    hashes = {"split_sha256": prepared.split_sha256}
    for name in ("kvasir_vqa_x1", "kvasir_seg"):
        root = Path(data_root) / name
        if root.is_dir():
            manifest = build_manifest(name, "local", root)
            write_manifest(manifest, out_dir / f"manifest_{name}.jsonl")
            hashes[f"manifest_{name}"] = manifest.manifest_sha256
    return hashes
