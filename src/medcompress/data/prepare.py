"""Data preparation for Kvasir-VQA-x1 + Kvasir-SEG (run card RC-D1).

Runs on a team member's machine against their own copy of the data. It writes
splits, manifests and mask links **outside** the repository, and only aggregate counts to
``results/data/summary.csv`` (experiment ``DATA``). No per-image row ever reaches the
repository.

Steps (D02 amendment of 2026-10-18: the official Kvasir-VQA-x1 split is by question, not by
image, so it is not used for training or evaluation):
1. read every question of both official splits of Kvasir-VQA-x1 (``data/kvasir.py``);
2. fingerprint every image (dHash) and group near-duplicates transitively (distance at most
   ``dedup_threshold`` bits, union-find);
3. link Kvasir-SEG masks to the images by fingerprint;
4. assign whole groups to train / validation / test (70 / 15 / 15 % of the images, the
   pre-registered seed), stratified by the answer values of the finding question and by
   the linked masks (``splits.make_image_disjoint_splits``);
5. check that no validation or test image is within ``dedup_threshold`` bits of a training
   image (none, by construction) and count the distances just above it (a data-card number,
   not a filter);
6. count QA rows, images, groups, masked images and positives per split, prevalence and
   lesion-size thresholds on the training split, answer lengths, answer values per split,
   and the overlap of the official split (for the data card and the appendix item).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from medcompress.data.base import VQASample
from medcompress.data.dedup import image_fingerprint, min_distances, near_duplicate_groups
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


def prepare(
    data_root: Path,
    dataset_config: Mapping[str, Any],
    *,
    split_seed: int,
    dedup_threshold: int,
    mask_threshold: int,
) -> PreparedData:
    """Run steps 1-6 and return the decisions plus the aggregate result rows."""
    data_root = Path(data_root)
    reader = KvasirVQAx1(dataset_config, data_root=data_root)
    samples = reader.load()
    images = sorted({s.image_path for s in samples})
    fingerprints = {p: image_fingerprint(str(data_root / p)) for p in images}
    image_group = near_duplicate_groups(fingerprints, threshold=dedup_threshold)

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
    train_fp = {p: fingerprints[p] for p in split_images["train"]}
    closest = {
        name: min_distances(train_fp, {p: fingerprints[p] for p in split_images[name]})
        for name in split_images
        if name != "train"
    }
    leaks = sum(1 for d in closest.values() for v in d.values() if v <= dedup_threshold)
    if leaks:  # impossible by construction: pairs within the threshold share a group
        raise AssertionError(f"{leaks} held-out images are near-duplicates of training images")

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

    rows: list[dict[str, Any]] = []
    group_size: dict[str, int] = {}
    for path in images:
        group_size[image_group[path]] = group_size.get(image_group[path], 0) + 1
    multi = [g for g, n in group_size.items() if n > 1]
    rows += [
        _row("images:all", len(images)),
        _row("groups:total", len(group_size)),
        _row("groups:multi_image", len(multi)),
        _row("groups:largest", max(group_size.values(), default=0)),
        _row("images_in_multi_image_groups", sum(group_size[g] for g in multi)),
        _row("grouping_threshold_bits", dedup_threshold),
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
    rows.append(_row("val_test_near_duplicates_of_train", leaks))
    for name, distances in closest.items():
        hist = _histogram(list(distances.values()), dedup_threshold + 1, TAIL_MAX_BITS)
        for key, value in hist.items():
            rows.append(_row(f"min_distance_to_train:{name}:{key}", value))
    rows.append(_row("masks:linked", len(links.links)))
    rows.append(_row("masks:ambiguous", len(links.ambiguous)))
    rows.append(_row("masks:unmatched", links.unmatched_masks))
    for k, value in enumerate(thresholds, start=1):
        rows.append(_row(f"lesion_area_threshold:q{k}", value))
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
    )


def _raw_rows(data_root: Path, split: str) -> list[dict[str, Any]]:
    path = Path(data_root) / "kvasir_vqa_x1" / f"{split}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_private_outputs(prepared: PreparedData, out_dir: Path, data_root: Path) -> dict[str, str]:
    """Write splits, exclusions, mask links and manifests to ``out_dir`` (outside the repo).

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
    hashes = {"split_sha256": prepared.split_sha256}
    for name in ("kvasir_vqa_x1", "kvasir_seg"):
        root = Path(data_root) / name
        if root.is_dir():
            manifest = build_manifest(name, "local", root)
            write_manifest(manifest, out_dir / f"manifest_{name}.jsonl")
            hashes[f"manifest_{name}"] = manifest.manifest_sha256
    return hashes
