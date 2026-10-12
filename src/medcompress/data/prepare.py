"""Data preparation for Kvasir-VQA-x1 + Kvasir-SEG (run card RC-D1).

Runs on a team member's machine against their own copy of the data. It writes
splits, manifests, mask links and exclusions **outside** the repository, and only
aggregate counts to ``results/data/summary.csv`` (experiment ``DATA``). No
per-image row ever reaches the repository.

Steps:
1. read Kvasir-VQA-x1 (official train / test) with ``data/kvasir.py``;
2. halve the official test split by image into validation and test (D02, Q1 = A)
   with the pre-registered split seed;
3. drop validation / test images that are near-duplicates of training images;
4. link Kvasir-SEG masks to VQA images by fingerprint; lesion areas and the
   size-stratum thresholds come from training images with a mask;
5. count QA rows, images, positives per finding, prevalence on the training
   split, and the answer values of the label questions (to fill the label rules).
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
from medcompress.data.dedup import image_fingerprint, near_duplicate_pairs
from medcompress.data.kvasir import (
    KvasirVQAx1,
    link_masks,
    load_kvasir_seg,
    parse_original,
)
from medcompress.data.manifests import build_manifest, write_manifest
from medcompress.data.masks import lesion_area_fraction, load_mask, size_strata_thresholds
from medcompress.data.splits import make_patient_splits, split_fingerprint

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


def prepare(
    data_root: Path,
    dataset_config: Mapping[str, Any],
    *,
    split_seed: int,
    dedup_threshold: int,
    mask_threshold: int,
) -> PreparedData:
    """Run steps 1-5 and return the decisions plus the aggregate result rows."""
    data_root = Path(data_root)
    reader = KvasirVQAx1(dataset_config, data_root=data_root)
    samples = reader.load()
    train = [s for s in samples if s.split == "train"]
    official_test = [s for s in samples if s.split == "test"]
    halves = make_patient_splits(official_test, {"val": 0.5, "test": 0.5}, seed=split_seed)

    # near-duplicates of training images in validation / test
    train_fp = {
        p: image_fingerprint(str(data_root / p)) for p in sorted({s.image_path for s in train})
    }
    held_out = {s.image_path for half in halves.values() for s in half}
    held_fp = {p: image_fingerprint(str(data_root / p)) for p in sorted(held_out)}
    duplicates = {
        c for c, _, _ in near_duplicate_pairs(train_fp, held_fp, threshold=dedup_threshold)
    }
    splits: dict[str, list[VQASample]] = {"train": train}
    for name, members in halves.items():
        splits[name] = [s for s in members if s.image_path not in duplicates]

    # masks
    seg = load_kvasir_seg(data_root)
    links = link_masks(
        sorted({s.image_path for s in samples}), seg, data_root=data_root, threshold=mask_threshold
    )
    areas = {
        img: lesion_area_fraction(load_mask(str(data_root / m))) for img, m in links.links.items()
    }
    train_areas = [areas[p] for p in sorted({s.image_path for s in train}) if p in areas]
    thresholds = tuple(size_strata_thresholds(train_areas)) if len(train_areas) >= 4 else ()
    splits = {
        name: [
            replace(s, mask_path=links.links.get(s.image_path), lesion_area=areas.get(s.image_path))
            for s in members
        ]
        for name, members in splits.items()
    }

    rows: list[dict[str, Any]] = []
    for name, members in splits.items():
        images = {s.image_path for s in members}
        rows.append(_row(f"qa_rows:{name}", len(members)))
        rows.append(_row(f"images:{name}", len(images)))
        rows.append(_row(f"masked_images:{name}", len(images & set(links.links))))
        by_image: dict[str, Mapping[str, bool]] = {s.image_path: s.findings for s in members}
        findings = sorted({f for labels in by_image.values() for f in labels})
        for finding in findings:
            positives = sum(1 for labels in by_image.values() if labels.get(finding))
            rows.append(_row(f"positives:{finding}:{name}", positives))
            if name == "train" and by_image:
                rows.append(_row(f"prevalence:{finding}", positives / len(by_image)))
    rows.append(_row("duplicates_removed:val_test", len(duplicates)))
    rows.append(_row("masks:linked", len(links.links)))
    rows.append(_row("masks:ambiguous", len(links.ambiguous)))
    rows.append(_row("masks:unmatched", links.unmatched_masks))
    for k, value in enumerate(thresholds, start=1):
        rows.append(_row(f"lesion_area_threshold:q{k}", value))
    raw = {split: _raw_rows(data_root, split) for split in ("train", "test")}
    for key, count in sorted(answer_value_counts(raw, dataset_config.get("label_mapping")).items()):
        rows.append(_row(f"answer_count:{key}", count))
    return PreparedData(
        splits=splits,
        excluded=tuple(sorted(duplicates)),
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
