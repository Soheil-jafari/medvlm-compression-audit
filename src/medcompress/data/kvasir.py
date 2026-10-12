"""Readers for Kvasir-VQA-x1 (main QA) and Kvasir-SEG (polyp masks), plus mask linking.

Local layout under ``paths.data_root`` (written by ``scripts/prepare_data.py export``
on a team member's own machine; never in the repository)::

    kvasir_vqa_x1/images/<img_id>.jpg        one file per Kvasir-VQA image
    kvasir_vqa_x1/train.jsonl, test.jsonl    one Kvasir-VQA-x1 row per line
    kvasir_seg/images/<name>.jpg             Kvasir-SEG images
    kvasir_seg/masks/<name>.jpg              masks with the same file names
    kvasir_seg/kavsir_bboxes.json            optional bounding boxes (unused here)

Kvasir-VQA-x1 rows have the fields ``img_id``, ``question``, ``answer``,
``complexity``, ``question_class`` and ``original`` (the source Kvasir-VQA
question-answer pairs, used to derive per-image finding labels). Images come from
Kvasir-VQA and are joined by ``img_id`` (dataset card).
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medcompress.data.base import VQADataset, VQASample
from medcompress.data.dedup import image_fingerprint, near_duplicate_pairs

X1_DIR = "kvasir_vqa_x1"
SEG_DIR = "kvasir_seg"
OFFICIAL_SPLITS: tuple[str, ...] = ("train", "test")


@dataclass(frozen=True)
class LabelRule:
    """Derive finding labels from one source question.

    ``question_contains`` selects source questions (lower-case substring). In
    ``multi_label`` mode the answer is split on ``;``; each value found in
    ``values`` marks that finding present, and every finding of the rule not named
    is absent. Answers in ``none_values`` mark all of the rule's findings absent.
    """

    question_contains: str
    values: Mapping[str, str] = field(default_factory=dict)
    none_values: tuple[str, ...] = ("none", "no")
    mode: str = "multi_label"


def parse_label_rules(mapping: Mapping[str, Any] | None) -> list[LabelRule]:
    """Parse ``dataset.label_mapping`` from ``configs/data/kvasir_vqa_x1.yaml``."""
    if not mapping or not isinstance(mapping, Mapping):
        return []
    rules = []
    for rule in mapping.get("rules") or []:
        if rule.get("mode", "multi_label") != "multi_label":
            raise ValueError(f"Unsupported label rule mode: {rule.get('mode')}")
        rules.append(
            LabelRule(
                question_contains=str(rule["question_contains"]).lower(),
                values={str(k).lower(): str(v) for k, v in (rule.get("values") or {}).items()},
                none_values=tuple(str(v).lower() for v in rule.get("none_values", ("none", "no"))),
            )
        )
    return rules


def parse_original(original: Any) -> list[tuple[str, str]]:
    """Return the source (question, answer) pairs of a Kvasir-VQA-x1 row."""
    if original is None or original == "":
        return []
    if isinstance(original, str):
        try:
            items = json.loads(original)
        except json.JSONDecodeError:
            items = ast.literal_eval(original)
    else:
        items = original
    pairs = []
    for item in items:
        if isinstance(item, Mapping):
            pairs.append((str(item.get("q", "")), str(item.get("a", ""))))
    return pairs


def derive_findings(
    pairs: Iterable[tuple[str, str]], rules: Sequence[LabelRule]
) -> dict[str, bool]:
    """Return ``{finding: present}`` for one image from its source QA pairs."""
    labels: dict[str, bool] = {}
    for question, answer in pairs:
        q = question.lower()
        for rule in rules:
            if rule.question_contains not in q:
                continue
            values = [v.strip().lower() for v in answer.split(";") if v.strip()]
            findings = set(rule.values.values())
            present = {rule.values[v] for v in values if v in rule.values}
            if values and all(v in rule.none_values for v in values):
                present = set()
            for finding in findings:
                labels[finding] = labels.get(finding, False) or finding in present
    return labels


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sample_id_for(img_id: str, question: str) -> str:
    """Return a stable sample ID from the image and the question text."""
    digest = hashlib.sha1(f"{img_id}\n{question}".encode()).hexdigest()[:12]
    return f"{img_id}:{digest}"


class KvasirVQAx1(VQADataset):
    """Kvasir-VQA-x1 rows as ``VQASample`` records with per-image finding labels."""

    def __init__(self, config: Mapping[str, Any], *, data_root: Path) -> None:
        super().__init__(config)
        self.data_root = Path(data_root)
        self.rules = parse_label_rules(config.get("label_mapping"))

    def load(self) -> list[VQASample]:
        """Read both official splits; ``split`` holds the official split name."""
        rows_by_split = {}
        pairs_by_image: dict[str, list[tuple[str, str]]] = {}
        for split in OFFICIAL_SPLITS:
            path = self.data_root / X1_DIR / f"{split}.jsonl"
            rows = _read_jsonl(path)
            rows_by_split[split] = rows
            for row in rows:
                pairs_by_image.setdefault(str(row["img_id"]), []).extend(
                    parse_original(row.get("original"))
                )
        findings = {
            img: derive_findings(pairs, self.rules) for img, pairs in pairs_by_image.items()
        }
        samples: list[VQASample] = []
        seen: set[str] = set()
        for split, rows in rows_by_split.items():
            for row in rows:
                img_id = str(row["img_id"])
                sid = sample_id_for(img_id, str(row["question"]))
                if sid in seen:
                    continue
                seen.add(sid)
                samples.append(
                    VQASample(
                        sample_id=sid,
                        dataset="kvasir_vqa_x1",
                        image_path=f"{X1_DIR}/images/{img_id}.jpg",
                        question=str(row["question"]),
                        answer=str(row["answer"]),
                        patient_id=None,
                        split=split,
                        findings=findings.get(img_id, {}),
                        metadata={
                            "img_id": img_id,
                            "complexity": row.get("complexity"),
                            "question_class": tuple(row.get("question_class") or ()),
                        },
                    )
                )
        return samples


@dataclass(frozen=True)
class SegRecord:
    """One Kvasir-SEG image with its mask (paths relative to the data root)."""

    name: str
    image_path: str
    mask_path: str


def load_kvasir_seg(data_root: Path) -> list[SegRecord]:
    """Return image/mask pairs that exist under ``<data_root>/kvasir_seg`` (sorted by name)."""
    root = Path(data_root) / SEG_DIR
    records = []
    for image in sorted((root / "images").glob("*")):
        mask = root / "masks" / image.name
        if image.is_file() and mask.is_file():
            records.append(
                SegRecord(
                    image.stem, f"{SEG_DIR}/images/{image.name}", f"{SEG_DIR}/masks/{image.name}"
                )
            )
    return records


@dataclass(frozen=True)
class MaskLinks:
    """Result of matching Kvasir-SEG images to VQA images by fingerprint."""

    links: Mapping[str, str]  # VQA image_path -> mask_path
    ambiguous: tuple[str, ...]  # VQA images with two or more equally close masks
    unmatched_masks: int


def link_masks(
    vqa_images: Sequence[str],
    seg_records: Sequence[SegRecord],
    *,
    data_root: Path,
    threshold: int,
) -> MaskLinks:
    """Match each VQA image to the closest Kvasir-SEG image within ``threshold`` bits.

    VQA images whose closest masks tie are reported as ambiguous and not linked.
    """
    root = Path(data_root)
    vqa_fp = {p: image_fingerprint(str(root / p)) for p in sorted(set(vqa_images))}
    seg_fp = {r.image_path: image_fingerprint(str(root / r.image_path)) for r in seg_records}
    mask_of = {r.image_path: r.mask_path for r in seg_records}
    best: dict[str, list[tuple[int, str]]] = {}
    for vqa_path, seg_path, distance in near_duplicate_pairs(seg_fp, vqa_fp, threshold=threshold):
        best.setdefault(vqa_path, []).append((distance, seg_path))
    links: dict[str, str] = {}
    ambiguous = []
    for vqa_path, candidates in sorted(best.items()):
        candidates.sort()
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            ambiguous.append(vqa_path)
            continue
        links[vqa_path] = mask_of[candidates[0][1]]
    used = set(links.values())
    return MaskLinks(
        links, tuple(ambiguous), sum(1 for r in seg_records if r.mask_path not in used)
    )
