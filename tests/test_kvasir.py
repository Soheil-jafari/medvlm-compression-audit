"""Kvasir-VQA-x1 / Kvasir-SEG readers and mask linking on a synthetic local layout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from medcompress.config import config_root, load_config
from medcompress.data.base import load_dataset
from medcompress.data.kvasir import (
    LabelRule,
    derive_findings,
    link_masks,
    load_kvasir_seg,
    parse_label_rules,
    parse_original,
    sample_id_for,
)

RULES = {
    "rules": [
        {
            "question_contains": "are there any abnormalities in the image",
            "values": {"polyp": "polyp", "ulcerative colitis": "ulcerative_colitis"},
        }
    ]
}


def _image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, size=(12, 16, 3), dtype=np.uint8)
    Image.fromarray(small).resize((128, 96), Image.Resampling.BILINEAR).save(path)


def _layout(root: Path) -> None:
    abnormal = "Are there any abnormalities in the image? Check all that are present."
    rows = {
        "train": [
            {
                "img_id": "a1",
                "question": "Is there a polyp?",
                "answer": "yes",
                "complexity": 1,
                "question_class": ["polyp_count"],
                "original": json.dumps([{"q": abnormal, "a": "polyp"}]),
            },
            {
                "img_id": "a2",
                "question": "What colour?",
                "answer": "pink",
                "complexity": 2,
                "question_class": ["abnormality_color"],
                "original": json.dumps([{"q": abnormal, "a": "none"}]),
            },
        ],
        "test": [
            {
                "img_id": "a3",
                "question": "Any findings?",
                "answer": "ulcerative colitis",
                "complexity": 1,
                "question_class": ["abnormality_presence"],
                "original": str([{"q": abnormal, "a": "polyp; ulcerative colitis"}]),
            },
        ],
    }
    for split, items in rows.items():
        path = root / "kvasir_vqa_x1" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in items))
    for i, img in enumerate(["a1", "a2", "a3"]):
        _image(root / "kvasir_vqa_x1" / "images" / f"{img}.jpg", seed=i)
    # Kvasir-SEG: a re-encoded copy of a1 and an unrelated image, each with a mask
    _image(root / "kvasir_seg" / "images" / "s1.jpg", seed=0)
    _image(root / "kvasir_seg" / "images" / "s9.jpg", seed=9)
    for name in ("s1.jpg", "s9.jpg"):
        _image(root / "kvasir_seg" / "masks" / name, seed=99)


def test_label_rules_and_original_parsing() -> None:
    rules = parse_label_rules(RULES)
    assert rules[0] == LabelRule(
        "are there any abnormalities in the image",
        {"polyp": "polyp", "ulcerative colitis": "ulcerative_colitis"},
    )
    pairs = parse_original('[{"q": "Are there any abnormalities in the image?", "a": "Polyp"}]')
    assert derive_findings(pairs, rules) == {"polyp": True, "ulcerative_colitis": False}
    assert derive_findings([("Are there any abnormalities in the image?", "none")], rules) == {
        "polyp": False,
        "ulcerative_colitis": False,
    }
    assert derive_findings([("Unrelated question", "polyp")], rules) == {}
    assert parse_label_rules(None) == []


def test_reader_builds_samples_with_labels(tmp_path: Path) -> None:
    _layout(tmp_path)
    dataset = load_dataset({"name": "Kvasir-VQA-x1", "label_mapping": RULES}, data_root=tmp_path)
    samples = dataset.load()
    assert [s.split for s in samples] == ["train", "train", "test"]
    by_img = {s.metadata["img_id"]: s for s in samples}
    assert by_img["a1"].findings == {"polyp": True, "ulcerative_colitis": False}
    assert by_img["a3"].findings == {"polyp": True, "ulcerative_colitis": True}
    assert by_img["a1"].image_path == "kvasir_vqa_x1/images/a1.jpg"
    assert by_img["a1"].sample_id == sample_id_for("a1", "Is there a polyp?")
    assert len({s.sample_id for s in samples}) == 3


def test_seg_records_and_mask_linking(tmp_path: Path) -> None:
    _layout(tmp_path)
    records = load_kvasir_seg(tmp_path)
    assert [r.name for r in records] == ["s1", "s9"]
    links = link_masks(
        ["kvasir_vqa_x1/images/a1.jpg", "kvasir_vqa_x1/images/a2.jpg"],
        records,
        data_root=tmp_path,
        threshold=4,
    )
    assert dict(links.links) == {"kvasir_vqa_x1/images/a1.jpg": "kvasir_seg/masks/s1.jpg"}
    assert links.unmatched_masks == 1 and links.ambiguous == ()


def test_repository_config_has_label_rules_section() -> None:
    dataset = load_config(config_root() / "data" / "kvasir_vqa_x1.yaml")["dataset"]
    rules = parse_label_rules(dataset["label_mapping"])
    assert rules and rules[0].values == {}  # values come from RC-D1, then pre-registration


def test_prepare_end_to_end(tmp_path: Path) -> None:
    from medcompress.data.prepare import prepare, write_private_outputs
    from medcompress.evaluation.aggregate import validate_results_tree
    from medcompress.utils.io import append_results

    data = tmp_path / "data"
    _layout(data)
    # a near-duplicate of a training image in the official test split must be dropped
    _image(data / "kvasir_vqa_x1" / "images" / "a4.jpg", seed=0)
    test_rows = (data / "kvasir_vqa_x1" / "test.jsonl").read_text()
    extra = {
        "img_id": "a4",
        "question": "Dup?",
        "answer": "no",
        "complexity": 1,
        "question_class": [],
        "original": "[]",
    }
    (data / "kvasir_vqa_x1" / "test.jsonl").write_text(test_rows + json.dumps(extra) + "\n")
    prepared = prepare(
        data,
        {"name": "Kvasir-VQA-x1", "label_mapping": RULES},
        split_seed=1,
        dedup_threshold=4,
        mask_threshold=4,
    )
    assert "kvasir_vqa_x1/images/a4.jpg" in prepared.excluded
    assert set(prepared.splits) == {"train", "val", "test"}
    assert all(
        s.image_path != "kvasir_vqa_x1/images/a4.jpg"
        for s in prepared.splits["val"] + prepared.splits["test"]
    )
    metrics = {r["metric"]: r["value"] for r in prepared.rows}
    assert metrics["qa_rows:train"] == 2 and metrics["masks:linked"] >= 1
    assert metrics["prevalence:polyp"] == 0.5
    assert metrics["answer_count:abnormalities:polyp"] == 2
    hashes = write_private_outputs(prepared, tmp_path / "private", data)
    assert {"split_sha256", "manifest_kvasir_vqa_x1", "manifest_kvasir_seg"} <= set(hashes)
    assert (tmp_path / "private" / "splits.json").is_file()
    append_results(prepared.rows, tmp_path / "results" / "data" / "summary.csv")
    assert validate_results_tree(tmp_path / "results") == []


def test_load_prepared_rebuilds_the_prepared_splits(tmp_path: Path) -> None:
    import pytest

    from medcompress.data.prepare import prepare, write_private_outputs
    from medcompress.data.prepared import load_prepared
    from medcompress.data.splits import split_fingerprint

    data = tmp_path / "data"
    _layout(data)
    config = {"name": "Kvasir-VQA-x1", "label_mapping": RULES}
    prepared = prepare(data, config, split_seed=1, dedup_threshold=4, mask_threshold=4)
    write_private_outputs(prepared, tmp_path / "private", data)

    rebuilt = load_prepared(
        data, config, tmp_path / "private", expected_split_sha256=prepared.split_sha256
    )
    assert set(rebuilt) == set(prepared.splits)
    for name, members in prepared.splits.items():
        assert [s.sample_id for s in rebuilt[name]] == [s.sample_id for s in members]
        assert all(s.split == name for s in rebuilt[name])
    assert split_fingerprint(rebuilt) == prepared.split_sha256
    linked = [s for members in rebuilt.values() for s in members if s.mask_path]
    assert linked and all(0 < (s.lesion_area or 0) <= 1 for s in linked)

    with pytest.raises(ValueError, match="split fingerprint"):
        load_prepared(data, config, tmp_path / "private", expected_split_sha256="0" * 64)
    (data / "kvasir_vqa_x1" / "train.jsonl").write_text("")  # a different dataset version
    with pytest.raises(ValueError, match="not in the dataset"):
        load_prepared(data, config, tmp_path / "private")
