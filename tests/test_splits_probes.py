"""Probe construction, image/patient-level splits and nested calibration subsets."""

from __future__ import annotations

import pytest

from medcompress.data.base import VQASample
from medcompress.data.probes import build_probes
from medcompress.data.splits import (
    CALIBRATION_SIZES,
    check_patient_disjoint,
    make_calibration_subset,
    make_patient_splits,
    split_fingerprint,
)

TEMPLATE = "Is there a {finding} in this image? Answer yes or no."


def _samples(n_images: int, questions_per_image: int = 2) -> list[VQASample]:
    out = []
    for i in range(n_images):
        findings = {"polyp": i % 3 == 0, "ulcer": i % 7 == 0, "esophagitis": i == 5}
        for q in range(questions_per_image):
            out.append(
                VQASample(
                    sample_id=f"s{i}_{q}",
                    dataset="synthetic",
                    image_path=f"img/{i}.jpg",
                    question=f"q{q}",
                    answer="yes",
                    findings=findings,
                )
            )
    return out


def test_build_probes_answers_and_display_names() -> None:
    sample = VQASample(
        "s1", "synthetic", "img/1.jpg", "q", "a", findings={"polyp": True, "ulcer": False}
    )
    probes = build_probes(
        [sample], ["polyp", "ulcer", "bleeding"], TEMPLATE, display_names={"ulcer": "ulcer lesion"}
    )
    assert [(p.finding, p.answer) for p in probes] == [("polyp", "yes"), ("ulcer", "no")]
    assert probes[1].question.startswith("Is there a ulcer lesion in this image?")


def test_splits_are_image_disjoint_and_deterministic() -> None:
    samples = _samples(40)
    splits = make_patient_splits(samples, {"val": 0.5, "test": 0.5}, seed=7)
    assert check_patient_disjoint(splits) == []
    assert sum(len(v) for v in splits.values()) == len(samples)
    assert {len({s.image_path for s in v}) for v in splits.values()} == {20}
    assert all(s.split == name for name, v in splits.items() for s in v)
    again = make_patient_splits(list(reversed(samples)), {"val": 0.5, "test": 0.5}, seed=7)
    assert split_fingerprint(splits) == split_fingerprint(again)
    other = make_patient_splits(samples, {"val": 0.5, "test": 0.5}, seed=8)
    assert split_fingerprint(splits) != split_fingerprint(other)


def test_patient_ids_group_images() -> None:
    samples = [
        VQASample(f"s{i}", "synthetic", f"img/{i}.jpg", "q", "a", patient_id=str(i // 4))
        for i in range(40)
    ]
    splits = make_patient_splits(samples, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=0)
    assert check_patient_disjoint(splits) == []
    leaked = {"a": samples[:2], "b": samples[1:3]}
    assert check_patient_disjoint(leaked) == ["patient:0"]


def test_split_ratios_validated() -> None:
    with pytest.raises(ValueError):
        make_patient_splits(_samples(4), {"a": 0.5, "b": 0.6}, seed=0)


def test_calibration_subsets_are_nested_and_cover_findings() -> None:
    train = _samples(300, questions_per_image=1)
    subsets = {n: make_calibration_subset(train, n, seed=1) for n in CALIBRATION_SIZES}
    for small, large in zip(CALIBRATION_SIZES, CALIBRATION_SIZES[1:], strict=False):
        assert [s.image_path for s in subsets[small]] == [
            s.image_path for s in subsets[large][:small]
        ]
    first = subsets[32]
    assert len({s.image_path for s in first}) == 32
    for finding in ("polyp", "ulcer", "esophagitis"):
        assert any(s.findings[finding] for s in first), finding


def test_calibration_subset_rules() -> None:
    train = _samples(10)
    assert len(make_calibration_subset(train, 10, seed=0)) == 10  # one sample per image
    with pytest.raises(ValueError):
        make_calibration_subset(train, 11, seed=0)
    test_sample = VQASample("t", "synthetic", "img/t.jpg", "q", "a", split="test")
    with pytest.raises(ValueError, match="training split"):
        make_calibration_subset([test_sample], 1, seed=0)
