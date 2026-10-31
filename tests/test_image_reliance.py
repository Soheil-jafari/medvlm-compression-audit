"""Stress tests, lesion-mask audit and the Gate 1 verdict with deterministic fake models.

The fake "reliant" model says yes exactly when the image contains a bright blob, so
blanking or masking the blob must flip its answer while masking a same-shaped region
elsewhere must not. The fake "blind" model ignores the image.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from medcompress.config import ConfigError
from medcompress.data.base import VQASample
from medcompress.evaluation.aggregate import validate_results_tree
from medcompress.evaluation.image_reliance import (
    StressTest,
    balanced_accuracy,
    blank_image,
    check_teacher_image_reliance,
    fill_region,
    gate1_verdict,
    macro_balanced_accuracy,
    mask_audit,
    pair_opposite_label,
    reliance_metrics,
    stress_image,
    subsample_per_finding,
    unique_image_probes,
)
from medcompress.models.base import YesNoScore

TEMPLATE = "Is there a {finding} in this image? Answer yes or no."
SIZE = (64, 48)


class Reliant:
    """Says yes iff some pixel is brighter than 200."""

    def score_yes_no(self, images, questions):  # noqa: ANN001, ANN201
        out = []
        for image in images:
            bright = np.asarray(image.convert("L")).max() > 200
            out.append(YesNoScore(-0.1, -3.0) if bright else YesNoScore(-3.0, -0.1))
        return out


class Blind:
    def score_yes_no(self, images, questions):  # noqa: ANN001, ANN201
        return [YesNoScore(-0.1, -3.0) for _ in images]


def _blob_image(path: Path, *, blob: bool, seed: int) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(20, 120, size=(SIZE[1], SIZE[0], 3), dtype=np.uint8)
    if blob:
        array[10:20, 10:22] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _mask(path: Path) -> None:
    mask = np.zeros((SIZE[1], SIZE[0]), dtype=np.uint8)
    mask[10:20, 10:22] = 255
    Image.fromarray(mask).save(path)


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, list[VQASample]]:
    samples = []
    for i in range(12):
        positive = i % 2 == 0
        rel = f"images/{i}.png"
        _blob_image(tmp_path / rel, blob=positive, seed=i)
        mask = None
        if positive:
            mask = f"masks/{i}.png"
            (tmp_path / "masks").mkdir(exist_ok=True)
            _mask(tmp_path / mask)
        samples.append(
            VQASample(f"s{i}", "d", rel, "q", "a", findings={"polyp": positive}, mask_path=mask)
        )
        samples.append(
            VQASample(f"s{i}b", "d", rel, "q2", "a", findings={"polyp": positive}, mask_path=mask)
        )
    return tmp_path, samples


def test_balanced_accuracy_handles_missing_classes() -> None:
    assert balanced_accuracy(
        [True, True, False, False], [True, False, False, False]
    ) == pytest.approx(0.75)
    assert math.isnan(balanced_accuracy([True, True], [True, True]))
    value = macro_balanced_accuracy(
        ["a", "a", "b", "b", "c"], [True, False, True, False, True], [True, False, True, True, True]
    )
    assert value == pytest.approx((1.0 + 0.5) / 2)  # finding c has one class only and is skipped


def test_image_edits() -> None:
    image = Image.fromarray(np.full((10, 20, 3), 50, dtype=np.uint8))
    assert set(np.asarray(blank_image(image, 99)).ravel()) == {99}
    region = np.zeros((10, 20), dtype=bool)
    region[2:4, 5:9] = True
    filled = np.asarray(fill_region(image, region, 200))
    assert (filled[region] == 200).all() and (filled[~region] == 50).all()
    coarse = np.zeros((5, 10), dtype=bool)
    coarse[:, :5] = True  # half the width, given at half resolution
    assert (np.asarray(fill_region(image, coarse, 7))[:, :10] == 7).all()
    with pytest.raises(ValueError, match="mask"):
        stress_image(image, StressTest.LESION_MASKED)
    with pytest.raises(ValueError, match="partner"):
        stress_image(image, StressTest.OTHER_PATIENT_IMAGE)


def test_random_region_control_matches_the_lesion_and_avoids_it() -> None:
    image = Image.fromarray(np.full((48, 64, 3), 50, dtype=np.uint8))
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:20, 10:22] = True
    control, overlap = stress_image(
        image, StressTest.RANDOM_REGION_MASKED, mask=mask, seed=3, fill=255
    )
    assert overlap == 0.0
    assert (np.asarray(control)[..., 0] == 255).sum() == mask.sum()


def test_probe_construction_pairing_and_subsampling(dataset: tuple[Path, list[VQASample]]) -> None:
    _, samples = dataset
    probes = unique_image_probes(
        samples, ["polyp", "missing"], TEMPLATE, display_names={"polyp": "polyp"}
    )
    assert len(probes) == 12  # two QA rows per image count once; unlabelled finding skipped
    assert probes[0].question == "Is there a polyp in this image? Answer yes or no."
    pairs = pair_opposite_label(probes, seed=1)
    assert len(pairs) == 12 and all(
        probes[i].present != probes[j].present for i, j in pairs.items()
    )
    assert pairs == pair_opposite_label(probes, seed=1)
    shared = {p.image_path: "same patient" for p in probes}
    assert pair_opposite_label(probes, seed=1, groups=shared) == {}
    kept = subsample_per_finding(probes, 5, seed=0)
    assert len(kept) == 5 and kept == subsample_per_finding(probes, 5, seed=0)


def test_mask_audit_separates_a_reliant_model_from_a_blind_one(
    dataset: tuple[Path, list[VQASample]],
) -> None:
    root, samples = dataset
    args = {"seed": 0, "data_root": root, "findings": ["polyp"], "template": TEMPLATE}
    reliant = mask_audit(Reliant(), samples, **args)
    summary = reliant.summary(n_resamples=200)
    assert summary["n_probes"] == 6
    assert summary["lesion_flip_rate"] == 1.0 and summary["random_flip_rate"] == 0.0
    assert summary["lesion_mask_effect"] == 1.0 and summary["lesion_mask_effect_ci_low"] == 1.0
    assert summary["lesion_mask_margin_effect"] > 0 and summary["control_overlap_mean"] == 0.0
    blind = mask_audit(Blind(), samples, **args).summary(n_resamples=200)
    assert blind["lesion_mask_effect"] == 0.0
    limited = mask_audit(Reliant(), samples, max_probes=2, **args)
    assert len(limited.sample_ids) == 2
    with pytest.raises(ValueError, match="No positive"):
        mask_audit(Reliant(), [s for s in samples if not s.mask_path], **args)


def test_reliance_metrics(dataset: tuple[Path, list[VQASample]]) -> None:
    root, samples = dataset
    args = {"seed": 0, "data_root": root, "findings": ["polyp"], "template": TEMPLATE}
    reliant = reliance_metrics(Reliant(), samples, **args)
    assert reliant["original_balanced_accuracy"] == 1.0
    assert reliant["blank_balanced_accuracy"] == 0.5 and reliant["blank_accuracy_drop"] == 0.5
    assert reliant["blank_yes_rate"] == 0.0 and reliant["other_image_follow_rate"] == 1.0
    blind = reliance_metrics(Blind(), samples, **args)
    assert blind["blank_accuracy_drop"] == 0.0 and blind["blank_yes_rate"] == 1.0
    assert blind["other_image_follow_rate"] == 0.5


THRESHOLDS = {
    "min_original_balanced_accuracy": 0.65,
    "min_blank_accuracy_drop": 0.10,
    "min_lesion_mask_effect_ci_low": 0.0,
}


def test_gate1_verdict() -> None:
    good = {
        "original_balanced_accuracy": 0.8,
        "blank_accuracy_drop": 0.2,
        "lesion_mask_effect_ci_low": 0.05,
    }
    assert gate1_verdict(good, THRESHOLDS)["gate1_pass"] == 1.0
    weak = gate1_verdict({**good, "lesion_mask_effect_ci_low": 0.0}, THRESHOLDS)
    assert weak["gate1_pass"] == 0.0 and weak["gate1_lesion_effect_ok"] == 0.0
    assert weak["gate1_accuracy_ok"] == 1.0 and weak["gate1_blank_drop_ok"] == 1.0
    assert gate1_verdict({**good, "blank_accuracy_drop": math.nan}, THRESHOLDS)["gate1_pass"] == 0.0
    with pytest.raises(ValueError, match="not set"):
        gate1_verdict(good, {})


def _config(tmp_path: Path, **overrides: object) -> dict:
    gate = dict(THRESHOLDS)
    section = {
        "fill": 128,
        "mask_findings": ["polyp"],
        "max_probes_per_finding": None,
        "max_mask_probes": None,
        "gate1": gate,
        **overrides,
    }
    return {
        "paths": {
            "data_root": str(tmp_path),
            "results_root": str(tmp_path / "results"),
            "private_root": str(tmp_path / "private"),
        },
        "components": {
            "domain": "domain/endoscopy.yaml",
            "datasets": {"train_qa": "data/kvasir_vqa_x1.yaml"},
        },
        "image_reliance_check": section,
    }


def test_stage_is_blocked_while_the_finding_list_is_open(
    dataset: tuple[Path, list[VQASample]], monkeypatch: pytest.MonkeyPatch
) -> None:
    import medcompress.config as config_module

    root, samples = dataset
    real_load = config_module.load_config

    def open_list(path, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        loaded = real_load(path, *args, **kwargs)
        if "domain" in loaded and "findings" in loaded["domain"]:
            loaded["domain"]["findings"] = ["TODO"]  # the list frozen on 2026-10-31, reopened
        return loaded

    monkeypatch.setattr(config_module, "load_config", open_list)
    with pytest.raises(ConfigError, match="domain.findings"):
        check_teacher_image_reliance(_config(root), None, model=Reliant(), samples=samples)
    with pytest.raises(ConfigError, match="gate1.min_blank_accuracy_drop"):
        check_teacher_image_reliance(
            _config(root, gate1={}), None, model=Reliant(), samples=samples
        )


def test_stage_writes_metrics_and_replaces_them_on_rerun(
    dataset: tuple[Path, list[VQASample]], monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas as pd

    import medcompress.config as config_module

    root, samples = dataset
    real_load = config_module.load_config

    def fake_load(path, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        loaded = real_load(path, *args, **kwargs)
        if "domain" in loaded and "findings" in loaded["domain"]:
            loaded["domain"]["findings"] = ["polyp"]
        return loaded

    monkeypatch.setattr(config_module, "load_config", fake_load)
    check_teacher_image_reliance(_config(root), None, model=Reliant(), samples=samples)
    check_teacher_image_reliance(_config(root), None, model=Reliant(), samples=samples)
    target = root / "results" / "E0" / "nobudget" / "seed0" / "metrics.csv"
    frame = pd.read_csv(target)
    assert frame["metric"].is_unique  # the second run replaced the first
    values = dict(zip(frame["metric"], frame["value"], strict=True))
    assert values["gate1_pass"] == 1.0 and values["lesion_mask_effect"] == 1.0
    row = frame[frame["metric"] == "lesion_mask_effect"].iloc[0]
    assert row["ci_low"] <= row["value"] <= row["ci_high"]
    assert validate_results_tree(root / "results") == []


def test_the_cache_waits_for_a_passed_gate1(tmp_path: Path) -> None:
    from medcompress.evaluation.image_reliance import require_gate1_pass
    from medcompress.utils.io import append_results

    config = {"paths": {"results_root": str(tmp_path / "results")}}
    with pytest.raises(ConfigError, match="has not run"):
        require_gate1_pass(config)
    target = tmp_path / "results" / "E0" / "nobudget" / "seed0" / "metrics.csv"
    row = {"experiment": "E0", "budget": "", "seed": 0, "metric": "gate1_pass", "value": 0.0}
    append_results([row], target)
    with pytest.raises(ConfigError, match="did not pass"):
        require_gate1_pass(config)
    target.unlink()
    append_results([{**row, "value": 1.0}], target)
    require_gate1_pass(config)  # passes quietly
