"""The fp16 calibration check (pre-registration section 10, amendment of 2026-11-13)."""

from __future__ import annotations

from pathlib import Path

import pytest
from eval_fixtures import write_dataset

from medcompress.config import ConfigError
from medcompress.data import calibration as cal
from medcompress.data.splits import make_calibration_subset


def _train(root: Path) -> list:
    return [s for s in write_dataset(root)]  # 12 images, 2 rows each; polyp on even images


def test_without_exclusions_the_subset_is_unchanged_and_with_them_the_next_image_moves_up(
    tmp_path: Path,
) -> None:
    train = _train(tmp_path)
    before = make_calibration_subset(train, 6, seed=3)
    assert make_calibration_subset(train, 6, seed=3, exclude=()) == before
    gone = before[2].sample_id  # not a stratification front image (polyp and bleeding first)
    after = make_calibration_subset(train, 6, seed=3, exclude=[gone])
    order = [s.sample_id for s in make_calibration_subset(train, 12, seed=3)]
    assert gone not in [s.sample_id for s in after]
    expected = [i for i in order if i != gone][:6]  # the same seeded order, the next one in
    assert [s.sample_id for s in after] == expected
    small = make_calibration_subset(train, 4, seed=3, exclude=[gone])  # still nested (A4)
    assert small == after[:4]


def test_an_excluded_front_image_is_replaced_by_the_next_positive_of_its_finding(
    tmp_path: Path,
) -> None:
    train = _train(tmp_path)
    full = make_calibration_subset(train, 12, seed=1)
    front = full[0]  # the first positive of the first finding (bleeding, sorted)
    assert front.findings["bleeding"]
    after = make_calibration_subset(train, 11, seed=1, exclude=[front.sample_id])
    assert after[0].findings["bleeding"] and after[0].sample_id != front.sample_id
    nxt = next(s for s in full[1:] if s.findings["bleeding"] and s.sample_id != front.sample_id)
    assert after[0].sample_id == nxt.sample_id  # the next positive in the same order


def test_the_rule_replaces_each_non_finite_image_once(tmp_path: Path) -> None:
    train = _train(tmp_path)
    original = make_calibration_subset(train, 5, seed=0)
    bad = {original[1].sample_id, original[3].sample_id}
    seen: list[str] = []

    def finite(sample: object) -> bool:
        seen.append(sample.sample_id)  # type: ignore[attr-defined]
        return sample.sample_id not in bad  # type: ignore[attr-defined]

    subset, replaced = cal.find_replacements(train, 5, 0, finite, log=lambda _m: None)
    assert set(replaced) == bad and len(subset) == 5
    assert not bad & {s.sample_id for s in subset}
    assert len(seen) == len(set(seen)) == 7  # every image checked once: 5 + 2 replacements
    assert subset == make_calibration_subset(train, 5, seed=0, exclude=replaced)


def test_the_committed_check_is_read_by_every_stage(tmp_path: Path) -> None:
    train = _train(tmp_path)
    config = {"paths": {"results_root": str(tmp_path / "results")},
              "calibration_check": {"required": True}}  # fmt: skip
    with pytest.raises(ConfigError, match="check_calibration.py"):
        cal.calibration_set(config, train, 4, 0)
    first = make_calibration_subset(train, 4, seed=0)
    path = cal.write_check(tmp_path / "results", 0, "student", 6, [first[2].sample_id])
    assert path == tmp_path / "results/calibration/seed0_replaced.csv"
    cal.write_check(tmp_path / "results", 0, "teacher", 4, [])  # kept next to the student's
    assert cal.read_check(tmp_path / "results", 0, "student") == (6, [first[2].sample_id])
    assert cal.read_check(tmp_path / "results", 0, "teacher") == (4, [])
    cleaned = cal.calibration_set(config, train, 4, 0)
    assert first[2].sample_id not in [s.sample_id for s in cleaned]
    assert cal.calibration_set(config, train, 4, 0, "teacher") == first  # its own check
    with pytest.raises(ConfigError, match="covers 6 images; this run needs 8"):
        cal.calibration_set(config, train, 8, 0)
    assert cal.calibration_set({"paths": config["paths"]}, train, 4, 0) == first  # tests


def test_the_check_on_a_tiny_model_replaces_an_injected_non_finite_example(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import tiny_vlm

    from medcompress.compression.quantization.ptq import calibration_batches

    wrapper = tiny_vlm.tiny_wrapper(seed=0)
    train = _train(tmp_path)
    poisoned = make_calibration_subset(train, 4, seed=0)[1].sample_id

    def is_finite(sample: object) -> bool:
        (batch,) = calibration_batches(wrapper, [sample], tmp_path)  # type: ignore[list-item]
        if sample.sample_id == poisoned:  # type: ignore[attr-defined]
            batch = {**batch, "pixel_values": batch["pixel_values"].clone()}
            batch["pixel_values"][0, 0] = float("inf")  # what an fp16 overflow leaves
        return cal.forward_is_finite(wrapper, batch)

    subset, replaced = cal.find_replacements(train, 4, 0, is_finite, log=lambda _m: None)
    assert replaced == [poisoned] and poisoned not in [s.sample_id for s in subset]
    (clean,) = calibration_batches(wrapper, [subset[0]], tmp_path)
    assert cal.forward_is_finite(wrapper, clean)
    assert torch.is_tensor(clean["pixel_values"])


def test_the_vision_tower_alone_gives_the_full_models_output(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import tiny_vlm

    from medcompress.models.loading import ModelSpec
    from medcompress.models.vision_only import load_vision_tower, output_tensor, tower_is_finite

    wrapper = tiny_vlm.tiny_wrapper(seed=0)
    wrapper.module.to(torch.float16).save_pretrained(tmp_path / "m")
    wrapper.processor.save_pretrained(tmp_path / "m")
    spec = ModelSpec(
        name="tiny", role="student", family="qwen2_5_vl", source=str(tmp_path / "m"),
        revision=None, sha_seen=None, adapter=None, adapter_revision=None, dtype="float16",
        status="DECIDED", tokenizer_family="qwen", vocab_size=48,
    )  # fmt: skip
    tower, processor = load_vision_tower(spec, device="cpu")
    pixels = processor.image_processor(images=[tiny_vlm.tiny_image(0)], return_tensors="pt")
    with torch.inference_mode():
        alone = output_tensor(tower(pixels["pixel_values"], grid_thw=pixels["image_grid_thw"]))
        visual = wrapper.module.model.visual
        half = pixels["pixel_values"].to(torch.float16)
        full = output_tensor(visual(half, grid_thw=pixels["image_grid_thw"]))
    assert alone.dtype == torch.float16 and torch.equal(alone, full)
    assert tower_is_finite(tower, pixels["pixel_values"], pixels["image_grid_thw"]) == (True, None)
    bad = pixels["pixel_values"].clone()
    bad[0, 0] = float("inf")
    finite, block = tower_is_finite(tower, bad, pixels["image_grid_thw"])
    assert not finite and block == "block 0"
