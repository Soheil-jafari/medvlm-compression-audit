"""Two-stage near-duplicate grouping (D02 amendment 2 of 2026-10-18, final values 2026-10-19)
on synthetic images (CPU).

The candidate stage (dHash), the confirmation by the crop (identical and shifted copies joined,
different images not, overlays ignored; no automatic join at 0-4 bits), the chaining guard at
10 %, determinism, the calibration step and the final contact sheet.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from medcompress.data.dedup import (
    ChainingError,
    CropSettings,
    NearDuplicateSettings,
    candidate_pairs,
    check_chaining,
    crop_similarities,
    crop_similarity,
    cross_similarity_matrix,
    hamming_distance,
    image_features,
    joined_pairs,
    near_duplicate_groups,
    two_stage_groups,
    union_groups,
)

REPO = Path(__file__).resolve().parents[1]
CROP = CropSettings()
W, H = 240, 200
MARGIN = 24


def _scene(seed: int) -> Image.Image:
    """A smooth random 'mucosa' with a dark centre, larger than one view (for shifted views)."""
    rng = np.random.default_rng(seed)
    size = (W + 2 * MARGIN, H + 2 * MARGIN)
    small = rng.integers(0, 255, size=(14, 16, 3), dtype=np.uint8)
    base = np.asarray(Image.fromarray(small).resize(size, Image.Resampling.BICUBIC), dtype=float)
    yy, xx = np.mgrid[0 : size[1], 0 : size[0]]
    fall_off = 0.4 + 1.2 * np.hypot(xx - size[0] / 2, yy - size[1] / 2) / max(size)
    return Image.fromarray(np.clip(base * fall_off[..., None], 0, 255).astype(np.uint8))


def _view(scene: Image.Image, dx: int = 0, dy: int = 0) -> Image.Image:
    return scene.crop((MARGIN + dx, MARGIN + dy, MARGIN + dx + W, MARGIN + dy + H))


def _overlay(image: Image.Image, box: tuple[int, int, int] = (0, 200, 0)) -> Image.Image:
    """What the endoscope draws: text at the top left, a green box at the bottom left, a black
    border on the right; all within the outer 16 % of the image."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, int(W * 0.15), int(H * 0.10)), fill=(255, 255, 255))
    draw.rectangle((0, int(H * 0.84), int(W * 0.16), H), fill=box)
    draw.rectangle((int(W * 0.94), 0, W, H), fill=(0, 0, 0))
    return out


def _save(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, quality=92)
    return path


def _features(images: dict[str, Image.Image], root: Path) -> tuple[dict, dict]:
    fingerprints, signatures = {}, {}
    for name, image in images.items():
        fingerprints[name], signatures[name] = image_features(str(_save(image, root / name)), CROP)
    return fingerprints, signatures


# ------------------------------------------------------------------ the candidate stage


def _fp(bits: int) -> str:
    return f"{bits:016x}"


def test_the_candidate_stage_lists_every_pair_within_the_distance_once() -> None:
    fingerprints = {"c": _fp(0b0111), "a": _fp(0), "b": _fp(0b0011), "d": _fp(0xFFFF << 48)}
    pairs = candidate_pairs(fingerprints, threshold=2)
    assert pairs == [("a", "b", 2), ("b", "c", 1)]  # sorted, smaller key first; a-c is 3 bits
    assert candidate_pairs(dict(reversed(list(fingerprints.items()))), threshold=2) == pairs
    assert len(candidate_pairs(fingerprints, threshold=64)) == 6
    assert candidate_pairs(fingerprints, threshold=0) == []
    # the dHash-only grouping of run 2 is union-find over these pairs
    assert near_duplicate_groups(fingerprints, threshold=2) == union_groups(
        fingerprints, ((a, b) for a, b, _ in pairs)
    )
    assert union_groups("abcd", [("c", "b"), ("b", "a")]) == {
        "a": "a",
        "b": "a",
        "c": "a",
        "d": "d",
    }


# ------------------------------------------------------------------ the confirmation stage


def test_identical_and_shifted_copies_are_joined_and_different_images_are_not(
    tmp_path: Path,
) -> None:
    a, b = _scene(1), _scene(2)
    images = {
        "a.jpg": _view(a),
        "a_copy.png": _view(a),  # re-encoded
        "a_shift.jpg": _view(a, dx=7, dy=4),  # about 3 % of the side
        "a_bright.jpg": _view(a).point(lambda v: min(255, int(v * 1.15) + 10)),
        "b.jpg": _view(b),
    }
    fingerprints, signatures = _features(images, tmp_path)
    score = {
        name: crop_similarity(signatures["a.jpg"], signatures[name], CROP)
        for name in images
        if name != "a.jpg"
    }
    assert score["a_copy.png"] > 0.99 and score["a_bright.jpg"] > 0.98
    assert score["a_shift.jpg"] > 0.95  # the shift search finds the moved frame
    assert score["b.jpg"] < 0.6
    no_shift = CropSettings(max_shift=0)
    unshifted = crop_similarity(
        image_features(str(tmp_path / "a.jpg"), no_shift)[1],
        image_features(str(tmp_path / "a_shift.jpg"), no_shift)[1],
        no_shift,
    )
    assert unshifted < score["a_shift.jpg"]  # without the shifts the moved frame scores lower

    settings = NearDuplicateSettings(  # every pair is a candidate; no chaining guard here
        min_correlation=0.9, candidate_bits=64, max_group_fraction=1.0
    )
    grouping = two_stage_groups(fingerprints, signatures, settings)
    groups = grouping.groups
    assert groups["a_copy.png"] == groups["a_shift.jpg"] == groups["a_bright.jpg"] == "a.jpg"
    assert groups["b.jpg"] == "b.jpg"
    assert len(grouping.candidates) == 10 and sum(grouping.joined) == 6


def test_the_crop_ignores_the_overlays(tmp_path: Path) -> None:
    a, b = _scene(3), _scene(4)
    images = {
        "a.jpg": _view(a),
        "a_overlay.jpg": _overlay(_view(a), box=(0, 120, 255)),
        "b_overlay.jpg": _overlay(_view(b)),
        "c_overlay.jpg": _overlay(_view(_scene(5))),
    }
    fingerprints, signatures = _features(images, tmp_path)
    same = crop_similarity(signatures["a.jpg"], signatures["a_overlay.jpg"], CROP)
    different = crop_similarity(signatures["b_overlay.jpg"], signatures["c_overlay.jpg"], CROP)
    assert same > 0.99  # the same scene, whatever is drawn at the edges
    assert different < 0.6  # different scenes under the same overlays
    # the overlays pull the 9 x 8 dHash of different scenes together; the crop does not follow
    bare = _features({"b.jpg": _view(b), "c.jpg": _view(_scene(5))}, tmp_path / "bare")[0]
    with_overlay = hamming_distance(fingerprints["b_overlay.jpg"], fingerprints["c_overlay.jpg"])
    assert with_overlay < hamming_distance(bare["b.jpg"], bare["c.jpg"])
    settings = NearDuplicateSettings(min_correlation=0.9, candidate_bits=64, max_group_fraction=1)
    grouping = two_stage_groups(fingerprints, signatures, settings)
    assert grouping.groups["a_overlay.jpg"] == "a.jpg"
    assert grouping.groups["b_overlay.jpg"] != grouping.groups["c_overlay.jpg"]


def _lookalike(seed: int) -> Image.Image:
    """A strong left-to-right brightness ramp with faint detail of its own: two of them have
    the same 9 x 8 dHash, like the lumen and the overlays in the calibration, but different
    mucosa."""
    rng = np.random.default_rng(seed)
    ramp = np.tile(np.linspace(30, 220, W), (H, 1))
    small = rng.integers(0, 255, size=(24, 30), dtype=np.uint8)
    texture = np.asarray(Image.fromarray(small).resize((W, H), Image.Resampling.BICUBIC), float)
    grey = np.clip(ramp + 0.25 * (texture - 127), 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([grey] * 3, axis=-1))


def test_pairs_within_four_bits_are_joined_only_by_the_crop(tmp_path: Path) -> None:
    fingerprints, signatures = _features(
        {"l1.jpg": _lookalike(1), "l2.jpg": _lookalike(2), "l1_copy.png": _lookalike(1)},
        tmp_path,
    )
    assert hamming_distance(fingerprints["l1.jpg"], fingerprints["l2.jpg"]) <= 4
    assert crop_similarity(signatures["l1.jpg"], signatures["l2.jpg"], CROP) < 0.3
    settings = NearDuplicateSettings(min_correlation=0.65, max_group_fraction=1.0)
    grouping = two_stage_groups(fingerprints, signatures, settings)
    assert grouping.groups["l2.jpg"] == "l2.jpg"  # close by the dHash, apart by the crop
    assert grouping.groups["l1_copy.png"] == "l1.jpg"  # a copy: joined by the crop
    assert all(d <= 4 for _, _, d in grouping.candidates) and sum(grouping.joined) == 1
    # synthetic scores, 0 bits apart: only the crop decides
    rng = np.random.default_rng(0)
    unrelated = {k: rng.normal(size=(CROP.grid, CROP.grid)) for k in "abc"}
    same = {"a": _fp(0), "b": _fp(0), "c": _fp(0b1111)}
    apart = two_stage_groups(same, unrelated, NearDuplicateSettings(min_correlation=0.65))
    assert apart.groups == {"a": "a", "b": "b", "c": "c"} and not any(apart.joined)
    assert joined_pairs([("a", "b", 0), ("a", "c", 12)], [0.64, 0.65], settings) == [False, True]


# ------------------------------------------------------------------ the chaining guard


def test_the_chaining_guard_stops_a_group_of_more_than_ten_percent() -> None:
    rng = np.random.default_rng(1)
    shared = rng.normal(size=(CROP.grid, CROP.grid))
    fingerprints, signatures = {}, {}
    for k in range(100):
        fingerprints[f"img{k:03d}"] = f"{int(rng.integers(0, 2**63)):016x}"
        signatures[f"img{k:03d}"] = rng.normal(size=(CROP.grid, CROP.grid))
    for k in range(10):  # a chain: each 2 bits from the next, the same crop (a~b~c...)
        fingerprints[f"img{k:03d}"] = _fp((1 << (2 * k)) - 1)
        signatures[f"img{k:03d}"] = shared
    settings = NearDuplicateSettings(min_correlation=0.65)  # the guard's default: 10 %
    assert settings.max_group_fraction == 0.10
    groups = two_stage_groups(fingerprints, signatures, settings).groups
    assert len({groups[f"img{k:03d}"] for k in range(10)}) == 1  # 10 of 100: not more than 10 %
    fingerprints["img010"], signatures["img010"] = _fp((1 << 20) - 1), shared  # 11 of 100
    with pytest.raises(ChainingError, match=r"11 of 100 images \(11.0%\), more than 10%.*STOP"):
        two_stage_groups(fingerprints, signatures, settings)
    loose = NearDuplicateSettings(min_correlation=0.65, max_group_fraction=0.20)
    assert max(two_stage_groups(fingerprints, signatures, loose).sizes.values()) == 11
    check_chaining({"a": "a", "b": "a", "c": "c"}, 0.10)  # a pair alone cannot chain
    with pytest.raises(ChainingError):
        check_chaining({"a": "a", "b": "a", "c": "a", "d": "d"}, 0.10)


# ------------------------------------------------------------------ determinism


def test_the_grouping_and_the_measure_are_deterministic_and_symmetric(tmp_path: Path) -> None:
    images = {f"{k}.jpg": _view(_scene(k % 3), dx=k, dy=0) for k in range(6)}
    fingerprints, signatures = _features(images, tmp_path)
    settings = NearDuplicateSettings(min_correlation=0.9, candidate_bits=64, max_group_fraction=1)
    first = two_stage_groups(fingerprints, signatures, settings)
    reordered = two_stage_groups(
        dict(reversed(list(fingerprints.items()))), dict(reversed(list(signatures.items()))),
        settings,
    )  # fmt: skip
    assert first == reordered
    assert first.groups["0.jpg"] == first.groups["3.jpg"] == "0.jpg"  # scene 0, shifted 3 px
    pairs = [(a, b) for a, b, _ in first.candidates]
    forward = crop_similarities(signatures, pairs, CROP)
    backward = crop_similarities(signatures, [(b, a) for a, b in pairs], CROP)
    assert np.allclose(forward, backward, atol=1e-12)
    names = sorted(images)
    matrix = cross_similarity_matrix(
        [signatures[n] for n in names], [signatures[n] for n in names], CROP
    )
    for (a, b), value in zip(pairs, forward, strict=True):
        assert abs(matrix[names.index(a), names.index(b)] - value) < 1e-4  # float32 products
    assert np.allclose(np.diag(matrix), 1.0, atol=1e-4)


# ------------------------------------------------------------------ prepare, calibrate, sheets


ABNORMAL = "Are there any abnormalities in the image? Check all that are present."


def _dataset(root: Path, *, overlay: bool = True, chain: int = 0, lookalike: bool = False) -> Path:
    """12 scenes; scene 0 also as a shifted copy and (``overlay``) an overlaid copy, in the
    official test split; scene 1 also as a brighter copy, in the train split. ``chain`` more
    shifted copies of scene 0 make its group larger. ``lookalike``: two different images with
    the same dHash (one train, one test), which the crop keeps apart."""
    data = root / "data"
    images = {f"s{k:02d}": _view(_scene(100 + k)) for k in range(12)}
    if lookalike:
        images["l1"], images["l2"] = _lookalike(1), _lookalike(2)
    images["s00_shift"] = _view(_scene(100), dx=6, dy=3)
    if overlay:
        images["s00_overlay"] = _overlay(_view(_scene(100)))
    images["s01_bright"] = _view(_scene(101)).point(lambda v: min(255, int(v * 1.1) + 8))
    for k in range(chain):
        images[f"s00_chain{k}"] = _view(_scene(100), dx=-2 * k, dy=k)
    test_only = {"s00", "s00_shift", "s00_overlay", "s05", "s06", "l2"}
    rows: dict[str, list[dict]] = {"train": [], "test": []}
    for k, (name, image) in enumerate(sorted(images.items())):
        _save(image, data / "kvasir_vqa_x1" / "images" / f"{name}.jpg")
        answer = "polyp" if k % 2 else "none"
        rows["test" if name in test_only else "train"].append({
            "img_id": name, "question": f"Is there a polyp in {name}?", "answer": "yes",
            "complexity": 1, "question_class": ["polyp_count"],
            "original": json.dumps([{"q": ABNORMAL, "a": answer}]),
        })  # fmt: skip
    for split, items in rows.items():
        path = data / "kvasir_vqa_x1" / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in items))
    _save(_view(_scene(103)), data / "kvasir_seg" / "images" / "m1.jpg")
    _save(Image.new("L", (W, H), 255), data / "kvasir_seg" / "masks" / "m1.jpg")
    return data


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "prepare_data.py"), *args],
        capture_output=True, text=True, cwd=REPO, check=False, timeout=300,
    )  # fmt: skip


def test_prepare_groups_in_two_stages_and_reports_the_checks(tmp_path: Path) -> None:
    from medcompress.data.prepare import prepare

    data = _dataset(tmp_path, lookalike=True)  # a group of 3 in 17 images: guard loosened here
    config = {"name": "Kvasir-VQA-x1", "label_mapping": None,
              "near_duplicates": {"max_group_fraction": 0.5}}  # fmt: skip
    prepared = prepare(
        data, config, split_seed=3, close_bits=4, mask_threshold=4, confirm_threshold=0.9
    )
    split_of = {s.image_path: name for name, rows in prepared.splits.items() for s in rows}
    image = "kvasir_vqa_x1/images/{}.jpg".format
    assert split_of[image("s00")] == split_of[image("s00_shift")] == split_of[image("s00_overlay")]
    assert split_of[image("s01")] == split_of[image("s01_bright")]
    metrics = {r["metric"]: r["value"] for r in prepared.rows}
    metrics = {r["metric"]: r["value"] for r in prepared.rows}
    assert metrics["groups:total"] == 14 and metrics["groups:largest"] == 3
    assert metrics["groups:size:3"] == 1 and metrics["groups:size:2"] == 1
    assert metrics["groups:size:1"] == 12  # l1 and l2 included: the crop keeps them apart
    assert metrics["near_duplicates:min_correlation"] == 0.9
    assert metrics["near_duplicates:crop_fraction"] == 0.6
    assert metrics["near_duplicates:close_bits"] == 4
    assert metrics["near_duplicates:close_pairs_not_joined"] >= 1  # l1 and l2
    assert metrics["val_test_confirmed_duplicates_of_train"] == 0
    for split in ("val", "test"):
        assert sum(v for k, v in metrics.items() if k.startswith(f"max_score_to_train:{split}:")) \
            == metrics[f"images:{split}"]  # fmt: skip
        assert f"similar_to_train_beyond_candidates:{split}" in metrics
    assert len(prepared.top_cross_split) <= 24
    scores = [t.score for t in prepared.top_cross_split]
    assert scores == sorted(scores, reverse=True) and scores[0] < 0.9  # no duplicate crosses
    assert all(t.first_split != t.second_split for t in prepared.top_cross_split)
    nearest = prepared.closest_dhash_cross_split
    assert 0 < len(nearest) <= 24 and [t.bits for t in nearest] == sorted(t.bits for t in nearest)
    assert all(t.first_split != t.second_split for t in nearest)
    apart = prepared.close_not_joined
    assert all(t.bits <= 4 and t.score < 0.9 and t.first_split != t.second_split for t in apart)
    assert len(apart) == metrics["near_duplicates:close_pairs_not_joined:cross_split"]
    l1, l2 = split_of[image("l1")], split_of[image("l2")]
    pair = {image("l1"), image("l2")}
    assert any({t.first, t.second} == pair for t in apart) == (l1 != l2)
    assert metrics["val_test_close_dhash_to_train"] == int(l1 != l2 and "train" in {l1, l2})
    again = prepare(
        data, config, split_seed=3, close_bits=4, mask_threshold=4, confirm_threshold=0.9
    )
    assert again.split_sha256 == prepared.split_sha256 and again.rows == prepared.rows


def test_the_threshold_is_required_before_a_split() -> None:
    from medcompress.config import ConfigError
    from medcompress.data.prepare import near_duplicate_settings

    with pytest.raises(ConfigError, match="calibrate"):
        near_duplicate_settings({"near_duplicates": {"min_correlation": "TODO"}}, close_bits=4)
    chosen = near_duplicate_settings({"near_duplicates": {"min_correlation": 0.65}}, close_bits=4)
    assert (chosen.min_correlation, chosen.max_group_fraction, chosen.close_bits) == (0.65, 0.1, 4)
    assert near_duplicate_settings({}, close_bits=4, confirm_threshold=0.8).min_correlation == 0.8


def test_the_committed_settings_are_the_decided_ones() -> None:
    from medcompress.config import config_root, load_config
    from medcompress.data.prepare import near_duplicate_settings

    dataset = load_config(config_root() / "data" / "kvasir_vqa_x1.yaml")["dataset"]
    decided = near_duplicate_settings(dataset, close_bits=4)  # D02 amendment 2, 2026-10-19
    assert decided.min_correlation == 0.65 and decided.max_group_fraction == 0.10
    assert decided.candidate_bits == 12 and decided.crop == CROP


def test_the_cli_stops_on_chaining_and_draws_the_final_sheet(tmp_path: Path) -> None:
    data = _dataset(tmp_path, overlay=False, lookalike=True)  # groups of 2 at most
    common = ("prepare", "--data-root", str(data), "--private-dir", str(tmp_path / "private"),
              "--split-seed", "3", "--close-bits", "4", "--mask-threshold", "4",
              "--results-root", str(tmp_path / "results"))  # fmt: skip
    sheets = tmp_path / "sheets"
    waiting = _run(*common)  # the committed rule is now the labelled one (D02 amendment 3)
    assert waiting.returncode == 4 and "labelled pairs" in waiting.stderr
    done = _run(*common, "--sheet-dir", str(sheets), "--confirm-threshold", "0.65")  # run 3
    assert done.returncode == 0, done.stderr
    assert "confirmed duplicate in train: 0" in done.stdout
    assert "pairs within 4 bits not joined: 1" in done.stdout  # l1 and l2
    assert (sheets / "rc_d1_final_cross_split_pairs.png").is_file()
    saved = json.loads((tmp_path / "private" / "near_duplicates.json").read_text())
    assert saved["min_correlation"] == 0.65 and saved["max_group_fraction"] == 0.10
    old_flag = _run(*common[:7], "--dedup-threshold", "4", *common[9:],
                    "--confirm-threshold", "0.65")  # fmt: skip
    assert old_flag.returncode == 0, old_flag.stderr  # the old name of --close-bits still works

    chained = _dataset(tmp_path / "chained")  # 3 of 15 images in one group: over 10 %
    stopped = _run("prepare", "--data-root", str(chained), "--private-dir",
                   str(tmp_path / "chained" / "private"), "--split-seed", "3",
                   "--close-bits", "4", "--mask-threshold", "4", "--confirm-threshold", "0.9",
                   "--results-root", str(tmp_path / "chained" / "results"))  # fmt: skip
    assert stopped.returncode == 7 and "STOP" in stopped.stderr and "3 of 15" in stopped.stderr
    assert not (tmp_path / "chained" / "private" / "splits.json").exists()


def test_the_calibration_prints_the_distribution_and_draws_the_sheet(tmp_path: Path) -> None:
    from medcompress.data.contact_sheet import render_pairs
    from medcompress.evaluation.aggregate import validate_results_tree

    data = _dataset(tmp_path, overlay=False)
    results, sheets = tmp_path / "results", tmp_path / "sheets"
    done = _run("calibrate", "--data-root", str(data), "--split-seed", "20261012",
                "--close-bits", "4", "--sheet-dir", str(sheets),
                "--results-root", str(results))  # fmt: skip
    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert "candidate pairs:" in out and "random controls:" in out
    assert "bits_0_4" in out and "bits_5_8" in out and "bits_9_12" in out
    assert "provisional proposal:" in out and "what each threshold gives" in out
    assert (sheets / "rc_d1_calibration_pairs.png").is_file()
    assert validate_results_tree(results) == []
    rows = (results / "data" / "near_duplicate_calibration.csv").read_text()
    assert "calibration:threshold:0.90:largest_group" in rows
    assert "calibration:score:random_controls:" in rows
    assert ".jpg" not in rows  # aggregate counts only, no image name

    given = _run("calibrate", "--data-root", str(data), "--split-seed", "20261012",
                 "--close-bits", "4", "--sheet-dir", str(sheets),
                 "--results-root", str(results), "--threshold", "0.72")  # fmt: skip
    assert given.returncode == 0 and "drawn at the given threshold 0.72" in given.stdout
    assert "      0.72" in given.stdout  # the given value is added to the table
    with pytest.raises(ValueError, match="outside the repo"):
        render_pairs((("t", ()),), REPO / "sheet.png", crop_fraction=0.6)


def test_the_calibration_and_the_pure_function_agree(tmp_path: Path) -> None:
    from medcompress.data.prepare import CALIBRATION_GRID, calibrate, score_bin, score_histogram

    data = _dataset(tmp_path, overlay=False, chain=1)  # scene 0: a copy and a shifted copy
    config = {"name": "Kvasir-VQA-x1", "label_mapping": None,
              "near_duplicates": {"max_group_fraction": 0.5}}  # fmt: skip
    first = calibrate(data, config, close_bits=4, seed=5)
    assert first == calibrate(data, config, close_bits=4, seed=5)  # deterministic
    metrics = {r["metric"]: r["value"] for r in first.rows}
    assert metrics["calibration:images"] == 15
    proposal = metrics["calibration:provisional_proposal"]
    assert proposal in CALIBRATION_GRID and first.threshold == proposal
    assert metrics[f"calibration:threshold:{proposal:.2f}:random_controls_at_or_above"] == 0
    candidates = sum(v for k, v in metrics.items() if k.startswith("calibration:candidate_pairs"))
    assert metrics["calibration:random_controls"] == 15 * 14 // 2 - candidates  # all the others
    assert metrics[f"calibration:threshold:{proposal:.2f}:largest_group"] == 3  # scene 0
    titles = [title for title, _ in first.bands]
    assert len(titles) == 4 and "NOT joined" in titles[0] and "JOINED" in titles[1]
    assert (
        score_bin(0.95) == "0.95" and score_bin(1.0) == "0.95" and score_bin(-0.2) == "below_0.00"
    )
    assert score_bin(0.35) == "0.35" and sum(score_histogram([0.1, 0.5, 0.99]).values()) == 3


def test_the_runner_plans_the_calibration_and_the_final_sheet(tmp_path: Path) -> None:
    from medcompress.cards import build_plan, card_file

    shared = Path("/kaggle/input/medcompress-data")
    attached = build_plan("RC-D1-calibrate", data_root=shared)  # run 2's shared data attached
    assert [s.label for s in attached.steps] == ["calibrate"] and not attached.internet
    argv = attached.steps[0].argv
    assert argv[:2] == ("scripts/prepare_data.py", "calibrate") and str(shared) in argv
    assert "--sheet-dir" in argv and "20261012" in argv and not attached.steps[0].inputs
    exported = build_plan("RC-D1-calibrate")  # nothing attached: export the images first
    assert [s.label for s in exported.steps] == ["export", "calibrate"] and exported.internet
    prepare = next(s for s in build_plan("RC-D1").steps if s.label == "prepare")
    assert "--sheet-dir" in prepare.argv  # the final contact sheet
    assert "--confirm-threshold" not in prepare.argv  # the threshold comes from the config
    assert card_file("RC-D1-calibrate", REPO / "docs" / "runs").name.startswith("RC-D1_")
    dry = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_card.py"), "--card", "RC-D1-calibrate",
         "--data-root", str(shared), "--out", str(tmp_path), "--dry-run",
         "--allow-blocked"],  # the calibration is done; the RC-D1 card now waits for run 4
        capture_output=True, text=True, cwd=REPO, check=False, timeout=120,
    )  # fmt: skip
    assert dry.returncode == 0, dry.stderr
    assert "calibrate --data-root /kaggle/input/medcompress-data" in dry.stdout
