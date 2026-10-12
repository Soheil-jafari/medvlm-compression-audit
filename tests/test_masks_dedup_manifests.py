"""Masks, near-duplicate fingerprints and manifests on synthetic images in a temp dir."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from medcompress.data.base import VQASample
from medcompress.data.dedup import (
    deduplicate,
    find_near_duplicates,
    hamming_distance,
    image_fingerprint,
    near_duplicate_pairs,
)
from medcompress.data.manifests import (
    build_manifest,
    load_manifest,
    verify_manifest,
    write_manifest,
)
from medcompress.data.masks import (
    area_matched_random_region,
    assign_size_stratum,
    lesion_area_fraction,
    load_mask,
    overlap_fraction,
    region_of,
    size_strata_thresholds,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _square_mask(
    shape: tuple[int, int] = (100, 120), box: tuple[int, int, int, int] = (10, 20, 30, 45)
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    y0, x0, y1, x1 = box
    mask[y0:y1, x0:x1] = True
    return mask


def test_area_and_region() -> None:
    mask = _square_mask()
    assert lesion_area_fraction(mask) == pytest.approx(20 * 25 / (100 * 120))
    region = region_of(mask)
    assert (region.x0, region.y0, region.x1, region.y1, region.n_pixels) == (20, 10, 45, 30, 500)
    with pytest.raises(ValueError):
        region_of(np.zeros((3, 3), dtype=bool))


def test_size_strata() -> None:
    thresholds = size_strata_thresholds([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    assert len(thresholds) == 3 and thresholds == sorted(thresholds)
    assert assign_size_stratum(0.005, thresholds) == 0
    assert assign_size_stratum(0.5, thresholds) == 3
    assert assign_size_stratum(thresholds[0], thresholds) == 1
    with pytest.raises(ValueError):
        size_strata_thresholds([0.1, 0.2])


def test_area_matched_region_is_disjoint_and_same_area() -> None:
    mask = _square_mask()
    region = area_matched_random_region(mask, seed=3)
    assert region.sum() == mask.sum()
    assert overlap_fraction(region, mask) == 0.0
    assert np.array_equal(region, area_matched_random_region(mask, seed=3))


def test_area_matched_region_falls_back_to_least_overlap() -> None:
    big = _square_mask((10, 10), (0, 0, 9, 9))
    region = area_matched_random_region(big, seed=0, max_tries=20)
    assert region.sum() == big.sum()
    assert 0 < overlap_fraction(region, big) < 1


def test_load_mask(tmp_path: Path) -> None:
    mask = _square_mask()
    path = tmp_path / "m.png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(path)
    assert np.array_equal(load_mask(str(path)), mask)


def _noise_image(path: Path, seed: int, size: tuple[int, int] = (96, 128)) -> Path:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(12, 16, 3), dtype=np.uint8)
    Image.fromarray(base).resize((size[1], size[0]), Image.Resampling.BILINEAR).save(path)
    return path


def test_fingerprint_detects_resized_recompressed_copy(tmp_path: Path) -> None:
    original = _noise_image(tmp_path / "a.png", seed=1)
    with Image.open(original) as im:
        im.resize((200, 150)).save(tmp_path / "a_copy.jpg", quality=70)
    other = _noise_image(tmp_path / "b.png", seed=2)
    fa = image_fingerprint(str(original))
    assert len(fa) == 16
    assert hamming_distance(fa, image_fingerprint(str(tmp_path / "a_copy.jpg"))) <= 6
    assert hamming_distance(fa, image_fingerprint(str(other))) > 10


def test_near_duplicate_pairs_and_dedup(tmp_path: Path) -> None:
    for name, seed in [("t1.png", 1), ("t2.png", 2), ("x1.png", 1), ("x3.png", 3)]:
        _noise_image(tmp_path / name, seed=seed)
    train = [VQASample(f"tr{i}", "syn", f"t{i}.png", "q", "a") for i in (1, 2)]
    external = [
        VQASample("ex1", "syn", "x1.png", "q", "a"),
        VQASample("ex3", "syn", "x3.png", "q", "a"),
    ]
    assert find_near_duplicates(train, external, threshold=4, root=tmp_path) == [("ex1", "tr1")]
    kept = deduplicate(train, external, threshold=4, root=tmp_path)
    assert [s.sample_id for s in kept] == ["ex3"]
    pairs = near_duplicate_pairs({"r": "ffff000000000000"}, {"c": "fffe000000000000"}, threshold=1)
    assert pairs == [("c", "r", 1)]


def test_manifest_round_trip_and_verification(tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "sub").mkdir(parents=True)
    (data / "a.txt").write_text("alpha")
    (data / "sub" / "b.txt").write_text("beta")
    manifest = build_manifest("syn", "v1", data)
    assert [e.relative_path for e in manifest.entries] == ["a.txt", "sub/b.txt"]
    out = tmp_path / "manifests" / "syn.jsonl"
    write_manifest(manifest, out)
    assert load_manifest(out) == manifest
    assert verify_manifest(manifest, data) == []
    (data / "a.txt").write_text("changed")
    (data / "sub" / "b.txt").unlink()
    assert verify_manifest(manifest, data) == ["a.txt", "sub/b.txt"]


def test_manifest_refuses_repository_paths(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x")
    manifest = build_manifest("syn", "v1", tmp_path)
    with pytest.raises(ValueError, match="outside the repository"):
        write_manifest(manifest, REPO_ROOT / "results" / "should_not_exist.jsonl")
    assert not (REPO_ROOT / "results" / "should_not_exist.jsonl").exists()
