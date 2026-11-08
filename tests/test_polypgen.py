"""PolypGen (RC-X2): the loader's layout rules, the counts and the overlap check (synthetic)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from medcompress.data import polypgen as pg
from medcompress.data.dedup import CropSettings, image_features
from medcompress.data.embeddings import EmbeddingSettings, embed_images, make_embedder


def _image(path: Path, seed: int, size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(path)


def _mask(path: Path, pixels: int, size: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask.flat[:pixels] = 255
    Image.fromarray(mask).save(path)


def _tree(root: Path) -> None:
    _image(root / "data_C1" / "images_C1" / "a.jpg", 1)
    _mask(root / "data_C1" / "masks_C1" / "a_mask.jpg", 64)  # 64 / 4096 = 1.6 % polyp
    _image(root / "data_C4" / "images_C4" / "b.jpg", 2)
    _mask(root / "data_C4" / "masks_C4" / "b.jpg", 0)  # an empty mask: a negative
    for k in range(3):  # one video sequence: one bootstrap cluster
        _image(root / "sequenceData" / "positive" / "seq7" / "images_seq7_C2" / f"f{k}.jpg", 10 + k)
        _mask(root / "sequenceData" / "positive" / "seq7" / "masks_seq7_C2" / f"f{k}.jpg", 100)
    _image(root / "sequenceData" / "negativeOnly" / "seq3_neg" / "n0.jpg", 20)  # no mask file
    _image(root / "data_C5" / "images_C5" / "orphan.jpg", 30)  # neither: unlabelled


def test_the_loader_finds_centres_sequences_masks_and_negatives(tmp_path: Path) -> None:
    _tree(tmp_path)
    items = {Path(i.image).name: i for i in pg.scan(tmp_path)}
    assert set(items) == {"a.jpg", "b.jpg", "f0.jpg", "f1.jpg", "f2.jpg", "n0.jpg", "orphan.jpg"}
    assert items["a.jpg"].centre == "C1" and items["a.jpg"].mask.endswith("a_mask.jpg")
    assert items["f0.jpg"].centre == "C2" and items["f0.jpg"].sequence == "seq7"
    assert items["f0.jpg"].cluster == items["f2.jpg"].cluster == "C2:seq7"  # frames together
    assert items["a.jpg"].cluster != items["b.jpg"].cluster  # single frames: their own clusters
    assert items["n0.jpg"].mask is None and items["n0.jpg"].labelled  # negative-only folder
    assert not items["orphan.jpg"].labelled  # no mask, not a negative folder: left out
    assert pg.polyp_area(tmp_path, items["a.jpg"]) == 64 / 4096
    assert pg.polyp_area(tmp_path, items["b.jpg"]) == 0.0


def test_the_counts_are_per_centre_and_aggregate_only(tmp_path: Path) -> None:
    _tree(tmp_path)
    items = pg.scan(tmp_path)
    areas = {i.image: pg.polyp_area(tmp_path, i) for i in items if i.labelled}
    rows = {r["metric"]: r["value"] for r in pg.describe_rows(items, areas)}
    assert rows["polypgen:images"] == 7 and rows["polypgen:clusters"] == 5
    assert rows["polypgen:positives:C2"] == 3 and rows["polypgen:sequences:C2"] == 1
    assert rows["polypgen:negatives:C4"] == 1 and rows["polypgen:unlabelled:C5"] == 1
    assert all(r["experiment"] == "DATA" for r in pg.describe_rows(items))  # no image names


def test_the_overlap_check_removes_a_copy_of_a_training_image(tmp_path: Path) -> None:
    ours = tmp_path / "ours"
    for k in range(4):
        _image(ours / f"t{k}.jpg", 100 + k)
    _tree(tmp_path / "pg")
    copy = tmp_path / "pg" / "data_C4" / "images_C4" / "b.jpg"
    copy.write_bytes((ours / "t2.jpg").read_bytes())  # a C4 image that is one of ours
    items = [i for i in pg.scan(tmp_path / "pg") if i.labelled]
    crop = CropSettings()
    keys = {f"kvasir/t{k}.jpg": ours / f"t{k}.jpg" for k in range(4)}
    keys |= {f"polypgen/{i.image}": tmp_path / "pg" / i.image for i in items}
    fingerprints, signatures = {}, {}
    for key, path in keys.items():
        fingerprints[key], signatures[key] = image_features(str(path), crop)
    settings = EmbeddingSettings("tiny", None, 32, 0.8)
    full = embed_images([str(p) for p in keys.values()], make_embedder("tiny"), settings)["full"]
    external = [k for k in keys if k.startswith("polypgen/")]
    train = [k for k in keys if k.startswith("kvasir/")]
    flags = pg.heldout_flags(external, train, fingerprints, signatures, {"full": full},
                             list(keys), crop)  # fmt: skip
    by_image = {k.split("/", 1)[1]: v for k, v in flags.items()}
    assert any(by_image["data_C4/images_C4/b.jpg"].values())  # the copy is removed
    rows = {r["metric"]: r["value"] for r in pg.overlap_rows(items, by_image)}
    assert rows["polypgen:heldout_removed:C4"] >= 1 and rows["polypgen:heldout_f"] == 0.86
