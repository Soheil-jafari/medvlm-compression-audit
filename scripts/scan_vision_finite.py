#!/usr/bin/env python
"""The fp16 vision tower on every image: which images give non-finite values (2026-11-13).

Runs the vision tower alone (``models/vision_only.py``: the student's, or the teacher's, which
fits one T4 without its language model) in fp16 on every image of the validation split, the
internal test (filtered and unfiltered) and the training split (the training pool and every
calibration image are in it), each image resized and processed exactly as the models see it.
Prints the counts per split and the sample ids of the non-finite images (with the first block
that is not finite), and writes ``<out>/vision_scan_<model>.csv`` (split, sample id, block).
Nothing is written to the repository.

Example (Kaggle, one T4, the shared data attached):
    python scripts/scan_vision_finite.py --model student
    python scripts/scan_vision_finite.py --model teacher
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SPLITS: tuple[str, ...] = ("val", "test", "test_unfiltered", "train")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("student", "teacher"), required=True)
    parser.add_argument("--data-root", type=Path, default=None, help="found by its manifest")
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    args = parser.parse_args(argv)

    from medcompress.cli import attach_inputs
    from medcompress.config import config_root, load_config, load_experiment
    from medcompress.data.prepared import load_prepared
    from medcompress.inputs import find_shared_data
    from medcompress.models.hf import _to_pil
    from medcompress.models.loading import ModelSpec
    from medcompress.models.vision_only import load_vision_tower, tower_is_finite
    from medcompress.utils.io import write_csv

    config, _experiment = load_experiment("E2")
    attach_inputs(config, args.data_root or find_shared_data(), ())
    paths = config["paths"]
    side = int(config["inputs"]["image_longest_side"])
    model_config = (
        config["components"]["teacher"] if args.model == "teacher" else config["model"]["config"]
    )
    spec = ModelSpec.from_config(load_config(config_root() / model_config))
    tower, processor = load_vision_tower(spec)
    dataset = load_config(config["components"]["datasets"]["train_qa"])["dataset"]
    data_root, private = Path(paths["data_root"]), Path(paths["private_root"])
    filtered = load_prepared(data_root, dataset, private)
    unfiltered = load_prepared(data_root, dataset, private, unfiltered_test=True)
    splits = {**filtered, "test_unfiltered": unfiltered["test_unfiltered"]}
    results: dict[str, tuple[bool, str | None]] = {}  # by image: scanned once
    rows, started = [], time.perf_counter()
    for split in args.splits:
        images: dict[str, str] = {}
        for sample in splits[split]:
            images.setdefault(sample.image_path, sample.sample_id)
        bad = []
        for image, sample_id in images.items():
            if image not in results:
                pil = _to_pil(data_root / image, side)
                pixels = processor.image_processor(images=[pil], return_tensors="pt")
                results[image] = tower_is_finite(tower, pixels["pixel_values"],
                                                 pixels["image_grid_thw"])  # fmt: skip
            finite, block = results[image]
            if not finite:
                bad.append(sample_id)
                rows.append((split, sample_id, block or ""))
        print(f"{args.model} vision tower, {split}: {len(bad)} of {len(images)} images give a "
              f"non-finite value in fp16: {bad}", flush=True)  # fmt: skip
    out = Path(args.out) / f"vision_scan_{args.model}.csv"
    write_csv(out, ("split", "sample_id", "first_non_finite"), rows)
    print(f"{len(results)} images scanned in {(time.perf_counter() - started) / 60:.1f} min; "
          f"wrote {out}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
