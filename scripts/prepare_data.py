#!/usr/bin/env python
"""Prepare Kvasir-VQA-x1 and Kvasir-SEG on your own machine (run card RC-D1).

Three commands:

  export   download Kvasir-VQA images and Kvasir-VQA-x1 rows into
           <data_root>/kvasir_vqa_x1/ (needs the "datasets" package; GPU not needed)
  prepare  splits, deduplication, mask linking and aggregate counts; private files go
           to --private-dir (outside the repository), counts to
           <results_root>/data/summary.csv
  overlap  how many images of an external set (any folder of images) are near-duplicates
           of our training, validation and test images; counts only, to
           <results_root>/data/<name>_overlap.csv (run card RC-X1)

Example:
    python scripts/prepare_data.py export --data-root /kaggle/working/data
    python scripts/prepare_data.py prepare --data-root /kaggle/working/data \\
        --private-dir /kaggle/working/private --split-seed 20261012 \\
        --dedup-threshold 4 --mask-threshold 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.config import config_root, load_config  # noqa: E402
from medcompress.utils.io import append_results  # noqa: E402

X1_REPO = "SimulaMet/Kvasir-VQA-x1"
HOST_REPO = "SimulaMet-HOST/Kvasir-VQA"


def export(data_root: Path, revision: str | None) -> None:
    """Download images (one per img_id) and Kvasir-VQA-x1 rows, as the dataset card describes."""
    from datasets import load_dataset

    out = data_root / "kvasir_vqa_x1"
    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for row in load_dataset(HOST_REPO, split="raw"):
        if row["img_id"] not in seen:
            seen.add(row["img_id"])
            row["image"].convert("RGB").save(images / f"{row['img_id']}.jpg")
    for split in ("train", "test"):
        rows = load_dataset(X1_REPO, split=split, revision=revision)
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for r in rows:
                record = {
                    k: r[k]
                    for k in (
                        "img_id",
                        "question",
                        "answer",
                        "complexity",
                        "question_class",
                        "original",
                    )
                }
                handle.write(json.dumps(record) + "\n")
    print(f"exported {len(seen)} images and both splits to {out}")


def overlap(args: argparse.Namespace, dataset: dict) -> int:
    """Count external images that are near-duplicates of each of our splits."""
    from medcompress.data.dedup import folder_fingerprints, image_fingerprint, overlap_rows
    from medcompress.data.prepared import load_prepared

    splits = load_prepared(args.data_root, dataset, args.private_dir)
    ours = {  # one fingerprint per image (an image has many questions)
        split: {p: image_fingerprint(str(args.data_root / p)) for p in {s.image_path for s in rows}}
        for split, rows in splits.items()
    }
    rows = overlap_rows(
        args.name, folder_fingerprints(args.images), ours, threshold=args.dedup_threshold
    )
    target = args.results_root / "data" / f"{args.name}_overlap.csv"
    target.unlink(missing_ok=True)
    append_results(rows, target)
    for row in rows:
        print(f"{row['metric']}: {row['value']}")
    print(f"wrote {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="download into <data-root>/kvasir_vqa_x1")
    exp.add_argument("--data-root", type=Path, required=True)
    prep = sub.add_parser("prepare", help="splits, dedup, mask links, aggregate counts")
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--private-dir", type=Path, required=True, help="outside the repository")
    prep.add_argument("--results-root", type=Path, default=Path("results"))
    prep.add_argument("--split-seed", type=int, required=True, help="pre-registered seed")
    prep.add_argument("--dedup-threshold", type=int, required=True, help="max Hamming distance")
    prep.add_argument("--mask-threshold", type=int, required=True, help="max Hamming distance")
    over = sub.add_parser("overlap", help="near-duplicates of an external image folder")
    over.add_argument("--data-root", type=Path, required=True)
    over.add_argument("--private-dir", type=Path, required=True, help="the RC-D1 private files")
    over.add_argument("--images", type=Path, required=True, help="folder of external images")
    over.add_argument("--name", required=True, help="e.g. endobench")
    over.add_argument("--results-root", type=Path, default=Path("results"))
    over.add_argument("--dedup-threshold", type=int, required=True, help="max Hamming distance")
    args = parser.parse_args(argv)

    dataset = load_config(config_root() / "data" / "kvasir_vqa_x1.yaml")["dataset"]
    if args.command == "export":
        export(args.data_root, dataset.get("revision"))
        return 0
    if args.command == "overlap":
        return overlap(args, dataset)

    from medcompress.data.prepare import prepare, write_private_outputs

    prepared = prepare(
        args.data_root,
        dataset,
        split_seed=args.split_seed,
        dedup_threshold=args.dedup_threshold,
        mask_threshold=args.mask_threshold,
    )
    hashes = write_private_outputs(prepared, args.private_dir, args.data_root)
    summary = args.results_root / "data" / "summary.csv"
    if summary.exists():
        summary.unlink()
    append_results(prepared.rows, summary)
    print(f"wrote {len(prepared.rows)} aggregate rows to {summary}")
    print("hashes for the run card:", json.dumps(hashes, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
