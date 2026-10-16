#!/usr/bin/env python
"""Prepare Kvasir-VQA-x1 and Kvasir-SEG on your own machine (run card RC-D1).

Commands:

  export   download Kvasir-VQA images and Kvasir-VQA-x1 rows into
           <data_root>/kvasir_vqa_x1/ (needs the "datasets" package; GPU not needed)
  prepare  splits, deduplication, mask linking and aggregate counts; private files go
           to --private-dir (outside the repository), counts to
           <results_root>/data/summary.csv
  bundle   the last RC-D1 step: write the manifest (SHA-256 of every file, the commit) of the
           shared-data folder <bundle-dir> = data/ + private/, ready to upload as ONE private
           Kaggle dataset
  verify   check an attached shared-data dataset against its manifest (every later card)
  counts   the second RC-D1 pass, on the attached shared data: the aggregate counts again,
           after the label values are filled in configs/data/kvasir_vqa_x1.yaml (positives
           and prevalences need them); the splits must hash as in the shared data
  overlap  how many images of an external set (any folder of images) are near-duplicates
           of our training, validation and test images; counts only, to
           <results_root>/data/<name>_overlap.csv (run card RC-X1)

Example (RC-D1):
    python scripts/prepare_data.py export --data-root /kaggle/working/medcompress_data/data
    python scripts/prepare_data.py prepare --data-root /kaggle/working/medcompress_data/data \\
        --private-dir /kaggle/working/medcompress_data/private --split-seed 20261012 \\
        --dedup-threshold 4 --mask-threshold 4
    python scripts/prepare_data.py bundle --bundle-dir /kaggle/working/medcompress_data
    python scripts/prepare_data.py verify --data-root /kaggle/working/medcompress_data
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


LICENCES = {
    "kvasir_vqa_x1": "Kvasir-VQA and Kvasir-VQA-x1: CC BY-NC 4.0 (non-commercial, attribution)",
    "kvasir_seg": "Kvasir-SEG: research and educational use only",
}
SHARING = "private Kaggle dataset, shared with the team's collaborators only; never public"


def bundle(args: argparse.Namespace, dataset: dict) -> int:
    """Write the manifest of the shared-data folder (the last RC-D1 step)."""
    from medcompress.data.manifests import load_manifest
    from medcompress.data.prepared import load_prepared
    from medcompress.data.splits import split_fingerprint
    from medcompress.utils.bundles import BundleError, check_data_layout, write_bundle_manifest

    root = args.bundle_dir
    lacking = check_data_layout(root)
    if lacking:
        print(f"{root} is not complete; missing: {', '.join(lacking)}", file=sys.stderr)
        return 2
    splits = load_prepared(root / "data", dataset, root / "private")
    meta = {
        "dataset": "kvasir_vqa_x1 + kvasir_seg",
        "kvasir_vqa_x1_revision": dataset.get("revision"),
        "split_sha256": split_fingerprint(splits),
        "samples": {name: len(rows) for name, rows in splits.items()},
        "licences": LICENCES,
        "sharing": SHARING,
    }
    for name in ("kvasir_vqa_x1", "kvasir_seg"):
        meta[f"manifest_{name}_sha256"] = load_manifest(
            root / "private" / f"manifest_{name}.jsonl"
        ).manifest_sha256
    try:
        made = write_bundle_manifest(root, kind="data", meta=meta)
    except BundleError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(
        f"wrote {root / 'bundle_manifest.json'}: {len(made.files)} files, "
        f"commit {made.meta['commit']}"
    )
    print(f"split_sha256 {meta['split_sha256']}; samples per split {meta['samples']}")
    print("upload this folder as ONE private Kaggle dataset (run card RC-D1, section 4)")
    return 0


def counts(args: argparse.Namespace, dataset: dict) -> int:
    """The second RC-D1 pass: aggregate counts from the verified shared data, nothing private."""
    from medcompress.data.prepare import prepare
    from medcompress.utils.bundles import BundleError, attach_data

    try:
        shared = attach_data({}, args.data_root)
    except BundleError as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return 5
    prepared = prepare(
        shared.root / "data",
        dataset,
        split_seed=args.split_seed,
        dedup_threshold=args.dedup_threshold,
        mask_threshold=args.mask_threshold,
    )
    if prepared.split_sha256 != shared.meta.get("split_sha256"):
        print(
            f"check failed: these settings give split_sha256 {prepared.split_sha256}, the shared "
            f"data have {shared.meta.get('split_sha256')}; use the settings of the first pass",
            file=sys.stderr,
        )
        return 5
    summary = args.results_root / "data" / "summary.csv"
    if summary.exists():
        summary.unlink()
    append_results(prepared.rows, summary)
    labelled = sum(1 for r in prepared.rows if str(r["metric"]).startswith("positives:"))
    print(f"wrote {len(prepared.rows)} aggregate rows ({labelled} positives rows) to {summary}")
    return 0


def verify(args: argparse.Namespace) -> int:
    """Check an attached shared-data dataset against its manifest."""
    from medcompress.utils.bundles import BundleError, check_data_layout, verify_bundle

    try:
        made = verify_bundle(args.data_root, kind="data")
    except BundleError as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return 5
    lacking = check_data_layout(made.root)
    if lacking:
        print(f"check failed: {made.root} lacks {', '.join(lacking)}", file=sys.stderr)
        return 5
    print(f"OK: {len(made.files)} files match the manifest at {made.root}")
    print(f"commit {made.meta.get('commit')}; split_sha256 {made.meta.get('split_sha256')}")
    return 0


def overlap(args: argparse.Namespace, dataset: dict) -> int:
    """Count external images that are near-duplicates of each of our splits."""
    from medcompress.data.dedup import folder_fingerprints, image_fingerprint, overlap_rows
    from medcompress.data.prepared import load_prepared

    if args.private_dir is None:  # the shared-data dataset: verify it, then use its folders
        from medcompress.utils.bundles import BundleError, attach_data

        try:
            paths = attach_data({}, args.data_root)
        except BundleError as exc:
            print(f"check failed: {exc}", file=sys.stderr)
            return 5
        args.data_root, args.private_dir = paths.root / "data", paths.root / "private"
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
    bund = sub.add_parser("bundle", help="manifest of the shared-data folder (data/ + private/)")
    bund.add_argument("--bundle-dir", type=Path, required=True, help="outside the repository")
    ver = sub.add_parser("verify", help="check an attached shared-data dataset")
    ver.add_argument("--data-root", type=Path, required=True)
    cnt = sub.add_parser("counts", help="second pass: aggregate counts from the shared data")
    cnt.add_argument("--data-root", type=Path, required=True, help="the shared-data dataset")
    cnt.add_argument("--results-root", type=Path, default=Path("results"))
    cnt.add_argument("--split-seed", type=int, required=True, help="as in the first pass")
    cnt.add_argument("--dedup-threshold", type=int, required=True, help="as in the first pass")
    cnt.add_argument("--mask-threshold", type=int, required=True, help="as in the first pass")
    over = sub.add_parser("overlap", help="near-duplicates of an external image folder")
    over.add_argument("--data-root", type=Path, required=True, help="the shared-data dataset")
    over.add_argument(
        "--private-dir", type=Path, default=None, help="only if --data-root is a bare data folder"
    )
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
    if args.command == "bundle":
        return bundle(args, dataset)
    if args.command == "verify":
        return verify(args)
    if args.command == "counts":
        return counts(args, dataset)

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
