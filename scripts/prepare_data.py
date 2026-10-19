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
  calibrate  the step before prepare (D02 amendment 2): the crop correlation of every
           candidate pair, what each threshold would give, and a contact sheet (a PNG in
           --sheet-dir, outside the repository); counts to
           <results_root>/data/near_duplicate_calibration.csv. --data-root is a data folder
           (kvasir_vqa_x1/ inside) or a shared-data dataset (verified first)

Exit codes: 4 a setting is open (the confirmation threshold before calibration), 5 an input
check failed, 7 the chaining guard stopped the run (one group holds too many images).

Example (RC-D1):
    python scripts/prepare_data.py export --data-root /kaggle/working/medcompress_data/data
    python scripts/prepare_data.py calibrate --data-root /kaggle/working/medcompress_data/data \\
        --split-seed 20261012 --close-bits 4 --sheet-dir /kaggle/working/rc_d1_sheets
    python scripts/prepare_data.py prepare --data-root /kaggle/working/medcompress_data/data \\
        --private-dir /kaggle/working/medcompress_data/private --split-seed 20261012 \\
        --close-bits 4 --mask-threshold 4 --sheet-dir /kaggle/working/rc_d1_sheets
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
SETTING_OPEN = 4
INPUT_CHECK_FAILED = 5
CHAINING_STOP = 7
CALIBRATION_SHEET = "rc_d1_calibration_pairs.png"
CLOSE_HELP = (
    "dHash bits of the band of very close pairs that is reported (those the crop leaves apart); "
    "no pair is joined on the dHash alone (D02 amendment 2, 2026-10-19)"
)
FINAL_SHEET = "rc_d1_final_cross_split_pairs.png"


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
    settings = root / "private" / "near_duplicates.json"
    if settings.is_file():  # D02 amendment 2: how the groups were made
        meta["near_duplicates"] = json.loads(settings.read_text(encoding="utf-8"))
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
        return INPUT_CHECK_FAILED
    prepared = prepare(
        shared.root / "data",
        dataset,
        split_seed=args.split_seed,
        close_bits=args.close_bits,
        mask_threshold=args.mask_threshold,
        confirm_threshold=args.confirm_threshold,
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


def _data_folder(data_root: Path) -> Path:
    """A bare data folder (``kvasir_vqa_x1/`` inside), or the ``data/`` of a verified
    shared-data dataset (raises ``BundleError`` if a file differs)."""
    if (data_root / "kvasir_vqa_x1").is_dir():
        return data_root
    from medcompress.utils.bundles import attach_data

    return attach_data({}, data_root).root / "data"


def calibrate(args: argparse.Namespace, dataset: dict) -> int:
    """The calibration step (D02 amendment 2): print the distribution, draw the sheet."""
    from medcompress.data.contact_sheet import render_pairs
    from medcompress.data.prepare import calibrate as run_calibration
    from medcompress.utils.bundles import BundleError

    try:
        data_root = _data_folder(args.data_root)
    except BundleError as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return INPUT_CHECK_FAILED
    result = run_calibration(
        data_root, dataset, close_bits=args.close_bits, seed=args.split_seed,
        threshold=args.threshold,
    )  # fmt: skip
    print("\n".join(result.lines))
    target = args.results_root / "data" / "near_duplicate_calibration.csv"
    target.unlink(missing_ok=True)
    append_results(result.rows, target)
    print(f"wrote {len(result.rows)} aggregate rows to {target}")
    if result.bands:
        sheet = render_pairs(
            result.bands, args.sheet_dir / CALIBRATION_SHEET, crop_fraction=result.crop_fraction
        )
        print(f"contact sheet (private, never commit it): {sheet}")
    return 0


def _final_sheet(prepared, sheet_dir: Path, data_root: Path) -> Path:  # type: ignore[no-untyped-def]
    """Three bands, all pairs across splits: the most similar by the crop, the closest by the
    dHash, and the closest pairs within ``close_bits`` that the crop left apart."""
    from medcompress.data.contact_sheet import render_pairs

    def cells(pairs):  # type: ignore[no-untyped-def]
        return tuple(
            (str(data_root / t.first), str(data_root / t.second),
             f"r={t.score:.3f}  {t.bits} bits  {t.first_split} / {t.second_split}")
            for t in pairs
        )  # fmt: skip

    metrics = {r["metric"]: r["value"] for r in prepared.rows}
    close = int(prepared.near_duplicates.get("close_bits", 4))
    apart = int(metrics.get("near_duplicates:close_pairs_not_joined", 0))
    across = int(metrics.get("near_duplicates:close_pairs_not_joined:cross_split", 0))
    bands = (
        (f"1. The {len(prepared.top_cross_split)} most similar pairs across splits by the crop "
         "correlation: all should be different scenes", cells(prepared.top_cross_split)),
        (f"2. The {len(prepared.closest_dhash_cross_split)} closest pairs across splits by the "
         "dHash: all should be different scenes", cells(prepared.closest_dhash_cross_split)),
        (f"3. Within {close} bits but not joined: {apart} pairs, {across} of them across splits; "
         f"the {len(prepared.close_not_joined)} across splits with the highest crop correlation",
         cells(prepared.close_not_joined)),
    )  # fmt: skip
    crop = float(prepared.near_duplicates.get("crop_fraction", 0.60))
    return render_pairs(bands, sheet_dir / FINAL_SHEET, crop_fraction=crop, columns=3)


def verify(args: argparse.Namespace) -> int:
    """Check an attached shared-data dataset against its manifest."""
    from medcompress.utils.bundles import BundleError, check_data_layout, verify_bundle

    try:
        made = verify_bundle(args.data_root, kind="data")
    except BundleError as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return INPUT_CHECK_FAILED
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
    prep.add_argument("--close-bits", "--dedup-threshold", dest="close_bits", type=int,
                      required=True, help=CLOSE_HELP)  # fmt: skip
    prep.add_argument("--mask-threshold", type=int, required=True, help="max Hamming distance")
    prep.add_argument(
        "--confirm-threshold", type=float, default=None,
        help="crop correlation that joins a candidate pair (default: the dataset config)",
    )  # fmt: skip
    prep.add_argument(
        "--sheet-dir", type=Path, default=None, help="final contact sheet (outside the repo)"
    )
    bund = sub.add_parser("bundle", help="manifest of the shared-data folder (data/ + private/)")
    bund.add_argument("--bundle-dir", type=Path, required=True, help="outside the repository")
    ver = sub.add_parser("verify", help="check an attached shared-data dataset")
    ver.add_argument("--data-root", type=Path, required=True)
    cnt = sub.add_parser("counts", help="second pass: aggregate counts from the shared data")
    cnt.add_argument("--data-root", type=Path, required=True, help="the shared-data dataset")
    cnt.add_argument("--results-root", type=Path, default=Path("results"))
    cnt.add_argument("--split-seed", type=int, required=True, help="as in the first pass")
    cnt.add_argument("--close-bits", "--dedup-threshold", dest="close_bits", type=int,
                     required=True, help="as in the first pass")  # fmt: skip
    cnt.add_argument("--mask-threshold", type=int, required=True, help="as in the first pass")
    cnt.add_argument("--confirm-threshold", type=float, default=None, help="as in the first pass")
    cal = sub.add_parser("calibrate", help="choose the crop threshold (before prepare)")
    cal.add_argument("--data-root", type=Path, required=True, help="data folder or shared data")
    cal.add_argument("--results-root", type=Path, default=Path("results"))
    cal.add_argument("--split-seed", type=int, required=True, help="seeds the random controls")
    cal.add_argument("--close-bits", "--dedup-threshold", dest="close_bits", type=int,
                     required=True, help=CLOSE_HELP)  # fmt: skip
    cal.add_argument("--sheet-dir", type=Path, required=True, help="outside the repository")
    cal.add_argument(
        "--threshold", type=float, default=None,
        help="draw the sheet at this value (default: the provisional proposal)",
    )  # fmt: skip
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
    from medcompress.config import ConfigError
    from medcompress.data.dedup import ChainingError

    try:
        if args.command == "calibrate":
            return calibrate(args, dataset)
        if args.command == "counts":
            return counts(args, dataset)
        return prepare_command(args, dataset)
    except ConfigError as exc:
        print(f"setting open: {exc}", file=sys.stderr)
        return SETTING_OPEN
    except ChainingError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return CHAINING_STOP


def prepare_command(args: argparse.Namespace, dataset: dict) -> int:
    """The first RC-D1 pass: splits, private files, counts and the final contact sheet."""
    from medcompress.data.prepare import prepare, write_private_outputs

    prepared = prepare(
        args.data_root,
        dataset,
        split_seed=args.split_seed,
        close_bits=args.close_bits,
        mask_threshold=args.mask_threshold,
        confirm_threshold=args.confirm_threshold,
    )
    hashes = write_private_outputs(prepared, args.private_dir, args.data_root)
    summary = args.results_root / "data" / "summary.csv"
    if summary.exists():
        summary.unlink()
    append_results(prepared.rows, summary)
    print(f"wrote {len(prepared.rows)} aggregate rows to {summary}")
    metrics = {r["metric"]: r["value"] for r in prepared.rows}
    print(
        f"groups {metrics['groups:total']:.0f}, largest {metrics['groups:largest']:.0f} "
        f"({metrics['groups:largest_fraction']:.1%}); held-out images with a confirmed "
        f"duplicate in train: {metrics['val_test_confirmed_duplicates_of_train']:.0f}; pairs "
        f"within {prepared.near_duplicates['close_bits']:.0f} bits not joined: "
        f"{metrics['near_duplicates:close_pairs_not_joined']:.0f} "
        f"({metrics['near_duplicates:close_pairs_not_joined:cross_split']:.0f} across splits)"
    )
    if args.sheet_dir is not None:
        sheet = _final_sheet(prepared, args.sheet_dir, args.data_root)
        print(f"final contact sheet (private, never commit it; attach it to the report): {sheet}")
    print("hashes for the run card:", json.dumps(hashes, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
