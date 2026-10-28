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
  embed    RC-D1-embed (D02 amendment 3): embeddings of every image, the candidate pairs, the
           labelling sample, its numbered sheets and the label template, in --out-dir (private,
           outside the repository; upload it as ONE private Kaggle dataset); counts to
           <results_root>/data/near_duplicate_embed.csv
  score-labels  RC-D1-label: Cohen's kappa, the third member's decisions, sensitivity and
           precision of every candidate rule, the joining rule chosen by the rule fixed in
           advance and the unsure analysis; aggregate numbers to
           <results_root>/data/near_duplicate_labels.csv and near_duplicate_rules.csv, which
           RC-D1 run 4 reads. If the third member has pairs to decide, their list goes to
           --private-dir and it stops (exit code 4). --no-third-member --kappa-pairs 11-298
           (D02 amendment 4): two raters, no adjudication, the kappa on those pairs, positive
           when both raters say same_scene, and the either-rater analysis (reported only)
  topup-labels  the top-up of D02 amendment 3: if fewer than 30 labelled pairs lie within
           +-0.03 of the chosen threshold, 100 more pairs from that band (seed 20261013), with
           sheets and a template in --out-dir (private); reuses RC-D1-embed's output (no new
           embedding); counts to <results_root>/data/near_duplicate_topup.csv
  commit-labels  check one rater's label file and add its SHA-256 to
           docs/runs/RC-D1-label_commitments.csv (blinding: commit that file before the labels
           are discussed)
  rule-sweep  D02 amendment 5: the rules full:T OR crop >= 0.80 for the given T, highest
           first, each split exactly as prepare builds it: the largest group, the joined pairs,
           the val/test confirmed duplicates and one three-band cross-split contact sheet per
           rule within the chaining guard (PNGs in --out-dir, private); counts to
           <results_root>/data/near_duplicate_sweep.csv. prepare and counts then take
           --rule <one of them> (run 4b); a rule not in that file is refused
  filter-sweep  D02 amendment 6: with the swept rule's groups and split, the held-out filter
           at each F (remove a val/test image whose training neighbour has a full-view cosine
           >= F, a crop >= 0.70, or a crop >= 0.50 within 4 dHash bits): what each F keeps and
           one private sheet of the closest remaining train/val and train/test pairs (PNGs in
           --out-dir); counts to <results_root>/data/near_duplicate_filter_sweep.csv. prepare and
           counts then take --heldout-filter <one F of it> with --rule

Exit codes: 4 a setting is open (the joining rule before the labels are scored, no rule
qualifies, the third member has pairs to decide, or a top-up is required), 5 an input check
failed, 7 the chaining guard stopped the run (one group holds too
many images), 8 the embedding model failed the check on synthetic copies.

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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from medcompress.config import config_root, load_config  # noqa: E402
from medcompress.utils.io import append_results  # noqa: E402

X1_REPO = "SimulaMet/Kvasir-VQA-x1"
HOST_REPO = "SimulaMet-HOST/Kvasir-VQA"
SETTING_OPEN = 4
INPUT_CHECK_FAILED = 5
CHAINING_STOP = 7
SELF_CHECK_FAILED = 8
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

    rule, heldout, code = _swept(args)
    if code:
        return code
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
        rule=rule,  # type: ignore[arg-type]
        heldout_filter=heldout,  # type: ignore[arg-type]
        results_root=args.results_root,
        embed_dir=_attached(args.embed_dir, dataset),
    )
    prepared = _from_sweep(prepared, rule)
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


def embed(args: argparse.Namespace, dataset: dict) -> int:
    """RC-D1-embed: embeddings, candidates, the labelling sample and sheets (private)."""
    from medcompress.config import repo_root
    from medcompress.data.embed_step import SelfCheckError, embed_step
    from medcompress.data.embeddings import EmbeddingSettings, make_embedder
    from medcompress.utils.bundles import EMBED_KIND, BundleError, write_bundle_manifest

    out = args.out_dir.resolve()
    if out.is_relative_to(repo_root().resolve()):
        print(f"--out-dir must be outside the repository: {out}", file=sys.stderr)
        return INPUT_CHECK_FAILED
    try:
        data_root = _data_folder(args.data_root)
    except BundleError as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return INPUT_CHECK_FAILED
    settings = EmbeddingSettings.from_config(dataset)
    model = args.embedding_model or settings.model
    try:
        embedder = make_embedder(model, settings.revision)
    except Exception as exc:  # noqa: BLE001 - the hub record or the model could not be read
        print(f"the embedding model {model!r} could not be loaded: {type(exc).__name__}: {exc}",
              file=sys.stderr)  # fmt: skip
        print("check the model id on the hub; if it is not there, a comparable model is chosen "
              "and recorded (D02 amendment 3)", file=sys.stderr)  # fmt: skip
        return INPUT_CHECK_FAILED
    try:
        result = embed_step(
            data_root, dataset, out, embedder, split_seed=args.split_seed,
            close_bits=args.close_bits, mask_threshold=args.mask_threshold,
        )  # fmt: skip
    except SelfCheckError as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return SELF_CHECK_FAILED
    target = args.results_root / "data" / "near_duplicate_embed.csv"
    target.unlink(missing_ok=True)
    append_results(result.rows, target)
    info = embedder.info
    made = write_bundle_manifest(
        out, kind=EMBED_KIND,
        meta={"model": info.get("model"), "revision": info.get("revision"),
              "licence": info.get("licence"), "data_split_seed": args.split_seed},
    )  # fmt: skip
    print(f"wrote {len(result.rows)} aggregate rows to {target}")
    print(f"wrote {out / 'bundle_manifest.json'}: {len(made.files)} files; upload {out} as ONE "
          "private Kaggle dataset (run card RC-D1-embed)")  # fmt: skip
    print(f"RECORD: model {info.get('model')}, revision {info.get('revision')}, licence "
          f"{info.get('licence')}")  # fmt: skip
    return 0


def score_labels(args: argparse.Namespace, dataset: dict) -> int:
    """RC-D1-label: kappa, the third member, every rule's scores, the chosen rule."""
    from medcompress.data.embed_step import score_labels_step
    from medcompress.data.labelling import LabelError
    from medcompress.utils.bundles import TOPUP_KIND, BundleError, verify_bundle

    if args.no_third_member != (args.kappa_pairs is not None):
        print("--no-third-member and --kappa-pairs go together (D02 amendment 4: "
              "--no-third-member --kappa-pairs 11-298)", file=sys.stderr)  # fmt: skip
        return INPUT_CHECK_FAILED
    try:
        embed_dir = _embed_output(args.embed_dir, dataset)
        topup = verify_bundle(args.topup_dir, kind=TOPUP_KIND).root if args.topup_dir else None
        scoring = score_labels_step(
            embed_dir, args.labels, dataset, topup_dir=topup, private_dir=args.private_dir,
            commitments=args.commitments, third_member=not args.no_third_member,
            kappa_pairs=args.kappa_pairs,
        )  # fmt: skip
    except (BundleError, LabelError, ValueError) as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return INPUT_CHECK_FAILED
    if scoring.adjudication is not None:
        return SETTING_OPEN  # nothing is written until the third member has decided
    for name, result in (("near_duplicate_labels.csv", scoring.labels),
                         ("near_duplicate_rules.csv", scoring.rules)):  # fmt: skip
        target = args.results_root / "data" / name
        target.unlink(missing_ok=True)
        if result.rows:
            append_results(result.rows, target)
            print(f"wrote {len(result.rows)} aggregate rows to {target}")
    return 0 if scoring.chosen is not None and not scoring.topup_required else SETTING_OPEN


def topup_labels(args: argparse.Namespace, dataset: dict) -> int:
    """The top-up of D02 amendment 3, from RC-D1-embed's output and the first chosen rule."""
    from medcompress.config import repo_root
    from medcompress.data.embed_step import topup_step
    from medcompress.data.labelling import LabelError
    from medcompress.utils.bundles import TOPUP_KIND, BundleError, write_bundle_manifest

    out = args.out_dir.resolve()
    if out.is_relative_to(repo_root().resolve()):
        print(f"--out-dir must be outside the repository: {out}", file=sys.stderr)
        return INPUT_CHECK_FAILED
    try:
        embed_dir = _embed_output(args.embed_dir, dataset)
        data_root = _data_folder(args.data_root)
        result, top = topup_step(embed_dir, data_root, out, args.results_root, dataset)
    except (BundleError, LabelError) as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return INPUT_CHECK_FAILED
    target = args.results_root / "data" / "near_duplicate_topup.csv"
    target.unlink(missing_ok=True)
    append_results(result.rows, target)
    print(f"wrote {len(result.rows)} aggregate rows to {target}")
    if top.pairs:
        made = write_bundle_manifest(out, kind=TOPUP_KIND, meta={"pairs": len(top.pairs)})
        print(f"wrote {out / 'bundle_manifest.json'}: {len(made.files)} files; upload {out} as "
              "ONE private Kaggle dataset; the raters get only its label_sheets/, "
              "label_images.csv and labels_template.csv (card RC-D1-label)")  # fmt: skip
    return 0


def commit_labels(args: argparse.Namespace) -> int:
    """Check a rater's own label file and add its SHA-256 to the commitments file."""
    from medcompress.data.labelling import LabelError, add_commitment

    try:
        row = add_commitment(args.commitments, args.labels, role=args.role, round_=args.round)
    except LabelError as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return INPUT_CHECK_FAILED
    print(f"{row['file']}: {row['labels']} labels of {row['role']}, round {row['round']}, "
          f"SHA-256 {row['sha256']}")  # fmt: skip
    print(f"now commit {args.commitments} on its own (git add {args.commitments}; git commit -m "
          f'"Commit the {row["role"]} labels of round {row["round"]}"; git push), then upload '
          "the label file to the private labels dataset")  # fmt: skip
    return 0


def rule_sweep_command(args: argparse.Namespace, dataset: dict) -> int:
    """D02 amendment 5: the rule sweep (counts, and one private sheet per rule in the guard)."""
    from medcompress.config import repo_root
    from medcompress.data.prepare import near_duplicate_settings
    from medcompress.data.rule_sweep import (
        SWEEP_FILE,
        SweepError,
        either_rater_rule,
        rule_sweep,
        sweep_rows,
        sweep_rules,
    )
    from medcompress.utils.bundles import BundleError

    out = args.out_dir.resolve()
    if out.is_relative_to(repo_root().resolve()):
        print(f"--out-dir must be outside the repository: {out}", file=sys.stderr)
        return INPUT_CHECK_FAILED
    try:
        embed_dir = _embed_output(args.embed_dir, dataset)
        data_root = _data_folder(args.data_root)
        rules = sweep_rules(args.thresholds, args.crop)
        label_rule = either_rater_rule(args.results_root)
        if label_rule != rules[0]:
            raise SweepError(
                f"the sweep starts from the rule chosen on the either-rater truth, "
                f"{label_rule.label} (labels:either_rater:rule:*), but its first rule is "
                f"{rules[0].label} (D02 amendment 5)"
            )
    except (BundleError, SweepError) as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return INPUT_CHECK_FAILED
    print(f"rule chosen on the either-rater truth (from the committed labels file): "
          f"{label_rule.label}")  # fmt: skip
    print("sweep: " + ", ".join(r.label for r in rules))
    results = rule_sweep(
        data_root, dataset, embed_dir, rules, split_seed=args.split_seed,
        close_bits=args.close_bits, mask_threshold=args.mask_threshold,
    )  # fmt: skip
    guard = near_duplicate_settings(dataset, close_bits=args.close_bits, confirm_threshold=1.0)
    for r in results:
        head = (f"{r.rule.label}: joined pairs {r.joined_pairs}; largest group "
                f"{r.largest_group} ({r.largest_fraction:.1%})")  # fmt: skip
        if r.prepared is None:
            print(f"{head}; OVER THE GUARD ({guard.max_group_fraction:.0%}): no split, no sheet")
            continue
        values = {row["metric"]: row["value"] for row in r.prepared.rows}
        sheet = _final_sheet(r.prepared, out, data_root,
                             name=f"rc_d1_sweep_{r.rule.label.replace(':', '_')}.png")  # fmt: skip
        print(f"{head}; within the guard; val/test confirmed duplicates of train "
              f"{values['val_test_confirmed_duplicates_of_train']:.0f}; images train / val / "
              f"test {values['images:train']:.0f} / {values['images:val']:.0f} / "
              f"{values['images:test']:.0f}; split_sha256 {r.prepared.split_sha256}; "
              f"sheet (private, never commit it): {sheet}")  # fmt: skip
    rows = sweep_rows(
        results, label_rule=label_rule, split_seed=args.split_seed, close_bits=args.close_bits,
        mask_threshold=args.mask_threshold, max_group_fraction=guard.max_group_fraction,
    )  # fmt: skip
    target = args.results_root / SWEEP_FILE
    target.unlink(missing_ok=True)
    append_results(rows, target)
    print(f"wrote {len(rows)} aggregate rows to {target}")
    print("the team now judges every sheet: per rule and band, how many pairs show the same "
          "scene; the rule is the highest T within the guard with none "
          "(D02 amendment 5)")  # fmt: skip
    return 0


def _filter_line(values: dict) -> str:
    """One line on what the held-out filter removed and kept."""
    parts = []
    for name in ("val", "test"):
        total = values[f"heldout_filter:unfiltered:images:{name}"]
        share = values.get(f"heldout_filter:kept_fraction:images:{name}")
        parts.append(
            f"{name} keeps {values[f'images:{name}']:.0f} of {total:.0f} images "
            f"({'n/a' if share is None else f'{share:.1%}'}), "
            f"{values[f'qa_rows:{name}']:.0f} of "
            f"{values[f'heldout_filter:unfiltered:qa_rows:{name}']:.0f} QA rows"
        )
    floor = "meets" if values["heldout_filter:test_floor_met"] else "is BELOW"
    masked_kept = values.get("heldout_filter:kept_fraction:masked_images:test")
    masked = "n/a" if masked_kept is None else f"{masked_kept:.1%}"
    pairs = values["heldout_filter:val_test_pairs_matching"]
    return (f"held-out filter F {values['heldout_filter:threshold']:.2f}: " + "; ".join(parts)
            + f"; masked test images kept {masked}; the test {floor} the "
            f"{values['heldout_filter:min_test_kept']:.0%} floor; val/test pairs meeting the "
            f"filter (not filtered) {pairs:.0f}")  # fmt: skip


def filter_sweep_command(args: argparse.Namespace, dataset: dict) -> int:
    """D02 amendment 6: the held-out filter sweep (counts, one private sheet per F)."""
    from medcompress.config import repo_root
    from medcompress.data.rule_sweep import (
        FILTER_SWEEP_FILE,
        SweepError,
        filter_sweep,
        filter_sweep_rows,
        heldout_filters,
        swept_rule,
    )
    from medcompress.utils.bundles import BundleError

    out = args.out_dir.resolve()
    if out.is_relative_to(repo_root().resolve()):
        print(f"--out-dir must be outside the repository: {out}", file=sys.stderr)
        return INPUT_CHECK_FAILED
    try:
        embed_dir = _embed_output(args.embed_dir, dataset)
        data_root = _data_folder(args.data_root)
        rule = swept_rule(args.results_root, args.rule)
        filters = heldout_filters(args.thresholds, crop=args.crop, bits=args.bits,
                                  bits_crop=args.bits_crop)  # fmt: skip
    except (BundleError, SweepError) as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return INPUT_CHECK_FAILED
    print(f"joining rule: {rule.label} (from the sweep, D02 amendment 5); filters: F "
          + ", ".join(f.label for f in filters)
          + f"; crop {args.crop:.2f}; crop {args.bits_crop:.2f} within {args.bits} "
          "bits")  # fmt: skip
    base, results = filter_sweep(
        data_root, dataset, embed_dir, rule, filters, split_seed=args.split_seed,
        close_bits=args.close_bits, mask_threshold=args.mask_threshold,
    )  # fmt: skip
    unfiltered = {row["metric"]: row["value"] for row in base.rows}
    print(f"unfiltered: images train / val / test {unfiltered['images:train']:.0f} / "
          f"{unfiltered['images:val']:.0f} / {unfiltered['images:test']:.0f}; split_sha256 "
          f"{base.split_sha256}")  # fmt: skip
    for r in results:
        sheet = _final_sheet(r.prepared, out, data_root, name=f"rc_d1_filter_{r.heldout.label}.png")
        print(f"{_filter_line(r.values)}; split_sha256 {r.prepared.split_sha256}; sheet "
              f"(private, never commit it): {sheet}")  # fmt: skip
    rows = filter_sweep_rows(
        base, results, rule=rule, split_seed=args.split_seed, close_bits=args.close_bits,
        mask_threshold=args.mask_threshold,
    )  # fmt: skip
    target = args.results_root / FILTER_SWEEP_FILE
    target.unlink(missing_ok=True)
    append_results(rows, target)
    print(f"wrote {len(rows)} aggregate rows to {target}")
    print("the team now judges every sheet: per F and band, how many pairs show the same scene; "
          "F is the highest with none whose test keeps at least half of its images "
          "(D02 amendment 6)")  # fmt: skip
    return 0


def _swept(args: argparse.Namespace) -> tuple[object | None, object | None, int]:
    """The rule of ``--rule`` and the filter of ``--heldout-filter`` if the committed sweeps
    allow them: (rule or None, filter or None, exit code)."""
    heldout = getattr(args, "heldout_filter", None)
    if args.rule is None:
        if heldout is not None:
            print("--heldout-filter needs --rule (D02 amendment 6)", file=sys.stderr)
            return None, None, INPUT_CHECK_FAILED
        return None, None, 0
    from medcompress.data.rule_sweep import SweepError, swept_filter, swept_rule

    if args.confirm_threshold is not None:
        print("--rule and --confirm-threshold exclude each other", file=sys.stderr)
        return None, None, INPUT_CHECK_FAILED
    try:
        rule = swept_rule(args.results_root, args.rule)
        found = swept_filter(args.results_root, heldout, rule) if heldout is not None else None
    except SweepError as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return None, None, INPUT_CHECK_FAILED
    print(f"joining rule: {rule.label} (from the sweep, D02 amendment 5)")
    if found is not None:
        print(f"held-out filter: F {found.threshold:.2f}, crop {found.crop:.2f}, crop "
              f"{found.bits_crop:.2f} within {found.bits} bits (from the filter sweep, D02 "
              "amendment 6)")  # fmt: skip
    return rule, found, 0


def _from_sweep(prepared, rule):  # type: ignore[no-untyped-def]
    """Record in the rows that the rule came from the sweep
    (``near_duplicates:rule_from_sweep`` = 1)."""
    from dataclasses import replace

    from medcompress.data.prepare import _row

    if rule is None:
        return prepared
    return replace(prepared, rows=(*prepared.rows, _row("near_duplicates:rule_from_sweep", 1)))


def _pair_range(text: str) -> tuple[int, int]:
    """``FIRST-LAST`` (both included), e.g. ``11-298``."""
    try:
        low, high = (int(part) for part in text.split("-"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r}: give FIRST-LAST, e.g. 11-298") from exc
    if not 1 <= low <= high:
        raise argparse.ArgumentTypeError(f"{text!r}: need 1 <= FIRST <= LAST")
    return low, high


def _attached(path: Path | None, dataset: dict) -> Path | None:
    """RC-D1-embed's output, verified (``None`` if not given)."""
    return None if path is None else _embed_output(path, dataset)


def _embed_output(path: Path, dataset: dict) -> Path:
    """RC-D1-embed's output, verified; its recorded model, revision and licence must be the
    pinned ones of the dataset config."""
    from medcompress.data.embed_step import pinned_problems
    from medcompress.utils.bundles import EMBED_KIND, BundleError, verify_bundle

    root = verify_bundle(path, kind=EMBED_KIND).root
    problems = pinned_problems(root, dataset)
    if problems:
        raise BundleError(f"{root} was not made with the pinned embedding model: "
                          + "; ".join(problems))  # fmt: skip
    return root


def _final_sheet(prepared, sheet_dir: Path, data_root: Path, name: str = FINAL_SHEET) -> Path:  # type: ignore[no-untyped-def]
    """Three bands, all pairs across splits. Under the cosine rule of amendment 3: the most
    similar by the cosine, by the crop, and the closest by the dHash. Under the crop rule:
    the most similar by the crop, the closest by the dHash, and the closest pairs within
    ``close_bits`` that the crop left apart."""
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
    if prepared.top_cosine_cross_split:
        where = "across splits"
        if prepared.heldout_filter:  # D02 amendment 6: train/val and train/test pairs only
            where = (f"train/val and train/test left after the held-out filter (F "
                     f"{prepared.heldout_filter['threshold']:.2f})")  # fmt: skip
        cosine = (
            (f"1. The {len(prepared.top_cosine_cross_split)} most similar pairs {where} "
             "by the cosine of the joining rule (caption: crop r, dHash bits, splits): all should "
             "be different scenes", cells(prepared.top_cosine_cross_split)),
            (f"2. The {len(prepared.top_cross_split)} most similar pairs {where} by the "
             "crop correlation: all should be different scenes", cells(prepared.top_cross_split)),
            (f"3. The {len(prepared.closest_dhash_cross_split)} closest pairs {where} by the"
             " dHash: all should be different scenes", cells(prepared.closest_dhash_cross_split)),
        )  # fmt: skip
        crop = float(prepared.near_duplicates.get("crop_fraction", 0.60))
        return render_pairs(cosine, sheet_dir / name, crop_fraction=crop, columns=3)
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
    return render_pairs(bands, sheet_dir / name, crop_fraction=crop, columns=3)


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
    prep.add_argument("--embed-dir", type=Path, default=None, help="RC-D1-embed's output")
    prep.add_argument("--embedding-model", default=None, help="tests only (tiny)")
    prep.add_argument(
        "--rule", default=None,
        help="run 4b (D02 amendment 5): a rule of the committed sweep, e.g. full:0.88:or_crop:0.80",
    )  # fmt: skip
    prep.add_argument(
        "--heldout-filter", type=float, default=None, metavar="F",
        help="run 4b (D02 amendment 6): an F of the committed filter sweep, with --rule",
    )  # fmt: skip
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
    cnt.add_argument("--embed-dir", type=Path, default=None, help="RC-D1-embed's output")
    cnt.add_argument("--rule", default=None, help="as in the first pass (run 4b: the swept rule)")
    cnt.add_argument("--heldout-filter", type=float, default=None, metavar="F",
                     help="as in the first pass (run 4b: the swept F)")  # fmt: skip
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
    emb = sub.add_parser("embed", help="RC-D1-embed: embeddings, candidates, labelling sheets")
    emb.add_argument("--data-root", type=Path, required=True, help="data folder or shared data")
    emb.add_argument("--out-dir", type=Path, required=True, help="private, outside the repo")
    emb.add_argument("--results-root", type=Path, default=Path("results"))
    emb.add_argument("--split-seed", type=int, required=True, help="to reproduce run 3")
    emb.add_argument("--close-bits", "--dedup-threshold", dest="close_bits", type=int,
                     required=True, help=CLOSE_HELP)  # fmt: skip
    emb.add_argument("--mask-threshold", type=int, required=True, help="to reproduce run 3")
    emb.add_argument("--embedding-model", default=None, help="tests only (tiny)")
    lab = sub.add_parser("score-labels", help="RC-D1-label: kappa, rules, the chosen rule")
    lab.add_argument("--embed-dir", type=Path, required=True, help="RC-D1-embed's output")
    lab.add_argument("--labels", type=Path, nargs="+", required=True,
                     help="the raters' files of every round and the third member's")  # fmt: skip
    lab.add_argument("--topup-dir", type=Path, default=None, help="the top-up's output, if any")
    lab.add_argument("--private-dir", type=Path, default=None,
                     help="where the third member's list goes (private)")  # fmt: skip
    lab.add_argument("--commitments", type=Path, default=None,
                     help="check every label file against these committed SHA-256")  # fmt: skip
    lab.add_argument("--no-third-member", action="store_true",
                     help="D02 amendment 4: two raters, no adjudication, the either-rater "
                          "analysis")  # fmt: skip
    lab.add_argument("--kappa-pairs", type=_pair_range, default=None, metavar="FIRST-LAST",
                     help="with --no-third-member: the kappa on these pairs only")  # fmt: skip
    lab.add_argument("--results-root", type=Path, default=Path("results"))
    top = sub.add_parser("topup-labels", help="the top-up pairs near the chosen threshold")
    top.add_argument("--data-root", type=Path, required=True, help="data folder or shared data")
    top.add_argument("--embed-dir", type=Path, required=True, help="RC-D1-embed's output")
    top.add_argument("--out-dir", type=Path, required=True, help="private, outside the repo")
    top.add_argument("--results-root", type=Path, default=Path("results"))
    swp = sub.add_parser("rule-sweep", help="D02 amendment 5: full:T OR crop rules, one sheet each")
    swp.add_argument("--data-root", type=Path, required=True, help="data folder or shared data")
    swp.add_argument("--embed-dir", type=Path, required=True, help="RC-D1-embed's output")
    swp.add_argument("--out-dir", type=Path, required=True, help="the sheets: outside the repo")
    swp.add_argument("--results-root", type=Path, default=Path("results"))
    swp.add_argument("--thresholds", type=float, nargs="+", required=True,
                     help="the full-view cosine levels T (D02 amendment 5: 0.92 0.90 0.88 0.86 "
                          "0.84)")  # fmt: skip
    swp.add_argument("--crop", type=float, required=True, help="the crop level (0.80)")
    swp.add_argument("--split-seed", type=int, default=20261012, help="as prepare (20261012)")
    swp.add_argument("--close-bits", type=int, default=4, help="as prepare (4)")
    swp.add_argument("--mask-threshold", type=int, default=4, help="as prepare (4)")
    fsw = sub.add_parser("filter-sweep", help="D02 amendment 6: the held-out filter at each F")
    fsw.add_argument("--data-root", type=Path, required=True, help="data folder or shared data")
    fsw.add_argument("--embed-dir", type=Path, required=True, help="RC-D1-embed's output")
    fsw.add_argument("--rule", required=True, help="the swept rule (full:0.92:or_crop:0.80)")
    fsw.add_argument("--out-dir", type=Path, required=True, help="the sheets: outside the repo")
    fsw.add_argument("--results-root", type=Path, default=Path("results"))
    fsw.add_argument("--thresholds", type=float, nargs="+", required=True,
                     help="the full-view cosine levels F (D02 amendment 6: 0.90 0.88 0.86 0.84 "
                          "0.82 0.80)")  # fmt: skip
    fsw.add_argument("--crop", type=float, default=0.70, help="crop level (0.70)")
    fsw.add_argument("--bits", type=int, default=4, help="dHash bits of the third condition (4)")
    fsw.add_argument("--bits-crop", type=float, default=0.50, help="crop within those bits (0.50)")
    fsw.add_argument("--split-seed", type=int, default=20261012, help="as prepare (20261012)")
    fsw.add_argument("--close-bits", type=int, default=4, help="as prepare (4)")
    fsw.add_argument("--mask-threshold", type=int, default=4, help="as prepare (4)")
    com = sub.add_parser("commit-labels", help="add a label file's SHA-256 to the commitments")
    com.add_argument("--labels", type=Path, required=True, help="your own filled file")
    com.add_argument("--role", required=True, choices=("rater_1", "rater_2", "rater_3"))
    com.add_argument("--round", required=True, choices=("first", "topup"))
    com.add_argument("--commitments", type=Path,
                     default=REPO / "docs" / "runs" / "RC-D1-label_commitments.csv")  # fmt: skip
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
    if args.command == "commit-labels":
        return commit_labels(args)
    from medcompress.config import ConfigError
    from medcompress.data.dedup import ChainingError
    from medcompress.utils.bundles import BundleError

    try:
        if args.command == "embed":
            return embed(args, dataset)
        if args.command == "score-labels":
            return score_labels(args, dataset)
        if args.command == "topup-labels":
            return topup_labels(args, dataset)
        if args.command == "rule-sweep":
            return rule_sweep_command(args, dataset)
        if args.command == "filter-sweep":
            return filter_sweep_command(args, dataset)
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
    except BundleError as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return INPUT_CHECK_FAILED


def prepare_command(args: argparse.Namespace, dataset: dict) -> int:
    """The first RC-D1 pass: splits, private files, counts and the final contact sheet."""
    from medcompress.data.embeddings import make_embedder
    from medcompress.data.prepare import prepare, write_private_outputs

    rule, heldout, code = _swept(args)
    if code:
        return code
    prepared = prepare(
        args.data_root,
        dataset,
        split_seed=args.split_seed,
        close_bits=args.close_bits,
        mask_threshold=args.mask_threshold,
        confirm_threshold=args.confirm_threshold,
        rule=rule,  # type: ignore[arg-type]
        heldout_filter=heldout,  # type: ignore[arg-type]
        results_root=args.results_root,
        embed_dir=_attached(args.embed_dir, dataset),
        embedder=make_embedder(args.embedding_model) if args.embedding_model else None,
    )
    prepared = _from_sweep(prepared, rule)
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
    if "strict_test:images" in metrics:
        print(f"strict test subset: {metrics['strict_test:images']:.0f} images "
              f"({metrics['strict_test:image_fraction_of_test']:.1%} of the test images), "
              f"{metrics['strict_test:qa_rows']:.0f} QA rows; margin "
              f"{metrics['strict_test:margin']:.2f}")  # fmt: skip
    if prepared.heldout_filter:
        print(_filter_line(metrics))
    if args.sheet_dir is not None:
        sheet = _final_sheet(prepared, args.sheet_dir, args.data_root)
        print(f"final contact sheet (private, never commit it; attach it to the report): {sheet}")
    print("hashes for the run card:", json.dumps(hashes, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
