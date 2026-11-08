#!/usr/bin/env python
"""PolypGen (RC-X2): describe the download, and check its overlap with our Kvasir images (CPU).

``describe``: counts per centre (images, masks, sequences, clusters, positives, negatives,
unlabelled frames) to ``results/data/polypgen_describe.csv``; run it first and read the counts
before anything is used (the loader's layout rules are in ``data/polypgen.py``).

``overlap``: the held-out filter of D02 amendment 6 against our training split, with the same
settings as for our filtered test (F 0.86), counts per centre to
``results/data/polypgen_heldout_filter.csv``; the list of removed images goes to
``--private-dir`` only (outside the repository, never committed). Our training embeddings come
from RC-D1-embed's output (``--embed-dir``) or are computed again with the pinned model.

Example (Kaggle CPU notebook, medcompress-data and the PolypGen dataset attached):
    DATA=$(python scripts/find_inputs.py data) || exit 1
    POLYPGEN=/kaggle/input/datasets/<user>/<polypgen-dataset>
    python scripts/polypgen.py describe --root $POLYPGEN
    python scripts/polypgen.py overlap --root $POLYPGEN --data-root $DATA \\
        --private-dir /kaggle/working/polypgen_private
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def describe(args: argparse.Namespace) -> int:
    from medcompress.data.polypgen import describe_rows, polyp_area, scan
    from medcompress.utils.io import append_results

    items = scan(args.root)
    if not items:
        print(f"STOP: no images under {args.root}", file=sys.stderr)
        return 5
    areas = {i.image: polyp_area(args.root, i) for i in items if i.labelled}
    rows = describe_rows(items, areas)
    target = args.results_root / "data" / "polypgen_describe.csv"
    target.unlink(missing_ok=True)
    append_results(rows, target)
    for row in rows:
        print(f"{row['metric']}: {row['value']:g}")
    print(f"wrote {target}")
    return 0


def overlap(args: argparse.Namespace) -> int:
    import numpy as np

    from medcompress.config import load_config
    from medcompress.data.dedup import image_features
    from medcompress.data.embeddings import (
        EmbeddingSettings,
        embed_images,
        load_embeddings,
        make_embedder,
    )
    from medcompress.data.polypgen import heldout_flags, overlap_rows, scan
    from medcompress.data.prepare import crop_settings
    from medcompress.data.prepared import load_prepared
    from medcompress.utils.bundles import BundleError, attach_data
    from medcompress.utils.io import append_results

    dataset = load_config("data/kvasir_vqa_x1.yaml")["dataset"]
    try:
        paths = attach_data({}, args.data_root)
    except BundleError as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return 5
    data_root, private = paths.root / "data", paths.root / "private"
    train = sorted({s.image_path for s in load_prepared(data_root, dataset, private)["train"]})
    items = [i for i in scan(args.root) if i.labelled]
    crop = crop_settings(dataset)
    ours = {f"kvasir/{p}": data_root / p for p in train}
    theirs = {f"polypgen/{i.image}": Path(args.root) / i.image for i in items}
    fingerprints, signatures = {}, {}
    for key, path in {**ours, **theirs}.items():
        fingerprints[key], signatures[key] = image_features(str(path), crop)
    settings = EmbeddingSettings.from_config(dataset)
    if args.model:
        settings = EmbeddingSettings(args.model, None, settings.side, settings.centre_fraction)
    embedder = make_embedder(settings.model, settings.revision)
    if args.embed_dir is not None:  # our training images from RC-D1-embed's output
        path = args.embed_dir / "embeddings.npz"
        with np.load(path, allow_pickle=False) as saved:
            saved_images = [str(x) for x in saved["images"]]
        saved_emb, _ = load_embeddings(path, saved_images, settings)
        row = {p: k for k, p in enumerate(saved_images)}
        missing = [p for p in train if p not in row]
        if missing:
            print(f"STOP: {path} lacks {len(missing)} training images", file=sys.stderr)
            return 5
        ours_full = saved_emb["full"][[row[p] for p in train]]
    else:
        ours_full = embed_images([str(p) for p in ours.values()], embedder, settings)["full"]
    theirs_full = embed_images([str(p) for p in theirs.values()], embedder, settings)["full"]
    images = [*ours, *theirs]
    full = np.concatenate([ours_full, theirs_full])
    flags = heldout_flags(list(theirs), list(ours), fingerprints, signatures, {"full": full},
                          images, crop)  # fmt: skip
    by_image = {key.split("/", 1)[1]: why for key, why in flags.items()}
    rows = overlap_rows(items, by_image)
    target = args.results_root / "data" / "polypgen_heldout_filter.csv"
    target.unlink(missing_ok=True)
    append_results(rows, target)
    args.private_dir.mkdir(parents=True, exist_ok=True)
    removed = sorted(k for k, why in by_image.items() if any(why.values()))
    (args.private_dir / "polypgen_removed.json").write_text(json.dumps(removed, indent=1))
    for row in rows:
        print(f"{row['metric']}: {row['value']:g}")
    print(f"wrote {target}; the removed images are listed in {args.private_dir} (private)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("describe", help="counts per centre")
    one.add_argument("--root", type=Path, required=True)
    one.add_argument("--results-root", type=Path, default=Path("results"))
    two = sub.add_parser("overlap", help="the held-out filter against our training images")
    two.add_argument("--root", type=Path, required=True)
    two.add_argument("--data-root", type=Path, required=True, help="the shared data (RC-D1)")
    two.add_argument("--private-dir", type=Path, required=True, help="outside the repository")
    two.add_argument("--embed-dir", type=Path, default=None, help="RC-D1-embed's output")
    two.add_argument("--model", default=None, help="embedding model (tests: tiny)")
    two.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    return describe(args) if args.command == "describe" else overlap(args)


if __name__ == "__main__":
    sys.exit(main())
