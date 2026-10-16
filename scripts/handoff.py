#!/usr/bin/env python
"""Hand-offs between Kaggle accounts: the output of one run, ready to become a private dataset.

  export  copy what a later run needs (by --kind) into --out and write its manifest: SHA-256
          of every file, the commit, the config hash, experiment, seed, budget, variant
  verify  check an attached hand-off against its manifest (the later run does this itself
          when it is given --from <path>)

Kinds: teacher_cache (RC-E0b), adapter (E1, E2), pruned (E5-E8, A1-A6), quantized (E3, E4,
E8, E9), outcomes (stage 06 of any run, for the comparison step), resume (a stopped training).

Examples (Kaggle):
    python scripts/handoff.py export --kind adapter --exp E2 --seed 0 \\
        --out /kaggle/working/handoff/E2_seed0_adapter
    python scripts/handoff.py export --kind teacher_cache --exp E0 \\
        --out /kaggle/working/handoff/E0_teacher_cache
    python scripts/handoff.py verify /kaggle/input/<dataset>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.cli import INPUT_CHECK_EXIT_CODE, build_parser, resolve_config  # noqa: E402
from medcompress.config import ConfigError, RunContext  # noqa: E402
from medcompress.handoffs import handoff_items  # noqa: E402
from medcompress.utils.bundles import (  # noqa: E402
    HANDOFF_KINDS,
    BundleError,
    export_handoff,
    handoff_meta,
    verify_bundle,
)


def export(argv: list[str]) -> int:
    parser = build_parser("export a hand-off")
    parser.add_argument("--kind", choices=HANDOFF_KINDS, required=True)
    parser.add_argument("--out", type=Path, required=True, help="a new folder outside the repo")
    args = parser.parse_args(argv)
    if args.exp is None:
        parser.error("--exp is required (E0 for the teacher cache)")
    run = RunContext(args.exp, seed=args.seed, budget=args.budget, variant=args.variant)
    config = resolve_config(args)
    try:
        items = handoff_items(args.kind, config, run)
        teacher_cache = args.kind == "teacher_cache"
        meta = handoff_meta(
            config,
            experiment=run.experiment_id,
            seed=None if teacher_cache else run.seed,
            budget=None if teacher_cache else run.budget,
            variant=None if teacher_cache else run.variant,
        )
        bundle = export_handoff(config, items, args.out, kind=args.kind, meta=meta)
    except (BundleError, ConfigError) as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return INPUT_CHECK_EXIT_CODE
    print(f"wrote {bundle.label} to {bundle.root}: {len(bundle.files)} files")
    print(f"commit {bundle.meta.get('commit')}; config {bundle.meta.get('config_sha256')}")
    print("make this folder ONE private Kaggle dataset (docs/team_setup.md, 'Hand-offs')")
    return 0


def verify(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="verify a hand-off")
    parser.add_argument("path", type=Path)
    parser.add_argument("--kind", choices=HANDOFF_KINDS, default=None)
    args = parser.parse_args(argv)
    try:
        bundle = verify_bundle(args.path, kind=args.kind or HANDOFF_KINDS)
    except BundleError as exc:
        print(f"check failed: {exc}", file=sys.stderr)
        return INPUT_CHECK_EXIT_CODE
    print(f"OK: {bundle.label}, {len(bundle.files)} files match the manifest at {bundle.root}")
    print(f"commit {bundle.meta.get('commit')}; config {bundle.meta.get('config_sha256')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: handoff.py {export,verify} ...  (each command takes --help)\n")
        print(__doc__)
        return 0
    command, rest = argv[0], argv[1:]
    if command == "export":
        return export(rest)
    if command == "verify":
        return verify(rest)
    print(f"unknown command {command!r}: use export or verify", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
