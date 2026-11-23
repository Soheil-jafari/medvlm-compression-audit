#!/usr/bin/env python
"""E2 seed 0's margins of every training probe of the FBR schedule (GPU, one T4; FBR cell (ii)).

Secondary study of the amendment of 2026-11-22 (docs/decisions/fbr_study_2026-11-22.md). Builds
the probe schedule (training split, one positive and one negative per finding and update, seed 0),
scores E2 seed 0 on every probe of it with the training pass's margin function (no gradient),
fixes the (yes, no) target tokens (the single-token spelling E2 ranks first most often) and
prints the spelling report (each spelling's tokens, decoded examples). Writes
``<out>/fbr_e2_margins.json`` (sample ids inside: keep it as a private dataset, never commit
it). Stops when E2's first answer tokens are not single-token yes / no spellings.

Example (Kaggle, one T4; the shared data and the E2 seed 0 adapter attached):
    python scripts/fbr_e2_margins.py --out /kaggle/working/fbr_margins
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.cli import build_parser, main_guard, resolve_config, run_stage  # noqa: E402
from medcompress.compression.fbr_stage import run_e2_margin_cache  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0], experiment_args=False)
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/fbr_margins"))
    args = parser.parse_args(argv)
    args.exp = "E5"  # a pruning arm's config: the registered recovery section (157 updates)
    # sets the schedule; E2's own config has none. E2 seed 0 itself is loaded (stage 05's way).
    config = resolve_config(args)
    return run_stage("fbr_e2_margins", run_e2_margin_cache, config=config, out=args.out)


if __name__ == "__main__":
    main_guard(main)
