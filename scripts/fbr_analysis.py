#!/usr/bin/env python
"""The FBR diagnostic and the pilot's gate, from saved validation arrays (CPU; FBR cells (i)
and (iv)).

Secondary study of the amendment of 2026-11-22 (docs/decisions/fbr_study_2026-11-22.md).
``diagnostic``: E2 and every B2 pruning arm with validation outcomes (and probe answers where
found): margins by label, token ranks, sign / decoded agreement, parse failures, AUROC, AP and the
two cross-fitted recalibrations. ``pilot``: every FBR run against its arm's registered recovery,
the gate, the sampling-only comparison. Validation only; writes CSV and a summary to ``--out``.

Example (Kaggle CPU notebook, the outcome and probe-answer datasets attached):
    python scripts/fbr_analysis.py diagnostic --outcomes /kaggle/input
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("diagnostic", "pilot"))
    parser.add_argument("--outcomes", type=Path, action="append", default=None,
                        help="a folder to search (repeatable; default /kaggle/input)")  # fmt: skip
    parser.add_argument("--out", type=Path, default=REPO / "results" / "exploratory" / "fbr")
    args = parser.parse_args(argv)

    from medcompress.cli import resolve_config
    from medcompress.evaluation import fbr_analysis
    from medcompress.evaluation.runner import resolve_settings
    from medcompress.registry import REGISTRY

    namespace = argparse.Namespace(config=None, exp="E2", override=[], data_root=None, handoffs=[])
    order = resolve_settings(resolve_config(namespace)).findings
    roots = args.outcomes or [Path("/kaggle/input")]
    if args.mode == "diagnostic":
        rows = fbr_analysis.run_diagnostic(roots, args.out, order, list(REGISTRY))
        print(f"wrote {rows} diagnostic rows to {args.out}")
    else:
        result = fbr_analysis.run_pilot(roots, args.out, order, list(REGISTRY))
        print(f"wrote the pilot to {args.out}; gate passed: {result['passed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
