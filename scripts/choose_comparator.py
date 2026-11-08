#!/usr/bin/env python
"""Choose the comparator at B2 and apply the futility rule, from seed 0's validation rows (CPU).

The seed plan of 2026-11-08 (pre-registration section 11; ``medcompress.seed_plan``): among
E5:magnitude, E5:activation, E5:taylor and E6 at B2, seed 0, validation, the comparator is the
first by (i) ``op_small_finding_recall``, (ii) ``op_mask_positive_recall``, (iii)
``op_small_margin_mean`` (the small polyps' mean margin over the arm's own polyp threshold,
from the saved margins), (iv) the name. Then seeds 1 and 2 of the main tier (E2, E7 and the
comparator at B2) run unless E7 at B2 ranks below the comparator by the same ordering.

Reads the committed ``results/`` only (RC-G2's five runs give these rows) and writes
``results/seed_plan/comparator_B2.json``; the run cards of seeds 1 and 2 read it. A run
evaluated before the tie-break rows existed gets them from its saved outcomes with
``scripts/rescore_probes.py``.

Example (CPU, from the repository root, after RC-G2 is committed):
    python scripts/choose_comparator.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.cli import build_parser, main_guard, resolve_config, run_stage  # noqa: E402


def choose(results_root: Path) -> None:
    from medcompress.config import ConfigError
    from medcompress.evaluation.aggregate import load_results
    from medcompress.seed_plan import (
        COMPARATOR_CANDIDATES,
        OURS,
        SeedPlanError,
        choice_record,
        scores_from_rows,
        write_choice,
    )

    rows = load_results(results_root)
    scores = scores_from_rows(rows, [*COMPARATOR_CANDIDATES, OURS])
    if OURS not in scores:
        raise ConfigError("E7 at B2, seed 0 has no validation rows: run RC-G2 first")
    try:
        record = choice_record({c: scores[c] for c in scores if c != OURS}, scores[OURS])
    except SeedPlanError as exc:
        raise ConfigError(str(exc)) from exc
    path = write_choice(record, results_root)
    for place, row in enumerate(record["ranking"], start=1):
        print(f"{place}. {row['label']}: small {row['small_recall']:.4f}, mask-positive "
              f"{row['mask_recall']:.4f}, margin {row['small_margin']:.4f}")  # fmt: skip
    print(f"comparator: {record['comparator']} (decided by {record['decided_by']})")
    print(f"E7: {record['seeds_1_2']}")
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0], experiment_args=False)
    parser.add_argument(
        "--results-root", type=Path, default=None, help="Defaults to paths.results_root."
    )
    args = parser.parse_args(argv)
    config = resolve_config(args)
    root = args.results_root or Path(config["paths"]["results_root"])
    return run_stage("choose_comparator", choose, results_root=root)


if __name__ == "__main__":
    main_guard(main)
