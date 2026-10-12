#!/usr/bin/env python
"""Student selection (decision D05): run one candidate through one round, rank a round, or report.

Runs on the validation split only. Round results go to results/selection/<round>/<candidate>/
metrics.csv; the per-probe answers that the ranking needs go to
$MEDCOMPRESS_OUTPUT_ROOT/selection/ and are never committed.

Examples (Kaggle):
    python scripts/select_student.py --round R0 --candidate qwen3_vl_2b
    python scripts/select_student.py --round R1 --candidate qwen3_vl_2b
    python scripts/select_student.py --rank R1       # after all outputs are in one place
    python scripts/select_student.py --report        # writes docs/results/SELECTION.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.cli import build_parser, main_guard, run_stage  # noqa: E402
from medcompress.config import load_config  # noqa: E402

ROUND_CHOICES = ("R0", "R1", "R2", "R3")


def make_parser() -> argparse.ArgumentParser:
    parser = build_parser(__doc__.splitlines()[0], experiment_args=False)
    parser.add_argument(
        "--seed", type=int, default=0, help="seed of the run (R3 has seeds 0 and 1)"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--round", choices=ROUND_CHOICES, help="run one candidate through a round")
    action.add_argument("--rank", choices=ROUND_CHOICES, help="rank a finished round")
    action.add_argument("--report", action="store_true", help="write docs/results/SELECTION.md")
    action.add_argument("--list", action="store_true", help="list the candidate pool")
    parser.add_argument("--candidate", help="candidate name, e.g. qwen3_vl_2b (with --round)")
    parser.add_argument(
        "--allow-remote-code",
        action="store_true",
        help="load the candidate's own model code (only after reading it)",
    )
    parser.add_argument("--report-path", type=Path, default=Path("docs/results/SELECTION.md"))
    return parser


def act(config: dict[str, Any], args: argparse.Namespace) -> None:
    from medcompress.selection import driver
    from medcompress.selection.rounds import run_round

    if args.list:
        for name in driver.pool(config):
            print(name)
    elif args.report:
        print(f"wrote {driver.write_report(config, args.report_path)}")
    elif args.rank:
        if args.rank == "R0":
            for name, failed in driver.screen_feasibility(config).items():
                print(f"{name}: {'pass' if not failed else 'FAIL ' + ', '.join(failed)}")
        else:
            result = driver.rank_selection(config, args.rank, seed=args.seed)
            for i, name in enumerate(result.candidates):
                mark = "advance" if name in result.advance else ""
                print(f"{i + 1:2d}  {name:28s} S={result.scores[i]:.3f}  {mark}")
            for name, gates in result.failed_gates.items():
                print(f"--  {name:28s} failed {', '.join(gates)}")
    else:
        run_round(
            config,
            args.round,
            args.candidate,
            seed=args.seed,
            allow_remote_code=args.allow_remote_code,
        )


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.round and not args.candidate:
        parser.error("--round needs --candidate")
    config = load_config(args.config or Path("student_selection.yaml"), args.override)
    return run_stage("select_student", act, config=config, args=args)


if __name__ == "__main__":
    main_guard(main)
