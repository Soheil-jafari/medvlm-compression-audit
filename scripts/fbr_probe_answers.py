#!/usr/bin/env python
"""Decoded answers and yes / no token ranks of a run's validation probes (GPU; FBR cell (i)).

Secondary study of the amendment of 2026-11-22 (docs/decisions/fbr_study_2026-11-22.md): the
diagnostic needs, per validation probe, the greedy decoded free answer (at most 8 tokens,
parsed into yes / no / parse failure) and the rank of the yes and no tokens among all tokens at
the answer position. The evaluation does not save them; this script loads the run's model as
stage 06 does and writes ``<output_root>/fbr/probe_answers/<run dir>/probe_answers_val.npz``.
Validation only. It writes no results file.

Example (Kaggle, one T4; the run's model hand-off and the E2 adapter attached):
    python scripts/fbr_probe_answers.py --exp E5 --variant magnitude --budget B2 --seed 0
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.cli import (  # noqa: E402
    build_parser,
    main_guard,
    parse_stage_args,
    resolve_config,
    run_stage,
)
from medcompress.compression.fbr_stage import run_probe_answers  # noqa: E402

SCRIPT = "scripts/06_evaluate.py"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    args, run = parse_stage_args(parser, argv, script=SCRIPT, default_exp=None,
                                 require_budget=False)  # fmt: skip
    config = resolve_config(args)
    return run_stage("fbr_probe_answers", run_probe_answers, config=config, run=run)


if __name__ == "__main__":
    main_guard(main)
