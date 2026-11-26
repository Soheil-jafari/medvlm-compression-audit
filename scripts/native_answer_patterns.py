#!/usr/bin/env python
"""The training split's answer patterns for the native-format readout (CPU, no model).

Note of 2026-11-25 (docs/decisions/native_readout_2026-11-25.md): the native parser ("no ..." ->
no; "evidence of ..." / "yes ..." -> yes; else unparsed) is checked against the training split's
answers to finding questions, never against an evaluation output. This lists their leading
two-word patterns with counts and the parser's coverage, and writes
``results/exploratory/native_answer_patterns.csv`` and ``..._coverage.csv`` (phrases and counts
only). If more than 5 % of those answers are unparsed, the note's rule says: report to the
research lead before the readout is used.

Example (Kaggle CPU notebook, the shared data attached):
    python scripts/native_answer_patterns.py --data-root "$DATA"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

UNPARSED_LIMIT = 0.05


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=REPO / "results" / "exploratory")
    args = parser.parse_args(argv)

    from medcompress.cli import resolve_config
    from medcompress.compression.pruning.stage import load_train_samples
    from medcompress.config import load_config
    from medcompress.evaluation.exploratory import _write
    from medcompress.evaluation.native_readout import answer_patterns, finding_questions

    namespace = argparse.Namespace(config=None, exp="E2", override=[], data_root=args.data_root,
                                   handoffs=[])  # fmt: skip
    config = resolve_config(namespace)
    domain = load_config(config["components"]["domain"])["domain"]
    names = list((domain.get("finding_names") or {}).values()) or list(domain["findings"])
    train = load_train_samples(config)
    answers = [s.answer for s in train if finding_questions(s.question, names)]
    rows, coverage = answer_patterns(answers)
    _write(args.out / "native_answer_patterns.csv", rows)
    _write(args.out / "native_answer_patterns_coverage.csv", [coverage])
    share = coverage["unparsed"] / coverage["answers"] if coverage["answers"] else float("nan")
    print(f"training answers to finding questions: {coverage}; unparsed share {share:.3f}")
    for row in rows:
        print(f"  {row['count']:>6}  {row['pattern']!r} -> {row['native_reading']}")
    if share > UNPARSED_LIMIT:
        print(f"MORE THAN {UNPARSED_LIMIT:.0%} UNPARSED: report to the research lead before the "
              "native readout is used (note of 2026-11-25)")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
