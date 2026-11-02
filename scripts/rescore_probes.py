#!/usr/bin/env python
"""Add the per-finding AUROC and "yes" rate to a run from its saved probe outcomes (CPU).

The secondary metrics pre-registered on 2026-11-02 (``auroc:<f>`` with its image-bootstrap
CI, ``yes_rate:<f>``, ``n_positives:<f>``, ``n_negatives:<f>``) are computed by every
evaluation from now on. A run scored before that keeps its per-probe outcomes
(``outcomes.npz`` under the output root), so they are recomputed here without a GPU, with the
same settings and bootstrap seed the evaluation uses.

Before writing, the outcomes are checked against the run's committed rows: the number of
probes and every finding's sensitivity and specificity must be the committed ones, so rows
from other outcomes are never mixed in. Only the new rows are written (upserted); no committed
number changes.

Example (Kaggle CPU notebook, the RC-E0a version's output attached):
    OUT=$(find /kaggle/input -path '*outputs/E0/nobudget/seed0/outcomes.npz' | head -n 1)
    python scripts/rescore_probes.py --exp E0 --seed 0 --outcomes "$OUT"
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.cli import (  # noqa: E402
    INPUT_CHECK_EXIT_CODE,
    build_parser,
    main_guard,
    resolve_config,
    run_stage,
)
from medcompress.config import ConfigError, RunContext  # noqa: E402

NEW_PREFIXES = ("auroc:", "yes_rate:", "n_positives:", "n_negatives:")
"""The rows this script writes (the pre-registered secondary metrics of 2026-11-02)."""
TOLERANCE = 1e-9


def committed_values(path: Path) -> dict[str, float]:
    """``{metric: value}`` of a committed results file (empty if it does not exist)."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {r["metric"]: float(r["value"]) for r in csv.DictReader(handle) if r["value"]}


def rescore(config: dict, run: RunContext | None, outcomes: Path) -> None:
    """Recompute the new rows from ``outcomes`` and upsert them (see the module docstring)."""
    import numpy as np

    from medcompress.evaluation.probe_suites import ProbeOutcomes, clinical_metrics
    from medcompress.evaluation.runner import RUN_FILES_BY_SPLIT, resolve_settings, to_rows
    from medcompress.utils.io import upsert_results

    if run is None:
        raise ConfigError("--exp is required")
    settings = resolve_settings(config)
    if not Path(outcomes).is_file():
        raise ConfigError(f"{outcomes} is not there: attach the run's output (outcomes.npz)")
    with np.load(outcomes, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    out = ProbeOutcomes(
        sample_ids=tuple(arrays["probe_sample_ids"].astype(str)),
        findings=tuple(arrays["probe_findings"].astype(str)),
        truth=arrays["probe_truth"].astype(bool),
        yes=arrays["probe_yes"].astype(bool),
        margin=arrays["probe_margin"].astype(float),
        area=arrays["probe_area"].astype(float),
        has_mask=arrays["probe_has_mask"].astype(bool),
    )
    metrics = clinical_metrics(
        out, settings.findings, n_resamples=int(settings.n_resamples or 10_000),
        alpha=settings.alpha, seed=run.seed,
    )  # fmt: skip
    target = run.run_dir(config["paths"]["results_root"]) / RUN_FILES_BY_SPLIT[settings.split][0]
    committed = committed_values(target)
    checked = [m for m in metrics if m.name == "n_probes_scored"
               or m.name.startswith(("sensitivity:", "specificity:"))]  # fmt: skip
    for metric in checked:
        if metric.name not in committed:
            raise ConfigError(f"{target} has no {metric.name} row: is it the run's results file?")
        same = math.isclose(committed[metric.name], metric.value, abs_tol=TOLERANCE)
        if not same:
            raise ConfigError(
                f"{outcomes} does not belong to the committed run: {metric.name} is "
                f"{metric.value} here but {committed[metric.name]} in {target}"
            )
    new = [m for m in metrics if m.name.startswith(NEW_PREFIXES)]
    rows = to_rows(new, run.experiment_tag, run.budget or "", run.seed)
    upsert_results(rows, target)
    print(f"checked {len(checked)} committed rows against {outcomes}: identical")
    for metric in new:
        ci = "" if metric.ci_low is None else f" [{metric.ci_low:.3f}, {metric.ci_high:.3f}]"
        print(f"  {metric.name} = {metric.value:.3f}{ci}")
    print(f"wrote {len(rows)} rows to {target} (split {settings.split})")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--outcomes", type=Path, required=True, help="the run's outcomes.npz")
    args = parser.parse_args(argv)
    if args.exp is None:
        parser.error("--exp is required")
    run = RunContext(args.exp, seed=args.seed, budget=args.budget, variant=args.variant)
    config = resolve_config(args)
    code = run_stage("rescore_probes", rescore, config=config, run=run, outcomes=args.outcomes)
    return INPUT_CHECK_EXIT_CODE if code == 4 and not args.outcomes.is_file() else code


if __name__ == "__main__":
    main_guard(main)
