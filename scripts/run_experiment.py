#!/usr/bin/env python
"""Run one experiment end to end: validate the ID, show the plan, run its stages.

The experiment ID, budget, variant and seed are validated against the registry
(src/medcompress/registry.py) and the experiment config. Stages run in order and
stop at the first stage that is not implemented yet.

Examples:
    python scripts/run_experiment.py --list
    python scripts/run_experiment.py --exp E7 --budget B2 --seed 0
    python scripts/run_experiment.py --exp E3 --variant w4 --seed 0 --plan-only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.cli import (  # noqa: E402
    build_parser,
    main_guard,
    report_todos,
    resolve_config,
    run_context,
    run_stage,
    stage_functions,
)
from medcompress.config import ExperimentConfig  # noqa: E402
from medcompress.registry import get_experiment, list_experiments  # noqa: E402


def print_registry() -> None:
    print(f"{'ID':<4} {'kind':<9} {'parent':<7} {'budgets':<10} {'variants':<28} title")
    for spec in list_experiments():
        print(
            f"{spec.id:<4} {spec.kind:<9} {spec.parent or '-':<7} "
            f"{','.join(spec.budgets) or '-':<10} {','.join(spec.variants) or '-':<28} {spec.title}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="List registered experiments.")
    parser.add_argument("--plan-only", action="store_true", help="Validate and print the plan.")
    args = parser.parse_args(argv)

    if args.list:
        print_registry()
        return 0
    if args.exp is None:
        parser.error("--exp is required (or use --list)")
    try:
        run = run_context(args)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc.args[0] if exc.args else exc))
    assert run is not None

    config = resolve_config(args)
    experiment = ExperimentConfig.from_dict(config["experiment"])
    if experiment.seeds and args.seed not in experiment.seeds:
        parser.error(f"seed {args.seed} is not in {args.exp} seeds {list(experiment.seeds)}")

    spec = get_experiment(args.exp)
    print(f"Experiment {spec.id}: {spec.title}")
    print(f"  run:      {run.experiment_tag} budget={run.budget or '-'} seed={run.seed}")
    print(f"  parent:   {spec.parent or 'pretrained checkpoint'}")
    print(f"  requires: {', '.join(spec.requires) or '-'}")
    print(f"  run dir:  {run.run_dir(config['paths']['results_root'])}")
    print("  stages:")
    for script in spec.scripts:
        print(f"    - {script}")
    report_todos(config)
    if args.plan_only:
        return 0

    functions = stage_functions()
    for script in spec.scripts:
        code = run_stage(Path(script).stem, functions[script], config=config, run=run)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    main_guard(main)
