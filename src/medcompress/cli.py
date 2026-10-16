"""Shared command-line helpers for the thin entry points in ``scripts/``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from medcompress.config import (
    ConfigError,
    RunContext,
    experiment_config_path,
    find_todos,
    load_config,
)
from medcompress.registry import BUDGET_IDS, REGISTRY, get_experiment, validate_run

NOT_IMPLEMENTED_EXIT_CODE = 3
"""Exit code used when a stage is still a stub, so pipelines never mistake it for success."""

CONFIG_INCOMPLETE_EXIT_CODE = 4
"""Exit code used when a stage is implemented but a setting it needs is still ``TODO``."""

INPUT_CHECK_EXIT_CODE = 5
"""Exit code used when an attached dataset (``--data-root``, ``--from``) fails its check."""


def build_parser(description: str, *, experiment_args: bool = True) -> argparse.ArgumentParser:
    """Create an argument parser with the options shared by all scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Config YAML. Defaults to configs/experiments/<exp>.yaml when --exp is given.",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. -o paths.data_root=/kaggle/input/x (repeatable).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="The attached shared-data dataset of RC-D1 (e.g. /kaggle/input/<name>); its "
        "manifest is verified before anything runs.",
    )
    parser.add_argument(
        "--from",
        dest="handoffs",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="An attached hand-off dataset (teacher cache, adapter, pruned or quantized model, "
        "outcomes); verified, then linked into the path roots (repeatable).",
    )
    if experiment_args:
        parser.add_argument("--exp", choices=list(REGISTRY), help="Experiment ID (E0-E9, A1-A6).")
        parser.add_argument("--budget", choices=list(BUDGET_IDS), default=None)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--variant", default=None, help="Experiment variant, e.g. w4.")
    return parser


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load the config named by ``--config`` or implied by ``--exp`` and apply overrides."""
    path = args.config
    if path is None and getattr(args, "exp", None):
        path = experiment_config_path(args.exp)
    if path is None:
        path = Path("base.yaml")
    config = load_config(path, args.override)
    attach_inputs(config, getattr(args, "data_root", None), getattr(args, "handoffs", ()))
    return config


def attach_inputs(config: dict[str, Any], data_root: Path | None, handoffs: Any = ()) -> None:
    """Verify the attached datasets and point the run at them; exit with code 5 if one fails.

    ``data_root`` is the shared data of RC-D1 (``paths.data_root`` and ``private_root`` are
    set to it); each of ``handoffs`` is the output of an earlier run, linked into the path
    roots. Every file's SHA-256 is checked against the dataset's manifest first.
    """
    from medcompress.utils.bundles import BundleError, attach_data, attach_handoff

    try:
        if data_root is not None:
            bundle = attach_data(config, Path(data_root))
            print(
                f"[inputs] shared data verified: {bundle.root} (commit {bundle.meta.get('commit')})"
            )
        for path in handoffs or ():
            bundle = attach_handoff(config, Path(path))
            print(f"[inputs] {bundle.label} verified and linked: {bundle.root}")
    except BundleError as exc:
        print(f"[inputs] check failed: {exc}", file=sys.stderr)
        sys.exit(INPUT_CHECK_EXIT_CODE)


def run_context(
    args: argparse.Namespace, *, require_budget: bool = True, require_variant: bool = True
) -> RunContext | None:
    """Validate ``--exp/--budget/--variant`` against the registry and build a RunContext."""
    if not getattr(args, "exp", None):
        return None
    validate_run(
        args.exp,
        args.budget,
        args.variant,
        require_budget=require_budget,
        require_variant=require_variant,
    )
    return RunContext(args.exp, seed=args.seed, budget=args.budget, variant=args.variant)


def parse_stage_args(
    parser: argparse.ArgumentParser,
    argv: list[str] | None,
    *,
    script: str,
    default_exp: str | None = None,
    require_budget: bool = True,
) -> tuple[argparse.Namespace, RunContext]:
    """Parse and validate arguments for a stage script.

    Checks that the experiment exists, that the budget/variant/seed are valid for
    it, and that ``script`` (e.g. ``"scripts/03_quantize.py"``) is one of the
    experiment's registered scripts. Invalid input exits via ``parser.error``.
    """
    args = parser.parse_args(argv)
    if args.exp is None:
        if default_exp is None:
            parser.error("--exp is required")
        args.exp = default_exp
    spec = get_experiment(args.exp)
    if script not in spec.scripts:
        users = [eid for eid, s in REGISTRY.items() if script in s.scripts]
        parser.error(f"{script} is not part of {args.exp}. Used by: {', '.join(users)}")
    try:
        run = run_context(args, require_budget=require_budget)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc.args[0] if exc.args else exc))
    assert run is not None
    return args, run


def stage_functions() -> dict[str, Callable[..., Any]]:
    """Map each stage script to the library function it calls (imported lazily)."""
    from medcompress.compression.distillation.offline_kd import run_student_training
    from medcompress.compression.pruning.stage import score_units_for_experiment
    from medcompress.compression.quantization.ptq import run_quantization
    from medcompress.compression.recovery import run_prune_and_recover
    from medcompress.evaluation.efficiency import run_efficiency_benchmark
    from medcompress.evaluation.image_reliance import check_teacher_image_reliance
    from medcompress.evaluation.runner import run_evaluation
    from medcompress.models.teacher_cache import run_teacher_caching
    from medcompress.registry import (
        BENCHMARK,
        CACHE_TEACHER,
        CHECK_TEACHER,
        EVALUATE,
        PRUNE_RECOVER,
        QUANTIZE,
        SCORE_UNITS,
        TRAIN_STUDENT,
    )

    return {
        CHECK_TEACHER: check_teacher_image_reliance,
        CACHE_TEACHER: run_teacher_caching,
        TRAIN_STUDENT: run_student_training,
        QUANTIZE: run_quantization,
        SCORE_UNITS: score_units_for_experiment,
        PRUNE_RECOVER: run_prune_and_recover,
        EVALUATE: run_evaluation,
        BENCHMARK: run_efficiency_benchmark,
    }


def report_todos(config: dict[str, Any], limit: int = 10) -> None:
    """Print how many config values are still ``TODO`` (first ``limit`` paths)."""
    todos = find_todos(config)
    if not todos:
        return
    print(f"{len(todos)} config value(s) still TODO, e.g.:")
    for path in todos[:limit]:
        print(f"  - {path}")


COMPRESSION_ARMS: tuple[str, ...] = (
    "E3", "E4", "E5", "E6", "E7", "E8", "E9", "A1", "A2", "A3", "A4", "A5", "A6",
)  # fmt: skip
"""Experiments that compress the student or the teacher: they run only after the freeze."""


def require_frozen_preregistration(config: Mapping[str, Any]) -> None:
    """Raise :class:`ConfigError` while ``preregistration.frozen_on`` is still open.

    Every compression run (E3-E9, A1-A6) comes after the pre-registration freeze
    (``docs/FINAL_GOAL.md`` section 4), so nothing that could be tuned to a result is still
    open when the first result exists.
    """
    path = (config.get("components") or {}).get("finding_weights")
    if path is None:
        raise ConfigError("components.finding_weights is missing (the pre-registration table)")
    frozen = (load_config(path).get("preregistration") or {}).get("frozen_on")
    if frozen is None or find_todos({"v": frozen}):
        raise ConfigError(
            "preregistration.frozen_on is still open: compression runs (E3-E9, A1-A6) start "
            "only after the pre-registration freeze"
        )


def run_stage(stage: str, func: Callable[..., Any], /, **kwargs: Any) -> int:
    """Call a stage function and turn expected stops into clean messages.

    Returns 0 on success, :data:`NOT_IMPLEMENTED_EXIT_CODE` if the stage is a stub and
    :data:`CONFIG_INCOMPLETE_EXIT_CODE` if a setting it needs is still open. A stage of a
    compression arm (:data:`COMPRESSION_ARMS`) first checks that the pre-registration is
    frozen (:func:`require_frozen_preregistration`).
    """
    try:
        run, config = kwargs.get("run"), kwargs.get("config")
        if config is not None and getattr(run, "experiment_id", None) in COMPRESSION_ARMS:
            require_frozen_preregistration(config)
        func(**kwargs)
    except NotImplementedError as exc:
        print(f"[{stage}] not implemented yet: {exc}")
        return NOT_IMPLEMENTED_EXIT_CODE
    except ConfigError as exc:
        print(f"[{stage}] configuration incomplete: {exc}")
        return CONFIG_INCOMPLETE_EXIT_CODE
    return 0


def main_guard(main: Callable[[], int]) -> None:
    """Run ``main`` and exit with its return code."""
    sys.exit(main())
