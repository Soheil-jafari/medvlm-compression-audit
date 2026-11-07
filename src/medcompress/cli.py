"""Shared command-line helpers for the thin entry points in ``scripts/``."""

from __future__ import annotations

import argparse
import os
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
    warn_roots_inside_repo(config)
    attach_inputs(config, getattr(args, "data_root", None), getattr(args, "handoffs", ()))
    return config


ROOTS_OUTSIDE_THE_CLONE: tuple[tuple[str, str], ...] = (
    ("output_root", "MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs"),
    ("checkpoint_root", "MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints"),
    ("cache_root", "MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache"),
)
"""The path roots whose files must outlive the clone (outcomes, adapters, the teacher cache)."""


def records_environment() -> bool:
    """Whether stages write ``environment.json``: on Kaggle, or with
    ``$MEDCOMPRESS_RECORD_ENVIRONMENT`` = 1 (the rented GPU of RC-T1); 0 turns it off."""
    flag = os.environ.get("MEDCOMPRESS_RECORD_ENVIRONMENT")
    if flag is not None:
        return flag == "1"
    return on_kaggle()


def note_environment(config: Mapping[str, Any], run: Any, stage: str) -> None:
    """Print the torch build, CUDA, driver, transformers and peft versions and add them to
    ``results/<run dir>/environment.json`` (``utils/environment.py``)."""
    if not records_environment() or not hasattr(run, "run_dir"):
        return
    from medcompress.utils.environment import environment_record, record_environment, summary_line

    record = environment_record()
    print(f"[{stage}] {summary_line(record)}")
    results_root = (config.get("paths") or {}).get("results_root")
    if results_root is not None:
        try:
            record_environment(run.run_dir(Path(str(results_root))), stage, record)
        except OSError as exc:  # never stop a run for its record
            print(f"[{stage}] could not write environment.json: {exc}", file=sys.stderr)


def on_kaggle() -> bool:
    """True in a Kaggle notebook (``$MEDCOMPRESS_ON_KAGGLE`` = 1 or 0 overrides, for tests)."""
    flag = os.environ.get("MEDCOMPRESS_ON_KAGGLE")
    if flag is not None:
        return flag == "1"
    return Path("/kaggle/working").is_dir()


def roots_inside_repo(config: Mapping[str, Any]) -> list[tuple[str, Path, str]]:
    """``(root, resolved path, export to set)`` of every root that lies inside the clone."""
    from medcompress.config import repo_root

    repo = repo_root().resolve()
    found = []
    for key, export in ROOTS_OUTSIDE_THE_CLONE:
        value = (config.get("paths") or {}).get(key)
        if value is None:
            continue
        path = Path(str(value)).expanduser()
        path = (path if path.is_absolute() else Path.cwd() / path).resolve()
        if path == repo or path.is_relative_to(repo):
            found.append((key, path, export))
    return found


def warn_roots_inside_repo(config: Mapping[str, Any]) -> list[str]:
    """On Kaggle, warn (stderr) for each path root inside the clone; return the warnings.

    A cell that removes the clone at the end would remove what was written there (RC-E0a's
    ``outcomes.npz`` was lost that way, 2026-11-04). Nothing is changed: the run goes on.
    """
    if not on_kaggle():
        return []
    warnings = [
        f"WARNING: paths.{key} is {path}, inside the repository clone: removing the clone "
        f"removes it. Set `export {export}` before the scripts run."
        for key, path, export in roots_inside_repo(config)
    ]
    for line in warnings:
        print(line, file=sys.stderr)
    return warnings


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
        if config is not None and run is not None:
            note_environment(config, run, stage)
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
