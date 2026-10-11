"""Shared command-line helpers for the thin entry points in ``scripts/``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from medcompress.config import (
    RunContext,
    experiment_config_path,
    find_todos,
    load_config,
)
from medcompress.registry import BUDGET_IDS, REGISTRY, validate_run

NOT_IMPLEMENTED_EXIT_CODE = 3
"""Exit code used when a stage is still a stub, so pipelines never mistake it for success."""


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
    return load_config(path, args.override)


def run_context(args: argparse.Namespace) -> RunContext | None:
    """Validate ``--exp/--budget/--variant`` against the registry and build a RunContext."""
    if not getattr(args, "exp", None):
        return None
    validate_run(args.exp, args.budget, args.variant)
    return RunContext(args.exp, seed=args.seed, budget=args.budget, variant=args.variant)


def report_todos(config: dict[str, Any], limit: int = 10) -> None:
    """Print how many config values are still ``TODO`` (first ``limit`` paths)."""
    todos = find_todos(config)
    if not todos:
        return
    print(f"{len(todos)} config value(s) still TODO, e.g.:")
    for path in todos[:limit]:
        print(f"  - {path}")


def run_stage(stage: str, func: Callable[..., Any], /, **kwargs: Any) -> int:
    """Call a stage function and turn ``NotImplementedError`` into a clean message.

    Returns 0 on success and :data:`NOT_IMPLEMENTED_EXIT_CODE` if the stage is a stub.
    """
    try:
        func(**kwargs)
    except NotImplementedError as exc:
        print(f"[{stage}] not implemented yet: {exc}")
        return NOT_IMPLEMENTED_EXIT_CODE
    return 0


def main_guard(main: Callable[[], int]) -> None:
    """Run ``main`` and exit with its return code."""
    sys.exit(main())
