"""Which files of a run another account needs, and where they live (hand-off datasets).

A run that a later run builds on writes its output under a path root (``paths`` in
``configs/base.yaml``); :func:`handoff_items` names it, by kind:

* ``teacher_cache``: ``cache_root/teacher`` (RC-E0b, stage 01);
* ``adapter``: ``checkpoint_root/<tag>/seed<k>/adapter`` (RC-E1, RC-E2, stage 02);
* ``pruned``: ``checkpoint_root/<tag>/<budget>/seed<k>/pruned`` (E5-E8, A1-A6, stage 05);
* ``quantized``: ``checkpoint_root/<tag>/<budget|nobudget>/seed<k>/quantized`` (E3, E4, E8,
  E9);
* ``outcomes``: ``output_root/<tag>/<budget|nobudget>/seed<k>`` (stage 06, any run);
* ``resume``: ``checkpoint_root/<tag>/seed<k>/resume`` (a training run that stopped).

The pruning scores (``unit_scores.csv``), the budget reference (RC-B0) and every metric are
results files: they reach the other accounts through git, not through a hand-off.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from medcompress.config import ConfigError, RunContext
from medcompress.utils.bundles import HANDOFF_KINDS

ADAPTER_EXPERIMENTS: tuple[str, ...] = ("E1", "E2")
QUANTIZED_EXPERIMENTS: tuple[str, ...] = ("E3", "E4", "E8", "E9")
PRUNED_EXPERIMENTS: tuple[str, ...] = (
    "E5", "E6", "E7", "E8", "A1", "A2", "A3", "A4", "A5", "A6",
)  # fmt: skip
RESUME_SUBDIR = "resume"


def _relative(path: Path, root: Any) -> str:
    return Path(path).relative_to(Path(str(root))).as_posix()


def adapter_relpath(run: RunContext) -> str:
    """``E2/seed0/adapter``: where stage 02 saves the adapter (``offline_kd``)."""
    return f"{run.experiment_tag.replace(':', '/')}/seed{run.seed}/adapter"


def resume_relpath(run: RunContext) -> str:
    """``E2/seed0/resume``: where a training run keeps its resume state."""
    return f"{run.experiment_tag.replace(':', '/')}/seed{run.seed}/{RESUME_SUBDIR}"


def handoff_items(
    kind: str, config: Mapping[str, Any], run: RunContext | None
) -> list[tuple[str, str]]:
    """The ``(path root, relative path)`` items of a hand-off of ``kind`` for ``run``."""
    if kind not in HANDOFF_KINDS:
        raise ConfigError(f"unknown hand-off kind {kind!r}; use one of {HANDOFF_KINDS}")
    if kind == "teacher_cache":
        return [("cache_root", "teacher")]
    if run is None:
        raise ConfigError(f"a {kind} hand-off belongs to a run: give --exp and --seed")
    paths = config["paths"]
    exp = run.experiment_id
    if kind == "adapter":
        if exp not in ADAPTER_EXPERIMENTS:
            raise ConfigError(f"{exp} saves no adapter of its own; adapters: {ADAPTER_EXPERIMENTS}")
        return [("checkpoint_root", adapter_relpath(run))]
    if kind == "resume":
        return [("checkpoint_root", resume_relpath(run))]
    if kind == "pruned":
        from medcompress.compression.recovery import pruned_dir

        if exp not in PRUNED_EXPERIMENTS:
            raise ConfigError(f"{exp} is not a pruning arm: {PRUNED_EXPERIMENTS}")
        return [("checkpoint_root", _relative(pruned_dir(config, run), paths["checkpoint_root"]))]
    if kind == "quantized":
        from medcompress.compression.quantization.ptq import quantized_dir

        if exp not in QUANTIZED_EXPERIMENTS:
            raise ConfigError(f"{exp} writes no quantized model: {QUANTIZED_EXPERIMENTS}")
        target = quantized_dir(config, run)
        return [("checkpoint_root", _relative(target, paths["checkpoint_root"]))]
    return [("output_root", run.run_dir("").as_posix())]  # outcomes


def run_from_meta(meta: Mapping[str, Any]) -> RunContext | None:
    """The run a hand-off came from (``None`` for the teacher cache)."""
    if not meta.get("experiment"):
        return None
    return RunContext(
        str(meta["experiment"]),
        seed=int(meta["seed"]) if meta.get("seed") is not None else 0,
        budget=meta.get("budget") or None,
        variant=meta.get("variant") or None,
    )
