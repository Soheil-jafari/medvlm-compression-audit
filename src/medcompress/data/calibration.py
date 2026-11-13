"""The calibration set after the fp16 check (pre-registration section 10, amendment of
2026-11-13, approved by Soheil and Amir before any compression result).

RC-E3 seed 0 stopped on a NaN Hessian: 2 of its 128 calibration images give a non-finite vision
tower output in fp16 (the frozen Qwen2.5-VL vision encoder overflows on them; the pixel values
are finite). The rule: **a calibration image whose fp16 forward through the model gives any
non-finite value is replaced by the next eligible image in the same seeded order**, keeping the
nested subsets (A4) and the finding stratification
(:func:`~medcompress.data.splits.make_calibration_subset` with ``exclude``). It is applied once
per seed, before any criterion or quantization, identically for every arm that uses the
calibration set (the pruning criteria, GPTQ of E3 and E8, E11 / E12 when built; E4 trains on the
teacher cache and draws no calibration set). The model is the student (E2 of that seed) for the
student arms and the teacher for E9. The vision tower stays fp16.

``scripts/check_calibration.py`` runs the check and writes
``results/calibration/seed<k>_replaced.csv`` (rows ``seed, model, kind, value``: one
``checked_size`` row, the number of leading images of the cleaned order that were checked
finite, and one ``replaced`` row per replaced image, its representative sample id). Every stage
reads the cleaned set (:func:`calibration_set`) and refuses to run without the file
(``calibration_check.required`` in ``configs/base.yaml``).
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from medcompress.config import ConfigError

if TYPE_CHECKING:
    from medcompress.data.base import VQASample

FOLDER = "calibration"
MODELS: tuple[str, ...] = ("student", "teacher")


def replaced_path(results_root: Path | str, seed: int) -> Path:
    """``results/calibration/seed<k>_replaced.csv``."""
    return Path(results_root) / FOLDER / f"seed{seed}_replaced.csv"


def read_check(results_root: Path | str, seed: int, model: str) -> tuple[int, list[str]] | None:
    """``(checked size, replaced sample ids)`` of ``model`` for ``seed``, or None (no check)."""
    path = replaced_path(results_root, seed)
    if not path.is_file():
        return None
    checked, replaced = None, []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != model or int(row["seed"]) != seed:
                continue
            if row["kind"] == "checked_size":
                checked = int(row["value"])
            elif row["kind"] == "replaced":
                replaced.append(row["value"])
    return None if checked is None else (checked, replaced)


def write_check(
    results_root: Path | str, seed: int, model: str, checked: int, replaced: Sequence[str]
) -> Path:
    """Write ``model``'s rows for ``seed`` (the other model's rows are kept)."""
    from medcompress.utils.io import write_csv

    path = replaced_path(results_root, seed)
    rows: list[tuple[Any, ...]] = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [(r["seed"], r["model"], r["kind"], r["value"]) for r in csv.DictReader(handle)
                    if r["model"] != model]  # fmt: skip
    rows.append((seed, model, "checked_size", checked))
    rows += [(seed, model, "replaced", sample_id) for sample_id in replaced]
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_csv(path, ("seed", "model", "kind", "value"), rows)


def check_required(config: Mapping[str, Any]) -> bool:
    """True when the stages must read the cleaned set (every real config, ``base.yaml``)."""
    return bool((config.get("calibration_check") or {}).get("required"))


def require_check(config: Mapping[str, Any], seed: int, size: int, model: str = "student") -> None:
    """Stop before any model is loaded if the check of ``seed`` does not cover ``size`` images.

    Raises:
        ConfigError: no check yet, or one that checked fewer images than needed.
    """
    if not check_required(config):
        return
    found = read_check(config["paths"]["results_root"], seed, model)
    path = replaced_path(config["paths"]["results_root"], seed)
    if found is None:
        raise ConfigError(
            f"the fp16 calibration check of seed {seed} ({model}) is not committed ({path}): run "
            "scripts/check_calibration.py first (amendment of 2026-11-13)"
        )
    if found[0] < size:
        raise ConfigError(f"the calibration check of seed {seed} ({model}) covers {found[0]} "
                          f"images; this run needs {size}")  # fmt: skip


def calibration_set(
    config: Mapping[str, Any],
    train: Sequence[VQASample],
    size: int,
    seed: int,
    model: str = "student",
) -> list[VQASample]:
    """The calibration images of a run: the seeded, stratified, nested subset without the images
    the fp16 check replaced (unchanged when the config does not require the check: tests)."""
    from medcompress.data.splits import make_calibration_subset

    exclude: list[str] = []
    if check_required(config):
        require_check(config, seed, size, model)
        found = read_check(config["paths"]["results_root"], seed, model)
        assert found is not None
        exclude = found[1]
    return make_calibration_subset(train, size, seed=seed, exclude=exclude)


def find_replacements(
    train: Sequence[VQASample],
    size: int,
    seed: int,
    is_finite: Callable[[VQASample], bool],
    *,
    log: Callable[[str], None] = print,
) -> tuple[list[VQASample], list[str]]:
    """Apply the rule: check the leading images of the cleaned order one by one; a non-finite
    one is excluded and the order is rebuilt (the next eligible image moves up, the
    stratification is recomputed), until the first ``size`` images are all finite.

    Returns ``(the cleaned subset, the replaced sample ids in the order found)``. Each image is
    checked once (``is_finite`` is called at most once per image).
    """
    from medcompress.data.splits import make_calibration_subset

    replaced: list[str] = []
    finite: set[str] = set()
    while True:
        subset = make_calibration_subset(train, size, seed=seed, exclude=replaced)
        unchecked = [s for s in subset if s.sample_id not in finite]
        if not unchecked:
            return subset, replaced
        for sample in unchecked:
            if is_finite(sample):
                finite.add(sample.sample_id)
                continue
            replaced.append(sample.sample_id)
            log(f"calibration seed {seed}: {sample.sample_id} gives a non-finite value in fp16; "
                "replaced by the next eligible image")  # fmt: skip
            break  # rebuild the order without it


def forward_is_finite(model: Any, batch: Mapping[str, Any]) -> bool:
    """True when the model's fp16 forward on ``batch`` keeps every checked value finite: the
    vision tower's output, every decoder layer's output and the logits."""
    import torch

    module = getattr(model, "module", model)
    from medcompress.compression.pruning.surgery import find_decoder_stack

    flags: list[Any] = []

    from medcompress.models.vision_only import output_tensor

    def check(_m: Any, _args: Any, output: Any) -> None:
        value = output_tensor(output)  # the vision tower may return a model output object
        if value is not None and value.is_floating_point():
            flags.append(torch.isfinite(value).all())

    inner = getattr(module, "model", module)
    watched = [m for m in (getattr(inner, "visual", None),) if m is not None]
    watched += list(find_decoder_stack(module).layers)
    handles = [m.register_forward_hook(check) for m in watched]
    try:
        with torch.inference_mode():
            device = module.device
            out = module(**{k: v.to(device) for k, v in batch.items()}, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    flags.append(torch.isfinite(out.logits).all())
    return bool(torch.stack([f.to(flags[0].device) for f in flags]).all())
