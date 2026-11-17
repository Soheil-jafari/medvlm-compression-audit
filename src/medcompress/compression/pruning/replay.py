"""Offline replay of the removal sets (amendment of 2026-11-17; CPU only, no model).

From the committed ``unit_scores.csv`` files this rebuilds the units, runs the same budget
search as stage 05 (:func:`medcompress.compression.recovery.search_budget`) with an
**estimated** footprint in place of the measured one, and reports what each arm would remove
at B1, B2 and B3: the number of units of each type, k for the ``uniform_mlp`` allocation, and
each decoder layer's share.

The estimate: E2's measured peak memory (RC-B0) minus 2 bytes for every removed parameter
(fp16 weights). It leaves out the activations that a narrower MLP no longer holds, so the real
search, which measures, can land on a slightly different k or prefix. It is an estimate for
planning and for the shape of each arm, not a result.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medcompress.compression.pruning.units import PrunableUnit, UnitType

BYTES_PER_PARAM = 2
"""fp16 weights: the estimate's bytes per removed parameter."""


def units_from_scores(
    path: Path | str, channel_group_size: int = 128
) -> tuple[list[PrunableUnit], dict[str, float]]:
    """The units and scores of a ``unit_scores.csv`` (the same unit ids, types, layers and
    sizes that stage 04 enumerated from the model)."""
    units: list[PrunableUnit] = []
    scores: dict[str, float] = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            kind = UnitType(row["unit_type"])
            index = None if row["index"] == "" else int(row["index"])
            start = count = None
            if kind is UnitType.MLP_CHANNEL_GROUP:
                assert index is not None
                start, count = index * channel_group_size, channel_group_size
            units.append(PrunableUnit(row["unit_id"], kind, int(row["layer_index"]), index,
                                      int(row["num_params"]), channel_start=start,
                                      channel_count=count))  # fmt: skip
            scores[row["unit_id"]] = float(row["score"])
    return units, scores


def layer_sizes_from_units(units: Sequence[PrunableUnit]) -> dict[int, int]:
    """``{layer: parameters}`` from the whole-layer units (every layer of the decoder)."""
    sizes = {u.layer_index: u.num_params for u in units if u.unit_type is UnitType.DECODER_LAYER}
    if not sizes:
        raise ValueError("the score file has no decoder_layer units: layer sizes are unknown")
    return sizes


def parameter_measurer(reference_peak: float) -> Callable[[Sequence[PrunableUnit]], float]:
    """The estimated footprint after a removal: the reference peak minus 2 bytes a parameter.

    Units inside a removed layer are not double-counted: :func:`removal_set` already lets a
    layer absorb its inner units, and the uniform allocation never removes a layer."""

    def measure(removal: Sequence[PrunableUnit]) -> float:
        return float(reference_peak) - BYTES_PER_PARAM * sum(u.num_params for u in removal)

    return measure


def measured_drop_per_parameter(run_dir: Path | str) -> tuple[float, float, int] | None:
    """For a committed pruning run: ``(measured drop of the peak, 2 x removed parameters, removed
    parameters)``, from its ``budget_search.csv`` (the unpruned and the accepted footprint) and
    ``metrics.csv`` (parameters before and after). None when a file is missing."""
    folder = Path(run_dir)
    trace, metrics = folder / "budget_search.csv", folder / "metrics.csv"
    if not trace.is_file() or not metrics.is_file():
        return None
    with trace.open(encoding="utf-8", newline="") as handle:
        rows = [(int(r[0]), float(r[1])) for r in list(csv.reader(handle))[1:]]
    values: dict[str, float] = {}
    with metrics.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values[row["metric"]] = float(row["value"])
    unpruned = dict(rows).get(0)
    if unpruned is None or "budget_footprint" not in values or "parameters_before" not in values:
        return None
    removed = int(values["parameters_before"] - values["parameters_after"])
    return unpruned - values["budget_footprint"], float(BYTES_PER_PARAM * removed), removed


@dataclass(frozen=True)
class ReplayRow:
    """One arm at one budget."""

    arm: str
    allocation: str
    budget: str
    target: float
    footprint: float
    within_tolerance: bool
    uniform_k: int | None
    removed_layers: int
    removed_kv_groups: int
    removed_channel_groups: int
    removed_params: int
    removed_in_protected: int
    by_layer: tuple[tuple[int, int, int, float, float], ...]
    unit_ids: frozenset[str] = frozenset()
    groups_per_layer: int = 0
    k_nominal: int | None = None
    """``uniform_mlp``: the smallest k at or below the target itself (no band)."""
    k_band: tuple[int, int] | None = None
    """``uniform_mlp``: the smallest and largest k whose estimate lies within the band."""

    @property
    def relative_gap(self) -> float:
        return self.footprint / self.target - 1.0


def replay_arm(
    arm: str,
    units: Sequence[PrunableUnit],
    scores: Mapping[str, float],
    removal: Any,
    spec: Any,
    target: float,
    measure: Callable[[Sequence[PrunableUnit]], float],
) -> ReplayRow:
    """The removal set of one arm at one budget, by the search stage 05 runs."""
    from medcompress.compression.recovery import removal_by_layer, search_budget

    search, protected = search_budget(units, scores, removal, spec, target, measure)
    chosen = list(search.removal)
    k_nominal = k_band = None
    if removal.allocation == "uniform_mlp":
        from medcompress.budgets import uniform_groups, uniform_max_k, uniform_removal

        groups = uniform_groups(units, scores, protected, removal.score_normalisation)
        footprints = [measure(uniform_removal(groups, k)) for k in range(uniform_max_k(groups) + 1)]
        k_nominal = next((k for k, f in enumerate(footprints) if f <= target), None)
        inside = [k for k, f in enumerate(footprints) if abs(f - target) <= spec.tolerance * target]
        k_band = (min(inside), max(inside)) if inside else None
    kinds = {t: sum(1 for u in chosen if u.unit_type is t) for t in UnitType}
    return ReplayRow(
        arm=arm,
        allocation=removal.allocation,
        budget=spec.id,
        target=search.target,
        footprint=search.footprint,
        within_tolerance=bool(search.within_tolerance),
        uniform_k=search.count if removal.allocation == "uniform_mlp" else None,
        removed_layers=kinds[UnitType.DECODER_LAYER],
        removed_kv_groups=kinds[UnitType.ATTENTION_HEAD],
        removed_channel_groups=kinds[UnitType.MLP_CHANNEL_GROUP],
        removed_params=sum(u.num_params for u in chosen),
        removed_in_protected=sum(1 for u in chosen if u.layer_index in set(protected)),
        by_layer=tuple(removal_by_layer(layer_sizes_from_units(units), chosen)),
        unit_ids=frozenset(u.unit_id for u in chosen),
        k_nominal=k_nominal,
        k_band=k_band,
        groups_per_layer=max(
            sum(
                1
                for u in units
                if u.unit_type is UnitType.MLP_CHANNEL_GROUP and u.layer_index == layer
            )
            for layer in {u.layer_index for u in units}
        ),  # fmt: skip
    )
