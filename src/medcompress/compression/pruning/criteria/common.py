"""Pieces shared by the criteria: the weights of a unit, the removal order and score files.

The removal order is the same for every criterion, so criteria differ only in the scores
they assign. Scores are additive over the weights of a unit wherever the criterion allows
it (L1 weight norm, Wanda sum, Taylor term, effect of removal), which makes the optional
division by the unit's parameter count mean "importance per weight".
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medcompress.compression.pruning import surgery
from medcompress.compression.pruning.units import (
    PrunableUnit,
    UnitType,
    validate_removal_set,
)

SCORE_NORMALISATIONS: tuple[str, ...] = ("none", "per_parameter")
SCORE_COLUMNS: tuple[str, ...] = (
    "unit_id",
    "unit_type",
    "layer_index",
    "index",
    "num_params",
    "score",
)


@dataclass(frozen=True)
class WeightSlice:
    """The part of one weight matrix that is removed together with a unit.

    ``stat`` names the input activation that multiplies the columns of the matrix
    (``attn_in``, ``attn_out``, ``mlp_in`` or ``mlp_act``; None for norms) and ``columns``
    the range of that activation vector this slice reads.
    """

    weight: Any
    bias: Any | None
    stat: str | None
    columns: slice | None = None

    @property
    def numel(self) -> int:
        """Parameters in the slice (weight and bias)."""
        return int(self.weight.numel()) + (int(self.bias.numel()) if self.bias is not None else 0)


def _layer_of(model: Any, unit: PrunableUnit) -> tuple[Any, dict[str, int]]:
    module = getattr(model, "module", model)
    stack = surgery.find_decoder_stack(module)
    layer = stack.layers[unit.layer_index]
    return layer, surgery.layer_shape(layer, stack.config)


def unit_slices(model: Any, unit: PrunableUnit) -> list[WeightSlice]:
    """The weights deleted with ``unit`` (views, no copies)."""
    layer, shape = _layer_of(model, unit)
    attn, mlp = layer.self_attn, layer.mlp
    if unit.unit_type is UnitType.DECODER_LAYER:
        out = [
            WeightSlice(attn.q_proj.weight, attn.q_proj.bias, "attn_in"),
            WeightSlice(attn.k_proj.weight, attn.k_proj.bias, "attn_in"),
            WeightSlice(attn.v_proj.weight, attn.v_proj.bias, "attn_in"),
            WeightSlice(attn.o_proj.weight, attn.o_proj.bias, "attn_out"),
            WeightSlice(mlp.gate_proj.weight, mlp.gate_proj.bias, "mlp_in"),
            WeightSlice(mlp.up_proj.weight, mlp.up_proj.bias, "mlp_in"),
            WeightSlice(mlp.down_proj.weight, mlp.down_proj.bias, "mlp_act"),
        ]
        seen = {id(s.weight) for s in out} | {id(s.bias) for s in out if s.bias is not None}
        out += [
            WeightSlice(p, None, None) for p in layer.parameters() if id(p) not in seen
        ]  # norms and anything else in the layer
        return out
    if unit.unit_type is UnitType.ATTENTION_HEAD:
        g, d, r = int(unit.index or 0), shape["head_dim"], shape["group_ratio"]
        q = slice(g * r * d, (g + 1) * r * d)
        kv = slice(g * d, (g + 1) * d)
        return [
            WeightSlice(attn.q_proj.weight[q], _rows(attn.q_proj.bias, q), "attn_in"),
            WeightSlice(attn.k_proj.weight[kv], _rows(attn.k_proj.bias, kv), "attn_in"),
            WeightSlice(attn.v_proj.weight[kv], _rows(attn.v_proj.bias, kv), "attn_in"),
            WeightSlice(attn.o_proj.weight[:, q], None, "attn_out", q),
        ]
    if unit.unit_type is UnitType.MLP_CHANNEL_GROUP:
        start = int(unit.channel_start or 0)
        cols = slice(start, start + int(unit.channel_count or 0))
        return [
            WeightSlice(mlp.gate_proj.weight[cols], _rows(mlp.gate_proj.bias, cols), "mlp_in"),
            WeightSlice(mlp.up_proj.weight[cols], _rows(mlp.up_proj.bias, cols), "mlp_in"),
            WeightSlice(mlp.down_proj.weight[:, cols], None, "mlp_act", cols),
        ]
    raise NotImplementedError("vision-block weights are not wired")


def _rows(bias: Any | None, rows: slice) -> Any | None:
    return None if bias is None else bias[rows]


# ------------------------------------------------------------------ removal order


def normalised_scores(
    scores: Mapping[str, float], units: Sequence[PrunableUnit], mode: str
) -> dict[str, float]:
    """Scores, or scores divided by the number of parameters the unit holds."""
    if mode not in SCORE_NORMALISATIONS:
        raise ValueError(f"score normalisation must be one of {SCORE_NORMALISATIONS}")
    if mode == "none":
        return dict(scores)
    size = {u.unit_id: u.num_params for u in units}
    return {k: v / max(1, size[k]) for k, v in scores.items()}


def removal_order(
    scores: Mapping[str, float],
    units: Sequence[PrunableUnit],
    *,
    normalisation: str = "none",
    protected_layers: Sequence[int] = (),
) -> list[PrunableUnit]:
    """Units from least to most important. Whole layers in ``protected_layers`` never appear.

    Ties are broken by unit id and a NaN score is an error, so the order is deterministic.
    """
    known = {u.unit_id: u for u in units}
    missing = sorted(set(known) - set(scores))
    if missing:
        raise ValueError(f"no score for units: {missing[:3]}")
    values = normalised_scores({k: scores[k] for k in known}, units, normalisation)
    bad = sorted(k for k, v in values.items() if v != v)
    if bad:
        raise ValueError(f"NaN importance for units: {bad[:3]}")
    protected = set(protected_layers)
    ordered = sorted(known.values(), key=lambda u: (values[u.unit_id], u.unit_id))
    return [
        u
        for u in ordered
        if not (u.unit_type is UnitType.DECODER_LAYER and u.layer_index in protected)
    ]


def _with_unit(removed: list[PrunableUnit], unit: PrunableUnit) -> list[PrunableUnit]:
    """The removal set after adding ``unit``; a layer absorbs the units inside it."""
    if unit.unit_type is UnitType.DECODER_LAYER:
        kept = [
            r
            for r in removed
            if r.unit_type is UnitType.DECODER_LAYER or r.layer_index != unit.layer_index
        ]
        return [*kept, unit]
    return [*removed, unit]


def feasible_sequence(
    units: Sequence[PrunableUnit],
    ordered: Sequence[PrunableUnit],
    *,
    protected_layers: Sequence[int] = (),
) -> list[PrunableUnit]:
    """The units of ``ordered`` that can be removed one after another without breaking the model.

    A unit is skipped when adding it would violate a structural rule (the last head group
    or channel group of a layer, the last layer, a unit inside an already removed layer). A
    layer added after some of its inner units absorbs them. Every prefix of the result is
    a valid removal set once :func:`removal_set` has applied the absorption.
    """
    accepted: list[PrunableUnit] = []
    current: list[PrunableUnit] = []
    for unit in ordered:
        trial = _with_unit(current, unit)
        if validate_removal_set(units, trial, protected_layers=protected_layers):
            continue
        accepted.append(unit)
        current = trial
    return accepted


def removal_set(accepted: Sequence[PrunableUnit], k: int) -> list[PrunableUnit]:
    """The removal set after the first ``k`` accepted units (layers absorb their inner units)."""
    current: list[PrunableUnit] = []
    for unit in accepted[:k]:
        current = _with_unit(current, unit)
    return current


# ------------------------------------------------------------------- score files


def write_unit_scores(
    path: Path | str,
    units: Sequence[PrunableUnit],
    scores: Mapping[str, float],
) -> Path:
    """Write ``unit_scores.csv`` (one row per unit, the raw score of the criterion)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SCORE_COLUMNS)
        for u in units:
            writer.writerow(
                [
                    u.unit_id,
                    u.unit_type.value,
                    u.layer_index,
                    "" if u.index is None else u.index,
                    u.num_params,
                    repr(float(scores[u.unit_id])),
                ]
            )
    return target


def read_unit_scores(path: Path | str) -> dict[str, float]:
    """Read the scores back as ``{unit_id: score}``."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return {row["unit_id"]: float(row["score"]) for row in csv.DictReader(handle)}
