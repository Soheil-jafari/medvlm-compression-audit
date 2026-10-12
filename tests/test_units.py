"""Unit identifiers, GQA grouping and removal-set validation (no torch needed; runs in CI)."""

from __future__ import annotations

import pytest

from medcompress.compression.pruning.units import (
    PrunableUnit,
    UnitType,
    gqa_head_groups,
    unit_id,
    validate_removal_set,
)


def test_gqa_groups() -> None:
    assert gqa_head_groups(8, 2) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    with pytest.raises(ValueError):
        gqa_head_groups(6, 4)


def _units(n_layers: int = 3, groups: int = 2) -> list[PrunableUnit]:
    out = []
    for i in range(n_layers):
        out.append(
            PrunableUnit(unit_id(i, UnitType.DECODER_LAYER), UnitType.DECODER_LAYER, i, None, 10)
        )
        for g in range(groups):
            out.append(
                PrunableUnit(
                    unit_id(i, UnitType.ATTENTION_HEAD, g), UnitType.ATTENTION_HEAD, i, g, 2
                )
            )
            out.append(
                PrunableUnit(
                    unit_id(i, UnitType.MLP_CHANNEL_GROUP, g),
                    UnitType.MLP_CHANNEL_GROUP,
                    i,
                    g,
                    3,
                    channel_start=g * 4,
                    channel_count=4,
                )
            )
    return out


def test_validate_removal_set_rules() -> None:
    units = _units()
    by_id = {u.unit_id: u for u in units}
    assert validate_removal_set(units, [by_id["lm.layer1"], by_id["lm.layer0.kv_group0"]]) == []
    problems = validate_removal_set(
        units,
        [
            by_id["lm.layer1"],
            by_id["lm.layer1.kv_group0"],
            by_id["lm.layer0.mlp_group0"],
            by_id["lm.layer0.mlp_group1"],
            by_id["lm.layer2"],
            by_id["lm.layer2"],
        ],
        protected_layers=[2],
    )
    text = "\n".join(problems)
    assert "inside removed layer 1" in text
    assert "layer 0 would lose every channel group" in text
    assert "removed twice" in text and "layer 2 is protected" in text
    all_layers = [u for u in units if u.unit_type is UnitType.DECODER_LAYER]
    assert "every decoder layer would be removed" in validate_removal_set(units, all_layers)
