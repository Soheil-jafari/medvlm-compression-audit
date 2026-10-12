"""Unit ablation, unit gates (first-order estimates) and probe objectives on the tiny model."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torchvision")
pytest.importorskip("tokenizers")

import numpy as np  # noqa: E402
import tiny_vlm  # noqa: E402
from eval_fixtures import write_dataset  # noqa: E402

from medcompress.compression.pruning import scoring  # noqa: E402
from medcompress.compression.pruning.apply import remove_units  # noqa: E402
from medcompress.compression.pruning.units import (  # noqa: E402
    PrunableUnit,
    UnitType,
    enumerate_units,
)
from medcompress.data.probes import FindingProbe  # noqa: E402
from medcompress.evaluation.image_reliance import ImageProbe  # noqa: E402

TYPES = [UnitType.DECODER_LAYER, UnitType.ATTENTION_HEAD, UnitType.MLP_CHANNEL_GROUP]
IMAGES = [tiny_vlm.tiny_image(i) for i in range(3)]
QUESTIONS = [tiny_vlm.PROBE] * 3


def _wrapper() -> object:
    return tiny_vlm.tiny_wrapper(seed=0)


def _units(wrapper: object, group: int = 16) -> list[PrunableUnit]:
    return enumerate_units(wrapper.module, TYPES, mlp_channel_group_size=group)


def _scores(wrapper: object) -> np.ndarray:
    return np.array([[s.logp_yes, s.logp_no] for s in wrapper.score_yes_no(IMAGES, QUESTIONS)])


def _one_of_each(units: list[PrunableUnit]) -> list[PrunableUnit]:
    return [next(u for u in units if u.unit_type is t and u.layer_index == 1) for t in TYPES]


# ------------------------------------------------------------------- ablation


def test_ablation_changes_the_scores_exactly_like_removing_the_unit() -> None:
    base = _scores(_wrapper())
    for unit in _one_of_each(_units(_wrapper())):
        wrapper = _wrapper()
        with scoring.ablate(wrapper, unit):
            ablated = _scores(wrapper)
        assert np.abs(ablated - base).max() > 1e-6, unit.unit_id  # it does something
        removed = _wrapper()
        remove_units(removed, [unit])
        assert np.allclose(ablated, _scores(removed), atol=1e-4), unit.unit_id
        assert np.allclose(_scores(wrapper), base)  # the hook is gone after the context


def test_ablation_needs_the_ranges_of_its_unit() -> None:
    wrapper = _wrapper()
    bad = PrunableUnit("lm.layer0.mlp_group0", UnitType.MLP_CHANNEL_GROUP, 0, 0, 1)
    with pytest.raises(ValueError, match="no channel range"), scoring.ablate(wrapper, bad):
        pass
    vision = PrunableUnit("v.block0", UnitType.VISION_BLOCK, 0, None, 1, component="vision")
    with pytest.raises(NotImplementedError), scoring.ablate(wrapper, vision):
        pass


# ---------------------------------------------------------------------- gates


def test_a_zero_gate_switches_the_unit_off_like_ablation() -> None:
    units = _units(_wrapper())
    for unit in _one_of_each(units):
        wrapper = _wrapper()
        with scoring.ablate(wrapper, unit):
            ablated = _scores(wrapper)
        gated = _wrapper()
        gates = scoring.UnitGates(gated, units)
        gates.set_value(unit.unit_id, 0.0)
        with gates.installed():
            assert np.allclose(_scores(gated), ablated, atol=1e-5), unit.unit_id
        assert np.allclose(_scores(gated), _scores(_wrapper()))  # hooks removed again


def test_gates_at_one_change_nothing() -> None:
    wrapper = _wrapper()
    base = _scores(wrapper)
    gates = scoring.UnitGates(wrapper, _units(wrapper))
    with gates.installed():
        assert np.allclose(_scores(wrapper), base, atol=1e-6)


def _objective(wrapper: object) -> torch.Tensor:
    yes, no = wrapper.yes_no_logprobs(IMAGES, QUESTIONS, grad=True)
    return (yes * 0.7 + no * 0.3).sum()


def test_gate_gradients_match_finite_differences() -> None:
    wrapper = _wrapper()
    units = _units(wrapper)
    grads = scoring.accumulate_gate_gradients(wrapper, units, [lambda: _objective(wrapper)])
    assert set(grads) == {u.unit_id for u in units}
    gates = scoring.UnitGates(wrapper, units)
    eps = 1e-2
    with gates.installed(), torch.no_grad():
        for unit in _one_of_each(units):
            values = []
            for sign in (1, -1):
                gates.set_value(unit.unit_id, 1.0 + sign * eps)
                values.append(float(_objective(wrapper)))
            gates.set_value(unit.unit_id, 1.0)
            numeric = (values[0] - values[1]) / (2 * eps)
            assert grads[unit.unit_id] == pytest.approx(numeric, rel=2e-2, abs=2e-3), unit.unit_id


def test_loss_scale_does_not_change_the_gradients() -> None:
    wrapper = _wrapper()
    units = _units(wrapper)
    plain = scoring.accumulate_gate_gradients(
        wrapper, units, [lambda: _objective(wrapper)], loss_scale=1.0
    )
    scaled = scoring.accumulate_gate_gradients(
        wrapper, units, [lambda: _objective(wrapper)], loss_scale=64.0
    )
    for key, value in plain.items():
        assert scaled[key] == pytest.approx(value, rel=1e-4, abs=1e-7)
    twice = scoring.accumulate_gate_gradients(
        wrapper, units, [lambda: _objective(wrapper)] * 2, loss_scale=1.0
    )
    assert twice[units[0].unit_id] == pytest.approx(2 * plain[units[0].unit_id], rel=1e-4)


def test_gate_gradients_refuse_unusable_objectives() -> None:
    wrapper = _wrapper()
    units = _units(wrapper)
    with pytest.raises(ValueError, match="not finite"):
        scoring.accumulate_gate_gradients(
            wrapper, units, [lambda: _objective(wrapper) * float("inf")], loss_scale=1.0
        )
    with pytest.raises(ValueError, match="every gate gradient is zero"):
        scoring.accumulate_gate_gradients(
            wrapper, units, [lambda: _objective(wrapper) * 0.0], loss_scale=1.0
        )


def test_weights_are_not_left_with_gradients() -> None:
    wrapper = _wrapper()
    flags = [p.requires_grad for p in wrapper.module.parameters()]
    scoring.accumulate_gate_gradients(
        wrapper, _units(wrapper), [lambda: _objective(wrapper)], loss_scale=1.0
    )
    assert [p.requires_grad for p in wrapper.module.parameters()] == flags
    assert all(p.grad is None for p in wrapper.module.parameters())


def test_the_first_order_estimate_ranks_small_units_like_exact_ablation() -> None:
    """For MLP channel groups the estimate follows the exact effect of removal."""
    wrapper = tiny_vlm.tiny_wrapper(seed=3)
    units = [u for u in _units(wrapper, 8) if u.unit_type is UnitType.MLP_CHANNEL_GROUP]
    grads = scoring.accumulate_gate_gradients(wrapper, units, [lambda: _objective(wrapper)])
    base = float(_objective(wrapper).detach())
    exact = []
    for unit in units:
        with scoring.ablate(wrapper, unit), torch.no_grad():
            exact.append(base - float(_objective(wrapper)))
    estimate = [grads[u.unit_id] for u in units]
    assert np.corrcoef(exact, estimate)[0, 1] > 0.9


def test_gate_groups_must_tile_the_mlp_width() -> None:
    wrapper = _wrapper()
    units = [u for u in _units(wrapper) if u.unit_type is UnitType.MLP_CHANNEL_GROUP]
    with pytest.raises(ValueError, match="must tile"):
        scoring.UnitGates(wrapper, [u for u in units if u.layer_index != 0 or u.index != 0])
    only_first = [u for u in units if u.layer_index == 0][:2]
    with pytest.raises(ValueError, match="cover 32 of 48"):
        scoring.UnitGates(wrapper, only_first)


# --------------------------------------------------------------------- probes


def _image_probes() -> list[ImageProbe]:
    out = []
    for i in range(6):
        for finding in ("polyp", "bleeding"):
            present = (i + (finding == "bleeding")) % 3 == 0
            out.append(ImageProbe(f"s{i}", f"images/{i}.png", finding, f"{finding}?", present))
    return out


def test_balancing_keeps_all_positives_and_a_seeded_share_of_negatives() -> None:
    probes = _image_probes()
    assert scoring.balance_probes(probes, None, seed=0) == probes
    kept = scoring.balance_probes(probes, 1, seed=0)
    for finding in ("polyp", "bleeding"):
        positives = [p for p in probes if p.finding == finding and p.present]
        negatives = [p for p in kept if p.finding == finding and not p.present]
        assert all(p in kept for p in positives)
        assert len(negatives) == len(positives)
    assert scoring.balance_probes(probes, 1, seed=0) == kept
    assert scoring.balance_probes(probes, 1, seed=1) != kept
    only_positive = scoring.balance_probes(probes, 0, seed=0)
    assert only_positive and all(p.present for p in only_positive)
    assert len(scoring.balance_probes(probes, 100, seed=0)) == len(probes)
    with pytest.raises(ValueError, match="must not be negative"):
        scoring.balance_probes(probes, -1, seed=0)


def test_probe_objective_spreads_each_finding_weight_over_its_probes() -> None:
    probes = scoring.to_finding_probes(_image_probes())
    assert probes[0] == FindingProbe("s0", "images/0.png", "polyp", "polyp?", "yes")
    objective = scoring.ProbeObjective.build(probes, {"polyp": 3.0, "bleeding": 0.5})
    for finding, weight in (("polyp", 3.0), ("bleeding", 0.5)):
        total = sum(
            c
            for p, c in zip(objective.probes, objective.coefficients, strict=True)
            if p.finding == finding
        )
        assert total == pytest.approx(weight)
    with pytest.raises(ValueError, match="No weight for finding"):
        scoring.ProbeObjective.build(probes, {"polyp": 1.0})


def test_probe_steps_add_up_to_the_weighted_log_probability_of_the_right_answer(
    tmp_path: Path,
) -> None:
    write_dataset(tmp_path)
    probes = scoring.to_finding_probes(
        [
            ImageProbe(f"s{i}", f"images/{i}.png", "polyp", tiny_vlm.PROBE, i % 2 == 0)
            for i in range(5)
        ]
    )
    objective = scoring.ProbeObjective.build(probes, {"polyp": 2.0})
    wrapper = _wrapper()
    steps = scoring.probe_objective_steps(wrapper, objective, tmp_path, batch_size=2)
    assert len(steps) == 3
    total = sum(float(step()) for step in steps)
    scores = scoring.score_probe_images(wrapper, probes, tmp_path)
    expected = sum(
        c * (s.logp_yes if p.answer == "yes" else s.logp_no)
        for p, c, s in zip(probes, objective.coefficients, scores, strict=True)
    )
    assert total == pytest.approx(expected, rel=1e-4)
    assert math.isfinite(total)
    assert scoring.calibration_images([]) == []
