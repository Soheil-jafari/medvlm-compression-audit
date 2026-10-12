"""The pruning criteria and the removal order on the tiny model (scores, not model quality)."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torchvision")
pytest.importorskip("tokenizers")

import numpy as np  # noqa: E402
import tiny_vlm  # noqa: E402
import yaml  # noqa: E402
from eval_fixtures import eval_config, write_dataset  # noqa: E402

from medcompress.compression.pruning import scoring  # noqa: E402
from medcompress.compression.pruning.criteria import CRITERIA, common  # noqa: E402
from medcompress.compression.pruning.criteria.activation import collect_input_norms  # noqa: E402
from medcompress.compression.pruning.criteria.clinical import (  # noqa: E402
    ClinicalCriterion,
    ClinicalSettings,
)
from medcompress.compression.pruning.criteria.output_similarity import (  # noqa: E402
    OutputSimilarityCriterion,
    protected_layer_indices,
)
from medcompress.compression.pruning.units import (  # noqa: E402
    UnitType,
    enumerate_units,
    validate_removal_set,
)
from medcompress.config import ConfigError  # noqa: E402

TYPES = [UnitType.DECODER_LAYER, UnitType.ATTENTION_HEAD, UnitType.MLP_CHANNEL_GROUP]
TINY_TEMPLATE = "Is there a {finding} in this image ? Answer yes or no ."


def _wrapper(seed: int = 0) -> Any:
    return tiny_vlm.tiny_wrapper(seed=seed)


def _units(wrapper: Any) -> list[Any]:
    return enumerate_units(wrapper.module, TYPES, mlp_channel_group_size=16)


def _config(root: Path, **clinical: Any) -> dict[str, Any]:
    config = eval_config(
        root, domain={"probe": {"template": TINY_TEMPLATE, "display_names": {}}}, max_new_tokens=4
    )
    config["pruning"] = {
        "clinical": {
            "delta": "logprob_drop", "weighting": "uniform", "negatives_per_positive": None,
            "negative_delta": "keep", "ablation_method": "exact", "batch_size": 4, **clinical,
        },
        "taylor": {},
    }  # fmt: skip
    return config


# ------------------------------------------------------------------- registry


def test_every_criterion_is_registered_and_implemented() -> None:
    assert set(CRITERIA) == {
        "magnitude", "activation", "taylor", "output_similarity", "random", "clinical",
    }  # fmt: skip
    for cls in CRITERIA.values():
        assert "TODO" not in (cls.score.__doc__ or "")


# ------------------------------------------------------------------ common helpers


def test_unit_slices_cover_exactly_the_parameters_of_the_unit() -> None:
    wrapper = _wrapper()
    for unit in _units(wrapper):
        slices = common.unit_slices(wrapper, unit)
        assert sum(s.numel for s in slices) == unit.num_params, unit.unit_id


def test_removal_order_sorts_normalises_and_protects() -> None:
    wrapper = _wrapper()
    units = _units(wrapper)
    scores = {u.unit_id: float(len(u.unit_id) % 5) for u in units}
    order = common.removal_order(scores, units)
    keys = [(scores[u.unit_id], u.unit_id) for u in order]
    assert keys == sorted(keys)
    per_param = common.removal_order(scores, units, normalisation="per_parameter")
    values = [scores[u.unit_id] / u.num_params for u in per_param]
    assert values == sorted(values)
    assert per_param != order
    layer0 = next(u for u in units if u.unit_type is UnitType.DECODER_LAYER and u.layer_index == 0)
    kept = common.removal_order(scores, units, protected_layers=[0])
    assert layer0 not in kept and len(kept) == len(units) - 1
    with pytest.raises(ValueError, match="NaN"):
        common.removal_order({**scores, units[0].unit_id: math.nan}, units)
    with pytest.raises(ValueError, match="no score"):
        common.removal_order({}, units)
    with pytest.raises(ValueError, match="normalisation"):
        common.removal_order(scores, units, normalisation="by_vibes")


def test_every_prefix_of_the_feasible_sequence_is_a_valid_removal_set() -> None:
    wrapper = _wrapper()
    units = _units(wrapper)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        scores = {u.unit_id: float(rng.random()) for u in units}
        order = common.removal_order(scores, units)
        accepted = common.feasible_sequence(units, order)
        assert len(accepted) < len(order)  # something was infeasible: not everything can go
        for k in range(len(accepted) + 1):
            assert validate_removal_set(units, common.removal_set(accepted, k)) == []


def test_a_layer_removed_after_its_inner_units_absorbs_them() -> None:
    units = _units(_wrapper())
    inner = [u for u in units if u.layer_index == 1 and u.unit_type is not UnitType.DECODER_LAYER]
    layer = next(u for u in units if u.unit_type is UnitType.DECODER_LAYER and u.layer_index == 1)
    accepted = common.feasible_sequence(units, [inner[0], inner[-1], layer])
    assert accepted[-1] == layer
    final = common.removal_set(accepted, len(accepted))
    assert final == [layer]
    # an inner unit that comes after its layer cannot be removed any more
    assert common.feasible_sequence(units, [layer, inner[0]]) == [layer]


def test_protected_layers_cannot_be_removed() -> None:
    units = _units(_wrapper())
    layers = [u for u in units if u.unit_type is UnitType.DECODER_LAYER]
    accepted = common.feasible_sequence(units, layers, protected_layers=[0])
    assert layers[0] not in accepted


def test_score_files_round_trip(tmp_path: Path) -> None:
    units = _units(_wrapper())
    scores = {u.unit_id: i * 0.1 + 1e-9 for i, u in enumerate(units)}
    path = common.write_unit_scores(tmp_path / "a" / "unit_scores.csv", units, scores)
    assert common.read_unit_scores(path) == scores
    with path.open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert tuple(header) == common.SCORE_COLUMNS


# --------------------------------------------------------------- generic criteria


def test_random_scores_depend_only_on_seed_and_unit() -> None:
    units = _units(_wrapper())
    a = CRITERIA["random"]({}, seed=3).score(None, units, [])
    assert a == CRITERIA["random"]({}, seed=3).score(None, units, [])
    assert a != CRITERIA["random"]({}, seed=4).score(None, units, [])
    subset = CRITERIA["random"]({}, seed=3).score(None, units[:5], [])
    assert all(subset[k] == a[k] for k in subset)
    assert all(0.0 <= v < 1.0 for v in a.values())


def test_magnitude_is_the_l1_norm_of_the_weights_of_the_unit() -> None:
    wrapper = _wrapper()
    units = _units(wrapper)
    scores = CRITERIA["magnitude"]({}).score(wrapper, units, [])
    layer = wrapper.module.model.language_model.layers[1]
    group = next(u for u in units if u.unit_id == "lm.layer1.mlp_group1")
    manual = sum(
        float(w[16:32].detach().abs().sum())
        for w in (layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight)
    ) + float(layer.mlp.down_proj.weight[:, 16:32].detach().abs().sum())
    assert scores[group.unit_id] == pytest.approx(manual, rel=1e-5)
    whole = float(sum(p.detach().abs().sum() for p in layer.parameters()))
    assert scores["lm.layer1"] == pytest.approx(whole, rel=1e-5)
    assert scores["lm.layer1"] > scores[group.unit_id]


def test_activation_norms_and_scores_match_a_direct_computation(tmp_path: Path) -> None:
    samples = write_dataset(tmp_path)[:4]
    wrapper = _wrapper()
    norms = collect_input_norms(wrapper, samples, tmp_path)
    captured = []
    layer = wrapper.module.model.language_model.layers[2]
    handle = layer.mlp.down_proj.register_forward_pre_hook(lambda m, a: captured.append(a[0]))
    for s in samples:
        inputs = wrapper.build_inputs([tmp_path / s.image_path], [s.question])
        with torch.inference_mode():
            wrapper.module(**inputs, use_cache=False, logits_to_keep=1)
    handle.remove()
    direct = torch.sqrt(sum(x.float().pow(2).sum(dim=(0, 1)) for x in captured))
    assert torch.allclose(norms[2]["mlp_act"], direct, rtol=1e-5)
    assert set(norms[0]) == {"attn_in", "attn_out", "mlp_in", "mlp_act"}
    units = _units(wrapper)
    scores = CRITERIA["activation"]({"paths": {"data_root": str(tmp_path)}}).score(
        wrapper, units, samples
    )
    group = next(u for u in units if u.unit_id == "lm.layer2.mlp_group0")
    gate, up, down = (
        layer.mlp.gate_proj.weight.detach(),
        layer.mlp.up_proj.weight.detach(),
        layer.mlp.down_proj.weight.detach(),
    )
    manual = float(
        (gate[:16].abs() * norms[2]["mlp_in"]).sum()
        + (up[:16].abs() * norms[2]["mlp_in"]).sum()
        + (down[:, :16].abs() * norms[2]["mlp_act"][:16]).sum()
    )
    assert scores[group.unit_id] == pytest.approx(manual, rel=1e-4)
    assert all(math.isfinite(v) and v > 0 for v in scores.values())
    with pytest.raises(ValueError, match="calibration samples"):
        CRITERIA["activation"]({"paths": {"data_root": str(tmp_path)}}).score(wrapper, units, [])


def test_the_gate_gradient_of_a_channel_group_is_the_taylor_term_of_its_down_weights() -> None:
    """d L / d gate = sum over the group's down-projection weights of w * dL/dw."""
    wrapper = _wrapper()
    images, questions = [tiny_vlm.tiny_image(0), tiny_vlm.tiny_image(1)], [tiny_vlm.PROBE] * 2

    def objective() -> torch.Tensor:
        yes, no = wrapper.yes_no_logprobs(images, questions, grad=True)
        return (yes + 0.5 * no).sum()

    units = [u for u in _units(wrapper) if u.unit_type is UnitType.MLP_CHANNEL_GROUP]
    grads = scoring.accumulate_gate_gradients(wrapper, units, [objective])
    down = wrapper.module.model.language_model.layers[1].mlp.down_proj.weight
    down.requires_grad_(True)
    objective().backward()
    for unit in units:
        if unit.layer_index != 1:
            continue
        cols = slice(unit.channel_start, unit.channel_start + unit.channel_count)
        taylor = float((down[:, cols].detach() * down.grad[:, cols]).sum())
        assert grads[unit.unit_id] == pytest.approx(taylor, rel=1e-3, abs=1e-6), unit.unit_id


def test_taylor_scores_are_nonnegative_and_reproducible(tmp_path: Path) -> None:
    samples = write_dataset(tmp_path)[:3]
    wrapper = _wrapper()
    units = _units(wrapper)
    config = {"paths": {"data_root": str(tmp_path)}, "pruning": {"taylor": {}}}
    first = CRITERIA["taylor"](config).score(wrapper, units, samples)
    second = CRITERIA["taylor"](config).score(wrapper, units, samples)
    assert first == second
    assert all(v >= 0 and math.isfinite(v) for v in first.values())
    assert len({round(v, 10) for v in first.values()}) > len(units) // 2
    with pytest.raises(ValueError, match="calibration samples"):
        CRITERIA["taylor"](config).score(wrapper, units, [])


# -------------------------------------------------------------- output similarity


_VOCAB: dict[str, int] = {}


def _text_embedder(texts: Any) -> np.ndarray:
    """Equal texts get equal unit vectors (also across calls), different texts orthogonal ones."""
    rows = [_VOCAB.setdefault(t, len(_VOCAB)) for t in texts]
    out = np.zeros((len(rows), 512))
    out[np.arange(len(rows)), rows] = 1.0
    return out


def test_output_similarity_is_the_share_of_answers_a_layer_changes(tmp_path: Path) -> None:
    samples = write_dataset(tmp_path)[:4]
    wrapper = _wrapper()
    units = [u for u in _units(wrapper) if u.unit_type is UnitType.DECODER_LAYER]
    config = _config(tmp_path)
    criterion = OutputSimilarityCriterion(config, embedder=_text_embedder)
    scores = criterion.score(wrapper, units, samples)
    images = [tmp_path / s.image_path for s in samples]
    questions = [s.question for s in samples]
    base = [o.text for o in wrapper.generate(images, questions, max_new_tokens=4)]
    for unit in units:
        with scoring.ablate(wrapper, unit):
            changed = [o.text for o in wrapper.generate(images, questions, max_new_tokens=4)]
        expected = sum(a != b for a, b in zip(base, changed, strict=True)) / len(base)
        assert scores[unit.unit_id] == pytest.approx(expected)
    assert any(v > 0 for v in scores.values())
    with pytest.raises(ValueError, match="whole decoder layers"):
        criterion.score(wrapper, _units(wrapper), samples)
    config["evaluation"]["max_new_tokens"] = "TODO"
    with pytest.raises(ConfigError, match="max_new_tokens"):
        OutputSimilarityCriterion(config, embedder=_text_embedder).score(wrapper, units, samples)
    with pytest.raises(ConfigError, match="embedder.revision"):
        OutputSimilarityCriterion(_config(tmp_path)).score(wrapper, units, samples)


def test_protected_layers_are_the_first_and_last() -> None:
    assert protected_layer_indices(8, 2, 1) == [0, 1, 7]
    assert protected_layer_indices(4, 0, 0) == []
    with pytest.raises(ValueError, match="at least one removable layer"):
        protected_layer_indices(3, 2, 1)


# ----------------------------------------------------------------------- clinical


def test_clinical_settings_name_every_open_value() -> None:
    with pytest.raises(ConfigError) as caught:
        ClinicalSettings.from_config({"pruning": {"clinical": {"delta": "logprob_drop"}}})
    for key in ("negatives_per_positive", "negative_delta", "ablation_method", "batch_size"):
        assert key in str(caught.value)
    ok = {"delta": "logprob_drop", "negatives_per_positive": 2, "negative_delta": "keep",
          "ablation_method": "exact", "batch_size": 4}  # fmt: skip

    def settings(**over: Any) -> ClinicalSettings:
        return ClinicalSettings.from_config({"pruning": {"clinical": {**ok, **over}}})

    assert settings().methods["decoder_layer"] == "exact"
    mixed = settings(ablation_method={"mlp_channel_group": "first_order", "decoder_layer": "exact"})
    assert mixed.methods == {"mlp_channel_group": "first_order", "decoder_layer": "exact"}
    for over, message in (
        ({"ablation_method": "approx"}, "unknown entries"),
        ({"ablation_method": {"vision": "exact"}}, "unknown entries"),
        ({"negative_delta": "drop"}, "keep"),
        ({"delta": "mse"}, "delta"),
        ({"weighting": "vibes"}, "weighting"),
        ({"ablation_method": "first_order", "delta": "kl"}, "does not support delta: kl"),
        ({"ablation_method": "first_order", "negative_delta": "clip0"}, "cannot clip"),
    ):
        with pytest.raises(ConfigError, match=message):
            settings(**over)
    assert settings(negatives_per_positive=None).negatives_per_positive is None
    assert settings(negative_delta="clip0").clip_negative is True


def _manual_importance(
    wrapper: Any, units: list[Any], samples: list[Any], root: Path, weights: dict[str, float]
) -> dict[str, float]:
    """I_j computed by hand: sum_c w_c * mean over the images of the drop in log p(correct)."""
    probes = []
    for s in samples:
        if s.image_path in {p[0] for p in probes}:
            continue
        for finding in ("polyp", "bleeding"):
            probes.append((s.image_path, finding, s.findings[finding]))
    questions = [TINY_TEMPLATE.format(finding=f) for _, f, _ in probes]
    paths = [root / p for p, _, _ in probes]

    def correct(model: Any) -> np.ndarray:
        return np.array(
            [
                s.logp_yes if present else s.logp_no
                for s, (_, _, present) in zip(
                    model.score_yes_no(paths, questions), probes, strict=True
                )
            ]
        )

    base = correct(wrapper)
    out = {}
    for unit in units:
        with scoring.ablate(wrapper, unit):
            drop = base - correct(wrapper)
        out[unit.unit_id] = sum(
            weights[f]
            * float(np.mean([d for d, (_, g, _) in zip(drop, probes, strict=True) if g == f]))
            for f in weights
        )
    return out


def test_exact_clinical_scores_match_a_hand_computation(tmp_path: Path) -> None:
    samples = write_dataset(tmp_path)
    wrapper = _wrapper()
    units = _units(wrapper)
    config = _config(tmp_path)
    scores = ClinicalCriterion(config).score(wrapper, units, samples)
    expected = _manual_importance(
        wrapper, units, samples, tmp_path, {"polyp": 1.0, "bleeding": 1.0}
    )
    for key, value in expected.items():
        assert scores[key] == pytest.approx(value, abs=1e-4), key
    assert len({round(v, 6) for v in scores.values()}) > len(units) // 2


def test_the_pre_registered_weights_scale_the_findings(tmp_path: Path) -> None:
    samples = write_dataset(tmp_path)
    wrapper = _wrapper()
    units = _units(wrapper)
    table = {
        "preregistration": {"frozen_on": "2026-11-01"},
        "weighting": {"normalisation": "none", "tiers": [{"tier": "high", "multiplier": 2.0}]},
        "findings": [
            {"finding": "polyp", "prevalence": 0.5, "importance_tier": "high", "cap": 10,
             "final_weight": 4.0},
            {"finding": "bleeding", "prevalence": 0.25, "importance_tier": "high", "cap": 2,
             "final_weight": 4.0},
        ],
    }  # fmt: skip
    config = _config(tmp_path, weighting="clinical")
    path = tmp_path / "weights.yaml"
    path.write_text(yaml.safe_dump(table))
    config["components"]["finding_weights"] = str(path)
    scores = ClinicalCriterion(config).score(wrapper, units, samples)
    expected = _manual_importance(
        wrapper, units, samples, tmp_path, {"polyp": 4.0, "bleeding": 4.0}
    )
    for key, value in expected.items():
        assert scores[key] == pytest.approx(value, abs=1e-3), key
    debug = _config(tmp_path, weighting="clinical", debug_unfrozen_weights=True)
    table["preregistration"]["frozen_on"] = "TODO"  # (_config rewrote the stub table above)
    path.write_text(yaml.safe_dump(table))
    with pytest.raises(ValueError, match="not frozen"):
        ClinicalCriterion(config).score(wrapper, units, samples)
    debug["components"]["finding_weights"] = str(path)
    assert ClinicalCriterion(debug).score(wrapper, units[:2], samples)


def test_first_order_scores_follow_the_exact_ones_for_channel_groups(tmp_path: Path) -> None:
    samples = write_dataset(tmp_path)
    wrapper = _wrapper()
    units = [u for u in _units(wrapper) if u.unit_type is UnitType.MLP_CHANNEL_GROUP]
    exact = ClinicalCriterion(_config(tmp_path)).score(wrapper, units, samples)
    approx = ClinicalCriterion(_config(tmp_path, ablation_method="first_order")).score(
        wrapper, units, samples
    )
    a, b = [exact[u.unit_id] for u in units], [approx[u.unit_id] for u in units]
    assert np.corrcoef(a, b)[0, 1] > 0.9


def test_a_mixed_method_scores_every_unit(tmp_path: Path) -> None:
    samples = write_dataset(tmp_path)
    wrapper = _wrapper()
    units = _units(wrapper)
    method = {
        "decoder_layer": "exact",
        "attention_head": "exact",
        "mlp_channel_group": "first_order",
    }
    mixed = ClinicalCriterion(_config(tmp_path, ablation_method=method)).score(
        wrapper, units, samples
    )
    exact = ClinicalCriterion(_config(tmp_path)).score(wrapper, units, samples)
    assert set(mixed) == {u.unit_id for u in units}
    for unit in units:
        if unit.unit_type is not UnitType.MLP_CHANNEL_GROUP:
            assert mixed[unit.unit_id] == pytest.approx(exact[unit.unit_id], abs=1e-5)
    partial = {"decoder_layer": "exact"}
    with pytest.raises(ConfigError, match="no entry for"):
        ClinicalCriterion(_config(tmp_path, ablation_method=partial)).score(wrapper, units, samples)


def test_balancing_changes_which_probes_are_scored(tmp_path: Path) -> None:
    samples = write_dataset(tmp_path)
    wrapper = _wrapper()
    units = _units(wrapper)[:3]
    full = ClinicalCriterion(_config(tmp_path)).score(wrapper, units, samples)
    balanced = ClinicalCriterion(_config(tmp_path, negatives_per_positive=0)).score(
        wrapper, units, samples
    )
    assert full != balanced


def test_clinical_ablate_is_a_reversible_context(tmp_path: Path) -> None:
    wrapper = _wrapper()
    unit = _units(wrapper)[0]
    before = wrapper.score_yes_no([tiny_vlm.tiny_image(0)], [tiny_vlm.PROBE])[0].logp_yes
    with ClinicalCriterion({}).ablate(wrapper, unit):
        inside = wrapper.score_yes_no([tiny_vlm.tiny_image(0)], [tiny_vlm.PROBE])[0].logp_yes
    after = wrapper.score_yes_no([tiny_vlm.tiny_image(0)], [tiny_vlm.PROBE])[0].logp_yes
    assert inside != pytest.approx(before, abs=1e-6) and after == pytest.approx(before)
