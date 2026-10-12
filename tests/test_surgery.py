"""Unit enumeration, validation, ablation and physical removal on tiny random models.

Heavy: needs torch and transformers (installed on GPU machines and in the agent's
CPU session, not in CI). Each ablation must give the same output as physically
removing the same unit, which is what makes ablation scores meaningful for pruning.
"""

from __future__ import annotations

import pytest

from medcompress.compression.pruning.units import (
    PrunableUnit,
    UnitType,
    gqa_head_groups,
    unit_id,
    validate_removal_set,
)

pytestmark = pytest.mark.heavy


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


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from medcompress.compression.pruning import surgery  # noqa: E402
from medcompress.compression.pruning.apply import (  # noqa: E402
    remove_units,
    verify_physical_removal,
)
from medcompress.compression.pruning.units import enumerate_units  # noqa: E402


def _tiny(kind: str):  # noqa: ANN202 - returns a transformers model
    torch.manual_seed(0)
    text = dict(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    if kind == "qwen2":
        model = transformers.Qwen2ForCausalLM(transformers.Qwen2Config(**text))
    elif kind == "qwen3":
        model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(head_dim=8, **text))
    else:
        cfg = transformers.Qwen2_5_VLConfig(
            text_config={**text, "rope_scaling": {"type": "mrope", "mrope_section": [1, 1, 2]}},
            vision_config=dict(
                depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32
            ),
        )
        model = transformers.Qwen2_5_VLForConditionalGeneration(cfg)
    return model.eval()


KINDS = ["qwen2", "qwen3", "qwen2_5_vl"]


def _logits(model, ids):  # noqa: ANN001, ANN202
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


@pytest.mark.parametrize("kind", KINDS)
def test_enumerate_units(kind: str) -> None:
    model = _tiny(kind)
    units = enumerate_units(model, list(UnitType)[:3], mlp_channel_group_size=16)
    counts = {t: sum(u.unit_type is t for u in units) for t in UnitType}
    assert counts[UnitType.DECODER_LAYER] == 3
    assert counts[UnitType.ATTENTION_HEAD] == 3 * 2
    assert counts[UnitType.MLP_CHANNEL_GROUP] == 3 * 3
    assert surgery.find_decoder_stack(model).path.endswith("layers")


@pytest.mark.parametrize("kind", KINDS)
def test_skip_layer_equals_removing_it(kind: str) -> None:
    model = _tiny(kind)
    ids = torch.randint(0, 90, (1, 9))
    full = _logits(model, ids)
    with surgery.skip_layer(model, 1):
        skipped = _logits(model, ids)
    assert not torch.allclose(full, skipped)
    assert torch.allclose(full, _logits(model, ids))  # hook removed
    units = enumerate_units(model, [UnitType.DECODER_LAYER])
    before = surgery.count_parameters(model)
    remove_units(model, [units[1]])
    assert torch.allclose(skipped, _logits(model, ids), atol=1e-5)
    report = verify_physical_removal(before, model, [units[1]])
    assert report["ok"], report
    assert surgery.find_decoder_stack(model).config.num_hidden_layers == 2


@pytest.mark.parametrize("kind", KINDS)
def test_zero_kv_group_equals_removing_it(kind: str) -> None:
    model = _tiny(kind)
    ids = torch.randint(0, 90, (1, 9))
    with surgery.zero_kv_group(model, 0, 1):
        ablated = _logits(model, ids)
    units = [
        u
        for u in enumerate_units(model, [UnitType.ATTENTION_HEAD])
        if u.unit_id == "lm.layer0.kv_group1"
    ]
    before = surgery.count_parameters(model)
    remove_units(model, units)
    assert torch.allclose(ablated, _logits(model, ids), atol=1e-5)
    assert verify_physical_removal(before, model, units)["ok"]
    shape = surgery.layer_shape(surgery.find_decoder_stack(model).layers[0], model.config)
    assert (shape["num_heads"], shape["num_kv_heads"]) == (2, 1)


@pytest.mark.parametrize("kind", KINDS)
def test_zero_mlp_channels_equals_removing_them(kind: str) -> None:
    model = _tiny(kind)
    ids = torch.randint(0, 90, (1, 9))
    groups = [
        u
        for u in enumerate_units(model, [UnitType.MLP_CHANNEL_GROUP], mlp_channel_group_size=16)
        if u.layer_index == 2
    ][:2]
    channels = [
        c for u in groups for c in range(u.channel_start, u.channel_start + u.channel_count)
    ]
    with surgery.zero_mlp_channels(model, 2, channels):
        ablated = _logits(model, ids)
    before = surgery.count_parameters(model)
    remove_units(model, groups)
    assert torch.allclose(ablated, _logits(model, ids), atol=1e-5)
    assert verify_physical_removal(before, model, groups)["ok"]


def test_mixed_removal_and_generation_still_work() -> None:
    model = _tiny("qwen2")
    units = enumerate_units(model, list(UnitType)[:3], mlp_channel_group_size=16)
    by_id = {u.unit_id: u for u in units}
    chosen = [by_id["lm.layer0.kv_group0"], by_id["lm.layer1.mlp_group2"], by_id["lm.layer2"]]
    assert validate_removal_set(units, chosen) == []
    before = surgery.count_parameters(model)
    remove_units(model, chosen)
    assert verify_physical_removal(before, model, chosen)["ok"]
    out = model.generate(torch.randint(0, 90, (1, 5)), max_new_tokens=3, do_sample=False)
    assert out.shape == (1, 8)
