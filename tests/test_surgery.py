"""Unit enumeration, validation, ablation and physical removal on tiny random models.

Heavy: needs torch and transformers (installed on GPU machines and in the agent's
CPU session, not in CI). Each ablation must give the same output as physically
removing the same unit, which is what makes ablation scores meaningful for pruning.
"""

from __future__ import annotations

import pytest

from medcompress.compression.pruning.units import UnitType, validate_removal_set

pytestmark = pytest.mark.heavy


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


def test_pruned_checkpoint_round_trip_over_two_steps(tmp_path) -> None:  # noqa: ANN001
    from medcompress.compression.pruning.checkpoint import (
        MANIFEST_NAME,
        load_pruned,
        read_manifest,
        save_pruned,
    )

    types = list(UnitType)[:3]
    model = _tiny("qwen2_5_vl")
    ids = torch.randint(0, 90, (1, 9))
    units = enumerate_units(model, types, mlp_channel_group_size=16)
    by_id = {u.unit_id: u for u in units}
    first = [by_id["lm.layer0.kv_group1"], by_id["lm.layer1.mlp_group0"]]
    remove_units(model, first)
    # a second step is enumerated on the already pruned model (indices are re-based)
    second = [
        u
        for u in enumerate_units(model, types, mlp_channel_group_size=16)
        if u.unit_id == "lm.layer1"
    ]
    assert second
    remove_units(model, second)
    expected = _logits(model, ids)
    save_pruned(model, tmp_path, [first, second], mlp_channel_group_size=16)
    assert read_manifest(tmp_path)["steps"] == [[u.unit_id for u in first], ["lm.layer1"]]

    restored = load_pruned(_tiny("qwen2_5_vl"), tmp_path)
    assert surgery.count_parameters(restored) == surgery.count_parameters(model)
    assert torch.allclose(_logits(restored, ids), expected, atol=1e-6)

    (tmp_path / MANIFEST_NAME).write_text(
        (tmp_path / MANIFEST_NAME).read_text().replace("lm.layer0.kv_group1", "lm.layer9.kv_group1")
    )
    with pytest.raises(ValueError, match="not in the base model"):
        load_pruned(_tiny("qwen2_5_vl"), tmp_path)
