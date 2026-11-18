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


def test_the_streamed_save_reads_back_like_save_model(tmp_path) -> None:  # noqa: ANN001
    """2026-11-14: the pruned model is written one tensor at a time; the file must hold what
    ``safetensors.torch.save_model`` holds (shared weights once, the dropped names in the
    metadata) and load back with ``load_model``."""
    from safetensors import safe_open
    from safetensors.torch import load_file, load_model, save_model

    from medcompress.compression.pruning.checkpoint import stream_save

    model = _tiny("qwen2_5_vl")
    model.lm_head.weight = model.model.language_model.embed_tokens.weight  # a shared tensor
    model.get_input_embeddings().weight.data[0, 0] = 3.0
    ours, theirs = tmp_path / "ours.safetensors", tmp_path / "theirs.safetensors"
    stream_save(model, ours)
    save_model(model, str(theirs))
    a, b = load_file(str(ours)), load_file(str(theirs))
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    with safe_open(str(ours), "pt") as x, safe_open(str(theirs), "pt") as y:
        assert x.metadata() == y.metadata() and x.metadata()  # the dropped name is recorded
    fresh = _tiny("qwen2_5_vl")
    fresh.lm_head.weight = fresh.model.language_model.embed_tokens.weight
    load_model(fresh, str(ours), strict=True)
    pairs = zip(fresh.state_dict().values(), model.state_dict().values(), strict=True)
    assert all(torch.equal(p, q) for p, q in pairs)
    assert not (tmp_path / "ours.safetensors.partial").exists()


def test_the_pruned_load_streams_one_tensor_at_a_time(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """2026-11-18 (RC-G2 session 5: stage 06 out of GPU memory): ``load_pruned`` holds the model
    plus at most one tensor of the file, never a second full copy of the state dict. Live
    tensor bytes (the CPU stand-in for the GPU's allocated memory) are read at every tensor the
    load takes from the file; a shared (tied) weight stored once still loads."""
    import safetensors

    from medcompress.compression.pruning import checkpoint
    from medcompress.utils.memory import live_tensor_bytes

    def tied() -> object:
        model = _tiny("qwen2_5_vl")
        model.lm_head.weight = model.model.language_model.embed_tokens.weight
        return model

    types = list(UnitType)[:3]
    model = tied()
    by_id = {u.unit_id: u for u in enumerate_units(model, types, mlp_channel_group_size=16)}
    removal = [by_id["lm.layer0.kv_group1"], by_id["lm.layer1.mlp_group0"], by_id["lm.layer2"]]
    remove_units(model, removal)
    ids = torch.randint(0, 90, (1, 9))
    expected = _logits(model, ids)
    checkpoint.save_pruned(model, tmp_path, [removal], mlp_channel_group_size=16)
    del model
    readings: list[int] = []
    largest = [0]
    real_open = safetensors.safe_open

    class Watched:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self.inner = real_open(*args, **kwargs)

        def __enter__(self):  # noqa: ANN204
            self.handle = self.inner.__enter__()
            return self

        def __exit__(self, *exc) -> None:  # noqa: ANN002
            self.inner.__exit__(*exc)

        def keys(self):  # noqa: ANN202
            return self.handle.keys()

        def metadata(self):  # noqa: ANN202
            return self.handle.metadata()

        def get_tensor(self, name: str):  # noqa: ANN202
            tensor = self.handle.get_tensor(name)
            largest[0] = max(largest[0], tensor.numel() * tensor.element_size())
            readings.append(live_tensor_bytes("cpu"))
            return tensor

    monkeypatch.setattr(safetensors, "safe_open", Watched)
    fresh = tied()
    baseline = live_tensor_bytes("cpu")  # the unpruned base: the most the load may start from
    restored = checkpoint.load_pruned(fresh, tmp_path)
    assert readings
    assert max(readings) <= baseline + largest[0]  # never a second copy of the state dict
    file_bytes = sum(p.numel() * p.element_size() for p in restored.state_dict().values())
    assert file_bytes > largest[0]  # a second full copy would break the bound above
    assert live_tensor_bytes("cpu") < baseline  # the replaced and dropped tensors were freed
    assert torch.allclose(_logits(restored, ids), expected, atol=1e-6)


def test_the_streamed_load_is_strict(tmp_path) -> None:  # noqa: ANN001
    from safetensors.torch import save_file

    from medcompress.compression.pruning.checkpoint import stream_load, stream_save

    model = _tiny("qwen2_5_vl")
    path = tmp_path / "w.safetensors"
    stream_save(model, path)
    assert stream_load(_tiny("qwen2_5_vl"), path) == len(model.state_dict())
    state = {k: v.clone() for k, v in model.state_dict().items()}
    extra = dict(state, extra=torch.zeros(2))
    save_file(extra, str(tmp_path / "extra.safetensors"))
    with pytest.raises(ValueError, match="unexpected"):
        stream_load(_tiny("qwen2_5_vl"), tmp_path / "extra.safetensors")
    name = next(iter(state))
    save_file({k: v for k, v in state.items() if k != name}, str(tmp_path / "less.safetensors"))
    with pytest.raises(ValueError, match="missing"):
        stream_load(_tiny("qwen2_5_vl"), tmp_path / "less.safetensors")
    key = next(k for k in state if k.endswith("mlp.gate_proj.weight"))
    save_file(
        dict(state, **{key: state[key][:-1].contiguous()}), str(tmp_path / "shape.safetensors")
    )
    with pytest.raises(ValueError, match="in the file"):
        stream_load(_tiny("qwen2_5_vl"), tmp_path / "shape.safetensors")
    save_file(dict(state, **{key: state[key].double()}), str(tmp_path / "dtype.safetensors"))
    with pytest.raises(ValueError, match="in the file"):
        stream_load(_tiny("qwen2_5_vl"), tmp_path / "dtype.safetensors")


def test_the_surgery_frees_what_it_replaces_or_drops() -> None:
    """2026-11-18: a replaced projection or a dropped layer is freed even when something else
    (an uncollected adapter wrapper) still holds the old module."""
    model = _tiny("qwen2_5_vl")
    units = enumerate_units(model, list(UnitType)[:3], mlp_channel_group_size=16)
    by_id = {u.unit_id: u for u in units}
    stack = surgery.find_decoder_stack(model)
    held = [stack.layers[1].mlp.gate_proj, stack.layers[0].self_attn.q_proj, stack.layers[2]]
    remove_units(model, [by_id["lm.layer1.mlp_group0"], by_id["lm.layer0.kv_group1"],
                         by_id["lm.layer2"]])  # fmt: skip
    assert all(p.numel() == 0 for module in held for p in module.parameters())
    new = surgery.find_decoder_stack(model).layers[1].mlp.gate_proj
    assert new.weight.shape[0] == 48 - 16 and new.weight.device.type != "meta"
