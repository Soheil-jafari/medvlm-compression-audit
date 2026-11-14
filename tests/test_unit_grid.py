"""The real student's unit grid: full channel-group coverage, and RC-SMOKE's whole-layer cap
(2026-11-13: A4 and A6 failed in the smoke because a unit cap cut a layer's channel groups)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("accelerate")


def _student_skeleton():  # type: ignore[no-untyped-def]
    """Qwen2.5-VL-3B (the student) without weights: its parameter count is the model config's
    ``parameters_stored`` and RC-B0's ``num_parameters``, 3,754,622,976."""
    from accelerate import init_empty_weights

    config = transformers.Qwen2_5_VLConfig(
        text_config=dict(vocab_size=151936, hidden_size=2048, intermediate_size=11008,
                         num_hidden_layers=36, num_attention_heads=16, num_key_value_heads=2,
                         tie_word_embeddings=True),
        vision_config=dict(depth=32, hidden_size=1280, intermediate_size=3420, num_heads=16,
                           out_hidden_size=2048),
        tie_word_embeddings=True,
    )  # fmt: skip
    with init_empty_weights():
        model = transformers.AutoModelForImageTextToText.from_config(config, dtype=torch.float16)
    model.tie_weights()
    assert sum(p.numel() for p in model.parameters()) == 3_754_622_976
    return model


def test_the_production_grid_covers_every_mlp_channel_and_the_smoke_cap_keeps_whole_layers() -> (
    None
):
    from medcompress.compression.pruning.scoring import UnitGates
    from medcompress.compression.pruning.stage import whole_layers_within
    from medcompress.compression.pruning.units import UnitType, enumerate_units

    model = _student_skeleton()
    types = [UnitType.DECODER_LAYER, UnitType.ATTENTION_HEAD, UnitType.MLP_CHANNEL_GROUP]
    units = enumerate_units(model, types, mlp_channel_group_size=128)
    assert len(units) == 36 * (1 + 2 + 86) == 3204  # 11,008 / 128 = 86 groups per layer
    channels = [u for u in units if u.unit_type is UnitType.MLP_CHANNEL_GROUP]
    UnitGates(model, channels)  # every layer's groups tile the 11,008 channels: no error
    for kind in (types, [UnitType.ATTENTION_HEAD, UnitType.MLP_CHANNEL_GROUP]):  # E7, A6
        grid = enumerate_units(model, kind, mlp_channel_group_size=128)
        UnitGates(model, [u for u in grid if u.unit_type is UnitType.MLP_CHANNEL_GROUP])
    with pytest.raises(ValueError, match="layer 3: channel groups cover 3840 of 11008"):
        UnitGates(model, [u for u in units[:300] if u.unit_type is UnitType.MLP_CHANNEL_GROUP])
    kept = whole_layers_within(units, 300)  # the smoke's cap now: 3 whole layers
    assert len(kept) == 267 and {u.layer_index for u in kept} == {0, 1, 2}
    UnitGates(model, [u for u in kept if u.unit_type is UnitType.MLP_CHANNEL_GROUP])
    assert whole_layers_within(units, 10) == [u for u in units if u.layer_index == 0]


def test_a5_kl_scores_layers_and_kv_groups_exactly() -> None:
    from medcompress.compression.pruning.criteria.clinical import ClinicalSettings
    from medcompress.compression.pruning.stage import ScoringSettings
    from medcompress.config import ConfigError, apply_variant, load_experiment

    config, _ = load_experiment("A5")
    kl = ScoringSettings.from_config(config, "kl")
    assert kl.unit_types == ("decoder_layer", "attention_head")
    settings = ClinicalSettings.from_config(apply_variant(config, "kl"))
    assert settings.delta == "kl"
    assert {settings.methods[t] for t in kl.unit_types} == {"exact"}
    e7, _ = load_experiment("E7")  # with first-order channel groups, KL is still refused
    clinical = {**e7["pruning"]["clinical"], "delta": "kl"}
    with pytest.raises(ConfigError, match="does not support delta: kl"):
        ClinicalSettings.from_config({**e7, "pruning": {**e7["pruning"], "clinical": clinical}})
    # both variants on the same units (clarification of 2026-11-13): A5 logprob_drop is not E7
    assert ScoringSettings.from_config(config, "logprob_drop").unit_types == kl.unit_types
