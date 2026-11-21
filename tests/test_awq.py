"""E3 ``awq_w4`` (amendment of 2026-11-20): the recipe, the card, the checkpoint check, and the
AWQ path through GPTQModel on a small Qwen2.5-VL (skipped where GPTQModel is not installed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from medcompress.cards import build_plan
from medcompress.config import ConfigError


def _config() -> dict:
    import argparse

    from medcompress.cli import resolve_config

    namespace = argparse.Namespace(config=None, exp="E3", override=[], data_root=None, handoffs=[])
    return resolve_config(namespace)


def test_the_awq_recipe_is_4_bit_groups_of_128_on_w4s_targets() -> None:
    from medcompress.compression.quantization.ptq import PTQRecipe

    config = _config()
    awq, w4 = PTQRecipe.from_config(config, "awq_w4"), PTQRecipe.from_config(config, "w4")
    assert (awq.backend, awq.weight_bits, awq.group_size) == ("awq", 4, 128)
    assert awq.targets == w4.targets and awq.calibration_size == w4.calibration_size == 128
    config["quantization"]["recipes"]["awq_w4"]["weight_bits"] = 8
    with pytest.raises(ConfigError, match="4-bit"):
        PTQRecipe.from_config(config, "awq_w4")
    config["quantization"]["recipes"]["awq_w4"]["weight_bits"] = 4
    config["quantization"]["targets"] = ["q_proj", "k_proj"]
    with pytest.raises(ConfigError, match="quantizes"):
        PTQRecipe.from_config(config, "awq_w4")


def test_the_card_runs_awq_only_when_asked_with_its_own_handoffs() -> None:
    plain = build_plan("RC-E3", 0)
    assert [s.label for s in plain.steps if s.label.startswith("run")] == [
        "run E3 w4 seed0", "run E3 w8 seed0", "run E3 w8a8 seed0",
    ]  # fmt: skip
    assert [h.name for h in plain.handoffs] == ["E3_seed0_quantized", "E3_seed0_outcomes"]
    awq = build_plan("RC-E3", 0, variants=["awq_w4"])
    assert [s.label for s in awq.steps] == [
        "run E3 awq_w4 seed0", "test evaluation E3 awq_w4 seed0",
        "unfiltered test evaluation E3 awq_w4 seed0",
    ]  # fmt: skip
    assert [h.name for h in awq.handoffs] == ["E3_seed0_awq_w4_quantized",
                                              "E3_seed0_awq_w4_outcomes"]  # fmt: skip
    from medcompress.smoke import smoke_jobs

    assert any(j.name == "E3 awq_w4" and j.card == "RC-E3" for j in smoke_jobs())
    from medcompress.scale_check import CHECKS, EXPERIMENT_OF

    assert "awq" in CHECKS and EXPERIMENT_OF["awq"] == "E3"


def test_a_quantized_checkpoint_is_complete_by_its_backend(tmp_path: Path) -> None:
    from medcompress.compression.quantization.ptq import (
        MANIFEST_NAME,
        TENSORS_NAME,
        quantized_checkpoint_complete,
    )

    def manifest(folder: Path, backend: str) -> Path:
        folder.mkdir(parents=True)
        (folder / MANIFEST_NAME).write_text(json.dumps({"recipe": {"backend": backend}}))
        return folder

    gptq = manifest(tmp_path / "gptq", "gptq")
    assert not quantized_checkpoint_complete(gptq)
    (gptq / TENSORS_NAME).write_bytes(b"x")
    assert quantized_checkpoint_complete(gptq)
    awq = manifest(tmp_path / "awq", "awq")
    assert not quantized_checkpoint_complete(awq)
    (awq / "awq").mkdir()
    (awq / "awq" / "model.safetensors").write_bytes(b"x")
    assert quantized_checkpoint_complete(awq)
    assert quantized_checkpoint_complete(manifest(tmp_path / "int8", "llm_int8"))
    assert not quantized_checkpoint_complete(tmp_path / "missing")


def test_awq_through_gptqmodel_on_a_small_qwen2_5_vl(tmp_path: Path) -> None:
    """The production path on CPU: our calibration inputs (image, question, answer), GPTQModel's
    AWQ in 4 bit with groups of 128, saved, reloaded with ``AwqTorchLinear`` and wrapped like
    the student; the wrapper scores yes/no probes."""
    pytest.importorskip("gptqmodel")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import tiny_vlm
    from eval_fixtures import write_dataset

    from medcompress.compression.quantization import awq
    from medcompress.compression.quantization.ptq import calibration_batches
    from medcompress.models.hf import HFVLMWrapper

    processor = tiny_vlm.tiny_processor()
    ids = processor.tokenizer.convert_tokens_to_ids
    config = transformers.Qwen2_5_VLConfig(
        text_config=dict(
            vocab_size=48, hidden_size=256, intermediate_size=1536, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
            rope_scaling={"type": "mrope", "mrope_section": [8, 12, 12]},
        ),
        vision_config=dict(
            depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=256
        ),
        image_token_id=ids("<|image_pad|>"), video_token_id=ids("<|video_pad|>"),
        vision_start_token_id=ids("<|vision_start|>"), vision_end_token_id=ids("<|vision_end|>"),
    )  # fmt: skip
    torch.manual_seed(0)
    module = transformers.Qwen2_5_VLForConditionalGeneration(config).eval()
    module.generation_config.eos_token_id = processor.tokenizer.eos_token_id  # as tiny_model
    module.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    wrapper = HFVLMWrapper(module, processor, batch_size=2)
    samples = write_dataset(tmp_path)
    batches = calibration_batches(wrapper, samples[:4], tmp_path)
    assert all({"input_ids", "pixel_values", "image_grid_thw"} <= set(b) for b in batches)
    source, target = tmp_path / "source", tmp_path / "awq"
    module.save_pretrained(str(source))
    processor.save_pretrained(str(source))
    info = awq.quantize_folder(source, target, batches, bits=4, group_size=128, device="cpu")
    assert info["quantized_modules"] == 2 * 7 and info["calibration_examples"] == 4
    assert awq.quantized_weight_bytes(target) > 0
    loaded = awq.load_folder(target, device="cpu", dtype=torch.float32)
    kinds = {type(m).__name__ for n, m in loaded.named_modules()
             if "language_model.layers." in n and n.endswith("mlp.down_proj")}  # fmt: skip
    assert kinds == {awq.KERNEL}
    assert type(loaded.model.visual.blocks[0].mlp.down_proj).__name__ == "Linear"  # not quantized
    scored = HFVLMWrapper(loaded, processor, batch_size=2).score_yes_no(
        [tiny_vlm.tiny_image(0), tiny_vlm.tiny_image(1)], [tiny_vlm.PROBE] * 2
    )
    assert all(torch.isfinite(torch.tensor([s.logp_yes, s.logp_no])).all() for s in scored)


def test_the_stage_saves_awq_and_the_runner_loads_it_without_the_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ptq.awq_quantize`` saves the parent, releases it, quantizes, removes the source copy;
    the manifest names the AWQ backend; ``load_run_quantized`` loads the AWQ model directly."""
    pytest.importorskip("gptqmodel")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    import dataclasses

    import tiny_vlm
    from eval_fixtures import write_dataset

    from medcompress.compression.quantization import awq, ptq
    from medcompress.config import RunContext
    from medcompress.models.hf import HFVLMWrapper

    processor = tiny_vlm.tiny_processor()
    ids = processor.tokenizer.convert_tokens_to_ids
    model_config = transformers.Qwen2_5_VLConfig(
        text_config=dict(
            vocab_size=48, hidden_size=256, intermediate_size=1536, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
            rope_scaling={"type": "mrope", "mrope_section": [8, 12, 12]},
        ),
        vision_config=dict(
            depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=256
        ),
        image_token_id=ids("<|image_pad|>"), video_token_id=ids("<|video_pad|>"),
        vision_start_token_id=ids("<|vision_start|>"), vision_end_token_id=ids("<|vision_end|>"),
    )  # fmt: skip
    torch.manual_seed(0)
    module = transformers.Qwen2_5_VLForConditionalGeneration(model_config).eval()
    module.generation_config.eos_token_id = processor.tokenizer.eos_token_id
    module.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    wrapper = HFVLMWrapper(module, processor, batch_size=2)
    samples = write_dataset(tmp_path)
    batches = ptq.calibration_batches(wrapper, samples[:4], tmp_path)
    config = _config()
    config["paths"] = {**config["paths"], "checkpoint_root": str(tmp_path / "checkpoints"),
                       "results_root": str(tmp_path / "results")}  # fmt: skip
    run = RunContext("E3", seed=0, variant="awq_w4")
    recipe = ptq.PTQRecipe.from_config(config, "awq_w4")
    info = ptq.awq_quantize(config, run, dataclasses.replace(recipe), wrapper, batches)
    folder = ptq.quantized_dir(config, run)
    assert not (folder.parent / awq.SOURCE_SUBDIR).exists()  # the source copy is removed
    assert info["quantized_modules"] == 14 and info["quantized_weight_bytes"] > 0
    assert all(p.device.type == "meta" for p in module.parameters())  # the parent was released
    assert info["symmetric"] is True and info["grid"]["zero_point"] is False  # as w4's grid
    assert info["modules_without_awq_scale"] == ["self_attn.o_proj"]  # GQA: no AWQ scale
    assert info["scaled_modules"] == 12 and info["generation_config"]["eos_token_id"] == (
        processor.tokenizer.eos_token_id
    )
    ptq.save_quantized({}, recipe, folder, start_from=ptq.parent_of(run), extra={"awq": info})
    assert ptq.quantized_checkpoint_complete(folder)
    loaded = ptq.load_awq_quantized(config, run, 56, processor=processor)
    assert loaded.module.generation_config.eos_token_id == processor.tokenizer.eos_token_id
    assert not loaded.module.generation_config.do_sample  # the parent's (GPTQModel: True)
    scores = loaded.score_yes_no([tiny_vlm.tiny_image(0)], [tiny_vlm.PROBE])
    assert torch.isfinite(torch.tensor([scores[0].logp_yes, scores[0].logp_no])).all()
    routed: list[object] = []
    monkeypatch.setattr(ptq, "load_awq_quantized", lambda *a, **k: routed.append(a) or "awq")
    assert ptq.load_run_quantized(config, run, 56) == "awq" and routed  # no parent load
