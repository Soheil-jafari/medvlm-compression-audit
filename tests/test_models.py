"""The model wrapper on a tiny offline vision-language model (heavy: needs torch).

Builds a tiny randomly initialised Qwen2.5-VL with a tiny word-level tokenizer and the
real processor classes, so the whole path (chat template, image tokens, left padding,
scoring, generation, unit removal, adapter loading and the RC-M1 script) runs on CPU
without any download. Skipped when torch, transformers or torchvision are missing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medcompress.models.loading import ModelConfigError

# ------------------------------------------------------------------ heavy tests

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("torchvision")
pytest.importorskip("tokenizers")
PIL_Image = pytest.importorskip("PIL.Image")

import tiny_vlm  # noqa: E402

PROBE = tiny_vlm.PROBE
_image = tiny_vlm.tiny_image
_tiny_model = tiny_vlm.tiny_model
_tiny_processor = tiny_vlm.tiny_processor

from medcompress.compression.pruning import surgery  # noqa: E402
from medcompress.compression.pruning.apply import (  # noqa: E402
    remove_units,
    verify_physical_removal,
)
from medcompress.compression.pruning.units import UnitType  # noqa: E402
from medcompress.models.base import GenerationOutput, YesNoScore  # noqa: E402
from medcompress.models.hf import HFVLMWrapper, _to_pil  # noqa: E402
from medcompress.models.loading import check_shared_tokenizer, load_adapter  # noqa: E402


@pytest.fixture
def wrapper() -> HFVLMWrapper:
    processor = _tiny_processor()
    return HFVLMWrapper(_tiny_model(processor), processor, batch_size=2, mlp_channel_group_size=16)


def test_wrapper_finds_answer_tokens(wrapper: HFVLMWrapper) -> None:
    ids = wrapper.tokenizer.convert_tokens_to_ids
    assert set(wrapper.yes_ids) == {ids("Yes"), ids("yes")}
    assert set(wrapper.no_ids) == {ids("No"), ids("no")}
    assert wrapper.tokenizer.padding_side == "left"


def test_score_yes_no_is_batch_independent(wrapper: HFVLMWrapper) -> None:
    images = [_image(1), _image(2, (56, 56)), _image(3)]
    questions = [PROBE, "What do you see ?", PROBE]
    together = wrapper.score_yes_no(images, questions)  # batches of 2 and 1, different padding
    assert len(together) == 3 and all(isinstance(s, YesNoScore) for s in together)
    for image, question, batched in zip(images, questions, together, strict=True):
        alone = wrapper.score_yes_no([image], [question])[0]
        assert alone.logp_yes == pytest.approx(batched.logp_yes, abs=1e-4)
        assert alone.logp_no == pytest.approx(batched.logp_no, abs=1e-4)
    assert all(s.logp_yes < 0 and s.logp_no < 0 for s in together)


def test_score_yes_no_reads_the_answer_position(wrapper: HFVLMWrapper) -> None:
    inputs = wrapper.build_inputs([_image(1)], [PROBE])
    with torch.no_grad():
        logits = wrapper.module(**inputs, use_cache=False).logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    expected_yes = torch.logsumexp(logp[list(wrapper.yes_ids)], dim=0).item()
    score = wrapper.score_yes_no([_image(1)], [PROBE])[0]
    assert score.logp_yes == pytest.approx(expected_yes, abs=1e-5)
    assert score.predicted in {"yes", "no"}


def test_the_image_changes_the_score(wrapper: HFVLMWrapper) -> None:
    a = wrapper.score_yes_no([_image(1)], [PROBE])[0]
    b = wrapper.score_yes_no([_image(2)], [PROBE])[0]
    assert a.logp_yes != pytest.approx(b.logp_yes, abs=1e-6)


def test_generate_returns_tokens_and_topk(wrapper: HFVLMWrapper) -> None:
    outputs = wrapper.generate(
        [_image(1), _image(2)], [PROBE, PROBE], max_new_tokens=4, return_topk=3
    )
    assert len(outputs) == 2 and all(isinstance(o, GenerationOutput) for o in outputs)
    for out in outputs:
        assert 1 <= len(out.token_ids) <= 4
        assert len(out.topk_token_ids) == len(out.topk_logprobs) == len(out.token_ids)
        assert all(len(row) == 3 for row in out.topk_token_ids)
        # greedy decoding picks the top-1 token of the raw distribution
        assert [row[0] for row in out.topk_token_ids] == list(out.token_ids)
        assert all(row == sorted(row, reverse=True) for row in out.topk_logprobs)
    plain = wrapper.generate([_image(1)], [PROBE], max_new_tokens=4)[0]
    assert plain.topk_token_ids is None and plain.token_ids == outputs[0].token_ids


def test_generate_stops_at_the_stop_token_and_keeps_it(wrapper: HFVLMWrapper) -> None:
    first = wrapper.generate([_image(1)], [PROBE], max_new_tokens=3)[0].token_ids[0]
    wrapper.module.generation_config.eos_token_id = first
    out = wrapper.generate([_image(1)], [PROBE], max_new_tokens=5, return_topk=2)[0]
    assert list(out.token_ids) == [first] and out.text == ""
    assert len(out.topk_token_ids) == 1


def test_units_removal_and_scoring_through_the_wrapper(wrapper: HFVLMWrapper) -> None:
    units = wrapper.list_prunable_units()
    kinds = {t: sum(u.unit_type is t for u in units) for t in UnitType}
    assert kinds[UnitType.DECODER_LAYER] == 3 and kinds[UnitType.ATTENTION_HEAD] == 6
    before_score = wrapper.score_yes_no([_image(1)], [PROBE])[0]
    before = wrapper.num_parameters()
    layer = next(u for u in units if u.unit_type is UnitType.DECODER_LAYER and u.layer_index == 1)
    with surgery.skip_layer(wrapper.module, 1):
        ablated = wrapper.score_yes_no([_image(1)], [PROBE])[0]
    assert ablated.logp_yes != pytest.approx(before_score.logp_yes, abs=1e-6)
    remove_units(wrapper, [layer])
    assert verify_physical_removal(before, wrapper, [layer])["ok"]
    assert wrapper.num_parameters() == before - layer.num_params
    removed = wrapper.score_yes_no([_image(1)], [PROBE])[0]
    assert removed.logp_yes == pytest.approx(ablated.logp_yes, abs=1e-4)
    assert wrapper.generate([_image(1)], [PROBE], max_new_tokens=2)  # cache works after removal


def test_tokenizer_fingerprint_and_sharing(wrapper: HFVLMWrapper) -> None:
    same = HFVLMWrapper(_tiny_model(wrapper.processor), _tiny_processor())
    different = HFVLMWrapper(_tiny_model(wrapper.processor), _tiny_processor(("extra",)))
    assert wrapper.tokenizer_fingerprint().startswith("fast:")
    check_shared_tokenizer(wrapper, same)
    with pytest.raises(ValueError, match="tokenizers differ"):
        check_shared_tokenizer(wrapper, different)


def test_images_are_shrunk_to_the_longest_side() -> None:
    big = _image(1, (200, 100))
    assert _to_pil(big, 50).size == (50, 25)
    assert _to_pil(big, None).size == (200, 100)
    assert _to_pil(big, 500).size == (200, 100)


def test_image_paths_are_opened(tmp_path: Path) -> None:
    path = tmp_path / "x.png"
    _image(1, (40, 30)).save(path)
    assert _to_pil(path, None).size == (40, 30)
    assert _to_pil(str(path), 20).size == (20, 15)


def test_mismatched_images_and_questions_are_rejected(wrapper: HFVLMWrapper) -> None:
    with pytest.raises(ValueError, match="one image per question"):
        wrapper.build_inputs([_image(1)], [PROBE, PROBE])


def test_adapter_merge_matches_the_adapted_model(tmp_path: Path) -> None:
    peft = pytest.importorskip("peft")
    processor = _tiny_processor()
    base = _tiny_model(processor)
    lora = peft.LoraConfig(
        r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], init_lora_weights=False
    )
    adapted = peft.get_peft_model(base, lora).eval()
    adapted.save_pretrained(tmp_path / "adapter")
    reference = HFVLMWrapper(adapted, processor)
    expected = reference.score_yes_no([_image(1)], [PROBE])[0]

    fresh = HFVLMWrapper(_tiny_model(processor), processor)
    plain = fresh.score_yes_no([_image(1)], [PROBE])[0]
    load_adapter(fresh, tmp_path / "adapter", merge=True)
    merged = fresh.score_yes_no([_image(1)], [PROBE])[0]
    assert merged.logp_yes == pytest.approx(expected.logp_yes, abs=1e-4)
    assert merged.logp_yes != pytest.approx(plain.logp_yes, abs=1e-6)
    assert not any("lora" in n for n, _ in fresh.module.named_parameters())


# ------------------------------------------------------- smoke measurements


def test_synthetic_images_and_probe_questions_are_reproducible() -> None:
    from medcompress.models.smoke import probe_questions, synthetic_images

    a, b = synthetic_images(2, size=(40, 30), seed=3), synthetic_images(2, size=(40, 30), seed=3)
    assert a[0].tobytes() == b[0].tobytes() and a[0].tobytes() != a[1].tobytes()
    assert a[0].size == (40, 30) and a[0].mode == "RGB"
    assert probe_questions(["polyp"])[0] == "Is there a polyp in this image? Answer yes or no."


def test_measure_model_and_layer_removal(wrapper: HFVLMWrapper) -> None:
    from medcompress.models.smoke import measure_layer_removal, measure_model, synthetic_images

    images = synthetic_images(3, size=(84, 56))
    questions = [PROBE, PROBE, "What do you see ?"]
    rows, texts = measure_model(wrapper, images, questions, "m", generate_tokens=3, repeats=1)
    values = {r["metric"]: r["value"] for r in rows}
    assert values["m_scores_finite"] == 1.0 and len(texts) == 1
    assert values["m_scoring_seconds_per_probe"] > 0 and values["m_generate_tokens_per_second"] > 0
    assert 0 < values["m_yes_no_probability_mass"] <= 1
    gpu_names = {name for name in values if "gpu" in name}
    if torch.cuda.is_available():  # GPU session: the peaks come from the CUDA allocator
        assert {"m_peak_gpu_bytes_scoring", "m_peak_gpu_bytes_generate"} <= gpu_names
    else:
        assert not gpu_names  # no CUDA device: no GPU numbers are invented
    assert all(r["experiment"] == "RC-M1" and r["budget"] == "" for r in rows)
    removal = {
        r["metric"]: r["value"] for r in measure_layer_removal(wrapper, 1, images, questions, "m")
    }
    assert removal["m_parameter_drop_matches"] == 1.0
    assert removal["m_scores_change_after_removal"] == 1.0
    assert wrapper.module.config.get_text_config().num_hidden_layers == 2
    with pytest.raises(ValueError, match="no decoder layer"):
        measure_layer_removal(wrapper, 9, images, questions, "m")


def _save_tiny_checkpoint(directory: Path) -> Path:
    processor = _tiny_processor()
    model = _tiny_model(processor)
    model.save_pretrained(directory)
    processor.save_pretrained(directory)
    return directory


def _write_model_config(path: Path, role: str, source: Path, **extra: object) -> Path:
    import yaml

    section = {
        "name": f"tiny-{role}",
        "role": role,
        "family": "qwen2_5_vl",
        "source": str(source),
        "status": "DECIDED",
        "dtype": "float32",
        **extra,
    }
    path.write_text(yaml.safe_dump({"model": section}), encoding="utf-8")
    return path


def test_smoke_script_end_to_end_on_cpu(tmp_path: Path) -> None:
    import importlib.util

    from medcompress.evaluation.aggregate import validate_results_tree

    teacher_dir = _save_tiny_checkpoint(tmp_path / "teacher")
    student_dir = _save_tiny_checkpoint(tmp_path / "student")
    teacher_cfg = _write_model_config(tmp_path / "teacher.yaml", "teacher", teacher_dir)
    student_cfg = _write_model_config(tmp_path / "student.yaml", "student", student_dir)
    script = Path(__file__).resolve().parents[1] / "scripts" / "smoke_models.py"
    spec = importlib.util.spec_from_file_location("smoke_models_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "results" / "smoke" / "RC-M1.csv"
    code = module.main(
        [
            "--teacher-config", str(teacher_cfg),
            "--student-config", str(student_cfg),
            "--out", str(out),
            "--image-longest-side", "56",
        ]
    )  # fmt: skip
    assert code == 0 and out.is_file()
    assert validate_results_tree(tmp_path / "results") == []
    import pandas as pd

    metrics = dict(zip(*pd.read_csv(out)[["metric", "value"]].T.values, strict=True))
    assert metrics["tokenizers_identical"] == 1.0
    assert (
        metrics["teacher_scores_finite"] == 1.0 and metrics["student_parameter_drop_matches"] == 1.0
    )
    assert metrics["student_params"] > 0 and metrics["student_removed_layer_parameters"] > 0


def test_fingerprint_ignores_padding_state() -> None:
    from medcompress.models.tokens import fingerprint_tokenizer

    fresh, used = _tiny_processor().tokenizer, _tiny_processor().tokenizer
    before = fingerprint_tokenizer(fresh)
    used(["Is there a polyp", "Yes"], padding=True, truncation=True, max_length=8)
    assert fingerprint_tokenizer(used) == before == fingerprint_tokenizer(fresh)


# ------------------------------------------------------------- adapter loading


def _save_adapter(tmp_path: Path, *, trained: bool, old_layout: bool = False) -> tuple[Path, float]:
    """Save a LoRA adapter for the tiny model; return its directory and its probe score."""
    peft = pytest.importorskip("peft")
    from safetensors.torch import load_file, save_file

    processor = _tiny_processor()
    regex = r"^(model).*\.(gate_proj|o_proj|k_proj|up_proj|down_proj|q_proj|v_proj)$"
    # default initialisation (lora_B = 0), as in a real adapter file; "trained" then
    # overwrites every LoRA matrix with values a fresh load cannot reproduce by chance
    config = peft.LoraConfig(r=4, lora_alpha=8, target_modules=regex)
    adapted = peft.get_peft_model(_tiny_model(processor), config).eval()
    if trained:
        generator = torch.Generator().manual_seed(123)
        with torch.no_grad():
            for name, param in adapted.named_parameters():
                if "lora_" in name:
                    param.copy_(0.2 * torch.randn(param.shape, generator=generator))
    score = HFVLMWrapper(adapted, processor).score_yes_no([_image(1)], [PROBE])[0].logp_yes
    adapted.save_pretrained(tmp_path)
    if old_layout:  # module names as written before the language_model / visual split
        weights = load_file(tmp_path / "adapter_model.safetensors")
        renamed = {
            k.replace("model.language_model.", "model.").replace("model.visual.", "visual."): v
            for k, v in weights.items()
        }
        save_file(renamed, tmp_path / "adapter_model.safetensors")
    return tmp_path, score


@pytest.mark.parametrize("old_layout", [False, True], ids=["current_names", "old_names"])
def test_adapter_regex_targets_and_old_module_names_load(tmp_path: Path, old_layout: bool) -> None:
    """The published teacher adapter targets modules by regex and was saved with older names."""
    directory, expected = _save_adapter(tmp_path, trained=True, old_layout=old_layout)
    processor = _tiny_processor()
    fresh = HFVLMWrapper(_tiny_model(processor), processor)
    load_adapter(fresh, directory, merge=True)
    got = fresh.score_yes_no([_image(1)], [PROBE])[0].logp_yes
    assert got == pytest.approx(expected, abs=1e-4)


def test_untrained_adapter_is_rejected(tmp_path: Path) -> None:
    directory, _ = _save_adapter(tmp_path, trained=False)  # lora_B stays at zero
    processor = _tiny_processor()
    fresh = HFVLMWrapper(_tiny_model(processor), processor)
    with pytest.raises(ModelConfigError, match="untrained"):
        load_adapter(fresh, directory)
    load_adapter(fresh, directory, check_trained=False)  # explicit opt-out still works


def test_teacher_cache_from_the_real_wrapper(wrapper: HFVLMWrapper, tmp_path: Path) -> None:
    from medcompress.data.base import VQASample
    from medcompress.models.teacher_cache import cache_teacher_outputs, load_teacher_cache

    samples = []
    for i in range(3):
        _image(i + 1).save(tmp_path / f"{i}.png")
        samples.append(VQASample(f"s{i}", "d", f"{i}.png", PROBE, "yes"))
    out = cache_teacher_outputs(
        wrapper,
        samples,
        tmp_path / "cache",
        top_k=4,
        max_new_tokens=3,
        shard_size=2,
        data_root=tmp_path,
    )
    records = list(load_teacher_cache(out))
    direct = wrapper.generate([tmp_path / "1.png"], [PROBE], max_new_tokens=3, return_topk=4)[0]
    assert [r.sample_id for r in records] == ["s0", "s1", "s2"]
    assert records[1].answer_token_ids.tolist() == list(direct.token_ids)
    assert records[1].topk_token_ids[:, 0].tolist() == list(direct.token_ids)  # greedy = top-1
    assert records[1].topk_token_ids.shape == (len(direct.token_ids), 4)
    assert (records[1].topk_logprobs[:, :-1] >= records[1].topk_logprobs[:, 1:]).all()
