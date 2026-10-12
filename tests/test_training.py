"""Losses, batching, adapters and training loops on a tiny offline model (heavy: needs torch).

The tiny Qwen2.5-VL of ``tiny_vlm.py`` has random weights, so the tests check
mechanics rather than quality: that batched padded forward passes agree with one-example
passes, that adapters leave the base model alone and merge exactly, that losses fall,
that runs are reproducible by seed, and that the stage entry writes what it should.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torchvision")
pytest.importorskip("tokenizers")
pytest.importorskip("peft")
pytest.importorskip("PIL.Image")

import tiny_vlm  # noqa: E402

from medcompress.compression import lora  # noqa: E402
from medcompress.compression.distillation import batching, losses, offline_kd  # noqa: E402
from medcompress.compression.distillation.batching import TrainExample  # noqa: E402
from medcompress.compression.distillation.losses import IGNORE_INDEX  # noqa: E402
from medcompress.compression.distillation.offline_kd import TrainConfig  # noqa: E402
from medcompress.compression.lora import LoRAConfig  # noqa: E402
from medcompress.config import RunContext  # noqa: E402
from medcompress.models.hf import HFVLMWrapper  # noqa: E402
from medcompress.models.loading import load_adapter  # noqa: E402
from medcompress.models.teacher_cache import CacheRecord  # noqa: E402

F = torch.nn.functional
PROBE = tiny_vlm.PROBE
ANSWERS = ["Yes", "No", "Yes a polyp", "No in this image"]
LORA = LoRAConfig(rank=4, alpha=8, dropout=0.0, target_modules=("decoder_attention", "decoder_mlp"))


def _wrapper(seed: int = 0) -> HFVLMWrapper:
    return tiny_vlm.tiny_wrapper(seed)


def _original_name(name: str) -> str:
    """Parameter name in the plain model for a name in the adapter-wrapped model."""
    return name.replace("base_model.model.", "", 1).replace("base_layer.", "")


def _config(**over: object) -> TrainConfig:
    base = TrainConfig(
        mode="sft",
        lora=LORA,
        steps=40,
        learning_rate=1e-2,
        batch_size=2,
        grad_accumulation=1,
        kd_weight=0.0,
        kd_temperature=1.0,
        data_subset="all",
    )
    return dataclasses.replace(base, **over)


def _examples(n: int = 4) -> list[TrainExample]:
    return [
        TrainExample(f"s{i}", tiny_vlm.tiny_image(i, (84, 56) if i % 2 else (56, 56)), PROBE,
                     ANSWERS[i % len(ANSWERS)])
        for i in range(n)
    ]  # fmt: skip


def _teacher_records(
    examples: list[TrainExample], *, answers: list[str] | None = None, top_k: int = 5
) -> list[TrainExample]:
    """Attach teacher-forced top-k records from a second random model (different seed)."""
    teacher = _wrapper(seed=1)
    pad = teacher.tokenizer.pad_token_id
    out = []
    for i, ex in enumerate(examples):
        text = answers[i] if answers else ex.answer
        ids = batching.ground_truth_ids(teacher, text)
        prompt = batching.prompt_inputs(teacher, ex.image, ex.question)
        batch = batching.assemble([prompt], [ids], pad)
        with torch.no_grad():
            logits = offline_kd.forward_answer_logits(teacher.module, batch)[0]
        top = F.log_softmax(logits.float(), dim=-1).topk(top_k, dim=-1)
        record = CacheRecord(
            ex.sample_id,
            text,
            np.array(ids, dtype=np.int64),
            top.indices.numpy().astype(np.int64),
            top.values.numpy().astype(np.float32),
        )
        out.append(dataclasses.replace(ex, teacher=record))
    return out


# ------------------------------------------------------------------------ losses


def test_answer_ce_matches_manual_cross_entropy_and_ignores_padding() -> None:
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 3, 7, generator=gen)
    targets = torch.tensor([[IGNORE_INDEX, 2, 4], [1, 3, IGNORE_INDEX]])
    valid = targets != IGNORE_INDEX
    expected = F.cross_entropy(logits[valid], targets[valid])
    assert torch.allclose(losses.answer_ce_loss(logits, targets), expected)
    changed = logits.clone()
    changed[~valid] += 5.0  # padded positions must not matter
    assert torch.allclose(losses.answer_ce_loss(changed, targets), expected)


def test_answer_ce_without_valid_targets_is_zero_and_differentiable() -> None:
    logits = torch.randn(1, 2, 5, requires_grad=True)
    loss = losses.answer_ce_loss(logits, torch.full((1, 2), IGNORE_INDEX))
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    with pytest.raises(ValueError, match="do not match"):
        losses.answer_ce_loss(torch.zeros(1, 2, 5), torch.zeros(1, 3, dtype=torch.long))


def _topk_case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 3, 9, generator=gen)
    ids = torch.stack([torch.randperm(9, generator=gen)[:4] for _ in range(6)]).reshape(2, 3, 4)
    logp = F.log_softmax(torch.randn(2, 3, 4, generator=gen), dim=-1)
    mask = torch.tensor([[False, True, True], [True, True, False]])
    return logits, ids, logp, mask


def test_topk_kl_is_zero_when_student_reproduces_the_teacher_top_k() -> None:
    _, ids, logp, mask = _topk_case()
    student = torch.full((2, 3, 9), -20.0).scatter(-1, ids, logp)
    assert losses.topk_kl_loss(student, ids, logp, mask).item() == pytest.approx(0.0, abs=1e-5)


def test_topk_kl_matches_manual_formula_and_temperature_scaling() -> None:
    logits, ids, logp, mask = _topk_case()
    for temperature in (1.0, 2.0):
        q = F.log_softmax(logits.gather(-1, ids) / temperature, dim=-1)
        p = F.log_softmax(logp / temperature, dim=-1)
        per_position = (p.exp() * (p - q)).sum(-1)
        expected = per_position[mask].mean() * temperature**2
        got = losses.topk_kl_loss(logits, ids, logp, mask, temperature=temperature)
        assert torch.allclose(got, expected, atol=1e-6)
        assert got.item() > 0


def test_topk_kl_ignores_masked_positions_and_trains_only_the_cached_tokens() -> None:
    logits, ids, logp, mask = _topk_case()
    base = losses.topk_kl_loss(logits, ids, logp, mask)
    other = logits.clone()
    other[~mask] += 3.0
    assert torch.allclose(losses.topk_kl_loss(other, ids, logp, mask), base)
    leaf = logits.clone().requires_grad_(True)
    losses.topk_kl_loss(leaf, ids, logp, mask).backward()
    untouched = torch.ones_like(leaf, dtype=torch.bool).scatter(-1, ids, False)
    assert leaf.grad[untouched].abs().sum() == 0 and leaf.grad.abs().sum() > 0


def test_topk_kl_edge_cases() -> None:
    logits, ids, logp, mask = _topk_case()
    assert losses.topk_kl_loss(logits, ids, logp, torch.zeros_like(mask)).item() == 0.0
    with pytest.raises(ValueError, match="temperature"):
        losses.topk_kl_loss(logits, ids, logp, mask, temperature=0.0)
    with pytest.raises(ValueError, match="share batch and length"):
        losses.topk_kl_loss(logits, ids[:, :2], logp, mask)


def test_combined_loss_weights() -> None:
    ce, kl = torch.tensor(2.0), torch.tensor(4.0)
    assert losses.combined_kd_loss(ce, kl, kd_weight=0.25).item() == pytest.approx(2.5)
    assert losses.combined_kd_loss(ce, kl, kd_weight=0.0).item() == 2.0
    assert losses.combined_kd_loss(ce, kl, kd_weight=1.0).item() == 4.0
    with pytest.raises(ValueError, match="between 0 and 1"):
        losses.combined_kd_loss(ce, kl, kd_weight=1.5)


# ---------------------------------------------------------------------- batching


def test_assemble_left_pads_prompts_and_right_aligns_answers() -> None:
    prompts = [
        {"input_ids": torch.tensor([[5, 6, 7]]), "pixel_values": torch.ones(2, 4)},
        {"input_ids": torch.tensor([[8, 9]]), "pixel_values": torch.zeros(3, 4)},
    ]
    batch = batching.assemble(prompts, [[1, 2], [3]], pad_id=0)
    assert batch.inputs["input_ids"].tolist() == [[5, 6, 7, 1, 2], [0, 0, 8, 9, 3]]
    assert batch.inputs["attention_mask"].tolist() == [[1] * 5, [0, 0, 1, 1, 1]]
    assert batch.targets.tolist() == [[1, 2], [IGNORE_INDEX, 3]]
    assert batch.answer_length == 2 and batch.topk_ids is None
    assert batch.inputs["pixel_values"].shape == (5, 4)  # image tensors are concatenated


def test_assemble_aligns_teacher_top_k_with_the_answer_slots() -> None:
    prompts = [{"input_ids": torch.tensor([[5, 6]])}, {"input_ids": torch.tensor([[7]])}]
    top = [
        (np.array([[1, 2], [3, 4]]), np.array([[-0.1, -2.0], [-0.2, -3.0]], dtype=np.float32)),
        (np.array([[9, 8]]), np.array([[-0.5, -1.5]], dtype=np.float32)),
    ]
    batch = batching.assemble(prompts, [[1, 3], [9]], pad_id=0, topk=top)
    assert batch.topk_mask.tolist() == [[True, True], [False, True]]
    assert batch.topk_ids[1, 1].tolist() == [9, 8] and batch.topk_ids[1, 0].tolist() == [0, 0]
    assert batch.topk_logprobs[0, 1, 1].item() == pytest.approx(-3.0)


def test_assemble_rejects_bad_input() -> None:
    prompt = {"input_ids": torch.tensor([[1, 2]])}
    with pytest.raises(ValueError, match="one answer per prompt"):
        batching.assemble([prompt], [[1], [2]], pad_id=0)
    with pytest.raises(ValueError, match="must not be empty"):
        batching.assemble([prompt], [[]], pad_id=0)
    with pytest.raises(ValueError, match="last answer_length"):
        batching.answer_logits(torch.zeros(1, 2, 5), 3)


def test_ground_truth_ids_end_with_the_stop_token() -> None:
    w = _wrapper()
    ids = batching.ground_truth_ids(w, "  Yes a polyp ")
    tok = w.tokenizer.convert_tokens_to_ids
    assert ids == [tok("Yes"), tok("a"), tok("polyp"), w.stop_token_id()]
    assert w.stop_token_id() == tok("<|im_end|>")


def test_padded_batch_logits_equal_one_example_forward_passes() -> None:
    w = _wrapper()
    pad = w.tokenizer.pad_token_id
    images = [tiny_vlm.tiny_image(1), tiny_vlm.tiny_image(2, (56, 56)), tiny_vlm.tiny_image(3)]
    answers = [
        batching.ground_truth_ids(w, text) for text in ("Yes a polyp", "No", "No in this image")
    ]
    prompts = [batching.prompt_inputs(w, img, PROBE) for img in images]
    batch = batching.assemble(prompts, answers, pad)
    with torch.no_grad():
        together = offline_kd.forward_answer_logits(w.module, batch)
        for b, ids in enumerate(answers):
            single = batching.assemble([prompts[b]], [ids], pad)
            alone = offline_kd.forward_answer_logits(w.module, single)[0]
            assert torch.allclose(together[b, -len(ids) :], alone, atol=1e-4)
            assert torch.equal(batch.targets[b, -len(ids) :], torch.tensor(ids))
            assert (batch.targets[b, : -len(ids)] == IGNORE_INDEX).all()


# -------------------------------------------------------------------------- lora


def test_attach_lora_trains_only_decoder_adapters() -> None:
    w = _wrapper()
    base_before = {n: p.detach().clone() for n, p in w.module.named_parameters()}
    lora.attach_lora(w, LORA)
    trainable = [n for n, p in w.module.named_parameters() if p.requires_grad]
    assert len(trainable) == 3 * 7 * 2  # layers x projections x (A, B)
    assert all("lora_" in n and "language_model.layers" in n for n in trainable)
    assert not any("visual" in n for n in trainable)
    for n, p in w.module.named_parameters():
        if "lora_" not in n:
            assert not p.requires_grad
            assert torch.equal(p, base_before[_original_name(n)])


def test_attach_lora_target_groups_and_errors() -> None:
    w = _wrapper()
    attention = LoRAConfig(4, 8, 0.0, ("decoder_attention",))
    lora.attach_lora(w, attention)
    names = [n for n, p in w.module.named_parameters() if p.requires_grad]
    assert len(names) == 3 * 4 * 2 and not any("mlp" in n for n in names)
    with pytest.raises(ValueError, match="already attached"):
        lora.attach_lora(w, attention)
    with pytest.raises(ValueError, match="unknown LoRA target"):
        lora.attach_lora(_wrapper(), LoRAConfig(4, 8, 0.0, ("vision",)))


def _logits(w: HFVLMWrapper) -> torch.Tensor:
    ids = batching.ground_truth_ids(w, "Yes a polyp")
    prompt = batching.prompt_inputs(w, tiny_vlm.tiny_image(5), PROBE)
    batch = batching.assemble([prompt], [ids], w.tokenizer.pad_token_id)
    with torch.no_grad():
        return offline_kd.forward_answer_logits(w.module, batch)


def _perturb(w: HFVLMWrapper) -> None:
    gen = torch.Generator().manual_seed(7)
    for n, p in w.module.named_parameters():
        if "lora_B" in n:
            p.data = torch.randn(p.shape, generator=gen) * 0.2


def test_fresh_adapter_changes_nothing() -> None:
    w = _wrapper()
    before = _logits(w)
    lora.attach_lora(w, LORA)
    assert torch.allclose(_logits(w), before, atol=1e-6)


def test_merge_adapter_reproduces_the_adapted_model_and_drops_the_layers() -> None:
    import peft

    w = _wrapper()
    base = _logits(w)
    lora.attach_lora(w, LORA)
    _perturb(w)
    adapted = _logits(w)
    assert not torch.allclose(adapted, base, atol=1e-3)
    lora.merge_adapter(w)
    assert not isinstance(w.module, peft.PeftModel)
    assert not any("lora_" in n for n, _ in w.module.named_parameters())
    assert torch.allclose(_logits(w), adapted, atol=1e-4)


def test_saved_adapter_loads_into_a_fresh_model_with_the_same_outputs(tmp_path: Path) -> None:
    w = _wrapper()
    lora.attach_lora(w, LORA)
    _perturb(w)
    expected = _logits(w)
    folder = lora.save_adapter(w, tmp_path / "adapter")
    assert (folder / "adapter_config.json").is_file()
    assert sum(f.stat().st_size for f in folder.iterdir()) < 200_000  # adapter only, no base
    fresh = _wrapper()
    load_adapter(fresh, folder)
    assert torch.allclose(_logits(fresh), expected, atol=1e-5)


def test_save_and_merge_need_an_adapter(tmp_path: Path) -> None:
    w = _wrapper()
    with pytest.raises(ValueError, match="no adapter"):
        lora.save_adapter(w, tmp_path / "x")
    with pytest.raises(ValueError, match="no adapter"):
        lora.merge_adapter(w)


# ---------------------------------------------------------------------- training


def test_answer_fine_tuning_lowers_the_loss_and_only_moves_the_adapter() -> None:
    w = _wrapper()
    frozen = {n: p.detach().clone() for n, p in w.module.named_parameters()}
    result = offline_kd.train_student_sft(w, _examples(), _config(), seed=0)
    assert result.steps == 40 and result.examples_seen == 80
    assert len(result.losses) == 40 and result.kd_losses == [0.0] * 40
    assert result.final_loss() < 0.5 * sum(result.losses[:3]) / 3
    assert result.seconds > 0 and not w.module.training
    for name, param in w.module.named_parameters():
        if "lora_" not in name:
            assert torch.equal(param, frozen[_original_name(name)]), name


def test_offline_distillation_pulls_the_student_toward_the_teacher() -> None:
    examples = _teacher_records(_examples())
    cfg = _config(mode="offline_kd", kd_weight=1.0, steps=40)
    result = offline_kd.train_offline_kd(_wrapper(), examples, cfg, seed=0)
    first, last = sum(result.kd_losses[:3]) / 3, result.final_loss()
    assert result.ce_losses[0] > 0 and first > 0
    assert last < 0.5 * first


def test_mixed_loss_reports_both_terms() -> None:
    examples = _teacher_records(_examples())
    cfg = _config(mode="offline_kd", kd_weight=0.5, steps=3)
    result = offline_kd.train_offline_kd(_wrapper(), examples, cfg, seed=0)
    for total, ce, kl in zip(result.losses, result.ce_losses, result.kd_losses, strict=True):
        assert total == pytest.approx(0.5 * ce + 0.5 * kl, rel=1e-4)


def test_gradient_accumulation_counts_examples() -> None:
    cfg = _config(steps=3, batch_size=1, grad_accumulation=3)
    result = offline_kd.train_student_sft(_wrapper(), _examples(), cfg, seed=0)
    assert result.steps == 3 and result.examples_seen == 9


def test_training_is_reproducible_by_seed() -> None:
    cfg = _config(steps=6)
    runs = [offline_kd.train_student_sft(_wrapper(), _examples(), cfg, seed=s) for s in (3, 3, 4)]
    assert runs[0].losses == runs[1].losses
    assert runs[0].losses != runs[2].losses


def test_one_forward_pass_serves_both_terms_when_teacher_and_ground_truth_agree() -> None:
    cfg = _config(mode="offline_kd", kd_weight=0.5)
    examples = _teacher_records(_examples(2))
    w = _wrapper()
    lora.attach_lora(w, LORA)
    _perturb(w)
    shared, extra = offline_kd._micro_batches(w, examples, cfg)
    assert extra is None and shared.topk_ids is not None
    prompts = [batching.prompt_inputs(w, e.image, e.question) for e in examples]
    pad = w.tokenizer.pad_token_id
    gt_ids = [batching.ground_truth_ids(w, e.answer) for e in examples]
    plain = batching.assemble(prompts, gt_ids, pad)
    teacher = batching.assemble(
        prompts,
        gt_ids,
        pad,
        topk=[(e.teacher.topk_token_ids, e.teacher.topk_logprobs) for e in examples],
    )
    with torch.no_grad():
        one = offline_kd._step_loss(w.module, shared, None, cfg)
        two = offline_kd._step_loss(w.module, plain, teacher, cfg)
    assert [float(x) for x in one] == pytest.approx([float(x) for x in two], abs=1e-5)


def test_different_teacher_answer_gets_its_own_pass_and_leaves_cross_entropy_alone() -> None:
    examples = _teacher_records(_examples(2), answers=["No", "Yes"])  # differs from ground truth
    w = _wrapper()
    lora.attach_lora(w, LORA)
    _perturb(w)
    mixed = _config(mode="offline_kd", kd_weight=0.5)
    shared, extra = offline_kd._micro_batches(w, examples, mixed)
    assert extra is not None and shared.topk_ids is None
    sft = _config(kd_weight=0.0)
    gt_only, none = offline_kd._micro_batches(w, examples, sft)
    assert none is None
    with torch.no_grad():
        _, ce_mixed, kl = offline_kd._step_loss(w.module, shared, extra, mixed)
        _, ce_plain, zero = offline_kd._step_loss(w.module, gt_only, None, sft)
    assert ce_mixed == pytest.approx(ce_plain, abs=1e-6) and zero == 0.0 and kl > 0


def test_training_refuses_inconsistent_requests() -> None:
    w = _wrapper()
    plain = _examples(2)
    with pytest.raises(ValueError, match="kd_weight = 0"):
        offline_kd.train_student_sft(w, plain, _config(kd_weight=0.5), seed=0)
    with pytest.raises(ValueError, match="kd_weight > 0"):
        offline_kd.train_offline_kd(w, plain, _config(mode="offline_kd"), seed=0)
    with pytest.raises(ValueError, match="no cached teacher record"):
        offline_kd.train_offline_kd(w, plain, _config(mode="offline_kd", kd_weight=0.5), seed=0)
    with pytest.raises(ValueError, match="no training examples"):
        offline_kd.train_student_sft(w, [], _config(), seed=0)
    with pytest.raises(ValueError, match="cached teacher record for every example"):
        offline_kd._micro_batches(w, plain, _config(mode="offline_kd", kd_weight=0.5))


# ------------------------------------------------------------------- stage entry


def _stage_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, mode: str, kd_weight: float
) -> tuple[dict, dict]:
    """Run ``run_student_training`` with the model, dataset and cache replaced by tiny stand-ins."""
    import medcompress.config as config_module
    import medcompress.data.prepared as prepared_module
    import medcompress.models.loading as loading_module
    import medcompress.models.teacher_cache as cache_module
    from medcompress.data.base import VQASample

    data_root = tmp_path / "data"
    data_root.mkdir()
    samples, examples = [], _examples(4)
    for ex in examples:
        ex.image.save(data_root / f"{ex.sample_id}.png")
        samples.append(
            VQASample(ex.sample_id, "tiny", f"{ex.sample_id}.png", ex.question, ex.answer)
        )
    records = [e.teacher for e in _teacher_records(examples)]

    real_load_config = config_module.load_config
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda path, *a, **k: {"dataset": {}, "model": {}}
        if str(path).startswith("fake/")
        else real_load_config(path, *a, **k),
    )
    monkeypatch.setattr(prepared_module, "load_prepared", lambda *a, **k: {"train": samples})
    monkeypatch.setattr(cache_module, "load_teacher_cache", lambda *a, **k: iter(records))
    monkeypatch.setattr(
        loading_module, "ModelSpec", SimpleNamespace(from_config=lambda cfg: "tiny-spec")
    )
    seen = {}

    def fake_load_model(spec: object, **kwargs: object) -> HFVLMWrapper:
        seen.update(kwargs)
        return _wrapper()

    monkeypatch.setattr(loading_module, "load_model", fake_load_model)
    config = {
        "training": {
            "mode": mode,
            "lora": {"rank": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["decoder_mlp"]},
            "steps": 3,
            "learning_rate": 1e-2,
            "batch_size": 2,
            "grad_accumulation": 1,
            "kd_weight": kd_weight,
            "kd_temperature": 1.0,
            "data_subset": 3,
        },
        "model": {"config": "fake/student.yaml"},
        "inputs": {"image_longest_side": 56},
        "seeds": {"data_split": 0},
        "components": {"datasets": {"train_qa": "fake/data.yaml"}},
        "paths": {
            "data_root": str(data_root),
            "private_root": str(tmp_path / "private"),
            "cache_root": str(tmp_path / "cache"),
            "checkpoint_root": str(tmp_path / "ckpt"),
            "results_root": str(tmp_path / "results"),
        },
    }
    return config, seen


@pytest.mark.parametrize(("mode", "kd_weight"), [("sft", 0.0), ("offline_kd", 0.5)])
def test_stage_entry_trains_saves_the_adapter_and_writes_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, kd_weight: float
) -> None:
    config, load_kwargs = _stage_setup(monkeypatch, tmp_path, mode=mode, kd_weight=kd_weight)
    offline_kd.run_student_training(config, RunContext("E2", seed=2))
    assert load_kwargs["allow_candidate"] is True and load_kwargs["image_longest_side"] == 56
    adapter = tmp_path / "ckpt" / "E2" / "seed2" / "adapter"
    assert (adapter / "adapter_config.json").is_file()
    with (tmp_path / "results" / "E2" / "nobudget" / "seed2" / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    metrics = {r["metric"]: float(r["value"]) for r in rows}
    assert metrics["train_steps"] == 3 and metrics["train_examples_seen"] == 6
    assert metrics["train_loss_final"] > 0 and metrics["train_seconds"] > 0
    assert {r["experiment"] for r in rows} == {"E2"} and {int(r["seed"]) for r in rows} == {2}
