"""E4 on torch: gated decoupled distillation, the LSQ quantizer and the stage (tiny model)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torchvision")
pytest.importorskip("tokenizers")
pytest.importorskip("peft")
pytest.importorskip("PIL.Image")

import tiny_vlm  # noqa: E402
from eval_fixtures import write_dataset  # noqa: E402
from kd_fixtures import teacher_records  # noqa: E402

from medcompress.compression.quantization import grid  # noqa: E402
from medcompress.compression.quantization.kernels import DECODER_TARGETS  # noqa: E402
from medcompress.compression.quantization.ptq import (  # noqa: E402
    REFERENCE_KERNEL,
    load_quantized,
    quantized_dir,
    read_quantized,
)
from medcompress.compression.quantization.qakd import (  # noqa: E402
    _qat_linear_class,
    export_qat,
    gated_dkd_loss,
    gated_dkd_numpy,
    install_qat,
    lsq_quantize,
    lsq_reference,
    run_qakd_training,
)
from medcompress.config import RunContext  # noqa: E402


def test_the_gated_loss_matches_the_reference_and_has_gradients() -> None:
    rng = np.random.default_rng(0)
    logits = torch.tensor(rng.normal(size=(2, 4, 30)), requires_grad=True)
    ids = torch.tensor(rng.permutation(30)[:5][None, None, :].repeat(2, 0).repeat(4, 1))
    logprobs = torch.log_softmax(torch.tensor(rng.normal(size=(2, 4, 5))), -1) - 0.1
    targets = ids[..., 0].clone()
    targets[1, 2] = next(t for t in range(30) if t not in ids[0, 0].tolist())  # not cached
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    got = gated_dkd_loss(logits, ids, logprobs, targets, mask)
    ref = gated_dkd_numpy(
        logits.detach().numpy(), ids.numpy(), logprobs.numpy(), targets.numpy(), mask.numpy()
    )
    assert float(got) == pytest.approx(ref, rel=1e-5)
    got.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_lsq_matches_the_reference_forward_and_step_gradient() -> None:
    rng = np.random.default_rng(1)
    w = torch.tensor(rng.normal(size=(3, 32)), dtype=torch.float32, requires_grad=True)
    start = grid.symmetric_scales(w.detach().numpy(), 4, 16) * 0.7  # some weights clip
    s = torch.tensor(start, dtype=torch.float32, requires_grad=True)
    out = lsq_quantize(w, s, 4, 16)
    fake, d_s = lsq_reference(w.detach().numpy(), s.detach().numpy(), 4, 16)
    assert np.allclose(out.detach().numpy(), fake, atol=1e-6)
    upstream = torch.tensor(rng.normal(size=(3, 32)), dtype=torch.float32)
    out.backward(upstream)
    g = 1 / np.sqrt(16 * 7)
    expect_s = (upstream.numpy() * d_s * g).reshape(3, 2, 16).sum(-1)
    assert np.allclose(s.grad.numpy(), expect_s, atol=1e-5)
    ratio = w.detach().numpy() / grid.expand_groups(start, 16)
    inside = (ratio > -8) & (ratio < 7)
    assert np.allclose(w.grad.numpy(), upstream.numpy() * inside)


def test_the_export_is_exactly_the_trained_layer() -> None:
    linear = torch.nn.Linear(32, 8)
    qat = _qat_linear_class()(linear, bits=4, group_size=16, rank=2, alpha=4.0)
    with torch.no_grad():
        qat.lora_B.normal_()
        qat.step.mul_(1.1)
    x = torch.randn(5, 32)
    weight = qat.merged().double().detach().numpy()
    scales = qat.step.abs().double().detach().numpy()
    codes = grid.quantize_codes(weight, scales, 4, 16)
    exported = torch.tensor(grid.dequantize(codes, scales, 4, 16), dtype=torch.float32)
    expected = torch.nn.functional.linear(x, exported, linear.bias)
    assert torch.allclose(qat(x), expected, atol=1e-5)


def _config(root: Path) -> dict:
    return {
        "seeds": {"data_split": 0},
        "paths": {
            "data_root": str(root),
            "checkpoint_root": str(root / "ckpt"),
            "results_root": str(root / "results"),
            "cache_root": str(root / "cache"),
        },
        "training": {
            "mode": "qakd",
            "lora": {"rank": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["decoder_mlp"]},
            "steps": 2,
            "learning_rate": 1e-2,
            "batch_size": 2,
            "grad_accumulation": 1,
            "kd_weight": 0.5,
            "kd_temperature": 2.0,
            "data_subset": 6,
        },
        "quantization": {
            "recipe": "w4",
            "targets": list(DECODER_TARGETS),
            "calibration": {"size": 4},
            "recipes": {"w4": {"backend": "gptq", "weight_bits": 4, "group_size": 16}},
            "distillation": {"loss": "gated_dkd", "alpha": 1.0, "beta": 4.0, "temperature": 2.0},
        },
    }


def test_the_e4_stage_trains_exports_and_reloads_the_same_model(tmp_path: Path) -> None:
    train = write_dataset(tmp_path)
    records = teacher_records(train, tmp_path)
    student = tiny_vlm.tiny_wrapper(seed=0)
    config = _config(tmp_path)
    run = RunContext("E4", seed=0)
    run_qakd_training(config, run, student=student, train=train, records=records)
    with (tmp_path / "results/E4/nobudget/seed0/metrics.csv").open() as handle:
        rows = {r["metric"]: float(r["value"]) for r in csv.DictReader(handle)}
    assert rows["train_steps"] == 2 and rows["quantized_matrices"] == 21
    manifest, saved = read_quantized(quantized_dir(config, run))
    assert manifest["start_from"]["experiment"] == "pretrained" and len(saved) == 21
    image = [tiny_vlm.tiny_image(0)]
    trained = student.score_yes_no(image, [tiny_vlm.PROBE])[0]
    reloaded = load_quantized(
        tiny_vlm.tiny_wrapper(seed=0), quantized_dir(config, run), kernel_class=REFERENCE_KERNEL
    ).score_yes_no(image, [tiny_vlm.PROBE])[0]
    assert reloaded.logp_yes == pytest.approx(trained.logp_yes, abs=1e-4)
    assert reloaded.logp_no == pytest.approx(trained.logp_no, abs=1e-4)


def test_install_freezes_everything_but_the_adapter_and_steps() -> None:
    wrapper = tiny_vlm.tiny_wrapper(seed=0)
    params = install_qat(wrapper, DECODER_TARGETS, bits=4, group_size=16, rank=2, alpha=4.0)
    trainable = [p for p in wrapper.module.parameters() if p.requires_grad]
    assert len(params) == 3 * 21 and {id(p) for p in trainable} == {id(p) for p in params}
    assert len(export_qat(wrapper, DECODER_TARGETS)) == 21
