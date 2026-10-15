"""GPTQ on torch, the quantization stage on the tiny model, and the GPU kernel modules.

Needs torch and transformers (Kaggle / a GPU session run the whole suite, RC-M1 and RC-Q0).
The GPTQModel and bitsandbytes tests also need a CUDA device and those libraries.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("torchvision")
pytest.importorskip("tokenizers")
pytest.importorskip("PIL.Image")

import tiny_vlm  # noqa: E402

from medcompress.compression.quantization import grid  # noqa: E402
from medcompress.compression.quantization.gptq import (  # noqa: E402
    HessianAccumulator,
    gptq_numpy,
    gptq_torch,
)
from medcompress.compression.quantization.kernels import (  # noqa: E402
    DECODER_TARGETS,
    TORCH_KERNEL_CLASS,
    build_gptq_linear,
    build_int8_linear,
    kernel_report,
)
from medcompress.compression.quantization.ptq import (  # noqa: E402
    REFERENCE_KERNEL,
    PTQRecipe,
    calibration_batches,
    gptq_quantize_decoder,
    install_quantized,
    load_quantized,
    quantized_dir,
    read_quantized,
    run_quantization,
    save_quantized,
)
from medcompress.config import RunContext  # noqa: E402
from medcompress.data.base import VQASample  # noqa: E402

RECIPE = PTQRecipe(
    name="w4",
    backend="gptq",
    weight_bits=4,
    activation_bits=None,
    group_size=16,
    targets=DECODER_TARGETS,
    calibration_size=4,
)


def test_gptq_on_torch_matches_the_reference() -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(size=(12, 64))
    acc = HessianAccumulator(64)
    acc.add(rng.normal(size=(300, 64)) @ rng.normal(size=(64, 64)))
    ref = gptq_numpy(w, acc.hessian, bits=4, group_size=16, block_size=24)
    got = gptq_torch(
        torch.tensor(w), torch.tensor(acc.hessian), bits=4, group_size=16, block_size=24
    )
    # float32 against float64: a handful of codes may sit exactly on a rounding boundary
    assert np.mean(got.codes == ref.codes) > 0.98
    assert np.allclose(got.scales[:, 0], ref.scales[:, 0], rtol=1e-5)


def _samples(root: Path, n: int = 4) -> list[VQASample]:
    out = []
    for i in range(n):
        tiny_vlm.tiny_image(i).save(root / f"img{i}.png")
        out.append(
            VQASample(f"s{i}", "tiny", f"img{i}.png", tiny_vlm.PROBE, "yes" if i % 2 else "no")
        )
    return out


def _scores(wrapper: object) -> np.ndarray:
    images = [tiny_vlm.tiny_image(i) for i in range(3)]
    scores = wrapper.score_yes_no(images, [tiny_vlm.PROBE] * 3)  # type: ignore[attr-defined]
    return np.array([[s.logp_yes, s.logp_no] for s in scores])


def test_the_decoder_is_quantized_saved_and_reloaded(tmp_path: Path) -> None:
    wrapper = tiny_vlm.tiny_wrapper(seed=0)
    original = _scores(wrapper)
    batches = calibration_batches(wrapper, _samples(tmp_path), tmp_path)
    records = gptq_quantize_decoder(wrapper, batches, RECIPE, log=lambda _m: None)
    assert len(records) == 3 * len(DECODER_TARGETS)
    for rec in records.values():
        assert rec.codes.max() <= grid.max_code(4)
    rounded = _scores(wrapper)  # the weights now hold the rounded values (16-bit tensors)
    assert not np.allclose(rounded, original)
    install_quantized(wrapper, records, RECIPE, kernel_class=REFERENCE_KERNEL)
    assert np.allclose(_scores(wrapper), rounded, atol=1e-4)
    report = kernel_report(wrapper)
    assert report.n_targets == 3 * len(DECODER_TARGETS) and not report.ok  # reference only
    folder = save_quantized(records, RECIPE, tmp_path / "q", start_from={"experiment": "E2"})
    fresh = load_quantized(tiny_vlm.tiny_wrapper(seed=0), folder, kernel_class=REFERENCE_KERNEL)
    # the scales are saved in 32 bit, so the reload is the installed model; with 16-bit storage
    # this 32-bit model moved by up to 2.5e-4 on a Kaggle machine (RC-Q0), not a GPU effect:
    # the tiny model runs on the CPU
    assert np.allclose(_scores(fresh), rounded, atol=1e-4)


def test_the_stage_writes_codes_manifest_and_rows(tmp_path: Path) -> None:
    config = {
        "paths": {
            "data_root": str(tmp_path),
            "checkpoint_root": str(tmp_path / "ckpt"),
            "results_root": str(tmp_path / "results"),
        },
        "quantization": {
            "recipe": "by_variant",
            "targets": list(DECODER_TARGETS),
            "calibration": {"size": 3},
            "recipes": {"w4": {"backend": "gptq", "weight_bits": 4, "group_size": 16}},
        },
    }
    run = RunContext("E3", seed=1, variant="w4")
    run_quantization(config, run, model=tiny_vlm.tiny_wrapper(seed=0), train=_samples(tmp_path))
    manifest, records = read_quantized(quantized_dir(config, run))
    assert manifest["start_from"] == {"experiment": "E2", "budget": None, "seed": 1}
    assert len(manifest["calibration_sample_ids"]) == 3 and len(records) == 21
    with (tmp_path / "results/E3/w4/nobudget/seed1/metrics.csv").open() as handle:
        rows = {r["metric"]: float(r["value"]) for r in csv.DictReader(handle)}
    assert rows["quantized_matrices"] == 21 and rows["quantization_weight_bits"] == 4
    assert rows["quantization_calibration_images"] == 3
    assert json.loads((quantized_dir(config, run) / "quant_manifest.json").read_text())


# ------------------------------------------------------------- GPU kernel modules

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@needs_cuda
@pytest.mark.parametrize("bits", [4, 8])
def test_the_gptqmodel_kernel_reproduces_our_codes(bits: int) -> None:
    pytest.importorskip("gptqmodel")
    rng = np.random.default_rng(bits)
    codes, scales = grid.round_to_nearest(rng.normal(size=(64, 256)), bits, 128)
    bias = torch.randn(64)
    module = build_gptq_linear(
        codes,
        scales,
        bias,
        bits=bits,
        group_size=128,
        kernel_class=TORCH_KERNEL_CLASS,
        dtype=torch.float16,
        device=torch.device("cuda"),
    )  # raises if the module does not compute with these weights
    assert "int" in str(module.qweight.dtype)


@needs_cuda
def test_llm_int8_stays_close_to_the_16_bit_layer() -> None:
    pytest.importorskip("bitsandbytes")
    linear = torch.nn.Linear(256, 64).half().cuda()
    x = torch.randn(8, 256, dtype=torch.float16, device="cuda")
    with torch.no_grad():
        ref = linear(x).float()
        out = build_int8_linear(linear)(x).float()
    assert float((out - ref).abs().max() / ref.abs().max()) < 0.05
