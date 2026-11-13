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


def _gptq_torch_before_2026_11_12(weight, hessian, *, bits, group_size, damping=0.01,
                                  block_size=128):  # fmt: skip
    """The torch GPTQ as it was before the in-place dampening (verbatim), for the check below."""
    w = weight.detach().to(torch.float32).clone()
    h = hessian.detach().to(device=w.device, dtype=torch.float32).clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    h += damping * torch.mean(torch.diag(h)) * torch.eye(h.shape[0], device=h.device)
    lower = torch.linalg.cholesky(h)
    h_inv = torch.linalg.cholesky(torch.cholesky_inverse(lower), upper=True)
    rows, columns = w.shape
    groups = grid.check_shape(columns, bits, group_size)
    zero, top = grid.zero_point(bits), grid.max_code(bits)
    codes = torch.zeros((rows, columns), dtype=torch.uint8, device=w.device)
    scales = torch.ones((rows, groups), dtype=torch.float32, device=w.device)
    loss = torch.zeros((), dtype=torch.float32, device=w.device)
    for start in range(0, columns, block_size):
        stop = min(start + block_size, columns)
        block = w[:, start:stop].clone()
        errors = torch.zeros_like(block)
        h_block = h_inv[start:stop, start:stop]
        for i in range(stop - start):
            column = start + i
            if column % group_size == 0:
                absmax = w[:, column : column + group_size].abs().amax(dim=1)
                scale = 2.0 * absmax / top
                scales[:, column // group_size] = torch.where(
                    scale == 0, torch.ones_like(scale), scale
                )
            s = scales[:, column // group_size]
            q = torch.clamp(torch.round(block[:, i] / s) + zero, 0, top)
            codes[:, column] = q.to(torch.uint8)
            rounded = (q - zero) * s
            d = h_block[i, i]
            err = (block[:, i] - rounded) / d
            loss += torch.sum((block[:, i] - rounded) ** 2) / d**2 / 2
            block[:, i:] -= err[:, None] * h_block[i, i:][None, :]
            errors[:, i] = err
        w[:, stop:] -= errors @ h_inv[start:stop, stop:]
    return codes.cpu().numpy(), scales.double().cpu().numpy(), float(loss)


@pytest.mark.parametrize("bits", [4, 8])
def test_the_in_place_dampening_gives_the_same_codes_as_before(bits: int) -> None:
    generator = torch.Generator().manual_seed(bits)
    for out, inp, group in ((24, 64, 16), (40, 256, 32), (8, 384, 128)):
        weight = torch.randn(out, inp, generator=generator, dtype=torch.float16)
        x = torch.randn(200, inp, generator=generator)
        x[:, 3] = 0.0  # a dead input: unit diagonal, zero weights
        hessian = 2.0 * x.T @ x / x.shape[0]
        before = _gptq_torch_before_2026_11_12(weight, hessian, bits=bits, group_size=group)
        kept = hessian.clone()
        now = gptq_torch(weight, hessian, bits=bits, group_size=group)
        assert np.array_equal(now.codes, before[0]) and np.array_equal(now.scales, before[1])
        assert now.output_error == before[2]
        assert torch.equal(hessian, kept)  # the caller's Hessian is not changed


def test_the_decoder_codes_are_the_same_as_before_the_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from medcompress.compression.quantization import gptq as gptq_mod

    batches = calibration_batches(tiny_vlm.tiny_wrapper(seed=0), _samples(tmp_path), tmp_path)
    assert all(v.device.type == "cpu" for b in batches for v in b.values())  # kept off the GPU
    now = gptq_quantize_decoder(tiny_vlm.tiny_wrapper(seed=0), batches, RECIPE, log=lambda _m: None)

    def old(weight, hessian, *, device=None, consume_hessian=False, **kwargs):  # type: ignore[no-untyped-def]
        assert device is None or torch.device(device) == weight.device  # the CPU here
        codes, scales, loss = _gptq_torch_before_2026_11_12(weight, hessian.clone(), **kwargs)
        return gptq_mod.GPTQResult(codes, scales, kwargs["bits"], kwargs["group_size"], loss)

    monkeypatch.setattr(gptq_mod, "gptq_torch", old)
    before = gptq_quantize_decoder(
        tiny_vlm.tiny_wrapper(seed=0), batches, RECIPE, log=lambda _m: None
    )
    assert set(now) == set(before)
    for key, record in now.items():
        assert np.array_equal(record.codes, before[key].codes), key
        assert np.array_equal(record.scales, before[key].scales), key


# ------------------------------------------------- placement and the GPTQ device (2026-11-13)


def test_the_plan_places_every_tensor_once_balanced_and_under_the_caps() -> None:
    from medcompress.compression.quantization.placement import (
        device_map_lines,
        placement_units,
        plan_device_map,
    )

    model = tiny_vlm.tiny_wrapper(seed=0).module
    units = placement_units(model)
    names = [n for n, _ in model.named_parameters()] + [n for n, _ in model.named_buffers()]
    for name in names:  # every tensor under exactly one unit
        hits = [u for u, _ in units if name == u or name.startswith(u + ".")]
        assert len(hits) == 1, (name, hits)
    total = sum(s for _, s in units)
    assert total == sum(
        t.numel() * t.element_size() for t in [*model.parameters(), *model.buffers()]
    )
    device_map, used = plan_device_map(units, {0: total, 1: total})
    order = [device_map[u] for u, _ in units]
    assert order == sorted(order) and set(order) == {0, 1}  # in model order, both GPUs used
    biggest = max(s for _, s in units)
    assert abs(used[0] - used[1]) <= biggest  # balanced within one unit
    assert len(device_map_lines(units, device_map)) == 2
    with pytest.raises(ValueError, match="exceeds the caps"):
        plan_device_map(units, {0: total // 4, 1: total // 4})


def test_the_teachers_plan_on_two_t4s() -> None:
    """The teacher's architecture (Qwen2.5-VL-7B: the dimensions of its model config and of
    RC-E9's log) built without weights: its plan, the one RC-E9 prints."""
    import transformers
    from accelerate import init_empty_weights

    from medcompress.compression.quantization.placement import placement_units, plan_device_map
    from medcompress.compression.quantization.ptq import gptq_max_memory

    config = transformers.Qwen2_5_VLConfig(
        text_config=dict(vocab_size=152064, hidden_size=3584, intermediate_size=18944,
                         num_hidden_layers=28, num_attention_heads=28, num_key_value_heads=4,
                         tie_word_embeddings=False),
        vision_config=dict(depth=32, hidden_size=1280, intermediate_size=3420, num_heads=16,
                           out_hidden_size=3584),
        tie_word_embeddings=False,
    )  # fmt: skip
    with init_empty_weights():
        model = transformers.AutoModelForImageTextToText.from_config(config, dtype=torch.float16)
    assert sum(p.numel() for p in model.parameters()) == 8_292_166_656  # parameters_stored
    units = placement_units(model)
    t4 = 15_636_037_632  # RC-B0's efficiency_setup.json
    caps = gptq_max_memory([t4, t4], sum(s for _, s in units), 18944, 3584)
    device_map, used = plan_device_map(units, caps)
    assert device_map["model.visual.patch_embed.proj"] == 0 and device_map["lm_head"] == 1
    assert device_map["model.language_model.layers.12"] == 0
    assert device_map["model.language_model.layers.13"] == 1
    assert used == {0: 8_502_597_792, 1: 8_081_736_192}  # 7.92 / 7.53 GiB
    assert all(used[g] <= caps[g] for g in used)


def test_the_gptq_device_is_the_layers_gpu_then_the_roomiest_then_the_cpu() -> None:
    from medcompress.compression.quantization.placement import (
        GPTQ_DEVICE_MARGIN,
        gptq_work_bytes,
        memory_report,
        pick_gptq_device,
    )

    need = gptq_work_bytes(18944, 3584) + GPTQ_DEVICE_MARGIN
    layer = torch.device("cuda", 1)

    def pick(rooms: dict[int, int]) -> object:
        return pick_gptq_device(layer, 18944, 3584, room=rooms.__getitem__, gpus=rooms)

    assert pick({0: need * 2, 1: need}) == torch.device("cuda", 1)  # its own GPU first
    assert pick({0: need, 1: need - 1}) == torch.device("cuda", 0)  # then the other one
    assert pick({0: need - 1, 1: need - 1}) == torch.device("cpu")  # neither: the CPU
    assert pick_gptq_device(torch.device("cpu"), 8, 8) == torch.device("cpu")
    if not torch.cuda.is_available():
        assert memory_report() == ""


def test_the_cpu_fallback_and_the_handed_over_hessian_give_the_same_codes(tmp_path: Path) -> None:
    """On this machine the tiny model is on the CPU, so this checks the plumbing (the device
    choice, the copies, the handed-over Hessians), not GPU-versus-CPU rounding."""
    batches = calibration_batches(tiny_vlm.tiny_wrapper(seed=0), _samples(tmp_path), tmp_path)
    plain = gptq_quantize_decoder(
        tiny_vlm.tiny_wrapper(seed=0), batches, RECIPE, log=lambda _m: None
    )
    choices: list[object] = []

    def to_cpu(layer_device: object, in_features: int, out_features: int) -> object:
        choices.append(layer_device)
        return torch.device("cpu")

    lines: list[str] = []
    fallback = gptq_quantize_decoder(
        tiny_vlm.tiny_wrapper(seed=0), batches, RECIPE, log=lines.append, pick_device=to_cpu
    )
    assert len(choices) == len(plain)
    assert any("start" in line for line in lines) and any("done" in line for line in lines)
    for key, record in plain.items():
        assert np.array_equal(record.codes, fallback[key].codes), key
        assert np.array_equal(record.scales, fallback[key].scales), key
    generator = torch.Generator().manual_seed(3)
    weight = torch.randn(24, 64, generator=generator)
    x = torch.randn(100, 64, generator=generator)
    hessian = 2.0 * x.T @ x / 100
    kept = gptq_torch(weight, hessian.clone(), bits=4, group_size=16)
    taken = gptq_torch(weight, hessian.clone(), bits=4, group_size=16, consume_hessian=True)
    moved = gptq_torch(weight, hessian, bits=4, group_size=16, device="cpu")
    for other in (taken, moved):
        assert np.array_equal(kept.codes, other.codes)
        assert np.array_equal(kept.scales, other.scales)
