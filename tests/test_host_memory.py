"""Host memory across many model loads in one process, and the memory helpers (2026-11-14).

RC-SMOKE at a237c5c: E7's stage 05 was killed with exit -9 (the kernel's out-of-memory killer)
after loading the student many times in one process. A production stage 05 loads it about 16
times (the base, about 14 budget measurements, the model to recover).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from medcompress.utils.memory import (
    GrowthCheck,
    MemoryGrowthError,
    host_available_bytes,
    host_rss_bytes,
    memory_line,
    trim_host_memory,
)


def test_growth_check_passes_a_spike_and_fails_a_climb() -> None:
    mib = 1 << 20
    spike = GrowthCheck("gpu", warmup=1, tolerance_bytes=100 * mib)
    for value in (900, 500, 510, 900, 505, 500, 512):
        spike.add(value * mib)
    assert not spike.grows()
    spike.require_flat()
    climb = GrowthCheck("gpu", warmup=1, tolerance_bytes=100 * mib)
    for step in range(8):
        climb.add((500 + 120 * step) * mib)
    assert climb.grows() and climb.growth_bytes() == 6 * 120 * mib
    with pytest.raises(MemoryGrowthError, match="GROWS"):
        climb.require_flat()
    empty = GrowthCheck("gpu")
    empty.add(None)
    assert not empty.grows() and "no readings" in empty.summary()


def test_memory_line_reads_the_host() -> None:
    line = memory_line("probe")
    assert line.startswith("MEDCOMPRESS_MEM probe: gpu allocated")
    if Path("/proc/self/status").exists():
        assert host_rss_bytes() and host_available_bytes()
        assert "host rss n/a" not in line
    trim_host_memory()  # safe without CUDA and on any libc


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("torchvision")
pytest.importorskip("tokenizers")


def _saved_model(folder: Path) -> Path:
    """A small Qwen2.5-VL (about 21 MiB in fp32) and its processor, saved like a hub snapshot.
    Loaded in fp16, so every load makes its own copy in host memory (fp32 would map the file)."""
    import tiny_vlm

    processor = tiny_vlm.tiny_processor()
    ids = processor.tokenizer.convert_tokens_to_ids
    config = transformers.Qwen2_5_VLConfig(
        text_config=dict(
            vocab_size=48, hidden_size=256, intermediate_size=1536, num_hidden_layers=4,
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
    model = transformers.Qwen2_5_VLForConditionalGeneration(config).eval()
    model.save_pretrained(folder)
    processor.save_pretrained(folder)
    return folder


def _spec(source: Path) -> Any:
    from medcompress.models.loading import ModelSpec

    return ModelSpec(
        name="small", role="student", family="qwen2_5_vl", source=str(source),
        revision="0" * 40, sha_seen=None, adapter=None, adapter_revision=None,
        dtype="float16", status="selected", tokenizer_family=None, vocab_size=None,
    )  # fmt: skip


def test_fifteen_loads_in_one_process_keep_host_memory_flat(tmp_path: Path) -> None:
    from medcompress.models.loading import load_model, release_model

    if host_rss_bytes() is None:
        pytest.skip("no /proc on this system")
    spec = _spec(_saved_model(tmp_path / "model"))
    size = sum(p.stat().st_size for p in (tmp_path / "model").glob("*.safetensors"))
    assert size > 20 << 20  # large enough that a copy kept per load would show
    from medcompress.utils.memory import live_tensor_bytes

    growth = GrowthCheck("host rss", warmup=3, tolerance_bytes=size)
    tensors = GrowthCheck("live tensors (the CPU's 'allocated')", warmup=0, tolerance_bytes=1 << 20)
    before = live_tensor_bytes("cpu")
    for _ in range(15):
        model = load_model(spec, device_map=None, allow_candidate=True)
        release_model(model)
        del model
        growth.add(host_rss_bytes())
        tensors.add(live_tensor_bytes("cpu") - before)
    # allocated memory returns to where it was after every release (2026-11-15), not only RSS
    assert max(tensors.readings) < 1 << 20, tensors.summary()
    # a model's worth of host memory kept per load would add about 5 x size by the end (fp16)
    assert growth.growth_bytes() < 2 * size, growth.summary()
    growth.require_flat()
    leak = GrowthCheck("host rss, models kept", warmup=3, tolerance_bytes=size)
    kept = []
    for _ in range(15):  # the same check catches loads that are never freed
        kept.append(load_model(spec, device_map=None, allow_candidate=True))
        leak.add(host_rss_bytes())
    assert leak.growth_bytes() >= 2 * size, leak.summary()


def test_later_loads_of_a_pinned_model_make_no_hub_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from medcompress.models import loading

    spec = _spec(_saved_model(tmp_path / "model"))
    monkeypatch.setattr(loading, "_FETCHED", set())
    seen: list[dict[str, Any]] = []
    real = transformers.AutoModelForImageTextToText.from_pretrained

    def spy(source: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return real(source, **kwargs)

    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", spy)
    for _ in range(2):
        loading.release_model(loading.load_model(spec, device_map=None, allow_candidate=True))
    assert "local_files_only" not in seen[0] and seen[1]["local_files_only"] is True
    unpinned = loading.ModelSpec(**{**spec.__dict__, "revision": None})
    assert not loading._already_fetched(unpinned.source, unpinned.resolved_revision)


def test_release_drops_the_processor_and_prints_host_memory(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from medcompress.models.loading import load_model, release_model

    model = load_model(_spec(_saved_model(tmp_path / "m")), device_map=None, allow_candidate=True)
    release_model(model)
    assert model.processor is None and model.tokenizer is None
    out = capsys.readouterr().out
    assert "MEDCOMPRESS_MEM before load small" in out
    assert "MEDCOMPRESS_MEM after load small" in out and "MEDCOMPRESS_MEM after release" in out


def test_a_watched_child_reports_its_peak_and_a_kill() -> None:
    import sys

    from medcompress.utils.memory import run_watched

    if host_rss_bytes() is None:
        pytest.skip("no /proc on this system")
    hold = "import time; b = bytearray(300 << 20); b[::4096] = b'x' * len(b[::4096]); " \
           "time.sleep(1.5); print('done')"  # fmt: skip
    done = run_watched([sys.executable, "-c", hold], interval=0.2)
    assert done.returncode == 0 and done.stdout.strip() == "done"
    assert done.peak_rss is not None and done.peak_rss > 250 << 20
    assert done.min_available is not None
    killed = run_watched(
        [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"],
        interval=0.1,
    )
    assert killed.returncode == -9


def test_the_leftover_report_is_quiet_without_a_gpu(capsys: pytest.CaptureFixture) -> None:
    from medcompress.utils.memory import live_cuda_tensors, report_leftover_gpu

    if torch.cuda.is_available():
        pytest.skip("checks the CPU path")
    assert report_leftover_gpu("probe") == 0 and live_cuda_tensors() == []
    assert capsys.readouterr().out == ""


def test_release_clears_tensors_kept_outside_parameters_and_buffers(tmp_path: Path) -> None:
    """A tensor kept as a plain attribute (like a rotary embedding's ``original_inv_freq``) is
    not moved by ``to("meta")``; the release clears it (2026-11-15)."""
    from medcompress.models.loading import load_model, release_model

    model = load_model(_spec(_saved_model(tmp_path / "m")), device_map=None, allow_candidate=True)
    loose = [(sub, n) for sub in model.module.modules() for n, v in vars(sub).items()
             if isinstance(v, torch.Tensor) and not isinstance(v, torch.nn.Parameter)]  # fmt: skip
    model.module.extra_copy = torch.ones(1000)
    release_model(model)
    assert model.module.extra_copy is None
    assert all(getattr(sub, n) is None for sub, n in loose)
    assert all(p.device.type == "meta" for p in model.module.parameters())
