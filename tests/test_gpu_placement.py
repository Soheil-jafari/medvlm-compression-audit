"""Clean GPUs between stages and no silent offload (RC-G2 session 1, 2026-11-13; CPU)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from medcompress.models.loading import (
    GPU_CLEAN_LIMIT,
    GPUPlacementError,
    explicit_device_map,
    gpu_memory_check,
    require_on_gpu,
)

REPO = Path(__file__).resolve().parents[1]
GIB = 2**30


def _run_experiment_module():
    spec = importlib.util.spec_from_file_location(
        "run_experiment", REPO / "scripts" / "run_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_every_stage_runs_in_its_own_process_and_the_first_failure_stops() -> None:
    module = _run_experiment_module()
    commands: list[list[str]] = []
    codes = iter([0, 4, 0])

    def runner(command: list[str]) -> int:
        commands.append(command)
        return next(codes)

    scripts = ["scripts/04_score_pruning_units.py", "scripts/05_prune_and_recover.py",
               "scripts/06_evaluate.py"]  # fmt: skip
    args = ["--exp", "E7", "--budget", "B2", "--seed", "0", "--data-root", "/d", "--from", "/a"]
    assert module.run_stages_in_processes(scripts, args, runner=runner) == 4
    assert [c[1] for c in commands] == [str(REPO / s) for s in scripts[:2]]  # stopped at 05
    assert all(c[0] == sys.executable and c[2:] == args for c in commands)


def test_a_gpu_that_still_holds_a_model_refuses_the_next_load() -> None:
    total = 15 * GIB
    lines: list[str] = []
    clean = {0: (total - GIB, total)}  # this process's context only
    gpu_memory_check("E2 student", cuda=True, mem_info=clean.__getitem__, count=1, log=lines.append)
    assert lines == ["[E2 student] GPU0: 14.00 of 15.00 GiB free before loading"]
    held = {0: (total - 8 * GIB, total)}  # the scoring stage's student still there
    with pytest.raises(GPUPlacementError, match="GPU0 has 8.00 GiB in use"):
        gpu_memory_check("E2 student", cuda=True, mem_info=held.__getitem__, count=1,
                         log=lines.append)  # fmt: skip
    limit = {0: (total - GPU_CLEAN_LIMIT, total)}
    gpu_memory_check("E2 student", cuda=True, mem_info=limit.__getitem__, count=1, log=lines.append)
    gpu_memory_check("no GPU", cuda=False, mem_info=held.__getitem__, count=1)  # CPU: no check


def test_loads_use_an_explicit_map_on_the_gpus_never_auto() -> None:
    assert explicit_device_map("auto", cuda=True, count=1) == {"": 0}
    assert explicit_device_map(None, cuda=True, count=1) == {"": 0}
    planned = {"model.visual": 0, "lm_head": 1}
    assert explicit_device_map("auto", cuda=True, count=2, planner=lambda: planned) == planned
    assert explicit_device_map({"": 0}, cuda=True, count=1) == {"": 0}
    for bad in ({"model": 0, "lm_head": "cpu"}, {"lm_head": "disk"}):
        with pytest.raises(GPUPlacementError, match="off the GPU"):
            explicit_device_map(bad, cuda=True, count=1)
    with pytest.raises(GPUPlacementError, match="use the GPU"):
        explicit_device_map("cpu", cuda=True, count=1)
    assert explicit_device_map("auto", cuda=False, count=0) == "auto"  # the CPU tests


def test_an_offloaded_model_is_refused() -> None:
    torch = pytest.importorskip("torch")

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    with pytest.raises(GPUPlacementError, match="not on the GPU"):
        require_on_gpu(model, "tiny", cuda=True)  # parameters on the CPU while a GPU is there
    require_on_gpu(model, "tiny", cuda=False)  # the CPU tests: nothing to check
    meta = torch.nn.Linear(4, 4, device="meta")  # what an offload leaves behind
    with pytest.raises(GPUPlacementError, match="on meta"):
        require_on_gpu(meta, "offloaded", cuda=True)
    mapped = torch.nn.Linear(4, 4, device="meta")
    mapped.hf_device_map = {"": "cpu"}  # type: ignore[attr-defined]
    with pytest.raises(GPUPlacementError, match="mapped to cpu"):
        require_on_gpu(mapped, "offloaded", cuda=True)


def test_a_refused_load_ends_the_stage_with_its_own_exit_code() -> None:
    from medcompress.cli import GPU_PLACEMENT_EXIT_CODE, run_stage

    def stage(**_kwargs: object) -> None:
        raise GPUPlacementError("GPU0 has 8.00 GiB in use")

    assert run_stage("05_prune_and_recover", stage) == GPU_PLACEMENT_EXIT_CODE == 6

    def other(**_kwargs: object) -> None:
        raise RuntimeError("something else")

    with pytest.raises(RuntimeError, match="something else"):
        run_stage("06_evaluate", other)


def test_load_model_checks_the_gpu_maps_explicitly_and_refuses_an_offload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The order inside ``load_model``: clean-GPU check, explicit map, load, offload check."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from medcompress.config import config_root, load_config
    from medcompress.models import loading

    calls: list[str] = []
    spec = loading.ModelSpec.from_config(
        load_config(config_root() / "models" / "qwen2_5_vl_3b_student.yaml")
    )

    class Loaded:
        def __init__(self, **kwargs: object) -> None:
            calls.append(f"load {kwargs['device_map']}")

    monkeypatch.setattr(loading, "gpu_memory_check", lambda what: calls.append("check gpu"))
    monkeypatch.setattr(loading, "explicit_device_map", lambda m, s: calls.append("map") or {"": 0})

    def offloaded(module: object, what: str) -> None:
        calls.append("check placement")
        raise GPUPlacementError(f"{what}: 3 tensors are not on the GPU (offloaded)")

    monkeypatch.setattr(loading, "require_on_gpu", offloaded)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained",
                        lambda source, **kw: Loaded(**kw))  # fmt: skip
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: object())
    import medcompress.models.hf as hf

    monkeypatch.setattr(hf, "HFVLMWrapper", lambda module, processor, spec, **kw: type(
        "W", (), {"module": module})())  # fmt: skip
    with pytest.raises(GPUPlacementError, match="offloaded"):
        loading.load_model(spec, allow_candidate=True, image_longest_side=672)
    assert calls == ["check gpu", "map", "load {'': 0}", "check placement"]
