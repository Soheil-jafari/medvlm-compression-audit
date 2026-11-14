"""The scale check: each heavy step at its real, full-size settings for a few steps, with memory
read after every step (2026-11-14).

RC-SMOKE runs every stage with tiny settings (two probes, two training steps), so it cannot see
memory that grows with the real batch, the real model depth or the number of steps. RC-G2
session 3 ran out of GPU memory in E7's first-order pass after the smoke had passed, and the
smoke's E7 stage 05 was killed by host RAM after many model loads. This check runs, one after
the other and each in its own process (``scripts/scale_check.py``):

======================  ====================================================================
check                   what runs (real settings unless stated)
======================  ====================================================================
``e7_first_order``      E2 student, seed 0's calibration set, E7's probes, batch 4, gates on
                        every first-order unit: the 3 batches with the largest images, then
                        the first ``steps`` batches in the production order.
``budget_search``       stage 05's model factory: ``loads`` fresh loads, each with a real
                        removal set (magnitude order, bisection-like sizes), a footprint
                        measurement with a shortened protocol (2 samples, no warm-up, 1
                        repeat), then the release.
``recovery``            a real pruned student (magnitude order, 30 % of the decoder's
                        parameters, the B2 level), the frozen recovery recipe, the first
                        ``updates`` optimizer updates; reports the updates the loss scaler
                        skipped.
``e4_training``         E4's quantization-aware training, the first ``updates`` updates.
``gptq``                E3 w4: 128 calibration images, the first ``layers`` decoder layers.
``e9_eval``             E9's quantized teacher scoring validation probes (only when the E9
                        hand-off is attached; WAIT otherwise).
======================  ====================================================================

A check fails when the GPU memory after each step, or the host RSS after each load, grows step
over step (:class:`medcompress.utils.memory.GrowthCheck`), or when the GPU peak leaves less than
``HEADROOM_GIB`` of a T4. Every check writes ``<out>/<check>.json``; nothing is written to the
results folder (the parent script points the path roots to ``<out>``).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medcompress.utils.memory import (
    GIB,
    GrowthCheck,
    gpu_bytes,
    host_rss_bytes,
    print_memory,
)

T4_GIB = 14.56
"""Memory of one T4 as PyTorch reports it (15,636,037,632 bytes)."""
HEADROOM_GIB = 1.0
"""A step must peak at least this far below a T4's memory (larger images than the ones seen)."""

CHECKS: tuple[str, ...] = (
    "e7_first_order",
    "budget_search",
    "recovery",
    "e4_training",
    "gptq",
    "e9_eval",
)
RECOVERY_UPDATES_IN_PRODUCTION = 157
"""Optimizer updates of one recovery (the frozen recipe, 2026-11-14 figure)."""


class StopCheck(Exception):
    """Raised from a hook to end a production loop after the few steps the check needs."""

    def __init__(self, progress: Any = None) -> None:
        super().__init__("scale check: enough steps")
        self.progress = progress


@dataclass
class CheckResult:
    check: str
    status: str = "PASS"  # PASS, FAIL or WAIT
    lines: list[str] = field(default_factory=list)
    numbers: dict[str, float] = field(default_factory=dict)

    def fail(self, reason: str) -> None:
        self.status = "FAIL"
        self.lines.append(f"FAIL: {reason}")

    def note(self, text: str) -> None:
        self.lines.append(text)
        print(f"[scale] {self.check}: {text}", flush=True)

    def write(self, out: Path) -> Path:
        path = Path(out) / f"{self.check}.json"
        path.write_text(json.dumps(self.__dict__, indent=2, sort_keys=True), encoding="utf-8")
        return path


def _peak_gib() -> float | None:
    reading = gpu_bytes()
    return None if reading is None else reading[1] / GIB


def _reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)


def _gpu_allocated() -> int | None:
    reading = gpu_bytes()
    return None if reading is None else reading[0]


def judge(result: CheckResult, growth: Sequence[GrowthCheck], peak_gib: float | None,
          *, gpus: int = 1) -> None:  # fmt: skip
    """Fail ``result`` on growth or on a peak above the headroom; record the numbers."""
    for check in growth:
        result.note(check.summary())
        result.numbers[f"{check.name}_growth_gib"] = check.growth_bytes() / GIB
        if check.grows():
            result.fail(f"{check.name} grows step over step")
    if peak_gib is not None:
        result.numbers["gpu_peak_gib"] = peak_gib
        limit = gpus * T4_GIB - HEADROOM_GIB * gpus
        result.note(f"GPU peak {peak_gib:.2f} GiB (limit {limit:.2f} GiB on {gpus} T4)")
        if peak_gib > limit:
            result.fail(f"GPU peak {peak_gib:.2f} GiB leaves less than {HEADROOM_GIB} GiB per T4")


# ------------------------------------------------------------------ shared pieces


def magnitude_removal(model: Any, config: Mapping[str, Any], fraction: float) -> list[Any]:
    """A real removal set: the magnitude order (E5's criterion, weights only) with stage 05's
    removal rules, the shortest
    prefix that removes at least ``fraction`` of the decoder's parameters (a parameter count,
    an estimate of the level: the stages measure the footprint instead)."""
    from medcompress.compression.pruning.criteria import CRITERIA
    from medcompress.compression.pruning.criteria.common import feasible_sequence, removal_set
    from medcompress.compression.pruning.units import UnitType, enumerate_units
    from medcompress.compression.recovery import RemovalSettings, plan_removal
    from medcompress.config import apply_variant

    effective = apply_variant(config, None)  # E7's units and removal rules
    pruning = effective["pruning"]
    types = [UnitType(t) for t in pruning["units"]["types"]]
    group = pruning["units"].get("mlp_channel_group_size")
    units = enumerate_units(model, types, mlp_channel_group_size=group)
    scores = CRITERIA["magnitude"](effective, seed=0).score(model, units, [])
    ordered, protected = plan_removal(units, scores, RemovalSettings.from_config(effective))
    accepted = feasible_sequence(units, ordered, protected_layers=protected)
    decoder = sum(u.num_params for u in units if u.unit_type is UnitType.DECODER_LAYER)
    low, high = 0, len(accepted)
    while high - low > 1:
        mid = (low + high) // 2
        if sum(u.num_params for u in removal_set(accepted, mid)) >= fraction * decoder:
            high = mid
        else:
            low = mid
    return removal_set(accepted, high)


def b2_fraction(config: Mapping[str, Any]) -> float:
    """The share of the decoder's bytes that B2 removes (``budgets.yaml``)."""
    from medcompress.config import load_config

    levels = load_config(config["components"]["budgets"])["budgets"]["levels"]
    return float(levels["B2"]["decoder_bytes_removed"])


def recovery_examples(config: Mapping[str, Any], section: str) -> list[Any]:
    """The examples of the frozen training pool with their cached teacher records."""
    from medcompress.compression.distillation.offline_kd import (
        data_seed,
        examples_from_samples,
        require_teacher_records,
        training_pool,
    )
    from medcompress.compression.pruning.stage import load_train_samples
    from medcompress.models.teacher_cache import load_pinned_teacher_cache

    paths = config["paths"]
    train = load_train_samples(config)
    cache = Path(paths["cache_root"]) / "teacher"
    records = {r.sample_id: r for r in load_pinned_teacher_cache(config, cache)}
    pool = training_pool(train, (config.get(section) or {}).get("data_subset"), data_seed(config))
    examples = examples_from_samples(pool, Path(paths["data_root"]), records=records)
    require_teacher_records(pool, examples)
    return examples


def _training_hook(updates: int, gpu: GrowthCheck, host: GrowthCheck) -> Callable[[Any], None]:
    def hook(progress: Any) -> None:
        gpu.add(_gpu_allocated())
        host.add(host_rss_bytes())
        print_memory(f"scale update {progress.step}")
        if progress.step >= updates:
            raise StopCheck(progress)

    return hook


def _run_training(run: Callable[[], Any], updates: int, gpu: GrowthCheck,
                  host: GrowthCheck) -> Any:  # fmt: skip
    """Run a production training call until ``updates`` updates are done; their progress."""
    from medcompress.compression.distillation import offline_kd

    hook = _training_hook(updates, gpu, host)
    offline_kd.AFTER_STEP_HOOKS.append(hook)
    try:
        run()
    except StopCheck as stop:
        return stop.progress
    finally:
        offline_kd.AFTER_STEP_HOOKS.remove(hook)
    raise RuntimeError(f"the training ended before {updates} updates")


# ------------------------------------------------------------------------ checks


def check_e7_first_order(config: Mapping[str, Any], result: CheckResult, *, steps: int) -> None:
    """E7's first-order pass at full size: the 3 largest-image batches, then ``steps`` more."""
    from PIL import Image

    from medcompress.compression.pruning import scoring, stage
    from medcompress.compression.pruning.criteria.clinical import (
        ClinicalCriterion,
        ClinicalSettings,
    )
    from medcompress.compression.pruning.units import UnitType, enumerate_units
    from medcompress.config import RunContext, apply_variant
    from medcompress.data.calibration import calibration_set, require_check

    run = RunContext("E7", seed=0)
    effective = apply_variant(config, None)
    settings = stage.ScoringSettings.from_config(config, None)
    clinical = ClinicalSettings.from_config(effective)
    assert settings.calibration_size is not None
    require_check(config, 0, settings.calibration_size)
    model = stage.load_scoring_model(config, run)
    calibration = calibration_set(config, stage.load_train_samples(config),
                                  settings.calibration_size, 0)  # fmt: skip
    criterion = ClinicalCriterion(effective, seed=0)
    probes, present, _weights = criterion._probe_set(clinical, calibration)
    types = [t for t in settings.unit_types if clinical.methods[t] == "first_order"]
    units = enumerate_units(model, [UnitType(t) for t in types],
                            mlp_channel_group_size=settings.mlp_channel_group_size)  # fmt: skip
    batch = clinical.batch_size
    per_finding = {f: [p for p in probes if p.finding == f] for f in present}
    production = sum(-(-len(v) // batch) for v in per_finding.values())
    result.note(f"{len(units)} first-order units ({types}), {len(probes)} probes, batch {batch}: "
                f"{production} steps in production")  # fmt: skip

    def pixels(probe: Any) -> int:
        with Image.open(criterion.data_root / probe.image_path) as image:
            return image.width * image.height

    largest = sorted(probes, key=pixels, reverse=True)[: 3 * batch]
    weights = {f: 1.0 for f in present}
    plan = scoring.probe_objective_steps(model, scoring.ProbeObjective.build(largest, weights),
                                         criterion.data_root, batch)  # fmt: skip
    for finding in present:
        if len(plan) >= 3 + steps:
            break
        objective = scoring.ProbeObjective.build(per_finding[finding], {finding: 1.0})
        plan += scoring.probe_objective_steps(model, objective, criterion.data_root, batch)
    plan = plan[: 3 + steps]
    gpu = GrowthCheck("gpu allocated after each step", warmup=3, tolerance_bytes=256 << 20)
    _reset_peak()
    started = time.perf_counter()
    scoring.accumulate_gate_gradients(model, units, plan, loss_scale=clinical.loss_scale,
                                      label="scale e7", growth=gpu, log_every=5)  # fmt: skip
    seconds = (time.perf_counter() - started) / len(plan)
    result.numbers["seconds_per_step"] = seconds
    result.note(f"{seconds:.1f} s per step: the first-order pass would take about "
                f"{production * seconds / 3600:.2f} h (estimate)")  # fmt: skip
    judge(result, [gpu], _peak_gib())


def check_budget_search(config: Mapping[str, Any], result: CheckResult, *, loads: int) -> None:
    """Stage 05's loads and footprint measurements, host RSS read after each release."""
    from dataclasses import replace

    from medcompress.budgets import make_removal_measurer
    from medcompress.compression.recovery import _student_factory, read_prune_settings
    from medcompress.config import RunContext, load_config
    from medcompress.data.prepared import load_prepared
    from medcompress.evaluation.efficiency import timing_samples
    from medcompress.models.loading import release_model

    run = RunContext("E7", seed=0, budget="B2")
    settings = read_prune_settings(config, run, need_protocol=True)
    factory = _student_factory(config, run)
    paths = config["paths"]
    model = factory()  # stage 05's first load: the units and their order
    removal = magnitude_removal(model, config, b2_fraction(config))
    release_model(model)
    del model
    protocol = replace(settings.protocol, samples=2, warmup_runs=0, repeats=1)
    dataset = load_config(config["components"]["datasets"]["train_qa"])["dataset"]
    pool = load_prepared(Path(paths["data_root"]), dataset, Path(paths["private_root"]))["val"]
    measure = make_removal_measurer(factory, timing_samples(pool, protocol), protocol,
                                    Path(paths["data_root"]), settings.spec.metric)  # fmt: skip
    host = GrowthCheck("host rss after each release", warmup=2, tolerance_bytes=512 << 20)
    host.add(host_rss_bytes())
    sizes = [len(removal) * k // 8 for k in (8, 0, 4, 6, 5, 7, 3, 2, 1)][: max(loads - 1, 1)]
    _reset_peak()
    for number, size in enumerate(sizes, 2):
        value = measure(removal[:size]) if size else measure([])
        host.add(host_rss_bytes())
        result.note(f"load {number}: {size} units removed, footprint {value:.4g}")
    result.numbers["loads"] = float(len(sizes) + 1)
    judge(result, [host], _peak_gib())


def check_recovery(config: Mapping[str, Any], result: CheckResult, *, updates: int) -> None:
    """The frozen recovery on a real pruned 3B for ``updates`` updates, then the save."""
    from medcompress.compression.pruning.apply import remove_units
    from medcompress.compression.recovery import _student_factory, run_recovery
    from medcompress.config import RunContext

    run = RunContext("E7", seed=0, budget="B2")
    model = _student_factory(config, run)()
    removal = magnitude_removal(model, config, b2_fraction(config))
    remove_units(model, removal)
    result.note(f"pruned student: {len(removal)} units removed (magnitude order, B2 level)")
    examples = recovery_examples(config, "recovery")
    gpu = GrowthCheck("gpu allocated after each update", warmup=2, tolerance_bytes=256 << 20)
    host = GrowthCheck("host rss after each update", warmup=2, tolerance_bytes=512 << 20)
    _reset_peak()
    started = time.perf_counter()
    progress = _run_training(
        lambda: run_recovery(model, examples, dict(config["recovery"]), seed=0), updates, gpu, host
    )
    seconds = (time.perf_counter() - started) / updates
    skipped = list(progress.skipped_steps)
    result.numbers.update(updates=float(updates), skipped_updates=float(len(skipped)),
                          nonfinite_steps=float(len(progress.nonfinite_steps)),
                          seconds_per_update=seconds)  # fmt: skip
    result.note(f"{len(skipped)} of {updates} updates skipped by the loss scaler (at updates "
                f"{skipped}); {len(progress.nonfinite_steps)} steps with a non-finite loss; "
                f"{seconds:.1f} s per update")  # fmt: skip
    if len(skipped) > 0.05 * updates:
        result.note(f"NOTE for Soheil: more than 5 % of these {updates} updates were skipped; "
                    f"a production recovery has {RECOVERY_UPDATES_IN_PRODUCTION}. Nothing is "
                    "changed: the scaler is part of the frozen recipe")  # fmt: skip
    judge(result, [gpu, host], _peak_gib())
    save_pruned_host_memory(model, removal, config, result)


def save_pruned_host_memory(model: Any, removal: Sequence[Any], config: Mapping[str, Any],
                            result: CheckResult) -> None:  # fmt: skip
    """Merge the adapter and save the pruned model as stage 05 does, and report the host memory
    the save needs (safetensors builds the file's tensors in host memory first)."""
    import shutil

    from medcompress.compression.lora import merge_adapter
    from medcompress.compression.pruning.checkpoint import save_pruned
    from medcompress.utils.memory import host_peak_rss_bytes, trim_host_memory

    merge_adapter(model)
    folder = Path(config["paths"]["checkpoint_root"]) / "scale_check_pruned"
    group = (config.get("pruning") or {}).get("units", {}).get("mlp_channel_group_size") or 0
    before = host_rss_bytes()
    save_pruned(model, folder, [list(removal)], mlp_channel_group_size=int(group))
    peak = host_peak_rss_bytes()
    size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    shutil.rmtree(folder, ignore_errors=True)
    trim_host_memory()
    if before is not None and peak is not None:
        result.numbers.update(save_rss_before_gib=before / GIB, process_peak_rss_gib=peak / GIB)
        result.note(f"save of the pruned model ({size / GIB:.2f} GiB on disk): host RSS "
                    f"{before / GIB:.2f} GiB before, process peak so far "
                    f"{peak / GIB:.2f} GiB")  # fmt: skip


def check_e4_training(config: Mapping[str, Any], result: CheckResult, *, updates: int) -> None:
    """E4's quantization-aware training for ``updates`` updates."""
    from medcompress.compression.quantization.qakd import run_qakd_training
    from medcompress.config import RunContext

    gpu = GrowthCheck("gpu allocated after each update", warmup=2, tolerance_bytes=256 << 20)
    host = GrowthCheck("host rss after each update", warmup=2, tolerance_bytes=512 << 20)
    _reset_peak()
    started = time.perf_counter()
    progress = _run_training(lambda: run_qakd_training(config, RunContext("E4", seed=0)),
                             updates, gpu, host)  # fmt: skip
    result.numbers["skipped_updates"] = float(len(progress.skipped_steps))
    result.note(f"{updates} updates in {time.perf_counter() - started:.0f} s (model load "
                f"included); {len(progress.skipped_steps)} skipped by the loss scaler")  # fmt: skip
    judge(result, [gpu, host], _peak_gib())


def check_gptq(config: Mapping[str, Any], result: CheckResult, *, layers: int) -> None:
    """E3 ``w4``'s GPTQ on the first ``layers`` decoder layers."""
    import re

    from medcompress.compression.pruning.stage import load_train_samples
    from medcompress.compression.quantization.ptq import (
        PTQRecipe,
        calibration_batches,
        gptq_quantize_decoder,
        load_parent,
    )
    from medcompress.config import RunContext
    from medcompress.data.calibration import calibration_set

    run = RunContext("E3", seed=0, variant="w4")
    recipe = PTQRecipe.from_config(config, "w4")
    assert recipe.calibration_size is not None
    calibration = calibration_set(config, load_train_samples(config), recipe.calibration_size,
                                  0, "student")  # fmt: skip
    model = load_parent(config, run, int(config["inputs"]["image_longest_side"]))
    batches = calibration_batches(model, calibration, Path(config["paths"]["data_root"]))
    gpu = GrowthCheck("gpu allocated at each layer start", warmup=1, tolerance_bytes=256 << 20)
    host = GrowthCheck("host rss at each layer start", warmup=1, tolerance_bytes=1 << 30)
    peaks: list[float] = []

    def log(text: str) -> None:
        print(text, flush=True)
        if re.match(r"GPTQ: layer \d+/\d+ start", text):
            gpu.add(_gpu_allocated())
            host.add(host_rss_bytes())
        done = re.match(r"GPTQ: layer (\d+)/\d+ done", text)
        if done:
            peaks.append(_peak_gib() or 0.0)
            if int(done.group(1)) >= layers:
                raise StopCheck()

    _reset_peak()
    started = time.perf_counter()
    try:
        gptq_quantize_decoder(model, batches, recipe, log=log,
                              batch_labels=[s.sample_id for s in calibration])  # fmt: skip
    except StopCheck:
        pass
    seconds = (time.perf_counter() - started) / layers
    result.note(f"{layers} layers, {seconds:.0f} s per layer; the host keeps the codes of the "
                "finished layers by design (a slow rise there is expected)")  # fmt: skip
    judge(result, [gpu], max(peaks) if peaks else _peak_gib())
    result.note(host.summary())


def check_e9_eval(config: Mapping[str, Any], result: CheckResult, *, batches: int) -> None:
    """E9's quantized teacher scoring ``batches`` batches of validation probes (WAIT without
    its hand-off)."""
    from medcompress.compression.quantization.ptq import load_run_quantized, quantized_dir
    from medcompress.config import RunContext
    from medcompress.evaluation.image_reliance import unique_image_probes
    from medcompress.handoffs import handoff_items

    run = RunContext("E9", seed=0)
    root, rel = handoff_items("teacher_adapter", config, None)[0]
    adapter = Path(str(config["paths"][root])) / rel
    if not quantized_dir(config, run).is_dir() or not adapter.is_dir():
        result.status = "WAIT"
        result.note("E9's quantized hand-off or the teacher adapter (needed to rebuild its "
                    "parent) is not attached: nothing to check")  # fmt: skip
        return
    from medcompress.config import load_config
    from medcompress.data.prepared import load_prepared

    paths = config["paths"]
    model = load_run_quantized(config, run, int(config["inputs"]["image_longest_side"]))
    dataset = load_config(config["components"]["datasets"]["train_qa"])["dataset"]
    val = load_prepared(Path(paths["data_root"]), dataset, Path(paths["private_root"]))["val"]
    domain = load_config(config["components"]["domain"])["domain"]
    probes = unique_image_probes(val, domain["findings"], domain["probe"]["template"],
                                 display_names=domain["probe"].get("display_names"))  # fmt: skip
    size = model.batch_size
    gpu = GrowthCheck("gpu allocated after each batch", warmup=2, tolerance_bytes=256 << 20)
    _reset_peak()
    root = Path(paths["data_root"])
    for start in range(0, min(len(probes), batches * size), size):
        part = probes[start : start + size]
        model.score_yes_no([root / p.image_path for p in part], [p.question for p in part])
        gpu.add(_gpu_allocated())
    import torch

    judge(result, [gpu], _peak_gib(), gpus=max(1, torch.cuda.device_count()))


EXPERIMENT_OF: dict[str, str] = {
    "e7_first_order": "E7",
    "budget_search": "E7",
    "recovery": "E7",
    "e4_training": "E4",
    "gptq": "E3",
    "e9_eval": "E9",
}


def run_check(name: str, config: Mapping[str, Any], out: Path, options: Mapping[str, int]) -> int:
    """Run one check in this process and write its JSON; 0 = PASS or WAIT, 1 = FAIL."""
    result = CheckResult(name)
    print_memory(f"scale {name} start")
    try:
        if name == "e7_first_order":
            check_e7_first_order(config, result, steps=options["steps"])
        elif name == "budget_search":
            check_budget_search(config, result, loads=options["loads"])
        elif name == "recovery":
            check_recovery(config, result, updates=options["updates"])
        elif name == "e4_training":
            check_e4_training(config, result, updates=options["e4_updates"])
        elif name == "gptq":
            check_gptq(config, result, layers=options["layers"])
        elif name == "e9_eval":
            check_e9_eval(config, result, batches=options["batches"])
        else:
            raise ValueError(f"unknown check {name!r}; known: {CHECKS}")
    except Exception as exc:  # the check's verdict, with the error, goes to the table
        result.fail(f"{type(exc).__name__}: {exc}")
        result.write(out)
        raise
    print_memory(f"scale {name} end")
    result.write(out)
    print(f"[scale] {name}: {result.status}", flush=True)
    return 1 if result.status == "FAIL" else 0


def report_table(results: Sequence[Mapping[str, Any]]) -> str:
    """One row per check: status, GPU peak, host RSS peak, the lines it printed."""
    lines = ["| Check | Status | GPU peak (GiB) | Host RSS peak / lowest available (GiB) "
             "| Minutes | Notes |", "|---|---|---|---|---|---|"]  # fmt: skip
    for r in results:
        numbers = r.get("numbers") or {}
        peak = numbers.get("gpu_peak_gib")
        host = r.get("host", "-")
        notes = "<br>".join(str(x) for x in r.get("lines") or []) or "-"
        lines.append(f"| {r['check']} | {r['status']} | "
                     f"{'-' if peak is None else f'{peak:.2f}'} | {host} | "
                     f"{r.get('minutes', 0.0):.1f} | {notes} |")  # fmt: skip
    return "\n".join(lines)
