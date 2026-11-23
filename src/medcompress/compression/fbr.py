"""Finding-balanced recovery (FBR): the secondary, exploratory study of the amendment of
2026-11-22 (``docs/decisions/fbr_study_2026-11-22.md``). Nothing here is tuned.

The recipe is the registered recovery (``configs/recovery.yaml``: LoRA, optimiser, schedule, 157
updates of effective batch 16, fp16 loss scaling, formatting, non-finite handling), with each
update split into

* 10 ordinary pool examples, in the registered seeded order, consumed 10 per update, each with
  the registered per-example loss ``(1 - w) CE + w KL(T)`` (``offline_kd._step_backward``);
* 6 labelled training probes: one positive and one negative per finding, from the training
  split (:func:`probe_schedule`),

and the update's loss

    L = (10/16) mean(L0) + (6/16) [ mean probe CE + c mean_f Huber_delta(d_f - d_f(E2)) ],

``d_f = m(x+) - m(x-)`` the paired margin difference of finding f (``c`` = 0.25, delta = 1; the
sampling-only control has ``c`` = 0). The margin ``m`` is the evaluation's
(:func:`answer_margin`). The paired term's gradient is exact with one graph alive at a time: each
probe's margin gradient over the trainable parameters is kept from its own forward pass (under
the same loss scale) and ``c (6/16) (1/F) Huber'(r_f) (grad m(x+) - grad m(x-))`` is added once
both members are done (:func:`train_fbr`).
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from medcompress.config import ConfigError

if TYPE_CHECKING:
    from medcompress.compression.distillation.batching import Batch, TrainExample
    from medcompress.compression.distillation.offline_kd import TrainConfig
    from medcompress.config import RunContext
    from medcompress.data.base import VQASample
    from medcompress.models.hf import HFVLMWrapper

CONFIG_FILE = "fbr.yaml"


# ----------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class FBRSettings:
    """The fixed FBR values (``configs/fbr.yaml``)."""

    general_per_update: int
    probes_per_finding: int
    margin_coefficient: float
    huber_delta: float
    seed: int
    pilot_arms: tuple[str, ...]
    control_arm: str
    decode_max_new_tokens: int
    fold_min_per_class: int
    folds: int
    bootstrap_resamples: int
    gpu_hours_cap: float

    @classmethod
    def load(cls, overrides: Sequence[str] = ()) -> FBRSettings:
        from medcompress.config import load_config

        section = load_config(CONFIG_FILE, overrides)["fbr"]
        return cls(
            general_per_update=int(section["general_per_update"]),
            probes_per_finding=int(section["probes_per_finding"]),
            margin_coefficient=float(section["margin_coefficient"]),
            huber_delta=float(section["huber_delta"]),
            seed=int(section["seed"]),
            pilot_arms=tuple(section["pilot_arms"]),
            control_arm=str(section["control_arm"]),
            decode_max_new_tokens=int(section["decode_max_new_tokens"]),
            fold_min_per_class=int(section["fold_min_per_class"]),
            folds=int(section["folds"]),
            bootstrap_resamples=int(section["bootstrap_resamples"]),
            gpu_hours_cap=float(section["gpu_hours_cap"]),
        )

    def check_batch(self, cfg: TrainConfig, findings: Sequence[str]) -> None:
        """The update holds exactly the registered effective batch (micro-batch 1)."""
        per_update = cfg.batch_size * cfg.grad_accumulation
        probes = self.probes_per_finding * len(findings)
        if cfg.batch_size != 1:
            raise ConfigError("FBR runs micro-batches of 1, as the registered recovery")
        if self.probes_per_finding != 2:
            raise ConfigError("FBR takes one positive and one negative probe per finding")
        if self.general_per_update + probes != per_update:
            raise ConfigError(f"{self.general_per_update} ordinary examples + {probes} probes "
                              f"!= the registered effective batch {per_update}")  # fmt: skip


def recovery_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """The registered recovery section, unchanged (FBR overrides none of its keys)."""
    recovery = dict(config.get("recovery") or {})
    if recovery.get("method") not in (None, "lora_plus_offline_kd"):
        raise ConfigError(f"unknown recovery method {recovery.get('method')!r}")
    return recovery


# ----------------------------------------------------------------------------- probes


@dataclass(frozen=True)
class TrainingProbe:
    """One labelled training probe: an image, a finding and whether the label says present."""

    sample_id: str
    image_path: str
    finding: str
    present: bool
    question: str

    @property
    def key(self) -> str:
        return f"{self.sample_id}|{self.finding}"


def training_probes(
    samples: Sequence[VQASample],
    findings: Sequence[str],
    template: str,
    display_names: Mapping[str, str] | None = None,
) -> list[TrainingProbe]:
    """One probe per (training image, finding) whose label is given, with the evaluation's
    template and display names. An image without a label for a finding gives no probe for it,
    so a negative is always a label saying absent, never a missing annotation."""
    from medcompress.evaluation.image_reliance import unique_image_probes

    return [
        TrainingProbe(p.sample_id, p.image_path, p.finding, bool(p.present), p.question)
        for p in unique_image_probes(samples, findings, template, display_names=display_names)
    ]


@dataclass
class ProbeSchedule:
    """The probes of every update (``updates[t]``: positive and negative of each finding, in
    the order of ``findings``) and what was reused."""

    findings: tuple[str, ...]
    updates: list[list[TrainingProbe]]
    log: dict[str, dict[str, int]] = field(default_factory=dict)

    def unique_keys(self) -> list[str]:
        return sorted({p.key for update in self.updates for p in update})

    def fingerprint(self) -> str:
        import hashlib

        text = "\n".join(",".join(p.key + f":{int(p.present)}" for p in u) for u in self.updates)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def probe_schedule(
    probes: Sequence[TrainingProbe], findings: Sequence[str], steps: int, seed: int
) -> ProbeSchedule:
    """One positive and one negative probe per finding for each of ``steps`` updates.

    Per (finding, label) stratum the unique images, sorted, are shuffled with
    ``numpy.random.default_rng(seed + cycle)``; every image is used once before any repeats,
    then the next cycle reshuffles. Independent of any model.

    Raises:
        ValueError: a stratum has no image.
    """
    strata: dict[tuple[str, bool], list[TrainingProbe]] = {}
    for probe in probes:
        strata.setdefault((probe.finding, probe.present), [])
    for probe in sorted(probes, key=lambda p: (p.finding, p.present, p.image_path, p.sample_id)):
        group = strata[(probe.finding, probe.present)]
        if not group or group[-1].image_path != probe.image_path:
            group.append(probe)
    empty = [f"{f} {'positive' if v else 'negative'}" for f in findings for v in (True, False)
             if not strata.get((f, v))]  # fmt: skip
    if empty:
        raise ValueError(f"no training image for {', '.join(empty)}")
    queues: dict[tuple[str, bool], list[TrainingProbe]] = {}
    cycles: dict[tuple[str, bool], int] = {}
    uses: dict[tuple[str, bool], dict[str, int]] = {}

    def take(key: tuple[str, bool]) -> TrainingProbe:
        if not queues.get(key):
            cycle = cycles.get(key, 0)
            group = strata[key]
            order = np.random.default_rng(seed + cycle).permutation(len(group))
            queues[key] = [group[int(i)] for i in order]
            cycles[key] = cycle + 1
        probe = queues[key].pop(0)
        uses.setdefault(key, {})
        uses[key][probe.image_path] = uses[key].get(probe.image_path, 0) + 1
        return probe

    updates = [[take((f, v)) for f in findings for v in (True, False)] for _ in range(steps)]
    log = {}
    for f in findings:
        for v in (True, False):
            counts = uses[(f, v)]
            log[f"{f}:{'positive' if v else 'negative'}"] = {
                "images_available": len(strata[(f, v)]),
                "unique_images_used": len(counts),
                "uses": int(sum(counts.values())),
                "max_reuse": int(max(counts.values())),
                "images_reused": int(sum(1 for n in counts.values() if n > 1)),
            }
    return ProbeSchedule(tuple(findings), updates, log)


def general_order(examples: Sequence[TrainExample], seed: int) -> list[int]:
    """The registered recovery's example order (``train_lora``: ``select_subset`` with the
    run's seed)."""
    from medcompress.compression.distillation.offline_kd import select_subset

    return select_subset([e.sample_id for e in examples], None, seed)


def general_indices(order: Sequence[int], step: int, per_update: int) -> list[int]:
    """The ordinary examples of update ``step``: positions ``per_update * step`` onwards of the
    registered order, cycling as the registered recovery does."""
    return [order[(per_update * step + i) % len(order)] for i in range(per_update)]


# ----------------------------------------------------------------------------- margin


def answer_margin(logits: Any, yes_ids: Sequence[int], no_ids: Sequence[int]) -> Any:
    """The evaluation's margin at one answer position, in fp32: logsumexp of the logits over
    the yes spellings minus over the no spellings (``= log p(yes) - log p(no)``, the
    normaliser cancels; ``HFVLMWrapper.yes_no_logprobs``). ``logits``: ``(..., vocab)``."""
    import torch

    z = logits.float()
    return torch.logsumexp(z[..., list(yes_ids)], dim=-1) - torch.logsumexp(
        z[..., list(no_ids)], dim=-1
    )


def probe_batch(
    wrapper: HFVLMWrapper, probe: TrainingProbe, data_root: Path, target_ids: Sequence[int]
) -> Batch:
    """The training row of a probe: its chat prompt, then the answer token and the stop token
    (the registered answer formatting)."""
    from medcompress.compression.distillation.batching import assemble, prompt_inputs

    prompt = prompt_inputs(wrapper, Path(data_root) / probe.image_path, probe.question)
    return assemble([prompt], [list(target_ids)], wrapper.tokenizer.pad_token_id)


def target_ids(wrapper: HFVLMWrapper, answer_id: int) -> list[int]:
    """``[answer token, stop token]``."""
    return [int(answer_id), wrapper.stop_token_id()]


def huber(r: float, delta: float) -> float:
    """Huber loss of ``r`` with threshold ``delta``."""
    a = abs(r)
    return 0.5 * r * r if a <= delta else delta * (a - 0.5 * delta)


def huber_slope(r: float, delta: float) -> float:
    """The derivative of :func:`huber` at ``r``."""
    return max(-delta, min(delta, r))


# ----------------------------------------------------------------------------- training


@dataclass
class FBRResult:
    """What an FBR training run did."""

    steps: int = 0
    seconds: float = 0.0
    losses: list[float] = field(default_factory=list)
    general_losses: list[float] = field(default_factory=list)
    probe_ce: list[float] = field(default_factory=list)
    margin_huber: list[float] = field(default_factory=list)
    nonfinite_steps: list[int] = field(default_factory=list)
    skipped_steps: list[int] = field(default_factory=list)
    nonfinite_microbatches: int = 0
    peak_gpu_bytes: int | None = None
    seconds_per_update: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        def mean(values: Sequence[float]) -> float | None:
            finite = [v for v in values if math.isfinite(v)]
            return sum(finite) / len(finite) if finite else None

        return {
            "steps": self.steps, "seconds": self.seconds,
            "final_loss": self.losses[-1] if self.losses else None,
            "mean_loss": mean(self.losses), "mean_general_loss": mean(self.general_losses),
            "mean_probe_ce": mean(self.probe_ce), "mean_margin_huber": mean(self.margin_huber),
            "nonfinite_steps": list(self.nonfinite_steps),
            "skipped_steps": list(self.skipped_steps),
            "nonfinite_microbatches": self.nonfinite_microbatches,
            "peak_gpu_bytes": self.peak_gpu_bytes,
        }  # fmt: skip


@dataclass
class UpdateContext:
    """What one FBR update needs besides its data (built once by :func:`train_fbr`)."""

    student: Any
    module: Any
    params: list[Any]
    cfg: Any
    settings: FBRSettings
    findings: tuple[str, ...]
    targets: dict[bool, list[int]]
    yes_ids: list[int]
    no_ids: list[int]
    data_root: Path
    device: Any
    scale: Any  # the loss scaler's scale() (identity without fp16)
    autocast: Any


def accumulate_update(
    ctx: UpdateContext,
    general: Sequence[TrainExample],
    probes: Sequence[TrainingProbe],
    e2_margins: Mapping[str, float],
) -> tuple[dict[str, float], bool]:
    """Forward and backward of one update into the parameters' ``.grad`` (scaled as the loss
    scaler scales): each ordinary example ``L0 / 16``; each probe ``CE / 16``; per finding the
    paired term ``c (6/16) (1/F) Huber(d - d(E2))`` through its exact gradient (each member's
    margin gradient kept from its own forward pass). Returns the loss parts and whether the
    update is non-finite."""
    import torch

    from medcompress.compression.distillation.batching import move
    from medcompress.compression.distillation.losses import answer_ce_loss
    from medcompress.compression.distillation.offline_kd import (
        _micro_batches,
        _step_backward,
        forward_answer_logits,
    )

    settings, findings, accumulation = ctx.settings, ctx.findings, ctx.cfg.grad_accumulation
    coefficient = settings.margin_coefficient
    probe_share = settings.probes_per_finding * len(findings) / accumulation  # 6/16
    pair_weight = coefficient * probe_share / len(findings)
    general_total, probe_total, huber_total, broken = 0.0, 0.0, 0.0, False
    for example in general:
        gt, teacher = _micro_batches(ctx.student, [example], ctx.cfg)
        gt = move(gt, ctx.device)
        teacher = move(teacher, ctx.device) if teacher is not None else None
        value, _ce, _kl = _step_backward(
            ctx.module, gt, teacher, ctx.cfg,
            backward=lambda t: ctx.scale(t / accumulation).backward(), autocast=ctx.autocast,
        )  # fmt: skip
        broken = broken or not math.isfinite(value)
        general_total += value / accumulation
    for f_index, finding in enumerate(findings):
        pair = probes[2 * f_index : 2 * f_index + 2]
        if [(p.finding, p.present) for p in pair] != [(finding, True), (finding, False)]:
            raise ValueError(f"the probes of {finding} are not a positive / negative pair")
        margins, grads = [], []
        for probe in pair:
            batch = probe_batch(ctx.student, probe, ctx.data_root, ctx.targets[probe.present])
            batch = move(batch, ctx.device)
            with ctx.autocast():
                logits = forward_answer_logits(ctx.module, batch)
                ce = answer_ce_loss(logits, batch.targets)
            margin = answer_margin(logits[:, 0, :], ctx.yes_ids, ctx.no_ids)[0]
            if coefficient > 0:
                found = torch.autograd.grad(ctx.scale(margin), ctx.params, retain_graph=True,
                                            allow_unused=True)  # fmt: skip
                grads.append([None if g is None else g.detach().float() for g in found])
            ctx.scale(ce / accumulation).backward()
            margins.append(float(margin.detach()))
            ce_value = float(ce.detach())
            broken = broken or not (math.isfinite(ce_value) and math.isfinite(margins[-1]))
            probe_total += ce_value / accumulation
            del logits, ce, margin, batch
        residual = (margins[0] - margins[1]) - (e2_margins[pair[0].key] - e2_margins[pair[1].key])
        if not math.isfinite(residual):
            broken = True
            continue
        huber_total += huber(residual, settings.huber_delta) / len(findings)
        if coefficient > 0:
            k = pair_weight * huber_slope(residual, settings.huber_delta)
            with torch.no_grad():
                for p, g_pos, g_neg in zip(ctx.params, grads[0], grads[1], strict=True):
                    if g_pos is None and g_neg is None:
                        continue
                    zero = torch.zeros_like(p, dtype=torch.float32)
                    delta = (g_pos if g_pos is not None else zero) - (
                        g_neg if g_neg is not None else zero
                    )
                    broken = broken or not bool(torch.isfinite(delta).all())
                    add = (k * delta).to(p.dtype)
                    p.grad = add if p.grad is None else p.grad + add
        del grads
    total = general_total + probe_total + coefficient * probe_share * huber_total
    parts = {"loss": total, "general": general_total, "probe_ce": probe_total,
             "huber": huber_total}  # fmt: skip
    return parts, broken


def train_fbr(
    student: HFVLMWrapper,
    examples: Sequence[TrainExample],
    schedule: ProbeSchedule,
    e2_margins: Mapping[str, float],
    cfg: TrainConfig,
    settings: FBRSettings,
    answer_ids: tuple[int, int],
    *,
    data_root: Path,
    seed: int,
    stop_after_steps: int | None = None,
) -> FBRResult:
    """Attach the registered LoRA and train ``cfg.steps`` FBR updates (fewer with
    ``stop_after_steps``: the timed trial). ``e2_margins``: E2's margin of every probe of the
    schedule (``TrainingProbe.key``); ``answer_ids``: the (yes, no) target tokens. The optimiser,
    schedule, clipping and loss scaling are the registered recovery's (``train_lora``).

    Raises:
        ValueError: a probe of the schedule has no cached E2 margin, or the batch does not add
            up to the registered effective batch.
    """
    import peft
    import torch

    from medcompress.compression.distillation.offline_kd import LOG_EVERY, _learning_rate
    from medcompress.compression.lora import attach_lora, trainable_parameters
    from medcompress.models.smoke import cuda_available, peak_gpu_bytes, reset_peak

    findings = schedule.findings
    settings.check_batch(cfg, findings)
    if len(schedule.updates) < cfg.steps:
        raise ValueError(f"the schedule has {len(schedule.updates)} updates, not {cfg.steps}")
    keys = schedule.unique_keys()
    missing = [k for k in keys if k not in e2_margins]
    if missing:
        raise ValueError(f"E2's margin cache lacks {len(missing)} probes, e.g. {missing[:2]}")
    if any(not math.isfinite(e2_margins[k]) for k in keys):
        raise ValueError("E2's margin cache holds a non-finite margin")
    torch.manual_seed(seed)
    if not isinstance(student.module, peft.PeftModel):
        attach_lora(student, cfg.lora)
    module = student.module
    module.train()
    params = trainable_parameters(module)
    optimizer = torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=float(cfg.extra.get("weight_decay", 0.0))
    )
    device = module.device
    amp = (cuda_available() and device.type == "cuda"
           and next(module.parameters()).dtype == torch.float16)  # fmt: skip
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    clip = float(cfg.extra.get("max_grad_norm", 1.0))
    order = general_order(examples, seed)
    ctx = UpdateContext(
        student, module, params, cfg, settings, tuple(findings),
        {True: target_ids(student, answer_ids[0]), False: target_ids(student, answer_ids[1])},
        list(student.yes_ids), list(student.no_ids), Path(data_root), device, scaler.scale,
        lambda: torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp),
    )  # fmt: skip
    result = FBRResult()
    reset_peak()
    start = time.perf_counter()
    steps = cfg.steps if stop_after_steps is None else min(cfg.steps, stop_after_steps)
    for step in range(steps):
        began = time.perf_counter()
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(cfg, step)
        general = [examples[i] for i in general_indices(order, step, settings.general_per_update)]
        parts, broken = accumulate_update(ctx, general, schedule.updates[step], e2_margins)
        if broken:
            result.nonfinite_microbatches += 1
            if params:  # the whole update is non-finite: the loss scaler skips it
                first = params[0]
                if first.grad is None:
                    first.grad = torch.zeros_like(first)
                first.grad.fill_(float("inf"))
        scale_before = scaler.get_scale() if amp else None
        if amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, clip)
            scaler.step(optimizer)
            scaler.update()
        elif not broken:
            torch.nn.utils.clip_grad_norm_(params, clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        result.steps = step + 1
        result.losses.append(parts["loss"] if not broken else float("nan"))
        result.general_losses.append(parts["general"])
        result.probe_ce.append(parts["probe_ce"])
        result.margin_huber.append(parts["huber"])
        if broken:
            result.nonfinite_steps.append(step + 1)
        if (scale_before is not None and scaler.get_scale() < scale_before) or (not amp and broken):
            result.skipped_steps.append(step + 1)
        result.seconds_per_update.append(time.perf_counter() - began)
        if result.steps % LOG_EVERY == 0 or result.steps == steps:
            print(f"FBR step {result.steps}/{cfg.steps}: loss {parts['loss']:.4f} (ordinary "
                  f"{parts['general']:.4f}, probe CE {parts['probe_ce']:.4f}, Huber "
                  f"{parts['huber']:.4f}); non-finite {len(result.nonfinite_steps)}, skipped "
                  f"{len(result.skipped_steps)}; {result.seconds_per_update[-1]:.1f} s/update",
                  flush=True)  # fmt: skip
    from medcompress.compression.distillation.offline_kd import _check_finite, named_trainable

    _check_finite(named_trainable(module), result.steps)  # never merge a broken adapter
    result.seconds = time.perf_counter() - start
    result.peak_gpu_bytes = peak_gpu_bytes()
    module.eval()
    return result


# ----------------------------------------------------------------------------- start model


def rebuild_pruned(
    config: Mapping[str, Any],
    run: RunContext,
    *,
    student_factory: Any = None,
    removal_json: Path | None = None,
    metrics_csv: Path | None = None,
) -> tuple[HFVLMWrapper, dict[str, Any]]:
    """The arm's pruned model **before** recovery: E2 (merged) with the committed
    ``removal.json``'s units removed by the pruning code. Checked: every listed unit exists,
    the parameters drop by the removed units' size, the count equals the committed
    ``parameters_after`` (when ``metrics_csv`` has it) and the per-layer removed parameters
    equal ``removal_by_layer.csv`` (when it is there).

    Raises:
        ValueError: a check fails.
    """
    from medcompress.compression.pruning.apply import remove_units, verify_physical_removal
    from medcompress.compression.pruning.units import UnitType, enumerate_units
    from medcompress.compression.recovery import _student_factory, read_prune_settings
    from medcompress.evaluation.efficiency import count_parameters

    run_dir = run.run_dir(config["paths"]["results_root"])
    removal_json = Path(removal_json or run_dir / "removal.json")
    metrics_csv = Path(metrics_csv or run_dir / "metrics.csv")
    removal = json.loads(removal_json.read_text(encoding="utf-8"))
    settings = read_prune_settings(config, run, need_protocol=False)
    scoring = settings.scoring
    factory = student_factory or _student_factory(config, run)
    model = factory()
    units = enumerate_units(model, [UnitType(t) for t in scoring.unit_types],
                            mlp_channel_group_size=scoring.mlp_channel_group_size)  # fmt: skip
    by_id = {u.unit_id: u for u in units}
    unknown = [u for u in removal["units"] if u not in by_id]
    if unknown:
        raise ValueError(f"{removal_json}: {len(unknown)} units not in the model, "
                         f"e.g. {unknown[:3]}")  # fmt: skip
    chosen = [by_id[u] for u in removal["units"]]
    before = count_parameters(model)
    remove_units(model, chosen)
    check = verify_physical_removal(before, model, chosen)
    if not check["ok"]:
        raise ValueError(f"the rebuild did not remove the weights it should: {check}")
    after = count_parameters(model)
    info: dict[str, Any] = {"units": len(chosen), "parameters_before": before,
                            "parameters_after": after}  # fmt: skip
    committed = _metric(metrics_csv, "parameters_after")
    if committed is not None:
        info["committed_parameters_after"] = committed
        if int(committed) != after:
            raise ValueError(f"rebuilt {after} parameters, the committed run has {int(committed)}")
    layers = removal_json.parent / "removal_by_layer.csv"
    if layers.is_file():
        per_layer: dict[int, int] = {}
        for unit in chosen:  # as recovery.removal_by_layer counts them
            per_layer[unit.layer_index] = per_layer.get(unit.layer_index, 0) + unit.num_params
        with layers.open(encoding="utf-8", newline="") as handle:
            rows = {int(r["layer_index"]): int(float(r["removed_params"]))
                    for r in csv.DictReader(handle)}  # fmt: skip
        wrong = sorted(i for i in set(rows) | set(per_layer)
                       if rows.get(i, 0) != per_layer.get(i, 0))  # fmt: skip
        if wrong:
            raise ValueError(f"removed parameters per layer differ from {layers} in layers "
                             f"{wrong[:5]}")  # fmt: skip
        info["layers_checked"] = len(rows)
    return model, info


def _metric(path: Path, name: str) -> float | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("metric") == name and row.get("value"):
                return float(row["value"])
    return None


# ----------------------------------------------------------------------------- E2's margins


def score_margins(
    wrapper: HFVLMWrapper,
    probes: Sequence[TrainingProbe],
    data_root: Path,
    answer_ids: tuple[int, int] | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    """The margin of every probe from the teacher-forced training pass (no gradient), the
    function FBR uses for the student, and the first answer token the model ranks highest
    (``{key: token id}``, the "decoded" first token). ``answer_ids`` only sets the teacher-forced
    target, which the first position does not see."""
    import torch

    from medcompress.compression.distillation.offline_kd import forward_answer_logits

    module = wrapper.module
    filler = answer_ids or (int(wrapper.yes_ids[0]), int(wrapper.no_ids[0]))
    yes_ids, no_ids = list(wrapper.yes_ids), list(wrapper.no_ids)
    margins, first = {}, {}
    was_training = module.training
    module.eval()
    with torch.inference_mode():
        for probe in probes:
            target = target_ids(wrapper, filler[0] if probe.present else filler[1])
            batch = probe_batch(wrapper, probe, data_root, target)
            batch.inputs = {k: v.to(module.device) for k, v in batch.inputs.items()}
            logits = forward_answer_logits(module, batch)[:, 0, :]
            margins[probe.key] = float(answer_margin(logits, yes_ids, no_ids)[0])
            first[probe.key] = int(logits[0].float().argmax())
    if was_training:
        module.train()
    return margins, first


def choose_answer_ids(
    first_tokens: Mapping[str, int], yes_ids: Sequence[int], no_ids: Sequence[int]
) -> tuple[int, int, dict[str, Any]]:
    """The (yes, no) target tokens: the single-token spelling of each answer that E2 ranks
    first most often on the schedule's probes (decision file, section 4). Ties go to the
    lower token id.

    Raises:
        ValueError: E2 never ranks a yes (or a no) spelling first: the study stops there.
    """
    counts = {"yes": {}, "no": {}}  # type: dict[str, dict[int, int]]
    other = 0
    for token in first_tokens.values():
        if token in yes_ids:
            counts["yes"][token] = counts["yes"].get(token, 0) + 1
        elif token in no_ids:
            counts["no"][token] = counts["no"].get(token, 0) + 1
        else:
            other += 1
    if not counts["yes"] or not counts["no"]:
        raise ValueError(f"E2's first answer token is never a single-token yes / no spelling "
                         f"(yes {counts['yes']}, no {counts['no']}, other {other}): stop and "
                         "tell the research lead")  # fmt: skip
    pick = {a: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for a, c in counts.items()}
    return pick["yes"], pick["no"], {"yes_counts": counts["yes"], "no_counts": counts["no"],
                                     "other_first_tokens": other}  # fmt: skip
