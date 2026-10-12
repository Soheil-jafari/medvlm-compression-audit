"""Student training loops: answer-only fine-tuning (E1) and offline distillation (E2).

Both train LoRA adapters on the decoder of a frozen-vision model. The loss is

    (1 - kd_weight) * CE(student, ground-truth answer)
      + kd_weight * KL(teacher top-k || student) on the teacher's cached answer

so E1 is the same loop with ``kd_weight = 0`` and no teacher. The two answers are
two teacher-forced sequences behind the same prompt; when they are token-identical
(common, because the teacher was fine-tuned on the training answers) one forward
pass serves both terms. The recipe is shared with the recovery of pruned models
(``compression/recovery.py``), so a change here changes every arm equally.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from medcompress.compression.distillation.batching import (
    Batch,
    TrainExample,
    answer_logits,
    assemble,
    ground_truth_ids,
    move,
    prompt_inputs,
)
from medcompress.compression.distillation.losses import (
    answer_ce_loss,
    combined_kd_loss,
    topk_kl_loss,
)
from medcompress.compression.lora import LoRAConfig, attach_lora, trainable_parameters

if TYPE_CHECKING:
    from medcompress.config import RunContext
    from medcompress.data.base import VQASample
    from medcompress.models.hf import HFVLMWrapper

TRAINING_MODES: tuple[str, ...] = ("sft", "offline_kd", "qakd")


@dataclass(frozen=True)
class TrainConfig:
    """Training settings shared by E1, E2 (and E4 via qakd) and by recovery.

    ``extra`` may hold ``warmup_steps`` (default 0), ``lr_schedule`` (``constant`` or
    ``linear``, default ``constant``), ``max_grad_norm`` (default 1.0),
    ``weight_decay`` (default 0) and ``gradient_checkpointing`` (default false).
    """

    mode: str
    lora: LoRAConfig
    steps: int
    learning_rate: float
    batch_size: int
    grad_accumulation: int
    kd_weight: float
    kd_temperature: float
    data_subset: str | int
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> TrainConfig:
        """Build a TrainConfig from a ``training:`` or ``recovery:`` config section.

        Raises:
            ConfigError: a needed value is missing or still ``TODO``.
        """
        from medcompress.config import ConfigError, find_todos

        lora = dict(section.get("lora", {}))
        keys = ("steps", "learning_rate", "batch_size", "grad_accumulation", "kd_weight")
        open_keys = sorted(
            {
                *(f"training.{k}" for k in keys if section.get(k) is None),
                *find_todos({"training": {k: section.get(k) for k in keys}}),
                *find_todos({"training.lora": lora}),
                *(
                    f"training.lora.{k}"
                    for k in ("rank", "alpha", "dropout", "target_modules")
                    if k not in lora
                ),
                *find_todos({"training.data_subset": section.get("data_subset")}),
            }
        )
        if open_keys:
            raise ConfigError(f"training settings still open: {', '.join(open_keys)}")
        mode = str(section.get("mode", "offline_kd"))
        if mode not in TRAINING_MODES:
            raise ConfigError(f"training.mode must be one of {TRAINING_MODES}, got {mode!r}")
        temperature = section.get("kd_temperature")
        if mode == "sft":
            temperature = 1.0  # unused without a teacher
        elif temperature is None or find_todos({"t": temperature}):
            raise ConfigError("training.kd_temperature is still open")
        return cls(
            mode=mode,
            lora=LoRAConfig(
                int(lora["rank"]),
                float(lora["alpha"]),
                float(lora["dropout"]),
                tuple(lora["target_modules"]),
            ),
            steps=int(section["steps"]),
            learning_rate=float(section["learning_rate"]),
            batch_size=int(section["batch_size"]),
            grad_accumulation=int(section["grad_accumulation"]),
            kd_weight=float(section["kd_weight"]),
            kd_temperature=float(temperature),
            data_subset=section["data_subset"],
            extra={k: v for k, v in section.items() if k not in _KNOWN_KEYS},
        )


_KNOWN_KEYS = frozenset(
    {
        "mode", "method", "lora", "steps", "learning_rate", "batch_size", "grad_accumulation",
        "kd_weight", "kd_temperature", "data_subset", "teacher_cache_from",
    }
)  # fmt: skip


@dataclass
class TrainResult:
    """What a training run did (kept in memory; summary rows go to the results CSV)."""

    steps: int = 0
    examples_seen: int = 0
    seconds: float = 0.0
    losses: list[float] = field(default_factory=list)
    ce_losses: list[float] = field(default_factory=list)
    kd_losses: list[float] = field(default_factory=list)
    peak_gpu_bytes: int | None = None

    def final_loss(self, last: int = 10) -> float:
        """Mean of the last ``last`` step losses (NaN if no step ran)."""
        tail = self.losses[-last:]
        return sum(tail) / len(tail) if tail else math.nan


def select_subset(sample_ids: Sequence[str], n: int | None, seed: int) -> list[int]:
    """First ``n`` indices of a seeded shuffle of the samples (all of them if ``n`` is None)."""
    order = list(range(len(sample_ids)))
    random.Random(f"{seed}:{len(sample_ids)}").shuffle(order)
    return order if n is None else order[:n]


def examples_from_samples(
    samples: Sequence[VQASample],
    data_root: Path,
    *,
    records: Mapping[str, Any] | None = None,
) -> list[TrainExample]:
    """Build training examples; with ``records`` (cached teacher outputs by sample id) attach them.

    Samples without a cached record are left out when ``records`` is given.
    """
    out = []
    for s in samples:
        teacher = None
        if records is not None:
            teacher = records.get(s.sample_id)
            if teacher is None:
                continue
        out.append(
            TrainExample(s.sample_id, Path(data_root) / s.image_path, s.question, s.answer, teacher)
        )
    return out


def _micro_batches(
    wrapper: HFVLMWrapper, chunk: Sequence[TrainExample], cfg: TrainConfig
) -> tuple[Batch, Batch | None]:
    """Prompt once, then build the ground-truth batch and (if distilling) the teacher batch."""
    prompts = [prompt_inputs(wrapper, ex.image, ex.question) for ex in chunk]
    pad = wrapper.tokenizer.pad_token_id
    gt_ids = [ground_truth_ids(wrapper, ex.answer) for ex in chunk]
    gt = assemble(prompts, gt_ids, pad)
    if cfg.kd_weight <= 0:
        return gt, None
    if any(ex.teacher is None for ex in chunk):
        raise ValueError("distillation needs a cached teacher record for every example")
    teacher_ids = [ex.teacher.answer_token_ids.astype("int64").tolist() for ex in chunk]
    teacher = assemble(
        prompts,
        teacher_ids,
        pad,
        topk=[(ex.teacher.topk_token_ids, ex.teacher.topk_logprobs) for ex in chunk],
    )
    same = teacher_ids == gt_ids
    if same:  # one forward pass serves both terms: reuse the ground-truth batch inputs
        gt.topk_ids, gt.topk_logprobs, gt.topk_mask = (
            teacher.topk_ids,
            teacher.topk_logprobs,
            teacher.topk_mask,
        )
        return gt, None
    return gt, teacher


def _forward_logits(module: Any, batch: Batch) -> Any:
    out = module(**batch.inputs, use_cache=False, logits_to_keep=batch.answer_length + 1)
    return answer_logits(out.logits, batch.answer_length)


def _step_loss(
    module: Any, gt: Batch, teacher: Batch | None, cfg: TrainConfig
) -> tuple[Any, float, float]:
    logits = _forward_logits(module, gt)
    ce = answer_ce_loss(logits, gt.targets)
    if cfg.kd_weight <= 0:
        return ce, float(ce.detach()), 0.0
    if teacher is None:
        kl = topk_kl_loss(
            logits, gt.topk_ids, gt.topk_logprobs, gt.topk_mask, temperature=cfg.kd_temperature
        )
    else:
        kl = topk_kl_loss(
            _forward_logits(module, teacher),
            teacher.topk_ids,
            teacher.topk_logprobs,
            teacher.topk_mask,
            temperature=cfg.kd_temperature,
        )
    return combined_kd_loss(ce, kl, kd_weight=cfg.kd_weight), float(ce.detach()), float(kl.detach())


def _learning_rate(cfg: TrainConfig, step: int) -> float:
    warmup = int(cfg.extra.get("warmup_steps", 0))
    if warmup and step < warmup:
        return cfg.learning_rate * (step + 1) / warmup
    if cfg.extra.get("lr_schedule", "constant") == "linear":
        return cfg.learning_rate * max(0.0, 1.0 - step / max(1, cfg.steps))
    return cfg.learning_rate


def train_lora(
    student: HFVLMWrapper, examples: Sequence[TrainExample], cfg: TrainConfig, *, seed: int
) -> TrainResult:
    """Attach LoRA to the student (if not attached) and train it for ``cfg.steps`` optimizer steps.

    One optimizer step uses ``batch_size * grad_accumulation`` examples taken in a
    seeded shuffled order (cycling if the data run out). Mixed precision is used when
    the model is fp16 on CUDA. The adapter stays attached; use ``lora.merge_adapter``
    or ``lora.save_adapter`` afterwards.
    """
    import peft
    import torch

    from medcompress.models.smoke import cuda_available, peak_gpu_bytes, reset_peak

    if not examples:
        raise ValueError("no training examples")
    torch.manual_seed(seed)
    if not isinstance(student.module, peft.PeftModel):
        attach_lora(student, cfg.lora)
    module = student.module
    if cfg.extra.get("gradient_checkpointing"):
        module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        module.config.use_cache = False
    module.train()
    params = trainable_parameters(module)
    optimizer = torch.optim.AdamW(
        params, lr=cfg.learning_rate, weight_decay=float(cfg.extra.get("weight_decay", 0.0))
    )
    device = module.device
    amp = (
        cuda_available()
        and device.type == "cuda"
        and next(module.parameters()).dtype == torch.float16
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    clip = float(cfg.extra.get("max_grad_norm", 1.0))
    order = select_subset([e.sample_id for e in examples], None, seed)
    per_step = cfg.batch_size * cfg.grad_accumulation
    result = TrainResult()
    reset_peak()
    start, cursor = time.perf_counter(), 0
    for step in range(cfg.steps):
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(cfg, step)
        totals = [0.0, 0.0, 0.0]
        for _ in range(cfg.grad_accumulation):
            idx = [order[(cursor + i) % len(order)] for i in range(cfg.batch_size)]
            cursor += cfg.batch_size
            gt, teacher = _micro_batches(student, [examples[i] for i in idx], cfg)
            gt = move(gt, device)
            teacher = move(teacher, device) if teacher is not None else None
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss, ce, kl = _step_loss(module, gt, teacher, cfg)
            scaler.scale(loss / cfg.grad_accumulation).backward()
            totals[0] += float(loss.detach()) / cfg.grad_accumulation
            totals[1] += ce / cfg.grad_accumulation
            totals[2] += kl / cfg.grad_accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        result.losses.append(totals[0])
        result.ce_losses.append(totals[1])
        result.kd_losses.append(totals[2])
        result.steps += 1
        result.examples_seen += per_step
    result.seconds = time.perf_counter() - start
    result.peak_gpu_bytes = peak_gpu_bytes()
    module.eval()
    return result


def train_student_sft(
    student: HFVLMWrapper, examples: Sequence[TrainExample], cfg: TrainConfig, *, seed: int
) -> TrainResult:
    """Fine-tune the student on ground-truth answers only (kd_weight = 0).

    Used by: E1 (3 seeds).
    """
    if cfg.kd_weight != 0:
        raise ValueError("answer-only fine-tuning needs kd_weight = 0")
    return train_lora(student, examples, cfg, seed=seed)


def train_offline_kd(
    student: HFVLMWrapper, examples: Sequence[TrainExample], cfg: TrainConfig, *, seed: int
) -> TrainResult:
    """Train the student on ground-truth answers and cached teacher outputs.

    Every example must carry its cached teacher record. Used by: E2 (3 seeds). E2 is
    the starting point of E3 and E5-E7.
    """
    if cfg.kd_weight <= 0:
        raise ValueError("offline distillation needs kd_weight > 0")
    missing = [e.sample_id for e in examples if e.teacher is None]
    if missing:
        raise ValueError(
            f"{len(missing)} examples have no cached teacher record (first: {missing[0]})"
        )
    return train_lora(student, examples, cfg, seed=seed)


def run_student_training(config: Mapping[str, Any], run: RunContext | None) -> None:
    """Stage entry point for ``scripts/02_train_student.py``.

    Dispatches on ``training.mode``: ``sft`` (E1) and ``offline_kd`` (E2) here;
    ``qakd`` (E4) is part of the quantization stage. Writes the adapter under
    ``paths.checkpoint_root`` and a few training numbers to the results CSV.
    """
    from medcompress.config import ConfigError, find_todos, load_config
    from medcompress.utils.io import upsert_results

    cfg = TrainConfig.from_config(config["training"])
    if cfg.mode == "qakd":
        raise NotImplementedError("TODO: quantization-aware distillation is part of S08 (E4).")
    student_cfg = config.get("model", {}).get("config")
    side = config.get("inputs", {}).get("image_longest_side")
    open_keys = [
        name
        for name, value in (("model.config", student_cfg), ("inputs.image_longest_side", side))
        if value is None or find_todos({name: value})
    ]
    if open_keys:
        raise ConfigError(f"settings still open: {', '.join(open_keys)}")
    paths = config["paths"]
    seed = run.seed if run is not None else 0

    from medcompress.data.prepared import load_prepared
    from medcompress.models.loading import ModelSpec, load_model

    dataset = load_config(config["components"]["datasets"]["train_qa"])["dataset"]
    train = load_prepared(Path(paths["data_root"]), dataset, Path(paths["private_root"]))["train"]
    records = None
    if cfg.mode == "offline_kd":
        from medcompress.models.teacher_cache import load_teacher_cache

        records = {
            r.sample_id: r for r in load_teacher_cache(Path(paths["cache_root"]) / "teacher")
        }
    examples = examples_from_samples(train, Path(paths["data_root"]), records=records)
    n = None if cfg.data_subset == "all" else int(cfg.data_subset)
    examples = [examples[i] for i in select_subset([e.sample_id for e in examples], n, seed)]
    student = load_model(
        ModelSpec.from_config(load_config(student_cfg)),
        device_map={"": 0},
        allow_candidate=True,
        image_longest_side=int(side),
    )
    train_fn = train_offline_kd if cfg.mode == "offline_kd" else train_student_sft
    result = train_fn(student, examples, cfg, seed=seed)

    from medcompress.compression.lora import save_adapter

    tag = run.experiment_tag if run is not None else cfg.mode
    save_adapter(
        student, Path(paths["checkpoint_root"]) / tag.replace(":", "/") / f"seed{seed}" / "adapter"
    )
    rows = [
        {"experiment": tag, "budget": "", "seed": seed, "metric": name, "value": float(value)}
        for name, value in (
            ("train_steps", result.steps),
            ("train_examples_seen", result.examples_seen),
            ("train_loss_final", result.final_loss()),
            ("train_seconds", result.seconds),
        )
    ]
    if result.peak_gpu_bytes is not None:
        rows.append(
            {
                "experiment": tag,
                "budget": "",
                "seed": seed,
                "metric": "train_peak_gpu_bytes",
                "value": float(result.peak_gpu_bytes),
            }
        )
    target = (
        run.run_dir(paths["results_root"]) if run is not None else Path(paths["results_root"])
    ) / "metrics.csv"
    upsert_results(rows, target)
    print(f"trained {result.steps} steps, final loss {result.final_loss():.4f}; wrote {target}")
