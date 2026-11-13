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

import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
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
        "teacher_cache_manifest_sha256",
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
    nonfinite_steps: list[int] = field(default_factory=list)
    nonfinite_microbatches: int = 0
    skipped_steps: list[int] = field(default_factory=list)

    def final_loss(self, last: int = 10) -> float:
        """Mean of the last ``last`` step losses (NaN if no step ran, or if one of them is)."""
        tail = self.losses[-last:]
        return sum(tail) / len(tail) if tail else math.nan

    def final_loss_finite(self, last: int = 10) -> float:
        """Mean of the last ``last`` finite step losses (NaN if there is none)."""
        tail = [x for x in self.losses if math.isfinite(x)][-last:]
        return sum(tail) / len(tail) if tail else math.nan

    def history(self) -> dict[str, Any]:
        """The per-step record kept next to the adapter (``train_history.json``)."""
        return {
            "steps": self.steps,
            "losses": [x if math.isfinite(x) else None for x in self.losses],
            "ce_losses": [x if math.isfinite(x) else None for x in self.ce_losses],
            "kd_losses": [x if math.isfinite(x) else None for x in self.kd_losses],
            "nonfinite_steps": list(self.nonfinite_steps),
            "nonfinite_microbatches": self.nonfinite_microbatches,
            "skipped_steps": list(self.skipped_steps),
        }


class NonFiniteAdapter(RuntimeError):
    """A trainable weight became NaN or inf: the run stops and saves nothing."""


HISTORY_FILE = "train_history.json"
"""Per-step losses and the non-finite / skipped steps, written into the adapter folder."""
LOG_EVERY = 25
"""Steps between the progress lines a training run prints."""


def select_subset(sample_ids: Sequence[str], n: int | None, seed: int) -> list[int]:
    """First ``n`` indices of a seeded shuffle of the samples (all of them if ``n`` is None)."""
    order = list(range(len(sample_ids)))
    random.Random(f"{seed}:{len(sample_ids)}").shuffle(order)
    return order if n is None else order[:n]


def data_seed(config: Mapping[str, Any]) -> int:
    """The pre-registered data seed (``seeds.data_split``) that fixes the training pool.

    Raises:
        ConfigError: the seed is missing or still ``TODO``.
    """
    from medcompress.config import ConfigError, find_todos

    value = (config.get("seeds") or {}).get("data_split")
    if value is None or find_todos({"v": value}):
        raise ConfigError("seeds.data_split is still open (it fixes the training pool)")
    return int(value)


def training_pool(samples: Sequence[VQASample], n: int | str | None, seed: int) -> list[VQASample]:
    """The training examples that every student run and every recovery draws from.

    One seeded draw of ``n`` samples (all of them for ``None`` or ``"all"``) from the **whole**
    training split, made with the data seed (``seeds.data_split``), not with the seed of the
    run. Consequences:

    * E1 and E2 train on the same examples; only the training signal differs (D06).
    * The teacher cache needs only these examples (``teacher_cache.pool_size``), not the whole
      split.
    * A smaller draw is the first part of a larger one, so the recovery examples (25 % of the
      E2 examples, D06) lie inside the E2 examples and are the same for every pruning arm.

    The seed of a run changes only the order in which the pool is visited and the start of
    the adapter.
    """
    count = None if n is None or n == "all" else int(n)
    return [samples[i] for i in select_subset([s.sample_id for s in samples], count, seed)]


def require_teacher_records(pool: Sequence[VQASample], examples: Sequence[TrainExample]) -> None:
    """Raise if some examples of the training pool have no cached teacher answer."""
    if len(examples) != len(pool):
        have = {e.sample_id for e in examples}
        missing = [s.sample_id for s in pool if s.sample_id not in have]
        raise ValueError(
            f"{len(missing)} of {len(pool)} examples of the training pool have no cached "
            f"teacher answer (first: {missing[0]}); the teacher cache must hold the training "
            "pool (teacher_cache.pool_size = training.data_subset, same seeds.data_split)"
        )


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


def forward_answer_logits(module: Any, batch: Batch) -> Any:
    """Logits that predict the answer slots of a batch (the last ``answer_length`` steps)."""
    out = module(**batch.inputs, use_cache=False, logits_to_keep=batch.answer_length + 1)
    return answer_logits(out.logits, batch.answer_length)


KD_LOSSES: tuple[str, ...] = ("topk_kl", "gated_dkd")
"""Distillation terms: ``topk_kl`` (E2, recovery) and ``gated_dkd`` (E4, D08 option B)."""


def _kd_term(logits: Any, batch: Batch, cfg: TrainConfig) -> Any:
    kind = cfg.extra.get("kd_loss", "topk_kl")
    if kind == "topk_kl":
        return topk_kl_loss(
            logits,
            batch.topk_ids,
            batch.topk_logprobs,
            batch.topk_mask,
            temperature=cfg.kd_temperature,
        )
    if kind == "gated_dkd":
        from medcompress.compression.quantization.qakd import gated_dkd_loss

        return gated_dkd_loss(
            logits,
            batch.topk_ids,
            batch.topk_logprobs,
            batch.targets,  # the teacher's cached answer tokens (teacher-forced)
            batch.topk_mask,
            alpha=float(cfg.extra["dkd_alpha"]),
            beta=float(cfg.extra["dkd_beta"]),
            temperature=float(cfg.extra["dkd_temperature"]),
        )
    raise ValueError(f"kd_loss must be one of {KD_LOSSES}, got {kind!r}")


def _step_loss(
    module: Any, gt: Batch, teacher: Batch | None, cfg: TrainConfig
) -> tuple[Any, float, float]:
    logits = forward_answer_logits(module, gt)
    ce = answer_ce_loss(logits, gt.targets)
    if cfg.kd_weight <= 0:
        return ce, float(ce.detach()), 0.0
    if teacher is None:  # the teacher's answer equals the ground truth: one pass serves both
        kl = _kd_term(logits, gt, cfg)
    else:
        kl = _kd_term(forward_answer_logits(module, teacher), teacher, cfg)
    return combined_kd_loss(ce, kl, kd_weight=cfg.kd_weight), float(ce.detach()), float(kl.detach())


def _step_backward(
    module: Any,
    gt: Batch,
    teacher: Batch | None,
    cfg: TrainConfig,
    *,
    backward: Callable[[Any], None],
    autocast: Callable[[], Any],
) -> tuple[float, float, float]:
    """Forward and backward of one micro-batch; returns ``(loss, ce, kl)`` as numbers.

    ``backward(t)`` runs the (scaled) backward of a loss tensor; ``autocast()`` opens the mixed
    precision context of the forward passes.

    * ``kd_weight <= 0``, or the teacher's answer equals the ground truth (``teacher is None``):
      one forward pass, then one backward of :func:`_step_loss`'s loss, exactly as before.
    * The teacher's answer differs: the loss ``(1 - w) * ce + w * kl`` is linear, so its
      gradient is the sum of the two terms' gradients. The CE term is backpropagated first and
      its graph freed, then the teacher batch is forwarded and the KL term backpropagated.
      Only one graph is alive at a time (E2 seed 0 ran out of memory on one T4 with both,
      2026-11-07). Same loss, same gradients up to rounding, same returned numbers.
    """
    w = cfg.kd_weight
    if w <= 0 or teacher is None:
        with autocast():
            loss, ce, kl = _step_loss(module, gt, teacher, cfg)
        backward(loss)
        return float(loss.detach()), ce, kl
    if not 0.0 <= w <= 1.0:  # the check of combined_kd_loss
        raise ValueError("kd_weight must be between 0 and 1")
    with autocast():
        ce_term = answer_ce_loss(forward_answer_logits(module, gt), gt.targets)
        part = (1.0 - w) * ce_term
    backward(part)
    ce = float(ce_term.detach())
    del ce_term, part  # free the ground-truth graph before the teacher's forward pass
    with autocast():
        kl_term = _kd_term(forward_answer_logits(module, teacher), teacher, cfg)
        part = w * kl_term
    backward(part)
    kl = float(kl_term.detach())
    del kl_term, part
    return (1.0 - w) * ce + w * kl, ce, kl


def _learning_rate(cfg: TrainConfig, step: int) -> float:
    warmup = int(cfg.extra.get("warmup_steps", 0))
    if warmup and step < warmup:
        return cfg.learning_rate * (step + 1) / warmup
    schedule = cfg.extra.get("lr_schedule", "constant")
    if schedule == "linear":
        return cfg.learning_rate * max(0.0, 1.0 - step / max(1, cfg.steps))
    if schedule == "cosine":  # the teacher (D01 amendment): cosine decay to 0 after the warm-up
        progress = (step - warmup) / max(1, cfg.steps - warmup)
        return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return cfg.learning_rate


def train_lora(
    student: HFVLMWrapper,
    examples: Sequence[TrainExample],
    cfg: TrainConfig,
    *,
    seed: int,
    attach_adapter: bool = True,
    resume_dir: Path | None = None,
    stop_after_seconds: float | None = None,
) -> TrainResult:
    """Attach LoRA to the student (if not attached) and train it for ``cfg.steps`` optimizer steps.

    With ``attach_adapter=False`` the trainable parameters are whatever already requires a
    gradient (E4 installs its own quantizer with LoRA before calling this).

    One optimizer step uses ``batch_size * grad_accumulation`` examples taken in a
    seeded shuffled order (cycling if the data run out). Mixed precision is used when
    the model is fp16 on CUDA. The adapter stays attached; use ``lora.merge_adapter``
    or ``lora.save_adapter`` afterwards.

    With ``resume_dir`` the run saves its state there every ``checkpoint_every`` steps
    (``cfg.extra``; default ``resume.DEFAULT_CHECKPOINT_EVERY``) and after the last step, and a
    run started again with the same settings and examples continues from the last saved step
    (``resume.py``). The caller removes the folder once the final output is saved.

    ``stop_after_seconds`` (needs ``resume_dir``): once that much wall time has passed in this
    call, the state is saved after the current step and the run returns early (fewer than
    ``cfg.steps`` steps), so a session can end cleanly before its time limit and the next one
    resumes. It is not part of the run's fingerprint.
    """
    import peft
    import torch

    from medcompress.compression.distillation.resume import (
        DEFAULT_CHECKPOINT_EVERY,
        Progress,
        ResumeStore,
        run_fingerprint,
        settings_for_fingerprint,
    )
    from medcompress.models.smoke import cuda_available, peak_gpu_bytes, reset_peak

    if not examples:
        raise ValueError("no training examples")
    if stop_after_seconds is not None and resume_dir is None:
        raise ValueError("stop_after_seconds needs resume_dir: the state must be saved")
    torch.manual_seed(seed)
    if attach_adapter and not isinstance(student.module, peft.PeftModel):
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
    spec = getattr(student, "spec", None)
    fingerprint = run_fingerprint(
        settings_for_fingerprint(cfg),
        seed,
        [examples[i].sample_id for i in order],
        model=f"{getattr(spec, 'name', None)}@{getattr(spec, 'resolved_revision', None)}",
    )
    progress = Progress(fingerprint)
    store = None
    if resume_dir is not None:
        every = int(cfg.extra.get("checkpoint_every", DEFAULT_CHECKPOINT_EVERY))
        store = ResumeStore(Path(resume_dir), fingerprint, every=every)
        named = {n: p for n, p in module.named_parameters() if p.requires_grad}

        def save_state(folder: Path) -> None:
            from medcompress.utils.disk import require_free_disk

            state_bytes = sum(p.numel() * p.element_size() for p in named.values())
            require_free_disk(folder, 3 * state_bytes, "the resume state")  # with Adam's moments
            torch.save({n: p.detach().cpu() for n, p in named.items()}, folder / "trainable.pt")
            torch.save(optimizer.state_dict(), folder / "optimizer.pt")
            torch.save(scaler.state_dict(), folder / "scaler.pt")
            cuda_rng = torch.cuda.get_rng_state_all() if cuda_available() else []
            torch.save({"cpu": torch.get_rng_state(), "cuda": cuda_rng}, folder / "rng.pt")

        def load_state(folder: Path) -> None:
            saved = torch.load(folder / "trainable.pt", map_location="cpu")
            if set(saved) != set(named):
                raise ValueError(f"the resume state in {folder} has other trainable weights")
            with torch.no_grad():
                for name, value in saved.items():
                    named[name].copy_(value)
            optimizer.load_state_dict(torch.load(folder / "optimizer.pt", map_location="cpu"))
            scaler.load_state_dict(torch.load(folder / "scaler.pt"))
            rng = torch.load(folder / "rng.pt")
            torch.set_rng_state(rng["cpu"])
            if rng["cuda"] and cuda_available():
                torch.cuda.set_rng_state_all(rng["cuda"])

        loaded = store.load(load_state)
        if loaded is not None:
            progress = loaded
            print(f"resuming from step {progress.step} of {cfg.steps} ({resume_dir})")
    reset_peak()
    start = time.perf_counter()
    for step in range(progress.step, cfg.steps):
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(cfg, step)
        totals = [0.0, 0.0, 0.0]
        for _ in range(cfg.grad_accumulation):
            idx = [order[(progress.cursor + i) % len(order)] for i in range(cfg.batch_size)]
            progress.cursor += cfg.batch_size
            gt, teacher = _micro_batches(student, [examples[i] for i in idx], cfg)
            gt = move(gt, device)
            teacher = move(teacher, device) if teacher is not None else None
            value, ce, kl = _step_backward(
                module, gt, teacher, cfg,
                backward=lambda t: scaler.scale(t / cfg.grad_accumulation).backward(),
                autocast=lambda: torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp
                ),
            )  # fmt: skip
            if not math.isfinite(value):
                progress.nonfinite_microbatches += 1
            totals[0] += value / cfg.grad_accumulation
            totals[1] += ce / cfg.grad_accumulation
            totals[2] += kl / cfg.grad_accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, clip)
        scale_before = scaler.get_scale() if amp else None
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        progress.losses.append(totals[0])
        progress.ce_losses.append(totals[1])
        progress.kd_losses.append(totals[2])
        progress.step = step + 1
        progress.examples_seen += per_step
        if not math.isfinite(totals[0]):
            progress.nonfinite_steps.append(progress.step)
        if scale_before is not None and scaler.get_scale() < scale_before:
            progress.skipped_steps.append(progress.step)  # the scaler skipped this update
        saved = store is not None and store.due(progress.step, cfg.steps)
        if saved or progress.step == cfg.steps or progress.step % LOG_EVERY == 0:
            _check_finite(named_trainable(module), progress.step)  # never save a broken state
        if progress.step % LOG_EVERY == 0 or progress.step == cfg.steps:
            print(f"step {progress.step}/{cfg.steps}: loss {totals[0]:.4f}; non-finite steps "
                  f"{len(progress.nonfinite_steps)}, skipped updates {len(progress.skipped_steps)}",
                  flush=True)  # fmt: skip
        if saved and store is not None:
            done = progress.seconds + time.perf_counter() - start
            store.save(Progress(**{**progress.__dict__, "seconds": done}), save_state)
        if (stop_after_seconds is not None and progress.step < cfg.steps
                and time.perf_counter() - start >= stop_after_seconds):  # fmt: skip
            if not saved and store is not None:  # end the session on a saved step
                done = progress.seconds + time.perf_counter() - start
                store.save(Progress(**{**progress.__dict__, "seconds": done}), save_state)
            break
    result = TrainResult(
        steps=progress.step,
        examples_seen=progress.examples_seen,
        seconds=progress.seconds + time.perf_counter() - start,
        losses=list(progress.losses),
        ce_losses=list(progress.ce_losses),
        kd_losses=list(progress.kd_losses),
        nonfinite_steps=list(progress.nonfinite_steps),
        nonfinite_microbatches=progress.nonfinite_microbatches,
        skipped_steps=list(progress.skipped_steps),
    )
    result.peak_gpu_bytes = peak_gpu_bytes()
    module.eval()
    return result


def named_trainable(module: Any) -> dict[str, Any]:
    """The trainable parameters by name (the LoRA weights)."""
    return {n: p for n, p in module.named_parameters() if p.requires_grad}


def nonfinite_tensors(tensors: Mapping[str, Any]) -> dict[str, int]:
    """``{name: number of NaN or inf values}`` of every tensor that has any."""
    import torch

    found = {}
    for name, value in tensors.items():
        bad = int((~torch.isfinite(value.detach().float())).sum())
        if bad:
            found[name] = bad
    return found


def _check_finite(tensors: Mapping[str, Any], step: int) -> None:
    bad = nonfinite_tensors(tensors)
    if bad:
        first = ", ".join(f"{n} ({k})" for n, k in list(bad.items())[:5])
        raise NonFiniteAdapter(
            f"the adapter became non-finite at step {step}: {len(bad)} tensor(s) hold NaN or "
            f"inf, e.g. {first}. Nothing is saved; tell the agent (keep the last 30 log lines)"
        )


def train_student_sft(
    student: HFVLMWrapper,
    examples: Sequence[TrainExample],
    cfg: TrainConfig,
    *,
    seed: int,
    resume_dir: Path | None = None,
) -> TrainResult:
    """Fine-tune the student on ground-truth answers only (kd_weight = 0).

    Used by: E1 (3 seeds).
    """
    if cfg.kd_weight != 0:
        raise ValueError("answer-only fine-tuning needs kd_weight = 0")
    return train_lora(student, examples, cfg, seed=seed, resume_dir=resume_dir)


def train_offline_kd(
    student: HFVLMWrapper,
    examples: Sequence[TrainExample],
    cfg: TrainConfig,
    *,
    seed: int,
    resume_dir: Path | None = None,
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
    return train_lora(student, examples, cfg, seed=seed, resume_dir=resume_dir)


def run_student_training(config: Mapping[str, Any], run: RunContext | None) -> None:
    """Stage entry point for ``scripts/02_train_student.py``.

    Dispatches on ``training.mode``: ``sft`` (E1) and ``offline_kd`` (E2) here;
    ``qakd`` (E4) in ``compression/quantization/qakd.py``. All modes train on the same
    :func:`training_pool`. Writes the adapter under ``paths.checkpoint_root`` and a few
    training numbers to the results CSV.
    """
    from medcompress.config import ConfigError, find_todos, load_config
    from medcompress.utils.io import upsert_results

    cfg = TrainConfig.from_config(config["training"])
    if cfg.mode == "qakd":
        from medcompress.compression.quantization.qakd import run_qakd_training

        run_qakd_training(config, run)
        return
    from medcompress.config import student_config_path

    student_cfg = student_config_path(config)
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
        from medcompress.models.teacher_cache import load_pinned_teacher_cache

        cache = Path(paths["cache_root"]) / "teacher"
        records = {r.sample_id: r for r in load_pinned_teacher_cache(config, cache)}
    pool = training_pool(train, cfg.data_subset, data_seed(config))
    examples = examples_from_samples(pool, Path(paths["data_root"]), records=records)
    if records is not None:
        require_teacher_records(pool, examples)
    student = load_model(
        ModelSpec.from_config(load_config(student_cfg)),
        device_map={"": 0},
        allow_candidate=True,
        image_longest_side=int(side),
    )
    train_fn = train_offline_kd if cfg.mode == "offline_kd" else train_student_sft
    tag = run.experiment_tag if run is not None else cfg.mode
    run_folder = Path(paths["checkpoint_root"]) / tag.replace(":", "/") / f"seed{seed}"
    resume_dir = run_folder / "resume"  # handoffs.resume_relpath; kept until the adapter is saved
    result = train_fn(student, examples, cfg, seed=seed, resume_dir=resume_dir)

    from medcompress.compression.distillation.resume import ResumeStore
    from medcompress.compression.lora import save_adapter

    save_adapter(student, run_folder / "adapter")
    (run_folder / "adapter" / HISTORY_FILE).write_text(
        json.dumps(result.history(), indent=1), encoding="utf-8"
    )  # the per-step losses stay with the adapter (and its hand-off)
    ResumeStore(resume_dir, "").clear()
    rows = [
        {"experiment": tag, "budget": "", "seed": seed, "metric": name, "value": float(value)}
        for name, value in (
            ("train_steps", result.steps),
            ("train_examples_seen", result.examples_seen),
            ("train_loss_final", result.final_loss()),
            ("train_loss_final_finite", result.final_loss_finite()),
            ("train_nonfinite_steps", len(result.nonfinite_steps)),
            ("train_nonfinite_microbatches", result.nonfinite_microbatches),
            ("train_skipped_steps", len(result.skipped_steps)),
            ("train_seconds", result.seconds),
        )
        if math.isfinite(float(value))  # a NaN final loss is left out; the counts say why
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
    print(f"trained {result.steps} steps, final loss {result.final_loss():.4f} (finite steps: "
          f"{result.final_loss_finite():.4f}); non-finite steps {len(result.nonfinite_steps)}, "
          f"skipped updates {len(result.skipped_steps)}; wrote {target}")  # fmt: skip
