"""Turn training examples into model inputs with answer targets.

A training row is the chat prompt (image and question) followed by answer tokens.
The prompt is built once per example with the model's processor; the answer tokens
are appended as ids, so the same prompt serves two answers: the ground-truth answer
(cross-entropy) and the teacher's cached answer (distillation). Rows are padded on
the left, so the answer tokens end at the last position of every row and the logits
for the last ``A + 1`` positions are enough to score answers of up to ``A`` tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from medcompress.compression.distillation.losses import IGNORE_INDEX

if TYPE_CHECKING:
    import torch

    from medcompress.models.hf import HFVLMWrapper
    from medcompress.models.teacher_cache import CacheRecord

_PASS_THROUGH = ("input_ids", "attention_mask", "mm_token_type_ids")


@dataclass(frozen=True)
class TrainExample:
    """One training example. ``image`` is a path or a PIL image."""

    sample_id: str
    image: Path | str | Any
    question: str
    answer: str
    teacher: CacheRecord | None = None


@dataclass
class Batch:
    """Model inputs plus right-aligned answer targets for one batch.

    ``targets`` holds answer ids (``-100`` where there is no answer token);
    ``topk_ids``, ``topk_logprobs`` and ``topk_mask`` are the teacher's cached top-k for
    the same positions, present only when the batch was built from teacher answers.
    """

    inputs: dict[str, torch.Tensor]
    targets: torch.Tensor
    answer_length: int
    topk_ids: torch.Tensor | None = None
    topk_logprobs: torch.Tensor | None = None
    topk_mask: torch.Tensor | None = None


def prompt_inputs(wrapper: HFVLMWrapper, image: Any, question: str) -> dict[str, torch.Tensor]:
    """Processor output (ids, mask, pixel values, ...) for the chat prompt of one example."""
    from medcompress.models.hf import _to_pil

    text = wrapper.processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}],
        add_generation_prompt=True,
        tokenize=False,
    )
    pil = _to_pil(image, wrapper.image_longest_side)
    return dict(wrapper.processor(text=[text], images=[pil], return_tensors="pt"))


def ground_truth_ids(wrapper: HFVLMWrapper, answer: str) -> list[int]:
    """Token ids of a ground-truth answer followed by the stop token."""
    ids = list(wrapper.tokenizer.encode(answer.strip(), add_special_tokens=False))
    return [int(i) for i in ids] + [wrapper.stop_token_id()]


def assemble(
    prompts: Sequence[dict[str, torch.Tensor]],
    answers: Sequence[Sequence[int]],
    pad_id: int,
    *,
    topk: Sequence[tuple[Any, Any] | None] | None = None,
) -> Batch:
    """Join prompts and answer ids into one left-padded batch.

    ``topk`` optionally gives, per row, the cached ``(ids, logprobs)`` arrays shaped
    ``(len(answer), k)`` for that row's answer.
    """
    import torch

    if len(prompts) != len(answers) or not prompts:
        raise ValueError("need one answer per prompt and at least one row")
    longest = max(len(a) for a in answers)
    if longest == 0:
        raise ValueError("answers must not be empty")
    rows, masks, types = [], [], []
    for prompt, answer in zip(prompts, answers, strict=True):
        ids = prompt["input_ids"][0]
        ans = torch.tensor(list(answer), dtype=ids.dtype)
        rows.append(torch.cat([ids, ans]))
        masks.append(torch.ones(len(ids) + len(ans), dtype=torch.long))
        if "mm_token_type_ids" in prompt:
            kinds = prompt["mm_token_type_ids"][0]
            types.append(torch.cat([kinds, torch.zeros(len(ans), dtype=kinds.dtype)]))
    width = max(len(r) for r in rows)

    def pad_left(values: list[torch.Tensor], fill: int) -> torch.Tensor:
        return torch.stack(
            [torch.cat([torch.full((width - len(v),), fill, dtype=v.dtype), v]) for v in values]
        )

    inputs: dict[str, torch.Tensor] = {
        "input_ids": pad_left(rows, pad_id),
        "attention_mask": pad_left(masks, 0),
    }
    if types:
        inputs["mm_token_type_ids"] = pad_left(types, 0)
    for key in prompts[0]:
        if key not in _PASS_THROUGH:
            inputs[key] = torch.cat([p[key] for p in prompts], dim=0)
    targets = torch.full((len(rows), longest), IGNORE_INDEX, dtype=torch.long)
    for b, answer in enumerate(answers):
        targets[b, longest - len(answer) :] = torch.tensor(list(answer))
    batch = Batch(inputs=inputs, targets=targets, answer_length=longest)
    if topk is not None:
        k = next(len(t[0][0]) for t in topk if t is not None)
        ids_t = torch.zeros(len(rows), longest, k, dtype=torch.long)
        lp_t = torch.zeros(len(rows), longest, k)
        mask = torch.zeros(len(rows), longest, dtype=torch.bool)
        for b, entry in enumerate(topk):
            if entry is None:
                continue
            n = len(entry[0])
            ids_t[b, longest - n :] = torch.as_tensor(entry[0], dtype=torch.long)
            lp_t[b, longest - n :] = torch.as_tensor(entry[1], dtype=torch.float32)
            mask[b, longest - n :] = True
        batch.topk_ids, batch.topk_logprobs, batch.topk_mask = ids_t, lp_t, mask
    return batch


def answer_logits(logits: torch.Tensor, answer_length: int) -> torch.Tensor:
    """Select the logits that predict the answer slots from the last ``A + 1`` positions."""
    if logits.shape[1] < answer_length + 1:
        raise ValueError("need logits for the last answer_length + 1 positions")
    return logits[:, -(answer_length + 1) : -1, :]


def move(batch: Batch, device: Any) -> Batch:
    """Move every tensor of a batch to ``device``."""
    batch.inputs = {k: v.to(device) for k, v in batch.inputs.items()}
    batch.targets = batch.targets.to(device)
    for name in ("topk_ids", "topk_logprobs", "topk_mask"):
        value = getattr(batch, name)
        if value is not None:
            setattr(batch, name, value.to(device))
    return batch
