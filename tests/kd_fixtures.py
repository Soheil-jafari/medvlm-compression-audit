"""Cached teacher records for the tiny model (a second random model plays the teacher)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tiny_vlm
import torch
import torch.nn.functional as F

from medcompress.compression.distillation import batching, offline_kd
from medcompress.data.base import VQASample
from medcompress.models.teacher_cache import CacheRecord


def teacher_records(
    samples: Sequence[VQASample], data_root: Path, *, top_k: int = 5
) -> dict[str, CacheRecord]:
    """Teacher-forced top-k records for the ground-truth answers, keyed by sample id."""
    teacher = tiny_vlm.tiny_wrapper(seed=1)
    pad = teacher.tokenizer.pad_token_id
    out: dict[str, CacheRecord] = {}
    for sample in samples:
        ids = batching.ground_truth_ids(teacher, sample.answer)
        prompt = batching.prompt_inputs(
            teacher, Path(data_root) / sample.image_path, sample.question
        )
        batch = batching.assemble([prompt], [ids], pad)
        with torch.no_grad():
            logits = offline_kd.forward_answer_logits(teacher.module, batch)[0]
        top = F.log_softmax(logits.float(), dim=-1).topk(top_k, dim=-1)
        out[sample.sample_id] = CacheRecord(
            sample.sample_id,
            sample.answer,
            np.array(ids, dtype=np.int64),
            top.indices.numpy().astype(np.int64),
            top.values.numpy().astype(np.float32),
        )
    return out
