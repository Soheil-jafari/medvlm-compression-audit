"""Teacher-output cache: run the teacher once, store answers and top-k probabilities.

The cache lives under ``paths.cache_root`` (never committed) and is reused by
offline distillation (E2), quantization-aware distillation (E4) and the shared
recovery training of every pruning arm (E5-E7, A1-A6).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medcompress.config import RunContext
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper

CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CacheRecord:
    """Cached teacher output for one sample."""

    sample_id: str
    answer_text: str
    answer_token_ids: Sequence[int]
    topk_token_ids: Sequence[Sequence[int]]
    topk_logprobs: Sequence[Sequence[float]]


def cache_teacher_outputs(
    teacher: VLMWrapper,
    samples: Sequence[VQASample],
    out_dir: Path,
    *,
    top_k: int,
    max_new_tokens: int,
    shard_size: int = 1000,
) -> Path:
    """Generate teacher answers once and write them to sharded files with a manifest.

    Must be resumable (skip shards that already exist), record the teacher
    config hash, tokenizer fingerprint and :data:`CACHE_FORMAT_VERSION` in a
    metadata file, and write nothing outside ``out_dir``.

    Used by: E0 (producer); E2, E4, E5-E7, A1-A6 (consumers).
    """
    raise NotImplementedError("TODO: sharded, resumable teacher-output cache (E0).")


def load_teacher_cache(cache_dir: Path) -> Iterator[CacheRecord]:
    """Iterate cached records, checking format version and metadata."""
    raise NotImplementedError("TODO: read cache shards and validate metadata.")


def cache_metadata(cache_dir: Path) -> Mapping[str, Any]:
    """Return the cache metadata (teacher config hash, tokenizer fingerprint, top-k, ...)."""
    raise NotImplementedError("TODO: read cache metadata file.")


def run_teacher_caching(config: Mapping[str, Any], run: RunContext | None) -> None:
    """Stage entry point for ``scripts/01_cache_teacher_outputs.py``.

    Loads the teacher from the config, builds the training samples, and calls
    :func:`cache_teacher_outputs`.

    Used by: E0.
    """
    raise NotImplementedError("TODO: wire config -> teacher -> samples -> cache (E0).")
