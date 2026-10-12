"""Teacher-output cache: run the teacher once, store answers and top-k probabilities.

The cache lives under ``paths.cache_root`` (never committed) and is reused by
offline distillation (E2), quantization-aware distillation (E4) and the shared
recovery training of every pruning arm (E5-E7, A1-A6).

Layout of a cache directory::

    metadata.json        settings the cache was made with (checked on resume)
    shard_00000.npz ...  one file per ``shard_size`` samples, written atomically
    manifest.json        written last: shard checksums and the manifest hash

A shard holds, for its samples in order: the sample ids and answer texts, the
number of generated tokens per sample, and the generated token ids with their top-k
token ids and log-probabilities (flattened over all tokens of all samples). The
stop token is part of the generated tokens. ``manifest.json`` exists only when every
shard is complete, so a directory without it is an unfinished cache.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from medcompress.config import RunContext
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper

CACHE_FORMAT_VERSION = 1
METADATA_NAME = "metadata.json"
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class CacheRecord:
    """Cached teacher output for one sample (arrays are views into the loaded shard)."""

    sample_id: str
    answer_text: str
    answer_token_ids: np.ndarray  # (T,)
    topk_token_ids: np.ndarray  # (T, k)
    topk_logprobs: np.ndarray  # (T, k)


def shard_name(index: int) -> str:
    """File name of shard number ``index``."""
    return f"shard_{index:05d}.npz"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sample_ids_hash(sample_ids: Sequence[str]) -> str:
    """Hash of the ordered sample ids (the cache is only valid for this exact list)."""
    return _sha256_text("\n".join(sample_ids))


def teacher_config_hash(teacher: VLMWrapper) -> str | None:
    """Hash of the teacher's model spec, or None for a wrapper without one."""
    spec = getattr(teacher, "spec", None)
    if spec is None:
        return None
    payload = json.dumps(dataclasses.asdict(spec), sort_keys=True, default=str)
    return _sha256_text(payload)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _pack_shard(
    sample_ids: Sequence[str], outputs: Sequence[Any], top_k: int
) -> dict[str, np.ndarray]:
    counts, tokens, topk_ids, topk_lp = [], [], [], []
    for sample_id, out in zip(sample_ids, outputs, strict=True):
        n = len(out.token_ids)
        if out.topk_token_ids is None or out.topk_logprobs is None:
            raise ValueError(f"{sample_id}: the teacher returned no top-k probabilities")
        if not (len(out.topk_token_ids) == len(out.topk_logprobs) == n):
            raise ValueError(f"{sample_id}: token and top-k lengths differ")
        if any(len(row) != top_k for row in out.topk_token_ids):
            raise ValueError(f"{sample_id}: expected {top_k} candidates per token")
        counts.append(n)
        tokens.extend(int(t) for t in out.token_ids)
        topk_ids.extend(out.topk_token_ids)
        topk_lp.extend(out.topk_logprobs)
    return {
        "sample_ids": np.array(list(sample_ids), dtype=str),
        "texts": np.array([o.text for o in outputs], dtype=str),
        "token_counts": np.array(counts, dtype=np.int32),
        "token_ids": np.array(tokens, dtype=np.int32),
        "topk_ids": np.array(topk_ids, dtype=np.int32).reshape(-1, top_k),
        "topk_logprobs": np.array(topk_lp, dtype=np.float32).reshape(-1, top_k),
    }


def _shard_matches(path: Path, expected_ids: Sequence[str]) -> bool:
    """True if ``path`` is a readable shard holding exactly ``expected_ids``."""
    try:
        with np.load(path, allow_pickle=False) as data:
            return list(data["sample_ids"]) == list(expected_ids)
    except (OSError, ValueError, KeyError):
        return False


def cache_teacher_outputs(
    teacher: VLMWrapper,
    samples: Sequence[VQASample],
    out_dir: Path,
    *,
    top_k: int,
    max_new_tokens: int,
    shard_size: int = 1000,
    data_root: Path | None = None,
) -> Path:
    """Generate teacher answers once and write them to sharded files with a manifest.

    Resumable: a shard that already holds the right samples is skipped. The settings
    are written to ``metadata.json`` and must match on resume. Nothing is written
    outside ``out_dir``. ``data_root`` is joined to each sample's relative image path.

    Used by: E0 (producer); E2, E4, E5-E7, A1-A6 (consumers).
    """
    if top_k < 1 or max_new_tokens < 1 or shard_size < 1:
        raise ValueError("top_k, max_new_tokens and shard_size must be positive")
    ids = [s.sample_id for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("sample ids must be unique")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "teacher_config_sha256": teacher_config_hash(teacher),
        "tokenizer_fingerprint": teacher.tokenizer_fingerprint(),
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "shard_size": shard_size,
        "num_samples": len(ids),
        "sample_ids_sha256": sample_ids_hash(ids),
        "image_longest_side": getattr(teacher, "image_longest_side", None),
    }
    meta_path = out / METADATA_NAME
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        changed = sorted(k for k in metadata if existing.get(k) != metadata[k])
        if changed:
            raise ValueError(f"{out} was made with different settings ({', '.join(changed)})")
    else:
        _write_json(meta_path, metadata)
    (out / MANIFEST_NAME).unlink(missing_ok=True)  # incomplete until the last shard is done

    root = Path(data_root) if data_root is not None else None
    shards = []
    for index, start in enumerate(range(0, len(samples), shard_size)):
        chunk = samples[start : start + shard_size]
        chunk_ids = [s.sample_id for s in chunk]
        path = out / shard_name(index)
        if not (path.exists() and _shard_matches(path, chunk_ids)):
            images = [(root / s.image_path) if root is not None else s.image_path for s in chunk]
            outputs = teacher.generate(
                images,
                [s.question for s in chunk],
                max_new_tokens=max_new_tokens,
                return_topk=top_k,
            )
            tmp = path.with_name(path.name + ".tmp")
            with tmp.open("wb") as handle:
                np.savez_compressed(handle, **_pack_shard(chunk_ids, outputs, top_k))
            os.replace(tmp, path)
        shards.append({"file": path.name, "num_samples": len(chunk), "sha256": _sha256_file(path)})
    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "metadata_sha256": _sha256_file(meta_path),
        "shards": shards,
    }
    manifest["manifest_sha256"] = _sha256_text(json.dumps(manifest, sort_keys=True))
    _write_json(out / MANIFEST_NAME, manifest)
    return out


def _read_manifest(cache_dir: Path) -> dict[str, Any]:
    path = Path(cache_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"{cache_dir} has no {MANIFEST_NAME}: the cache is unfinished")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != CACHE_FORMAT_VERSION:
        raise ValueError(f"unsupported cache format {manifest.get('format_version')!r}")
    return manifest


def cache_manifest_hash(cache_dir: Path) -> str:
    """The manifest hash to write into the run card."""
    return str(_read_manifest(cache_dir)["manifest_sha256"])


def load_teacher_cache(cache_dir: Path, *, verify: bool = True) -> Iterator[CacheRecord]:
    """Iterate cached records in the order they were written.

    With ``verify`` each shard's checksum is compared with the manifest before it is
    read. Raises if the cache is unfinished or a shard was changed.
    """
    cache_dir = Path(cache_dir)
    manifest = _read_manifest(cache_dir)
    for entry in manifest["shards"]:
        path = cache_dir / entry["file"]
        if verify and _sha256_file(path) != entry["sha256"]:
            raise ValueError(f"{path} does not match the manifest checksum")
        with np.load(path, allow_pickle=False) as data:
            counts = data["token_counts"]
            offsets = np.concatenate([[0], np.cumsum(counts)])
            for i, sample_id in enumerate(data["sample_ids"]):
                a, b = int(offsets[i]), int(offsets[i + 1])
                yield CacheRecord(
                    sample_id=str(sample_id),
                    answer_text=str(data["texts"][i]),
                    answer_token_ids=data["token_ids"][a:b],
                    topk_token_ids=data["topk_ids"][a:b],
                    topk_logprobs=data["topk_logprobs"][a:b],
                )


def cache_metadata(cache_dir: Path) -> Mapping[str, Any]:
    """Return the cache metadata (teacher config hash, tokenizer fingerprint, top-k, ...)."""
    path = Path(cache_dir) / METADATA_NAME
    if not path.exists():
        raise FileNotFoundError(f"{cache_dir} has no {METADATA_NAME}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_teacher_caching(
    config: Mapping[str, Any],
    run: RunContext | None,
    *,
    teacher: VLMWrapper | None = None,
    samples: Sequence[VQASample] | None = None,
) -> None:
    """Stage entry point for ``scripts/01_cache_teacher_outputs.py``.

    Loads the teacher from the config, reads the prepared splits, and calls
    :func:`cache_teacher_outputs` for the configured splits. With ``teacher_cache.pool_size``
    only the training pool of that size is cached (the examples E2, E4 and the recoveries
    train on, drawn with ``seeds.data_split``; see
    :func:`~medcompress.compression.distillation.offline_kd.training_pool`), instead of the
    whole split. ``teacher`` and ``samples`` can be passed in (tests do); otherwise they are
    built from the config, and ``pool_size`` is applied to them either way.

    Used by: E0.
    """
    from medcompress.config import ConfigError, find_todos

    section = config.get("teacher_cache", {})
    needed = {k: section.get(k) for k in ("splits", "top_k", "max_new_tokens", "shard_size")}
    open_keys = sorted(k for k, v in needed.items() if v is None or find_todos({k: v}))
    if open_keys:
        raise ConfigError(f"teacher_cache settings still open: {', '.join(open_keys)}")
    side = config.get("inputs", {}).get("image_longest_side")
    if teacher is None and (side is None or find_todos({"image_longest_side": side})):
        raise ConfigError("inputs.image_longest_side is still open (fixed at the S05b freeze)")
    paths = config["paths"]
    if samples is None:
        from medcompress.data.prepared import load_prepared

        splits = load_prepared(
            Path(paths["data_root"]),
            _dataset_section(config),
            Path(paths["private_root"]),
        )
        samples = [s for name in needed["splits"] for s in splits[name]]
    if "pool_size" in section:
        from medcompress.compression.distillation.offline_kd import data_seed, training_pool

        size = section["pool_size"]
        if size is None or find_todos({"pool_size": size}):
            raise ConfigError("teacher_cache.pool_size is still open (= training.data_subset)")
        samples = training_pool(samples, size, data_seed(config))
    if teacher is None:
        from medcompress.config import load_config
        from medcompress.models.loading import ModelSpec, load_model

        spec = ModelSpec.from_config(load_config(config["components"]["teacher"]))
        teacher = load_model(spec, image_longest_side=int(side))
    out = cache_teacher_outputs(
        teacher,
        samples,
        Path(paths["cache_root"]) / "teacher",
        top_k=int(needed["top_k"]),
        max_new_tokens=int(needed["max_new_tokens"]),
        shard_size=int(needed["shard_size"]),
        data_root=Path(paths["data_root"]),
    )
    print(f"teacher cache complete: {len(samples)} samples, manifest {cache_manifest_hash(out)}")


def _dataset_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
    from medcompress.config import load_config

    dataset_config = load_config(config["components"]["datasets"]["train_qa"])
    return dataset_config["dataset"]
