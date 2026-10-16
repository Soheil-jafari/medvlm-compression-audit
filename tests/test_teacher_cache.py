"""Teacher-output cache: sharding, resume, checks and the stage entry point (no torch needed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from medcompress.cli import CONFIG_INCOMPLETE_EXIT_CODE, run_stage
from medcompress.config import ConfigError
from medcompress.data.base import VQASample
from medcompress.models.base import GenerationOutput
from medcompress.models.teacher_cache import (
    CACHE_FORMAT_VERSION,
    cache_manifest_hash,
    cache_metadata,
    cache_teacher_outputs,
    load_teacher_cache,
    run_teacher_caching,
    shard_name,
)

TOP_K = 3


class FakeTeacher:
    """Deterministic stand-in: the answer length depends on the question."""

    spec = None

    def __init__(self, fingerprint: str = "fast:abc", salt: float = 0.0) -> None:
        self.fingerprint, self.salt = fingerprint, salt
        self.calls: list[list[str]] = []
        self.images: list[object] = []

    def tokenizer_fingerprint(self) -> str:
        return self.fingerprint

    def generate(self, images, questions, *, max_new_tokens, return_topk=None):  # noqa: ANN001, ANN201
        self.calls.append(list(questions))
        self.images.extend(images)
        outputs = []
        for q in questions:
            n = min(max_new_tokens, 1 + len(q) % 3)
            ids = [(len(q) + i) % 50 for i in range(n)]
            topk = [[(t + j) % 50 for j in range(return_topk)] for t in ids]
            lp = [[-0.1 * (j + 1) + self.salt for j in range(return_topk)] for _ in ids]
            outputs.append(GenerationOutput(f"answer {q}", ids, topk, lp))
        return outputs


def _samples(n: int) -> list[VQASample]:
    return [VQASample(f"s{i:03d}", "d", f"images/{i}.jpg", "q" * (i + 1), "a") for i in range(n)]


def _cache(tmp_path: Path, teacher: FakeTeacher, n: int = 5, **kwargs) -> Path:  # noqa: ANN003
    return cache_teacher_outputs(
        teacher,
        _samples(n),
        tmp_path / "cache",
        top_k=TOP_K,
        max_new_tokens=4,
        shard_size=2,
        **kwargs,
    )


def test_round_trip_in_order(tmp_path: Path) -> None:
    teacher = FakeTeacher()
    out = _cache(tmp_path, teacher)
    assert [p.name for p in sorted(out.glob("shard_*.npz"))] == [shard_name(i) for i in range(3)]
    records = list(load_teacher_cache(out))
    assert [r.sample_id for r in records] == [f"s{i:03d}" for i in range(5)]
    expected = teacher.generate(["q" * 3], ["q" * 3], max_new_tokens=4, return_topk=TOP_K)[0]
    third = records[2]
    assert third.answer_text == expected.text
    assert third.answer_token_ids.tolist() == list(expected.token_ids)
    assert third.topk_token_ids.shape == (len(expected.token_ids), TOP_K)
    assert np.allclose(third.topk_logprobs, np.array(expected.topk_logprobs, dtype=np.float32))
    meta = cache_metadata(out)
    assert meta["format_version"] == CACHE_FORMAT_VERSION and meta["top_k"] == TOP_K
    assert meta["tokenizer_fingerprint"] == "fast:abc" and meta["num_samples"] == 5


def test_images_are_joined_to_the_data_root(tmp_path: Path) -> None:
    teacher = FakeTeacher()
    _cache(tmp_path, teacher, n=2, data_root=tmp_path / "data")
    assert teacher.images == [
        tmp_path / "data" / "images" / "0.jpg",
        tmp_path / "data" / "images" / "1.jpg",
    ]


def test_resume_only_regenerates_missing_shards(tmp_path: Path) -> None:
    first = FakeTeacher()
    out = _cache(tmp_path, first)
    assert len(first.calls) == 3
    full_hash = cache_manifest_hash(out)
    (out / shard_name(2)).unlink()
    (out / "manifest.json").unlink()
    second = FakeTeacher()
    _cache(tmp_path, second)
    assert len(second.calls) == 1 and second.calls[0] == ["q" * 5]
    assert cache_manifest_hash(out) == full_hash  # same inputs, same hash


def test_changed_settings_or_samples_are_refused(tmp_path: Path) -> None:
    out = _cache(tmp_path, FakeTeacher())
    with pytest.raises(ValueError, match="top_k"):
        cache_teacher_outputs(
            FakeTeacher(), _samples(5), out, top_k=TOP_K + 1, max_new_tokens=4, shard_size=2
        )
    with pytest.raises(ValueError, match="tokenizer_fingerprint"):
        cache_teacher_outputs(
            FakeTeacher("fast:other"), _samples(5), out, top_k=TOP_K, max_new_tokens=4, shard_size=2
        )
    with pytest.raises(ValueError, match="sample_ids_sha256"):
        cache_teacher_outputs(
            FakeTeacher(), _samples(5)[::-1], out, top_k=TOP_K, max_new_tokens=4, shard_size=2
        )


def test_unfinished_and_tampered_caches_are_rejected(tmp_path: Path) -> None:
    out = _cache(tmp_path, FakeTeacher())
    (out / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="unfinished"):
        list(load_teacher_cache(out))
    _cache(tmp_path, FakeTeacher())  # resume finishes it again without calling the teacher
    shard = out / shard_name(0)
    shard.write_bytes(shard.read_bytes() + b"x")
    with pytest.raises(ValueError, match="checksum"):
        list(load_teacher_cache(out))


def test_manifest_hash_follows_the_content(tmp_path: Path) -> None:
    a = cache_manifest_hash(_cache(tmp_path / "a", FakeTeacher()))
    b = cache_manifest_hash(_cache(tmp_path / "b", FakeTeacher()))
    c = cache_manifest_hash(_cache(tmp_path / "c", FakeTeacher(salt=0.5)))
    assert a == b and a != c


def test_bad_teacher_outputs_are_rejected(tmp_path: Path) -> None:
    class NoTopK(FakeTeacher):
        def generate(self, images, questions, *, max_new_tokens, return_topk=None):  # noqa: ANN001, ANN201
            return [GenerationOutput("x", [1, 2]) for _ in questions]

    with pytest.raises(ValueError, match="no top-k"):
        _cache(tmp_path, NoTopK())
    with pytest.raises(ValueError, match="unique"):
        cache_teacher_outputs(
            FakeTeacher(), _samples(2) + _samples(1), tmp_path / "x", top_k=1, max_new_tokens=1
        )


def _config(tmp_path: Path, **cache: object) -> dict:
    section = {"splits": ["train"], "top_k": TOP_K, "max_new_tokens": 4, "shard_size": 2, **cache}
    return {
        "paths": {
            "data_root": str(tmp_path / "data"),
            "cache_root": str(tmp_path / "cache"),
            "private_root": str(tmp_path / "private"),
        },
        "teacher_cache": section,
    }


def test_stage_entry_point_writes_under_the_cache_root(tmp_path: Path) -> None:
    run_teacher_caching(_config(tmp_path), None, teacher=FakeTeacher(), samples=_samples(3))
    root = tmp_path / "cache" / "teacher"
    assert json.loads((root / "manifest.json").read_text())["manifest_sha256"]
    assert len(list(load_teacher_cache(root))) == 3


def test_open_settings_stop_the_stage_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    config = _config(tmp_path, top_k="TODO")
    with pytest.raises(ConfigError, match="top_k"):
        run_teacher_caching(config, None, teacher=FakeTeacher(), samples=_samples(1))
    code = run_stage(
        "01",
        run_teacher_caching,
        config=config,
        run=None,
        teacher=FakeTeacher(),
        samples=_samples(1),
    )
    assert code == CONFIG_INCOMPLETE_EXIT_CODE
    assert "configuration incomplete" in capsys.readouterr().out


def test_the_cache_can_hold_just_the_training_pool(tmp_path: Path) -> None:
    from medcompress.compression.distillation.offline_kd import training_pool

    samples = _samples(7)
    config = {**_config(tmp_path, pool_size=3), "seeds": {"data_split": 5}}
    run_teacher_caching(config, None, teacher=FakeTeacher(), samples=samples)
    cached = [r.sample_id for r in load_teacher_cache(tmp_path / "cache" / "teacher")]
    assert cached == [s.sample_id for s in training_pool(samples, 3, 5)]
    for bad, match in (({"pool_size": "TODO"}, "pool_size"), ({}, "seeds.data_split")):
        config = {**_config(tmp_path / "x", pool_size=3), "seeds": {"data_split": 5}}
        config["teacher_cache"].update(bad)
        if not bad:
            config.pop("seeds")
        with pytest.raises(ConfigError, match=match):
            run_teacher_caching(config, None, teacher=FakeTeacher(), samples=samples)


def test_the_cache_metadata_records_the_image_size(tmp_path: Path) -> None:
    teacher = FakeTeacher()
    teacher.image_longest_side = 448  # type: ignore[attr-defined]
    cache_teacher_outputs(teacher, _samples(2), tmp_path / "c", top_k=TOP_K, max_new_tokens=2)
    teacher.image_longest_side = 512  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="image_longest_side"):
        cache_teacher_outputs(teacher, _samples(2), tmp_path / "c", top_k=TOP_K, max_new_tokens=2)


def test_the_first_answers_are_timed_on_whole_batches_and_recorded_at_once(
    tmp_path: Path,
) -> None:
    from medcompress.config import RunContext

    plain = cache_teacher_outputs(
        FakeTeacher(), _samples(7), tmp_path / "plain", top_k=TOP_K, max_new_tokens=4, shard_size=5
    )
    timed, seen = FakeTeacher(), []
    timed.batch_size = 2  # type: ignore[attr-defined]
    out = cache_teacher_outputs(
        timed, _samples(7), tmp_path / "timed", top_k=TOP_K, max_new_tokens=4, shard_size=5,
        time_first=3, on_timed=lambda n, s: seen.append((n, s)),
    )  # fmt: skip
    assert [len(c) for c in timed.calls] == [4, 1, 2]  # 3 rounded up to two batches of 2
    assert len(seen) == 1 and seen[0][0] == 4 and seen[0][1] >= 0
    same = [r.answer_text for r in load_teacher_cache(out)]
    assert same == [r.answer_text for r in load_teacher_cache(plain)]
    again = FakeTeacher()
    cache_teacher_outputs(
        again, _samples(7), tmp_path / "timed", top_k=TOP_K, max_new_tokens=4, shard_size=5,
        time_first=3, on_timed=lambda n, s: seen.append((n, s)),
    )  # fmt: skip
    assert again.calls == [] and len(seen) == 1  # a resumed cache is not timed again

    config = _config(tmp_path / "stage")
    config["paths"]["results_root"] = str(tmp_path / "results")
    run_teacher_caching(
        config, RunContext("E0", seed=0), teacher=FakeTeacher(), samples=_samples(3)
    )
    metrics = (tmp_path / "results" / "E0" / "nobudget" / "seed0" / "metrics.csv").read_text()
    assert "teacher_cache_first_answers," in metrics
    assert "teacher_cache_first_answers_seconds" in metrics
