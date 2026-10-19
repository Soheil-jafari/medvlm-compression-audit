"""RC-D1-embed and RC-D1-label's scoring step (D02 amendment 3 of 2026-10-19).

``embed_step``: reads every image once (dHash and crop signature), reproduces RC-D1 run 3's
final contact sheet pairs (same data, code and seed; its split hash is compared with run 3's),
checks the embedding model on synthetic copies, embeds every image in two views, builds the
candidate pairs, draws the labelling sample and writes, to a private folder, everything the
labelling, the scoring and run 4 need. Only aggregate counts go to the repository.

``score_labels_step``: reads the two raters' labels, computes Cohen's kappa and the sensitivity
and precision of every rule of ``candidates.rule_family()``, and chooses the joining rule by the
rule fixed in advance (``labelling.choose_rule``), with the chaining guard checked over all
images. The chosen rule is written as numbers to ``results/data/near_duplicate_labels.csv``,
from which RC-D1 run 4 reads it.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from medcompress.data.candidates import (
    RULE_CROP_LEVELS,
    CandidateTable,
    CropPairs,
    JoinRule,
    build_candidates,
    component_groups,
    joined_pairs_all,
    largest_group,
    rule_family,
)
from medcompress.data.dedup import CHAIN_MIN_IMAGES
from medcompress.data.embeddings import (
    Embedder,
    EmbeddingSettings,
    embed_images,
    load_embeddings,
    save_embeddings,
    self_check,
    self_check_passes,
)
from medcompress.data.labelling import (
    LABELS,
    MIN_PRECISION,
    SAMPLE_SEED,
    STRATA,
    choose_rule,
    cohen_kappa,
    labelling_sample,
    read_labels,
    read_pairs,
    score_rule,
    stratum_sizes,
    truth_of,
    weighted_scores,
    write_labelling_files,
)
from medcompress.data.prepare import (
    _features,
    _row,
    crop_settings,
    near_duplicate_settings,
    prepare,
    score_histogram,
)

RUN3_THRESHOLD = 0.65
"""The crop rule of RC-D1 run 3 (D02 amendment 2, final values of 2026-10-19)."""
RUN3_SPLIT_PREFIX = "4a595fdf"
"""The start of run 3's split hash as reported on 2026-10-19 (the reproduction is checked)."""
EMBED_FILES = {
    "embeddings": "embeddings.npz",
    "candidates": "candidates.npz",
    "crop_pairs": "crop_pairs.npz",
    "meta": "embed_meta.json",
}


class SelfCheckError(ValueError):
    """The embedding model does not put synthetic copies nearer than different images."""


def candidate_settings(dataset_config: Mapping[str, Any]) -> dict[str, int]:
    """``near_duplicates.candidates``: k nearest neighbours, dHash bits, crop neighbours."""
    cfg = (dataset_config.get("near_duplicates") or {}).get("candidates") or {}
    return {
        "knn_k": int(cfg.get("knn_k", 20)),
        "dhash_bits": int(cfg.get("dhash_bits", 12)),
        "crop_k": int(cfg.get("crop_k", 5)),
    }


@dataclass(frozen=True)
class StepResult:
    """What a step printed, the aggregate rows it writes, and its private files."""

    lines: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]
    files: Mapping[str, Path]


def run3_band_pairs(
    data_root: Path,
    dataset_config: Mapping[str, Any],
    features: tuple[Any, Any, Any, Any],
    *,
    split_seed: int,
    close_bits: int,
    mask_threshold: int,
) -> tuple[list[tuple[str, str]], str]:
    """The pairs of RC-D1 run 3's final contact sheet (three bands) and the split hash, from
    the crop rule of amendment 2 on the same data with the same seed."""
    prepared = prepare(
        data_root, dataset_config, split_seed=split_seed, close_bits=close_bits,
        mask_threshold=mask_threshold, rule=JoinRule("crop", RUN3_THRESHOLD), features=features,
    )  # fmt: skip
    pairs = []
    for band in (
        prepared.top_cross_split,
        prepared.closest_dhash_cross_split,
        prepared.close_not_joined,
    ):
        pairs += [(min(t.first, t.second), max(t.first, t.second)) for t in band]
    return sorted(set(pairs)), prepared.split_sha256


def embed_step(
    data_root: Path,
    dataset_config: Mapping[str, Any],
    out_dir: Path,
    embedder: Embedder,
    *,
    split_seed: int,
    close_bits: int,
    mask_threshold: int,
    log: Callable[[str], None] = print,
) -> StepResult:
    """RC-D1-embed (module docstring). Raises :class:`SelfCheckError` if the model fails the
    synthetic check."""
    data_root, out_dir = Path(data_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    lines: list[str] = []

    def say(text: str) -> None:
        lines.append(text)
        log(text)

    settings = EmbeddingSettings.from_config(dataset_config)
    crop = crop_settings(dataset_config)
    started = time.perf_counter()
    features = _features(data_root, dataset_config, crop)
    _, images, fingerprints, signatures = features
    rows.append(_row("embed:images", len(images)))
    say(
        f"read {len(images)} images (dHash, crop signature) in "
        f"{time.perf_counter() - started:.0f} s"
    )

    run3: list[tuple[str, str]] = []
    if (data_root / "kvasir_seg" / "images").is_dir():
        run3, split_hash = run3_band_pairs(
            data_root, dataset_config, features, split_seed=split_seed, close_bits=close_bits,
            mask_threshold=mask_threshold,
        )  # fmt: skip
        matches = split_hash.startswith(RUN3_SPLIT_PREFIX)
        verdict = "matches" if matches else "DIFFERS FROM"
        say(f"run 3 reproduced: split_sha256 {split_hash} ({verdict} run 3's "
            f"{RUN3_SPLIT_PREFIX}...); {len(run3)} sheet pairs")  # fmt: skip
        rows += [_row("embed:run3:split_hash_matches", int(matches)),
                 _row("embed:run3:band_pairs", len(run3))]  # fmt: skip
    else:
        say("no Kvasir-SEG folder under the data root: run 3's sheet pairs are not added")

    info_ = embedder.info
    say(f"embedding model: {info_.get('model')} at revision {info_.get('revision')}; "
        f"licence (hub record): {info_.get('licence')}")  # fmt: skip
    check = self_check(embedder, settings)
    for view, scores in check.items():
        say(f"self-check ({view}): " + ", ".join(f"{k} {v:.3f}" for k, v in scores.items()))
        rows += [_row(f"embed:self_check:{view}:{k}", v) for k, v in scores.items()]
    if not self_check_passes(check):
        raise SelfCheckError(
            "a synthetic copy is not nearer to its original than a different image: this model "
            "does not separate copies (D02 amendment 3); report the lines above"
        )

    started = time.perf_counter()
    paths = [str(data_root / p) for p in images]
    every = max(1, len(paths) // 10)

    def progress(done: int, total: int) -> None:
        if done % every < settings.batch_size or done == total:
            log(f"  embedded {done} of {total} images ({time.perf_counter() - started:.0f} s)")

    embeddings = embed_images(paths, embedder, settings, progress=progress)
    seconds = time.perf_counter() - started
    rows.append(_row("embed:seconds", seconds))
    rows.append(_row("embed:seconds_per_image", seconds / max(1, len(images))))
    say(f"embedded {len(images)} images, two views, in {seconds:.0f} s")
    info = {**dict(embedder.info), "settings": asdict(settings)}
    save_embeddings(out_dir / EMBED_FILES["embeddings"], images, embeddings, info)

    cands = candidate_settings(dataset_config)
    started = time.perf_counter()
    table, crop_pairs = build_candidates(
        images, fingerprints, signatures, embeddings, crop,
        knn_k=cands["knn_k"], dhash_bits=cands["dhash_bits"], crop_k=cands["crop_k"],
        crop_floor=min(RULE_CROP_LEVELS), extra=run3,
    )  # fmt: skip
    table.save(out_dir / EMBED_FILES["candidates"])
    crop_pairs.save(out_dir / EMBED_FILES["crop_pairs"])
    say(f"candidate pairs: {len(table)} ({time.perf_counter() - started:.0f} s); "
        + ", ".join(f"{s} {int(table.has(s).sum())}" for s in
                    ("knn_full", "knn_centre", "dhash", "crop_knn", "run3_band")))  # fmt: skip
    rows.append(_row("embed:candidate_pairs", len(table)))
    for source in ("knn_full", "knn_centre", "dhash", "crop_knn", "run3_band"):
        rows.append(_row(f"embed:candidate_pairs:{source}", int(table.has(source).sum())))
    rows.append(_row("embed:crop_pairs_at_or_above_0.65", len(crop_pairs.a)))
    for view in ("full", "centre", "max"):
        for key, value in score_histogram(table.cosine(view).tolist()).items():
            rows.append(_row(f"embed:cosine:{view}:{key}", value))

    pairs = labelling_sample(table, seed=_labelling_seed(dataset_config))
    sizes = stratum_sizes(table)
    drawn = Counter(p.stratum for p in pairs)
    for name, _ in STRATA:
        rows.append(_row(f"embed:stratum:{name}:pairs", sizes[name]))
        rows.append(_row(f"embed:stratum:{name}:drawn", drawn.get(name, 0)))
    rows.append(_row("embed:labelling_pairs", len(pairs)))
    files = write_labelling_files(pairs, data_root, out_dir)
    meta = {
        "embedding": info,
        "candidates": cands,
        "labelling_seed": _labelling_seed(dataset_config),
        "images": len(images),
        "run3_pairs": len(run3),
    }
    (out_dir / EMBED_FILES["meta"]).write_text(json.dumps(meta, indent=1, default=str))
    say(f"{len(pairs)} pairs to label: {files['sheets']} ({math.ceil(len(pairs) / 10)} sheets), "
        f"template {files['template']}")  # fmt: skip
    return StepResult(
        tuple(lines), tuple(rows),
        {**{k: out_dir / v for k, v in EMBED_FILES.items()}, **files},
    )  # fmt: skip


def _labelling_seed(dataset_config: Mapping[str, Any]) -> int:
    cfg = (dataset_config.get("near_duplicates") or {}).get("labelling") or {}
    return int(cfg.get("seed", SAMPLE_SEED))


def _min_precision(dataset_config: Mapping[str, Any]) -> float:
    cfg = (dataset_config.get("near_duplicates") or {}).get("labelling") or {}
    return float(cfg.get("min_precision", MIN_PRECISION))


def score_labels_step(
    embed_dir: Path,
    label_files: Sequence[Path],
    dataset_config: Mapping[str, Any],
    *,
    log: Callable[[str], None] = print,
) -> tuple[StepResult, StepResult, JoinRule | None]:
    """RC-D1-label's scoring (module docstring). Returns (the labels file rows, the rules file
    rows, the chosen rule or ``None``)."""
    embed_dir = Path(embed_dir)
    pairs = read_pairs(embed_dir / "label_pairs.csv")
    ids = [p.pair_id for p in pairs]
    first, second = read_labels(ids, *label_files)
    truth = truth_of(first, second, ids)
    lines: list[str] = []

    def say(text: str) -> None:
        lines.append(text)
        log(text)

    a = [first[i] for i in ids]
    b = [second[i] for i in ids]
    kappa = cohen_kappa(a, b, LABELS)
    binary = cohen_kappa(
        ["same" if x == "same_scene" else "not" for x in a],
        ["same" if x == "same_scene" else "not" for x in b],
        ("same", "not"),
    )
    either = sum(1 for x, y in zip(a, b, strict=True) if "same_scene" in (x, y))
    rows: list[dict[str, Any]] = [
        _row("labels:pairs", len(ids)),
        _row("labels:kappa", kappa),
        _row("labels:kappa_same_scene_vs_rest", binary),
        _row("labels:agreement", sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(ids)),
        _row("labels:positives_both_same_scene", int(truth.sum())),
        _row("labels:either_same_scene", either),
    ]
    for rater, labels in (("rater_1", a), ("rater_2", b)):
        for label in LABELS:
            rows.append(_row(f"labels:{rater}:{label}", sum(1 for x in labels if x == label)))
    say(f"{len(ids)} pairs; Cohen's kappa {kappa:.3f} (same scene vs the rest {binary:.3f});"
        f" positives (both same_scene) {int(truth.sum())}; either rater {either}")  # fmt: skip

    scores = [score_rule(rule, pairs, truth) for rule in rule_family()]
    rule_rows: list[dict[str, Any]] = []
    for s in scores:
        for name, (value, low, high) in (("sensitivity", s.sensitivity),
                                         ("precision", s.precision)):  # fmt: skip
            row = _row(f"rule:{s.rule.label}:{name}", value)
            row.update(ci_low=low, ci_high=high)
            rule_rows.append(row)
        rule_rows.append(_row(f"rule:{s.rule.label}:joined_labelled", s.joined))

    embeddings, images, crop_pairs = _embed_inputs(embed_dir, dataset_config)
    settings = near_duplicate_settings(dataset_config, close_bits=4, confirm_threshold=1.0)

    def guard(rule: JoinRule) -> tuple[bool, int]:
        groups = component_groups(images, *joined_pairs_all(rule, embeddings, crop_pairs))
        largest = largest_group(groups)
        ok = largest < CHAIN_MIN_IMAGES or largest <= settings.max_group_fraction * len(images)
        return ok, largest

    chosen, checks = choose_rule(scores, guard, min_precision=_min_precision(dataset_config))
    for rule, ok, largest in checks:
        rule_rows.append(_row(f"rule:{rule.label}:largest_group", largest))
        rule_rows.append(_row(f"rule:{rule.label}:guard_ok", int(ok)))
    for rule, ok, largest in checks:
        say(f"guard {rule.label}: largest group {largest} ({largest / len(images):.1%}) "
            f"{'ok' if ok else 'over the guard'}")  # fmt: skip
    if chosen is None:
        say("NO rule has precision >= the minimum and passes the guard: report (a decision is "
            "needed, D02 amendment 3)")  # fmt: skip
        return StepResult(tuple(lines), tuple(rows), {}), StepResult((), tuple(rule_rows), {}), None
    rows += [_row(f"labels:chosen:{k}", v) for k, v in chosen.rule.code().items()]
    for name, (value, low, high) in (("sensitivity", chosen.sensitivity),
                                     ("precision", chosen.precision)):  # fmt: skip
        row = _row(f"labels:chosen:{name}", value)
        row.update(ci_low=low, ci_high=high)
        rows.append(row)
    weighted = weighted_scores(chosen.rule, pairs, truth)
    for name, (value, low, high) in weighted.items():
        row = _row(f"labels:chosen:{name}_weighted", value)
        row.update(ci_low=low, ci_high=high)
        rows.append(row)
    largest = next(lg for rule, ok, lg in checks if rule == chosen.rule)
    rows += [
        _row("labels:chosen:largest_group", largest),
        _row("labels:chosen:largest_fraction", largest / len(images)),
        _row("labels:min_precision", _min_precision(dataset_config)),
    ]
    sens, prec = chosen.sensitivity, chosen.precision
    say(f"chosen rule: {chosen.rule.label}: sensitivity {sens[0]:.3f} "
        f"({sens[1]:.3f}-{sens[2]:.3f}), precision {prec[0]:.3f} ({prec[1]:.3f}-{prec[2]:.3f}) "
        "on the labelled pairs "
        f"({chosen.true_positive} of {chosen.positives} positives joined, {chosen.joined} joined); "
        f"over all candidates (weighted): sensitivity {weighted['sensitivity'][0]:.3f}, precision "
        f"{weighted['precision'][0]:.3f}; largest group {largest}")  # fmt: skip
    return (
        StepResult(tuple(lines), tuple(rows), {}),
        StepResult((), tuple(rule_rows), {}),
        chosen.rule,
    )


def _embed_inputs(
    embed_dir: Path, dataset_config: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], list[str], CropPairs]:
    with np.load(embed_dir / EMBED_FILES["embeddings"], allow_pickle=False) as data:
        images = [str(x) for x in data["images"]]
    settings = EmbeddingSettings.from_config(dataset_config)
    embeddings, _ = load_embeddings(
        embed_dir / EMBED_FILES["embeddings"], images,
        EmbeddingSettings(settings.model, None, settings.side, settings.centre_fraction),
    )  # fmt: skip
    return embeddings, images, CropPairs.load(embed_dir / EMBED_FILES["crop_pairs"])


__all__ = [
    "CandidateTable",
    "SelfCheckError",
    "StepResult",
    "embed_step",
    "score_labels_step",
]
