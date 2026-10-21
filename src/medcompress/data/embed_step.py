"""RC-D1-embed and RC-D1-label's scoring step (D02 amendment 3 of 2026-10-19).

``embed_step``: reads every image once (dHash and crop signature), reproduces RC-D1 run 3's
final contact sheet pairs (same data, code and seed; its split hash is compared with run 3's),
checks the embedding model on synthetic copies, embeds every image in two views, builds the
candidate pairs, draws the labelling sample and writes, to a private folder, everything the
labelling, the scoring and run 4 need. Only aggregate counts go to the repository.

``score_labels_step``: reads the two raters' labels, computes Cohen's kappa, asks the third
member to decide the disagreements (every pair of a round whose kappa is below 0.40), computes
the sensitivity and precision of every rule of ``candidates.rule_family()`` and chooses the
joining rule by the rule fixed in advance (``labelling.choose_rule``), with the chaining guard
checked over all images; then the unsure analysis. The chosen rule is written as numbers to
``results/data/near_duplicate_labels.csv``, from which RC-D1 run 4 reads it, unless a top-up
is required first.

``topup_step``: the top-up of the amendment (``labelling.topup_sample``) for the chosen rule,
from RC-D1-embed's candidates (no new embedding): the extra pairs, their sheets and template.
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
    count_pairs_in_band,
    embed_images,
    load_embeddings,
    save_embeddings,
    self_check,
    self_check_passes,
)
from medcompress.data.labelling import (
    BAND_TOLERANCE,
    LABELS,
    MIN_PRECISION,
    SAMPLE_SEED,
    STRATA,
    TOPUP_BAND,
    TOPUP_MIN_IN_BAND,
    TOPUP_PAIRS,
    TOPUP_SEED,
    Adjudication,
    LabelError,
    LabelPair,
    RuleScore,
    TopUp,
    adjudicate,
    check_commitments,
    choose_rule,
    cohen_kappa,
    labelled_in_band,
    labelling_sample,
    read_labels,
    read_pairs,
    read_third,
    score_rule,
    stratum_sizes,
    topup_sample,
    truth_from,
    truth_of,
    weighted_scores,
    write_adjudication_template,
    write_labelling_files,
)
from medcompress.data.prepare import (
    _features,
    _row,
    crop_settings,
    labelled_rule,
    labels_values,
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


def topup_settings(dataset_config: Mapping[str, Any]) -> dict[str, float]:
    """``near_duplicates.labelling.topup``: band, min_in_band, pairs, seed (the top-up)."""
    labelling = (dataset_config.get("near_duplicates") or {}).get("labelling") or {}
    cfg = labelling.get("topup") or {}
    return {
        "band": float(cfg.get("band", TOPUP_BAND)),
        "min_in_band": int(cfg.get("min_in_band", TOPUP_MIN_IN_BAND)),
        "pairs": int(cfg.get("pairs", TOPUP_PAIRS)),
        "seed": int(cfg.get("seed", TOPUP_SEED)),
    }


def pinned_problems(embed_dir: Path, dataset_config: Mapping[str, Any]) -> list[str]:
    """Where RC-D1-embed's recorded model, revision and licence differ from the pinned ones of
    ``near_duplicates.embedding`` (nothing is checked while a value is ``TODO``)."""
    info = json.loads((Path(embed_dir) / "embeddings.json").read_text(encoding="utf-8"))
    cfg = (dataset_config.get("near_duplicates") or {}).get("embedding") or {}
    pinned = {key: cfg.get(key) for key in ("model", "revision", "licence")}
    if pinned["revision"] in (None, "TODO"):
        return []
    return [
        f"{key}: {info.get(key)!r} in {embed_dir}, {value!r} pinned in the config"
        for key, value in pinned.items()
        if value not in (None, "TODO") and str(info.get(key, "")).lower() != str(value).lower()
    ]


@dataclass(frozen=True)
class Scoring:
    """What RC-D1-label's scoring gives: the rows of the labels and rules files, the chosen
    rule (``None`` if none qualifies or the third member has pairs to decide), whether the
    top-up is required, and the private list of pairs for the third member (if any)."""

    labels: StepResult
    rules: StepResult
    chosen: JoinRule | None
    topup_required: bool = False
    adjudication: Path | None = None


def _round_pairs(embed_dir: Path, topup_dir: Path | None) -> dict[str, list[LabelPair]]:
    rounds = {"first": read_pairs(Path(embed_dir) / "label_pairs.csv")}
    if topup_dir is not None:
        rounds["topup"] = read_pairs(Path(topup_dir) / "label_pairs.csv")
        clash = {p.pair_id for p in rounds["first"]} & {p.pair_id for p in rounds["topup"]}
        if clash:
            raise LabelError(f"the top-up reuses {len(clash)} pair numbers of the first round")
    return rounds


def _adjudicated_labels(
    first: Mapping[int, str],
    second: Mapping[int, str],
    third: Mapping[int, str],
    rounds: Mapping[str, Sequence[LabelPair]],
    private_dir: Path | None,
    say: Callable[[str], None],
) -> Path | tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """D02 amendment 3: the kappa before adjudication, the third member's decisions, the
    positives. Returns the rows and the two truths, or the private list of pairs still to be
    decided."""
    ids = [p.pair_id for round_pairs in rounds.values() for p in round_pairs]
    unknown = sorted(set(third) - set(ids))
    if unknown:
        raise LabelError(f"rater_3 labelled {len(unknown)} unknown pairs (first: {unknown[0]})")
    decided: dict[str, Adjudication] = {
        name: adjudicate(first, second, third, [p.pair_id for p in round_pairs])
        for name, round_pairs in rounds.items()
    }
    a = [first[i] for i in ids]
    b = [second[i] for i in ids]
    kappa = cohen_kappa(a, b, LABELS)
    binary = cohen_kappa(
        ["same" if x == "same_scene" else "not" for x in a],
        ["same" if x == "same_scene" else "not" for x in b],
        ("same", "not"),
    )
    either = sum(1 for x, y in zip(a, b, strict=True) if "same_scene" in (x, y))
    both = truth_of(first, second, ids)
    say(f"{len(ids)} pairs; Cohen's kappa before adjudication {kappa:.3f} (same scene vs "
        f"the rest {binary:.3f}); both same_scene {int(both.sum())}; either rater "
        f"{either}")  # fmt: skip
    for name, adj in decided.items():
        why = ("kappa below 0.40: the third member decides every pair" if adj.all_pairs
               else "the third member decides the disagreements")  # fmt: skip
        say(f"round {name}: {len(adj.ids)} pairs, kappa {adj.kappa:.3f}; {why}: "
            f"{len(adj.required)} pairs ({len(adj.missing)} still without a third "
            "label)")  # fmt: skip
    missing = [i for adj in decided.values() for i in adj.missing]
    if missing:
        if private_dir is None:
            raise LabelError(f"{len(missing)} pairs need the third member's label: pass "
                             "--private-dir to write their list")  # fmt: skip
        path = write_adjudication_template(missing, private_dir)
        say(f"ADJUDICATION NEEDED: {len(missing)} pairs for the third member in {path} "
            "(private, no score, no rater's label); score again with their file added to "
            "--labels")  # fmt: skip
        return path

    final = {i: label for adj in decided.values() for i, label in adj.final.items()}
    truth = truth_from(final, ids)
    unsure_truth = truth_from(final, ids, unsure_positive=True)
    rows: list[dict[str, Any]] = [
        _row("labels:pairs", len(ids)),
        _row("labels:kappa", kappa),
        _row("labels:kappa_same_scene_vs_rest", binary),
        _row("labels:agreement", sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(ids)),
        _row("labels:positives_both_same_scene", int(both.sum())),
        _row("labels:either_same_scene", either),
        _row("labels:adjudicated", sum(len(adj.required) for adj in decided.values())),
        _row("labels:positives", int(truth.sum())),
        _row("labels:positives_unsure_as_same_scene", int(unsure_truth.sum())),
    ]
    for name in ("first", "topup"):
        adj_ = decided.get(name)
        rows.append(_row(f"labels:pairs:{name}", len(adj_.ids) if adj_ else 0))
        if adj_ is not None:
            rows += [_row(f"labels:kappa:{name}", adj_.kappa),
                     _row(f"labels:adjudicated:{name}", len(adj_.required)),
                     _row(f"labels:adjudicated_all:{name}", int(adj_.all_pairs))]  # fmt: skip
    for rater, labels in (("rater_1", a), ("rater_2", b)):
        for label in LABELS:
            rows.append(_row(f"labels:{rater}:{label}", sum(1 for x in labels if x == label)))
    for label in LABELS:
        rows.append(_row(f"labels:final:{label}", sum(1 for x in final.values() if x == label)))
    say(f"after adjudication: {int(truth.sum())} positives (same_scene); "
        f"{int(unsure_truth.sum())} with unsure counted as same_scene")  # fmt: skip
    return rows, truth, unsure_truth


def _two_rater_labels(
    first: Mapping[int, str],
    second: Mapping[int, str],
    rounds: Mapping[str, Sequence[LabelPair]],
    kappa_pairs: tuple[int, int],
    say: Callable[[str], None],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """D02 amendment 4: two raters, no third member. Cohen's kappa on the pairs numbered
    ``kappa_pairs`` (both ends included: the pairs both raters labelled blind), reported only.
    Returns the rows and three truths: primary (both ``same_scene``), the unsure analysis's
    sensitivity side (both ``same_scene`` or ``unsure``) and either rater ``same_scene``."""
    ids = [p.pair_id for round_pairs in rounds.values() for p in round_pairs]
    low, high = kappa_pairs
    blind = [i for i in ids if low <= i <= high]
    if not blind:
        raise LabelError(f"no labelled pair is numbered {low} to {high} (--kappa-pairs)")

    def agreement(pick: Sequence[int]) -> tuple[float, float, float]:
        a, b = [first[i] for i in pick], [second[i] for i in pick]
        binary = cohen_kappa(["same" if x == "same_scene" else "not" for x in a],
                             ["same" if x == "same_scene" else "not" for x in b],
                             ("same", "not"))  # fmt: skip
        raw = sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(pick)
        return cohen_kappa(a, b, LABELS), binary, raw

    kappa, binary, raw = agreement(blind)
    both = truth_of(first, second, ids)
    unsure_truth = np.array([first[i] in ("same_scene", "unsure") and
                             second[i] in ("same_scene", "unsure") for i in ids])  # fmt: skip
    either_truth = np.array([first[i] == "same_scene" or second[i] == "same_scene" for i in ids])
    rows: list[dict[str, Any]] = [
        _row("labels:pairs", len(ids)),
        _row("labels:kappa", kappa),
        _row("labels:kappa_same_scene_vs_rest", binary),
        _row("labels:agreement", raw),
        _row("labels:kappa:first_pair", low),
        _row("labels:kappa:last_pair", high),
        _row("labels:kappa:pairs", len(blind)),
        _row("labels:third_member", 0),
        _row("labels:positives_both_same_scene", int(both.sum())),
        _row("labels:either_same_scene", int(either_truth.sum())),
        _row("labels:positives", int(both.sum())),
        _row("labels:positives_unsure_as_same_scene", int(unsure_truth.sum())),
    ]
    for name in ("first", "topup"):
        rows.append(_row(f"labels:pairs:{name}", len(rounds.get(name, ()))))
    if rounds.get("topup"):  # a top-up round's agreement, on all its pairs (reported only)
        rows.append(_row("labels:kappa:topup", agreement([p.pair_id for p in rounds["topup"]])[0]))
    for rater, labels in (("rater_1", first), ("rater_2", second)):
        for label in LABELS:
            rows.append(_row(f"labels:{rater}:{label}",
                             sum(1 for i in ids if labels[i] == label)))  # fmt: skip
    say(f"{len(ids)} pairs, two raters, no third member (D02 amendment 4); Cohen's kappa on "
        f"pairs {low}-{high} ({len(blind)} pairs) {kappa:.3f} (same scene vs the rest "
        f"{binary:.3f}), raw agreement {raw:.3f}; reported only")  # fmt: skip
    say(f"positives (both same_scene) {int(both.sum())}; either rater same_scene "
        f"{int(either_truth.sum())}; both same_scene or unsure "
        f"{int(unsure_truth.sum())}")  # fmt: skip
    return rows, both, unsure_truth, either_truth


def score_labels_step(
    embed_dir: Path,
    label_files: Sequence[Path],
    dataset_config: Mapping[str, Any],
    *,
    topup_dir: Path | None = None,
    private_dir: Path | None = None,
    commitments: Path | None = None,
    third_member: bool = True,
    kappa_pairs: tuple[int, int] | None = None,
    log: Callable[[str], None] = print,
) -> Scoring:
    """RC-D1-label's scoring (module docstring). ``label_files``: the raters' files of every
    round and the third member's files (column ``rater_3``); ``topup_dir``: the top-up's output
    (its pairs are scored with the first round's, and the rule is chosen again on all labels);
    ``private_dir``: where the list of pairs for the third member is written; ``commitments``:
    the committed SHA-256 of each label file, checked first.

    ``third_member=False`` (D02 amendment 4): no adjudication; the kappa on the pairs numbered
    ``kappa_pairs``; positive = both raters ``same_scene``; the either-rater analysis is added
    (reported only). The rule, the guard, the weights, the unsure analysis and the top-up are
    as with a third member."""
    embed_dir = Path(embed_dir)
    lines: list[str] = []

    def say(text: str) -> None:
        lines.append(text)
        log(text)

    if commitments is not None:
        for path, match in zip(label_files, check_commitments(label_files, commitments),
                               strict=True):  # fmt: skip
            say(f"{Path(path).name}: matches the commitment of {match['role']}, round "
                f"{match['round']} (SHA-256 {match['sha256'][:12]}...)")  # fmt: skip
    rounds = _round_pairs(embed_dir, topup_dir)
    pairs = [p for round_pairs in rounds.values() for p in round_pairs]
    ids = [p.pair_id for p in pairs]
    first, second = read_labels(ids, *label_files)
    third = read_third(*label_files)
    either_truth: np.ndarray | None = None
    if third_member:
        head = _adjudicated_labels(first, second, third, rounds, private_dir, say)
        if isinstance(head, Path):
            return Scoring(StepResult(tuple(lines), (), {}), StepResult((), (), {}), None,
                           adjudication=head)  # fmt: skip
        rows, truth, unsure_truth = head
    else:
        if third:
            raise LabelError(f"{len(third)} rater_3 labels given, but there is no third member "
                             "(D02 amendment 4)")  # fmt: skip
        if kappa_pairs is None:
            raise LabelError("without a third member the kappa needs its pairs (--kappa-pairs; "
                             "D02 amendment 4: 11-298)")  # fmt: skip
        rows, truth, unsure_truth, either_truth = _two_rater_labels(
            first, second, rounds, kappa_pairs, say
        )

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
    guarded: dict[JoinRule, tuple[bool, int]] = {}

    def guard(rule: JoinRule) -> tuple[bool, int]:
        if rule not in guarded:
            groups = component_groups(images, *joined_pairs_all(rule, embeddings, crop_pairs))
            largest = largest_group(groups)
            ok = largest < CHAIN_MIN_IMAGES or largest <= settings.max_group_fraction * len(images)
            guarded[rule] = (ok, largest)
        return guarded[rule]

    min_precision = _min_precision(dataset_config)
    chosen, checks = choose_rule(scores, guard, min_precision=min_precision)
    for rule, ok, largest in checks:
        rule_rows.append(_row(f"rule:{rule.label}:largest_group", largest))
        rule_rows.append(_row(f"rule:{rule.label}:guard_ok", int(ok)))
    for rule, ok, largest in checks:
        say(f"guard {rule.label}: largest group {largest} ({largest / len(images):.1%}) "
            f"{'ok' if ok else 'over the guard'}")  # fmt: skip
    if chosen is None:
        say("NO rule has precision >= the minimum and passes the guard: report (a decision is "
            "needed, D02 amendment 3)")  # fmt: skip
        return Scoring(StepResult(tuple(lines), tuple(rows), {}),
                       StepResult((), tuple(rule_rows), {}), None)  # fmt: skip
    rows += [_row(f"labels:chosen:{k}", v) for k, v in chosen.rule.code().items()]
    rows += _score_rows("labels:chosen", chosen)
    first_round = len(rounds["first"])  # the stratified design: the first round only
    weighted = weighted_scores(chosen.rule, rounds["first"], truth[:first_round])
    for name, (value, low, high) in weighted.items():
        row = _row(f"labels:chosen:{name}_weighted", value)
        row.update(ci_low=low, ci_high=high)
        rows.append(row)
    largest = guard(chosen.rule)[1]
    top = topup_settings(dataset_config)
    in_band = labelled_in_band(pairs, chosen.rule, top["band"])
    required = False
    if topup_dir is None:  # the same draw as topup-labels: needed, and pairs left to draw
        table = CandidateTable.load(embed_dir / EMBED_FILES["candidates"])
        draw = topup_sample(
            table, rounds["first"], chosen.rule, band=top["band"],
            min_in_band=int(top["min_in_band"]), size=int(top["pairs"]), seed=int(top["seed"]),
        )  # fmt: skip
        required = draw.needed and bool(draw.pairs)
        if draw.needed and not draw.pairs:
            say("fewer than the minimum labelled pairs near the threshold, but no candidate "
                "pair is left in the band: no top-up, the rule stands")  # fmt: skip
    rows += [
        _row("labels:chosen:largest_group", largest),
        _row("labels:chosen:largest_fraction", largest / len(images)),
        _row("labels:min_precision", min_precision),
        _row("labels:in_band_labelled", in_band),
        _row("labels:topup_required", int(required)),
    ]
    sens, prec = chosen.sensitivity, chosen.precision
    say(f"chosen rule: {chosen.rule.label}: sensitivity {sens[0]:.3f} "
        f"({sens[1]:.3f}-{sens[2]:.3f}), precision {prec[0]:.3f} ({prec[1]:.3f}-{prec[2]:.3f}) "
        "on the labelled pairs "
        f"({chosen.true_positive} of {chosen.positives} positives joined, {chosen.joined} joined); "
        f"over all candidates (weighted, first round): sensitivity "
        f"{weighted['sensitivity'][0]:.3f}, precision {weighted['precision'][0]:.3f}; "
        f"largest group {largest}")  # fmt: skip

    # the unsure analysis: unsure is not same_scene for precision, same_scene for sensitivity
    analysis = [score_rule(r, pairs, truth, sensitivity_truth=unsure_truth) for r in rule_family()]
    as_chosen = next(s for s in analysis if s.rule == chosen.rule)
    rows += _score_rows("labels:unsure_analysis:chosen", as_chosen)
    other, _ = choose_rule(analysis, guard, min_precision=min_precision)
    if other is not None:
        rows += [_row(f"labels:unsure_analysis:rule:{k}", v) for k, v in other.rule.code().items()]
        rows += _score_rows("labels:unsure_analysis:rule", other)
    same = other is not None and other.rule == chosen.rule
    rows.append(_row("labels:unsure_analysis:same_rule", int(same)))
    say(f"unsure analysis (sensitivity with unsure as same_scene): the chosen rule's sensitivity "
        f"{as_chosen.sensitivity[0]:.3f}; the rule it would choose: "
        f"{other.rule.label if other else 'none'}")  # fmt: skip
    if either_truth is not None:  # D02 amendment 4: either rater same_scene, reported only
        either = [score_rule(r, pairs, either_truth) for r in rule_family()]
        either_chosen = next(s for s in either if s.rule == chosen.rule)
        rows += _score_rows("labels:either_rater:chosen", either_chosen)
        alt, _ = choose_rule(either, guard, min_precision=min_precision)
        if alt is not None:
            rows += [_row(f"labels:either_rater:rule:{k}", v) for k, v in alt.rule.code().items()]
            rows += _score_rows("labels:either_rater:rule", alt)
        same_either = alt is not None and alt.rule == chosen.rule
        rows.append(_row("labels:either_rater:same_rule", int(same_either)))
        say(f"either-rater analysis (same_scene for either rater): the chosen rule's sensitivity "
            f"{either_chosen.sensitivity[0]:.3f}, precision {either_chosen.precision[0]:.3f}; "
            f"the rule it would choose: {alt.rule.label if alt else 'none'}")  # fmt: skip
        say(f"top-up: {'REQUIRED' if required else 'not needed'} ({in_band} labelled pairs "
            f"within +-{top['band']:.2f} of {chosen.rule.threshold:.2f}, {chosen.rule.view} "
            f"view; the minimum is {top['min_in_band']})")  # fmt: skip
    if required:
        say(f"TOP-UP REQUIRED: {in_band} labelled pairs within +-{top['band']:.2f} of "
            f"{chosen.rule.threshold:.2f} ({chosen.rule.view} view), fewer than "
            f"{top['min_in_band']}: run topup-labels, label its pairs, score again (D02 "
            "amendment 3); run 4 waits")  # fmt: skip
    return Scoring(
        StepResult(tuple(lines), tuple(rows), {}),
        StepResult((), tuple(rule_rows), {}),
        chosen.rule,
        topup_required=required,
    )


def _score_rows(prefix: str, score: RuleScore) -> list[dict[str, Any]]:
    out = []
    for name, (value, low, high) in (("sensitivity", score.sensitivity),
                                     ("precision", score.precision)):  # fmt: skip
        row = _row(f"{prefix}:{name}", value)
        row.update(ci_low=low, ci_high=high)
        out.append(row)
    return out


def topup_step(
    embed_dir: Path,
    data_root: Path,
    out_dir: Path,
    results_root: Path,
    dataset_config: Mapping[str, Any],
    *,
    log: Callable[[str], None] = print,
) -> tuple[StepResult, TopUp]:
    """The top-up (``labelling.topup_sample``) for the rule chosen on the first round, read
    from the committed labels file. Writes the extra pairs, their sheets and template to
    ``out_dir`` (private) when the top-up is needed; refuses if a top-up was scored already
    (the rule is chosen again only once)."""
    embed_dir, out_dir = Path(embed_dir), Path(out_dir)
    values = labels_values(results_root)
    if values.get("labels:pairs:topup", 0.0) > 0:
        raise LabelError("the top-up was scored already: the rule is chosen again only once "
                         "(D02 amendment 3)")  # fmt: skip
    rule = labelled_rule(results_root, for_topup=True)
    cfg = topup_settings(dataset_config)
    table = CandidateTable.load(embed_dir / EMBED_FILES["candidates"])
    labelled = read_pairs(embed_dir / "label_pairs.csv")
    top = topup_sample(
        table, labelled, rule, band=cfg["band"], min_in_band=int(cfg["min_in_band"]),
        size=int(cfg["pairs"]), seed=int(cfg["seed"]),
    )  # fmt: skip
    embeddings, _, _ = _embed_inputs(embed_dir, dataset_config)
    low, high = rule.threshold - cfg["band"], rule.threshold + cfg["band"]
    every = count_pairs_in_band(embeddings, rule.view, low - BAND_TOLERANCE,
                                high + BAND_TOLERANCE)  # fmt: skip
    rows = [_row(f"topup:first_rule:{k}", v) for k, v in rule.code().items()]
    rows += [
        _row("topup:band", cfg["band"]),
        _row("topup:min_in_band", cfg["min_in_band"]),
        _row("topup:seed", cfg["seed"]),
        _row("topup:in_band_labelled", top.in_band_labelled),
        _row("topup:in_band_candidates_unlabelled", top.in_band_unlabelled),
        _row("topup:in_band_pairs_all_images", every),
        _row("topup:needed", int(top.needed)),
        _row("topup:drawn", len(top.pairs)),
    ]
    lines = [
        f"rule chosen on the first round: {rule.label}; {top.in_band_labelled} labelled pairs "
        f"within +-{cfg['band']:.2f} of {rule.threshold:.2f} ({rule.view} view); in the band: "
        f"{top.in_band_unlabelled} candidate pairs not labelled, {every} pairs of all images"
    ]
    files: dict[str, Path] = {}
    if not top.needed:
        lines.append(f"no top-up: at least {cfg['min_in_band']} labelled pairs in the band; the "
                     "rule stands")  # fmt: skip
    elif not top.pairs:
        lines.append("fewer than the minimum labelled pairs near the threshold, but no candidate "
                     "pair is left in the band: no top-up, the rule stands")  # fmt: skip
    else:
        files = write_labelling_files(list(top.pairs), data_root, out_dir)
        lines.append(f"top-up: {len(top.pairs)} pairs to label (pairs {top.pairs[0].pair_id} to "
                     f"{top.pairs[-1].pair_id}, seed {cfg['seed']}): {files['sheets']}, "
                     f"template {files['template']}")  # fmt: skip
    for line in lines:
        log(line)
    return StepResult(tuple(lines), tuple(rows), files), top


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
    "Scoring",
    "SelfCheckError",
    "StepResult",
    "embed_step",
    "pinned_problems",
    "score_labels_step",
    "topup_step",
]
