"""Near-duplicates by embeddings and human labels (D02 amendment 3 of 2026-10-19), on CPU.

The real model (DINOv2-small, proposed) runs on Kaggle; here a numpy stand-in
(``embeddings.TinyEmbedder``) checks the pipeline: the views, the self-check on synthetic
copies (shifted, blurred, rescaled, brightened copies near, different images not), the
candidates, the rules, the labelling sample and sheets, the scores and the rule choice, run 4
with the chosen rule, the strict test subset, and the runner plans.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from medcompress.data import candidates as cand
from medcompress.data import labelling as lab
from medcompress.data.dedup import CropSettings, union_groups
from medcompress.data.embeddings import (
    EmbeddingSettings,
    TinyEmbedder,
    embed_images,
    knn_pairs,
    l2_normalise,
    normalise,
    pairs_at_or_above,
    self_check,
    self_check_passes,
    synthetic_variants,
    view_images,
)

REPO = Path(__file__).resolve().parents[1]
SETTINGS = EmbeddingSettings(model="tiny", side=224, centre_fraction=0.60, batch_size=4)
CONFIG = {
    "name": "Kvasir-VQA-x1",
    "label_mapping": None,
    "near_duplicates": {
        "max_group_fraction": 0.5,
        "candidates": {"knn_k": 3, "dhash_bits": 12, "crop_k": 2},
        "embedding": {"model": "tiny", "side": 224, "centre_fraction": 0.60, "batch_size": 4},
    },
}


def _dataset(root: Path, **kwargs) -> Path:  # type: ignore[no-untyped-def]
    from test_near_duplicates import _dataset as make

    return make(root, **kwargs)


# ------------------------------------------------------------------ embeddings


def test_the_two_views_and_the_normalisation() -> None:
    image = Image.new("RGB", (300, 200), (255, 0, 0))
    image.paste((0, 255, 0), (0, 0, 50, 30))  # an overlay in the top-left corner
    views = view_images(image, SETTINGS)
    assert {k: v.size for k, v in views.items()} == {"full": (224, 224), "centre": (224, 224)}
    assert views["full"].getpixel((5, 5))[1] > 200  # the full view keeps the corner overlay
    assert views["centre"].getpixel((5, 5)) == (255, 0, 0)  # the central 60 % leaves it out
    array = normalise(views["centre"], (0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    assert array.shape == (3, 224, 224) and array.dtype == np.float32
    assert np.allclose(array[0], 2.0) and np.allclose(array[1], -2.0)
    unit = l2_normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(unit, [[0.6, 0.8], [0.0, 0.0]])


def test_copies_are_near_in_cosine_and_different_images_are_not() -> None:
    result = self_check(TinyEmbedder(), SETTINGS)
    for view, scores in result.items():
        copies = {k: v for k, v in scores.items() if k.startswith("copy:")}
        others = [v for k, v in scores.items() if k.startswith("different:")]
        assert set(copies) == {"copy:shifted", "copy:blurred", "copy:rescaled", "copy:brightened"}
        assert min(copies.values()) > max(others), (view, scores)
    assert self_check_passes(result)
    assert not self_check_passes({"full": {"copy:a": 0.3, "different:b": 0.4}})
    assert set(synthetic_variants()) >= {"original", "different:0"}


def test_embeddings_are_deterministic_and_unit_length(tmp_path: Path) -> None:
    paths = []
    for name, image in synthetic_variants().items():
        paths.append(tmp_path / f"{name.replace(':', '_')}.png")
        image.save(paths[-1])
    first = embed_images(paths, TinyEmbedder(), SETTINGS)
    second = embed_images(list(reversed(paths)), TinyEmbedder(), SETTINGS)
    for view in ("full", "centre"):
        assert first[view].shape == (len(paths), 64)
        assert np.allclose(np.linalg.norm(first[view], axis=1), 1.0, atol=1e-5)
        assert np.array_equal(first[view], second[view][::-1])


def test_knn_and_the_exact_pairs_above_a_threshold() -> None:
    rng = np.random.default_rng(3)
    emb = l2_normalise(rng.normal(size=(40, 8)))
    sims = emb @ emb.T
    pairs = knn_pairs(emb, 2, block=7)
    for i in range(40):
        row = sims[i].copy()
        row[i] = -np.inf
        for j in np.argsort(-row, kind="stable")[:2]:
            assert (min(i, j), max(i, j)) in pairs
    assert all(i < j for i, j in pairs)
    a, b, c = pairs_at_or_above(emb, 0.5, block=9)
    expected = {(i, j) for i in range(40) for j in range(i + 1, 40) if sims[i, j] >= 0.5}
    assert set(zip(a.tolist(), b.tolist(), strict=True)) == expected
    assert np.allclose(c, sims[a, b], atol=1e-6)


# ------------------------------------------------------------------ candidates and rules


def _table(tmp_path: Path) -> tuple[cand.CandidateTable, cand.CropPairs, dict, list[str]]:
    from medcompress.data.prepare import _features, crop_settings

    data = _dataset(tmp_path, chain=1, lookalike=True)
    _, images, fingerprints, signatures = _features(data, CONFIG, crop_settings(CONFIG))
    emb = embed_images([str(data / p) for p in images], TinyEmbedder(), SETTINGS)
    table, crop_pairs = cand.build_candidates(
        images, fingerprints, signatures, emb, CropSettings(), knn_k=3, dhash_bits=12,
        crop_k=2, crop_floor=0.65, extra=[(images[0], images[-1])],
    )  # fmt: skip
    return table, crop_pairs, emb, images


def test_the_candidates_carry_every_source_and_all_four_scores(tmp_path: Path) -> None:
    table, crop_pairs, emb, images = _table(tmp_path)
    assert len(table) > 0 and np.all(table.a < table.b)
    pairs = list(zip(table.a.tolist(), table.b.tolist(), strict=True))
    assert pairs == sorted(set(pairs))
    for source in ("knn_full", "knn_centre", "dhash", "crop_knn", "run3_band"):
        assert table.has(source).any(), source
    k = int(np.flatnonzero(table.has("run3_band"))[0])
    assert (table.a[k], table.b[k]) == (0, len(images) - 1)
    assert np.allclose(table.cos_full, np.sum(emb["full"][table.a] * emb["full"][table.b], 1))
    assert np.all(table.cos_max >= table.cos_centre)
    assert np.all(crop_pairs.value >= 0.65) and np.all(crop_pairs.a < crop_pairs.b)
    saved = cand.CandidateTable.load(table.save(tmp_path / "c.npz"))
    assert saved.images == table.images and np.array_equal(saved.crop, table.crop)


def test_the_rule_family_is_fixed_and_a_rule_reads_back_from_its_numbers() -> None:
    family = cand.rule_family()
    assert len(family) == 3 * 50 * 3  # 3 views x 0.50..0.99 x (alone, or crop 0.65, or 0.80)
    assert family[0] == cand.JoinRule("full", 0.50)
    rule = cand.JoinRule("max", 0.88, 0.65)
    assert rule.label == "max:0.88:or_crop:0.65" and rule.complexity == 1
    assert cand.JoinRule.from_code(**rule.code()) == rule
    assert cand.JoinRule.from_code(2.0, 0.91, -1.0) == cand.JoinRule("centre", 0.91)


def test_a_rule_over_all_pairs_and_its_groups(tmp_path: Path) -> None:
    table, crop_pairs, emb, images = _table(tmp_path)
    rule = cand.JoinRule("centre", 0.95, 0.80)
    a, b = cand.joined_pairs_all(rule, emb, crop_pairs)
    sims = emb["centre"] @ emb["centre"].T
    by_cos = {(i, j) for i in range(len(images)) for j in range(i + 1, len(images))
              if sims[i, j] >= 0.95}  # fmt: skip
    by_crop = {(i, j) for i, j, v in zip(crop_pairs.a, crop_pairs.b, crop_pairs.value,
                                         strict=True) if v >= 0.80}  # fmt: skip
    assert set(zip(a.tolist(), b.tolist(), strict=True)) == by_cos | by_crop
    groups = cand.component_groups(images, a, b)
    expected = union_groups(images, ((images[i], images[j]) for i, j in by_cos | by_crop))
    assert groups == expected and cand.largest_group(groups) >= 2
    joined = rule.joins(table)  # the same rule on the candidate pairs
    for k in np.flatnonzero(joined):
        assert groups[images[table.a[k]]] == groups[images[table.b[k]]]
    with pytest.raises(ValueError, match="crop level"):
        cand.joined_pairs_all(cand.JoinRule("full", 0.9, 0.5), emb, crop_pairs)


# ------------------------------------------------------------------ labelling


def test_the_labelling_sample_is_stratified_seeded_and_weighted(tmp_path: Path) -> None:
    table, *_ = _table(tmp_path)
    strata = lab.assign_strata(table)
    assert set(strata.tolist()) <= {name for name, _ in lab.STRATA}
    pairs = lab.labelling_sample(table, seed=11)
    assert [p.pair_id for p in pairs] == list(range(1, len(pairs) + 1))
    assert lab.labelling_sample(table, seed=11) == pairs
    sizes = lab.stratum_sizes(table)
    quotas = dict(lab.STRATA)
    for name, size in sizes.items():
        drawn = [p for p in pairs if p.stratum == name]
        quota = quotas[name]
        assert len(drawn) == (size if quota is None else min(quota, size)), name
        assert all(p.weight == pytest.approx(size / len(drawn)) for p in drawn)
    assert all(p.stratum == "run3_band" for p in pairs if (p.image_a, p.image_b) ==
               (table.images[0], table.images[-1]))  # fmt: skip
    assert sum(q for _, q in lab.STRATA if q) == 242  # + all run-3 pairs: about 300


def test_the_sheets_show_no_score_and_the_files_round_trip(tmp_path: Path) -> None:
    table, *_ = _table(tmp_path)
    data = tmp_path / "data"
    pairs = lab.labelling_sample(table)
    files = lab.write_labelling_files(pairs, data, tmp_path / "out")
    sheets = sorted(files["sheets"].glob("*.png"))
    assert len(sheets) == -(-len(pairs) // lab.PAIRS_PER_SHEET)
    with files["template"].open() as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["pair_id", "rater_1", "rater_2"] and len(rows) == len(pairs) + 1
    assert all(r[1:] == ["", ""] for r in rows[1:])
    assert ".jpg" not in files["template"].read_text()  # no image name in the template
    back = lab.read_pairs(files["pairs"])
    assert [(p.pair_id, p.image_a, p.stratum) for p in back] == [
        (p.pair_id, p.image_a, p.stratum) for p in pairs
    ]
    with pytest.raises(ValueError, match="outside the repo"):
        lab.write_labelling_files(pairs, data, REPO / "tmp_labels")


def _labels(path: Path, ids: list[int], first: list[str], second: list[str]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "rater_1", "rater_2"])
        for row in zip(ids, first, second, strict=True):
            writer.writerow(row)
    return path


def test_the_label_files_are_read_and_checked(tmp_path: Path) -> None:
    ids = [1, 2, 3]
    both = _labels(tmp_path / "both.csv", ids, ["same_scene", "different", "unsure"],
                   ["Same scene", "different", "same_scene"])  # fmt: skip
    first, second = lab.read_labels(ids, both)
    assert first == {1: "same_scene", 2: "different", 3: "unsure"}
    assert second[1] == "same_scene" and second[3] == "same_scene"
    one = _labels(tmp_path / "r1.csv", ids, ["same_scene"] * 3, [""] * 3)
    two = _labels(tmp_path / "r2.csv", ids, [""] * 3, ["different"] * 3)
    first, second = lab.read_labels(ids, one, two)
    assert set(first.values()) == {"same_scene"} and set(second.values()) == {"different"}
    with pytest.raises(lab.LabelError, match="rater_2 has no label for 3"):
        lab.read_labels(ids, one)
    bad = _labels(tmp_path / "bad.csv", ids, ["same", "different", "unsure"], ["x"] * 3)
    with pytest.raises(lab.LabelError, match="unknown label"):
        lab.read_labels(ids, bad)


def test_kappa_and_wilson() -> None:
    a = ["same_scene", "different", "different", "unsure"]
    assert lab.cohen_kappa(a, a, lab.LABELS) == 1.0
    first = ["same_scene"] * 5 + ["different"] * 5
    second = ["same_scene"] * 4 + ["different"] * 4 + ["same_scene", "different"]
    # observed 0.8, chance 0.5: kappa 0.6
    assert lab.cohen_kappa(first, second, lab.LABELS) == pytest.approx(0.6)
    p, low, high = lab.wilson(5, 10)
    assert (
        p == 0.5
        and low == pytest.approx(0.2366, abs=1e-4)
        and high == pytest.approx(0.7634, abs=1e-4)
    )
    assert np.isnan(lab.wilson(0, 0)[0])


def _pair(pid: int, cos: float, crop: float = 0.2, stratum: str = "s", weight: float = 1.0):  # type: ignore[no-untyped-def]
    return lab.LabelPair(pid, pid, f"a{pid}", f"b{pid}", stratum, weight, 8, crop, cos, cos)


def test_the_rule_choice_fixed_in_advance() -> None:
    cosines = [0.99, 0.95, 0.93, 0.90, 0.85, 0.80, 0.70, 0.60]
    truth = np.array([True, True, True, False, True, False, False, False])
    pairs = [_pair(i + 1, c) for i, c in enumerate(cosines)]
    rules = [cand.JoinRule("full", t) for t in (0.95, 0.90, 0.85, 0.84, 0.70, 0.55)]
    scores = [lab.score_rule(r, pairs, truth) for r in rules]
    s85 = scores[2]
    assert (s85.sensitivity[0], s85.precision[0]) == (1.0, 0.8)
    assert scores[5].precision[0] == 0.5  # 4 of 8
    chosen, checks = lab.choose_rule(scores, lambda r: (True, 2))
    # 0.85 and 0.84 tie (sensitivity 1.0, precision 0.8): the lower threshold catches more
    assert chosen is not None and chosen.rule == cand.JoinRule("full", 0.84)
    chosen, checks = lab.choose_rule(scores, lambda r: (r.threshold >= 0.86, 99))
    assert chosen.rule == cand.JoinRule("full", 0.90)  # the guard turns down every lower one
    assert [c[0].threshold for c in checks] == [0.84, 0.85, 0.70, 0.55, 0.90]
    strict = lab.choose_rule(scores, lambda r: (True, 2), min_precision=0.95)[0]
    assert strict.rule == cand.JoinRule("full", 0.95)
    assert lab.choose_rule(scores, lambda r: (False, 99))[0] is None
    equal = lab.weighted_scores(rules[2], pairs, truth, resamples=50)
    assert equal["sensitivity"][0] == pytest.approx(1.0)
    assert equal["precision"][0] == pytest.approx(0.8)


# ------------------------------------------------------------------ top-up, third member, blinding


def _band_table(images: int = 40, seed: int = 0) -> cand.CandidateTable:
    rng = np.random.default_rng(seed)
    a, b = np.triu_indices(images, 1)
    return cand.CandidateTable(
        tuple(f"im{i:03d}" for i in range(images)), a, b, np.full(len(a), 20),
        np.zeros(len(a), np.float32), rng.uniform(0.8, 1.0, len(a)).astype(np.float32),
        rng.uniform(0.8, 1.0, len(a)).astype(np.float32), np.ones(len(a), np.int64),
    )  # fmt: skip


def _labelled(table: cand.CandidateTable, rows: list[int]) -> list[lab.LabelPair]:
    return [
        lab.LabelPair(pid, k, table.images[table.a[k]], table.images[table.b[k]], "s", 1.0,
                      int(table.bits[k]), 0.0, float(table.cos_full[k]),
                      float(table.cos_centre[k]))
        for pid, k in enumerate(rows, start=1)
    ]  # fmt: skip


def test_the_topup_sampler(tmp_path: Path) -> None:
    table = _band_table()
    rule = cand.JoinRule("full", 0.95)
    inside = np.flatnonzero(lab.in_band(table.cos_full, 0.95))
    outside = np.flatnonzero(~lab.in_band(table.cos_full, 0.95))
    labelled = _labelled(table, [*inside[:29].tolist(), *outside[:269].tolist()])  # 298 pairs
    assert lab.labelled_in_band(labelled, rule) == 29
    top = lab.topup_sample(table, labelled, rule)
    assert top.needed and top.in_band_labelled == 29
    assert top.in_band_unlabelled == len(inside) - 29 >= 100
    assert len(top.pairs) == lab.TOPUP_PAIRS == 100
    assert [p.pair_id for p in top.pairs] == list(range(299, 399))  # numbered after the first
    assert all(abs(p.cos_full - 0.95) <= 0.03 + 1e-6 for p in top.pairs)  # all in the band
    taken = {(p.image_a, p.image_b) for p in labelled}
    drawn = [(p.image_a, p.image_b) for p in top.pairs]
    assert not taken & set(drawn) and len(set(drawn)) == 100  # new pairs, each once
    assert all(p.stratum == "topup_band" for p in top.pairs)
    assert all(p.weight == pytest.approx(top.in_band_unlabelled / 100) for p in top.pairs)
    assert lab.topup_sample(table, labelled, rule) == top  # seed 20261013: reproducible
    assert lab.topup_sample(table, labelled, rule, seed=1).pairs != top.pairs
    enough = _labelled(table, [*inside[:30].tolist(), *outside[:268].tolist()])
    assert not lab.topup_sample(table, enough, rule).needed  # 30 in the band: no top-up
    assert lab.topup_sample(table, enough, rule).pairs == ()
    every = lab.topup_sample(table, labelled, rule, size=10_000)
    assert len(every.pairs) == every.in_band_unlabelled  # fewer than asked: all of them
    centre = lab.topup_sample(table, labelled, cand.JoinRule("max", 0.95, 0.65))
    assert all(abs(max(p.cos_full, p.cos_centre) - 0.95) <= 0.03 + 1e-6 for p in centre.pairs)
    edge = np.array([0.92, 0.98, 0.9199, 0.9801], dtype=np.float32)
    assert lab.in_band(edge, 0.95).tolist() == [True, True, False, False]  # edges inside
    from medcompress.data.embeddings import count_pairs_in_band

    emb = {"full": np.eye(3, dtype=np.float32), "centre": np.eye(3, dtype=np.float32)}
    emb["full"][1] = emb["full"][0]
    assert count_pairs_in_band(emb, "full", 0.9, 1.1) == 1
    assert count_pairs_in_band(emb, "max", -0.1, 0.1) == 2
    assert count_pairs_in_band(emb, "centre", -0.1, 0.1) == 3


def test_the_third_member_and_the_unsure_analysis(tmp_path: Path) -> None:
    ids = list(range(1, 11))
    first = {i: "different" for i in ids}
    second = dict(first)
    first[1] = second[1] = "same_scene"
    first[2], second[2] = "same_scene", "unsure"  # disagreement
    first[3] = second[3] = "unsure"
    first[4], second[4] = "different", "unsure"  # disagreement
    adj = lab.adjudicate(first, second, {}, ids)
    assert adj.kappa == pytest.approx(0.33 / 0.53) and not adj.all_pairs  # kappa 0.62
    assert adj.required == (2, 4) and adj.missing == (2, 4)
    with pytest.raises(lab.LabelError, match="not to be decided"):
        lab.adjudicate(first, second, {5: "same_scene"}, ids)
    adj = lab.adjudicate(first, second, {2: "same_scene", 4: "unsure"}, ids)
    assert adj.missing == () and (adj.final[1], adj.final[2], adj.final[4]) == (
        "same_scene", "same_scene", "unsure")  # fmt: skip
    assert np.flatnonzero(lab.truth_of(first, second, ids)).tolist() == [0]  # before
    truth = lab.truth_from(adj.final, ids)
    unsure = lab.truth_from(adj.final, ids, unsure_positive=True)
    assert np.flatnonzero(truth).tolist() == [0, 1]  # primary: same_scene only
    assert np.flatnonzero(unsure).tolist() == [0, 1, 2, 3]
    # kappa below 0.40: the third member decides every pair, agreed ones too
    poor_first = {i: "same_scene" if i % 2 else "different" for i in ids}
    poor_second = {i: "same_scene" if i <= 5 else "different" for i in ids}
    poor = lab.adjudicate(poor_first, poor_second, {}, ids)
    assert poor.kappa == pytest.approx(0.2) and poor.all_pairs and poor.required == tuple(ids)
    decided = lab.adjudicate(poor_first, poor_second, {i: "different" for i in ids}, ids)
    assert not lab.truth_from(decided.final, ids).any()
    # the unsure analysis: the same precision, the sensitivity over same_scene + unsure
    cosines = [0.99, 0.97, 0.80, 0.96, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    pairs = [_pair(i, c) for i, c in zip(ids, cosines, strict=True)]
    rule = cand.JoinRule("full", 0.95)
    primary = lab.score_rule(rule, pairs, truth)
    analysis = lab.score_rule(rule, pairs, truth, sensitivity_truth=unsure)
    assert primary.sensitivity[0] == 1.0 and primary.precision[0] == pytest.approx(2 / 3)
    assert analysis.precision == primary.precision and analysis.sensitivity[0] == 0.75
    path = lab.write_adjudication_template([4, 2], tmp_path / "private")
    assert path.read_text().splitlines() == ["pair_id,rater_3", "2,", "4,"]
    with pytest.raises(ValueError, match="private"):
        lab.write_adjudication_template([2], REPO / "tmp_adjudication")
    three = _third(tmp_path / "r3.csv", {2: "same_scene", 4: "Unsure"})
    assert lab.read_third(three) == {2: "same_scene", 4: "unsure"}


def test_the_label_commitments(tmp_path: Path) -> None:
    from medcompress.utils.bundles import file_sha256

    ids = [1, 2, 3]
    one = _labels(tmp_path / "r1.csv", ids, ["same_scene", "different", "unsure"], [""] * 3)
    both = _labels(tmp_path / "both.csv", ids, ["different"] * 3, ["different"] * 3)
    commitments = tmp_path / "commitments.csv"
    row = lab.add_commitment(commitments, one, role="rater_1", round_="first")
    assert row["labels"] == "3" and row["sha256"] == file_sha256(one) and row["file"] == "r1.csv"
    with pytest.raises(lab.LabelError, match="already committed"):
        lab.add_commitment(commitments, one, role="rater_1", round_="first")
    with pytest.raises(lab.LabelError, match="rater_2 labels too"):
        lab.add_commitment(commitments, both, role="rater_1", round_="topup")
    with pytest.raises(lab.LabelError, match="needs the columns"):
        lab.add_commitment(commitments, one, role="rater_3", round_="first")
    assert lab.check_commitments([one], commitments)[0]["role"] == "rater_1"
    one.write_text(one.read_text().replace("unsure", "different"))  # changed after the commit
    with pytest.raises(lab.LabelError, match="not in"):
        lab.check_commitments([one], commitments)
    assert commitments.read_text().splitlines()[0] == ",".join(lab.COMMITMENT_FIELDS)
    repo_file = REPO / lab.COMMITMENTS_FILE  # committed with the header only, before any label
    assert repo_file.read_text().splitlines()[0] == ",".join(lab.COMMITMENT_FIELDS)


def test_the_topup_step_and_the_rule_chosen_again(tmp_path: Path) -> None:
    from medcompress.config import ConfigError
    from medcompress.data.embed_step import embed_step, score_labels_step, topup_step
    from medcompress.data.prepare import labelled_rule
    from medcompress.utils.io import append_results

    def quiet(_: str) -> None:
        return None

    data = _dataset(tmp_path, overlay=False, lookalike=True)
    out = tmp_path / "embed"
    wide = {"band": 0.5, "min_in_band": 1000, "pairs": 5, "seed": 1}  # always needed here
    config = {**CONFIG, "near_duplicates": {**CONFIG["near_duplicates"],
                                            "labelling": {"topup": wide}}}  # fmt: skip
    embed_step(data, config, out, TinyEmbedder(), split_seed=3, close_bits=4, mask_threshold=4,
               log=quiet)  # fmt: skip
    path = out / "label_pairs.csv"  # keep the positives and every other negative
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    pairs = {p.pair_id: p for p in lab.read_pairs(path)}
    kept = [r for r in rows if _same(pairs[int(r["pair_id"])]) == "same_scene" or
            int(r["pair_id"]) % 2]  # fmt: skip
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    labelled = lab.read_pairs(path)
    labels = _fake_labels(out, tmp_path / "labels.csv")
    first = score_labels_step(out, [labels], config, log=quiet)
    assert first.chosen is not None and first.topup_required
    results = tmp_path / "results"
    append_results(first.labels.rows, results / "data" / "near_duplicate_labels.csv")
    with pytest.raises(ConfigError, match="top-up"):
        labelled_rule(results)  # run 4 waits for the top-up
    assert labelled_rule(results, for_topup=True) == first.chosen
    result, top = topup_step(out, data, tmp_path / "topup", results, config, log=quiet)
    start = max(p.pair_id for p in labelled) + 1
    assert top.needed and [p.pair_id for p in top.pairs] == list(range(start, start + 5))
    assert not {(p.image_a, p.image_b) for p in labelled} & {(p.image_a, p.image_b)
                                                             for p in top.pairs}  # fmt: skip
    assert (tmp_path / "topup" / "labels_template.csv").is_file()
    assert sorted((tmp_path / "topup" / "label_sheets").glob("*.png"))
    counts = {r["metric"]: r["value"] for r in result.rows}
    assert counts["topup:drawn"] == 5 and counts["topup:needed"] == 1
    assert counts["topup:in_band_pairs_all_images"] >= counts["topup:in_band_candidates_unlabelled"]
    assert counts["topup:first_rule:threshold"] == first.chosen.threshold
    extra = _fake_labels(tmp_path / "topup", tmp_path / "topup_labels.csv")
    again = score_labels_step(out, [labels, extra], config, topup_dir=tmp_path / "topup",
                              log=quiet)  # fmt: skip
    assert again.chosen is not None and not again.topup_required  # chosen again, once
    values = {r["metric"]: r["value"] for r in again.labels.rows}
    assert values["labels:pairs:topup"] == 5 and values["labels:pairs"] == len(labelled) + 5
    target = results / "data" / "near_duplicate_labels.csv"
    target.unlink()
    append_results(again.labels.rows, target)
    assert labelled_rule(results) == again.chosen
    with pytest.raises(lab.LabelError, match="only once"):
        topup_step(out, data, tmp_path / "topup2", results, config, log=quiet)


# ------------------------------------------------------------------ the steps, run 4, cards


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "prepare_data.py"), *args],
        capture_output=True, text=True, cwd=REPO, check=False, timeout=600,
    )  # fmt: skip


def _same(p: lab.LabelPair) -> str:
    a, b = Path(p.image_a).stem, Path(p.image_b).stem
    return "same_scene" if a[:3] == b[:3] and a.startswith("s") else "different"


def _fake_labels(embed_dir: Path, path: Path) -> Path:
    pairs = lab.read_pairs(embed_dir / "label_pairs.csv")
    return _labels(path, [p.pair_id for p in pairs], [_same(p) for p in pairs],
                   [_same(p) for p in pairs])  # fmt: skip


def _pin(embed_dir: Path) -> None:
    """Record the tiny stand-in's output as made with the model pinned in the real config."""
    from medcompress.config import config_root, load_config
    from medcompress.utils.bundles import EMBED_KIND, write_bundle_manifest

    dataset = load_config(config_root() / "data" / "kvasir_vqa_x1.yaml")["dataset"]
    pinned = dataset["near_duplicates"]["embedding"]
    path = embed_dir / "embeddings.json"
    info = json.loads(path.read_text())
    info.update({k: pinned[k] for k in ("model", "revision", "licence")})
    path.write_text(json.dumps(info))
    write_bundle_manifest(embed_dir, kind=EMBED_KIND, meta={})


def _third(path: Path, labels: dict[int, str]) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "rater_3"])
        for pid, label in sorted(labels.items()):
            writer.writerow([pid, label])
    return path


def test_embed_label_and_run_4_end_to_end(tmp_path: Path) -> None:
    from medcompress.data.prepare import prepare, write_private_outputs

    data = _dataset(tmp_path, overlay=False, lookalike=True)
    results, out = tmp_path / "results", tmp_path / "embed"
    common = ("--split-seed", "3", "--close-bits", "4", "--mask-threshold", "4")
    done = _run("embed", "--data-root", str(data), "--out-dir", str(out), "--results-root",
                str(results), *common, "--embedding-model", "tiny")  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    assert "run 3 reproduced" in done.stdout and "pairs to label" in done.stdout
    assert "RECORD: model tiny" in done.stdout
    assert (out / "bundle_manifest.json").is_file() and (out / "labels_template.csv").is_file()
    rows = (results / "data" / "near_duplicate_embed.csv").read_text()
    assert "embed:candidate_pairs:knn_full" in rows and ".jpg" not in rows
    labels = _fake_labels(out, tmp_path / "labels.csv")
    refused = _run("score-labels", "--embed-dir", str(out), "--labels", str(labels),
                   "--results-root", str(results))  # fmt: skip
    assert refused.returncode == 5 and "pinned" in refused.stderr  # not the pinned model
    _pin(out)

    # two raters, each committing their own file; one disagreement for the third member
    pairs = lab.read_pairs(out / "label_pairs.csv")
    ids, truth = [p.pair_id for p in pairs], [_same(p) for p in pairs]
    disputed = ids[0]
    second = ["unsure" if i == disputed else t for i, t in zip(ids, truth, strict=True)]
    one = _labels(tmp_path / "labels_rater_1.csv", ids, truth, [""] * len(ids))
    two = _labels(tmp_path / "labels_rater_2.csv", ids, [""] * len(ids), second)
    commitments = tmp_path / "commitments.csv"
    for path, role in ((one, "rater_1"), (two, "rater_2")):
        done = _run("commit-labels", "--labels", str(path), "--role", role, "--round", "first",
                    "--commitments", str(commitments))  # fmt: skip
        assert done.returncode == 0 and "SHA-256" in done.stdout, done.stderr
    base = ("score-labels", "--embed-dir", str(out), "--results-root", str(results),
            "--commitments", str(commitments))  # fmt: skip
    no_dir = _run(*base, "--labels", str(one), str(two))
    assert no_dir.returncode == 5 and "--private-dir" in no_dir.stderr
    private = tmp_path / "adjudication"
    pending = _run(*base, "--private-dir", str(private), "--labels", str(one), str(two))
    assert pending.returncode == 4, pending.stdout + pending.stderr
    assert "ADJUDICATION NEEDED: 1 pairs" in pending.stdout
    assert not (results / "data" / "near_duplicate_labels.csv").exists()  # nothing written yet
    listed = (private / "adjudication_template.csv").read_text().splitlines()
    assert listed == ["pair_id,rater_3", f"{disputed},"]
    three = _third(tmp_path / "labels_rater_3.csv", {disputed: truth[0]})
    unknown = _run(*base, "--private-dir", str(private), "--labels", str(one), str(two),
                   str(three))  # fmt: skip
    assert unknown.returncode == 5 and "not in" in unknown.stderr  # not committed yet
    done = _run("commit-labels", "--labels", str(three), "--role", "rater_3", "--round", "first",
                "--commitments", str(commitments))  # fmt: skip
    assert done.returncode == 0, done.stderr
    scored = _run(*base, "--private-dir", str(private), "--labels", str(one), str(two),
                  str(three))  # fmt: skip
    assert scored.returncode == 0, scored.stdout + scored.stderr
    assert "matches the commitment of rater_1" in scored.stdout and "chosen rule" in scored.stdout
    assert "unsure analysis" in scored.stdout
    chosen = (results / "data" / "near_duplicate_labels.csv").read_text()
    assert "labels:chosen:view_code" in chosen and "labels:kappa" in chosen
    from medcompress.data.prepare import labels_values

    values = labels_values(results)
    assert values["labels:adjudicated"] == 1 and values["labels:kappa"] < 1.0
    assert values["labels:topup_required"] == 0 and values["labels:pairs:topup"] == 0
    assert "labels:unsure_analysis:chosen:sensitivity" in values
    assert values["labels:positives"] == truth.count("same_scene")
    topped = _run("topup-labels", "--data-root", str(data), "--embed-dir", str(out), "--out-dir",
                  str(tmp_path / "topup"), "--results-root", str(results))  # fmt: skip
    assert topped.returncode == 0, topped.stdout + topped.stderr
    assert "no candidate pair is left in the band" in topped.stdout  # every candidate labelled
    assert "topup:drawn" in (results / "data" / "near_duplicate_topup.csv").read_text()
    assert (
        "rule:full:0.50:sensitivity" in (results / "data" / "near_duplicate_rules.csv").read_text()
    )

    from medcompress.evaluation.aggregate import validate_results_tree

    assert validate_results_tree(results) == []
    config = {**CONFIG, "near_duplicates": {**CONFIG["near_duplicates"], "rule": "labelled"}}
    prepared = prepare(
        data, config, split_seed=3, close_bits=4, mask_threshold=4,
        results_root=results, embed_dir=out,
    )  # fmt: skip
    metrics = {r["metric"]: r["value"] for r in prepared.rows}
    assert metrics["val_test_confirmed_duplicates_of_train"] == 0
    assert metrics["near_duplicates:rule:view_code"] in (1.0, 2.0, 3.0)
    assert "strict_test:images" in metrics and metrics["strict_test:margin"] == 0.05
    assert len(prepared.top_cosine_cross_split) > 0
    assert all(t.first_split != t.second_split for t in prepared.top_cosine_cross_split)
    split_of = {s.image_path: n for n, rows_ in prepared.splits.items() for s in rows_}
    image = "kvasir_vqa_x1/images/{}.jpg".format
    assert split_of[image("s00")] == split_of[image("s00_shift")]
    test_ids = {Path(p).stem for p, n in split_of.items() if n == "test"}
    assert set(prepared.strict_test_images) <= test_ids
    again = prepare(
        data, config, split_seed=3, close_bits=4, mask_threshold=4,
        results_root=results, embedder=TinyEmbedder(),
    )  # fmt: skip
    assert again.split_sha256 == prepared.split_sha256  # attached or computed: the same split
    write_private_outputs(prepared, tmp_path / "private", data)
    strict_file = tmp_path / "private" / "strict_test.json"
    if prepared.strict_test_images:
        assert json.loads(strict_file.read_text()) == list(prepared.strict_test_images)
    cli = _run("prepare", "--data-root", str(data), "--private-dir", str(tmp_path / "p2"),
               "--results-root", str(results), *common, "--embed-dir", str(out),
               "--sheet-dir", str(tmp_path / "sheets"))  # fmt: skip
    assert cli.returncode == 0, cli.stdout + cli.stderr  # the real config: rule labelled
    assert "strict test subset" in cli.stdout
    assert (tmp_path / "sheets" / "rc_d1_final_cross_split_pairs.png").is_file()


def test_run_4_waits_for_the_labels_and_the_guard_stops_chaining(tmp_path: Path) -> None:
    from medcompress.config import ConfigError
    from medcompress.data.dedup import ChainingError
    from medcompress.data.prepare import join_rule, prepare

    labelled = {"near_duplicates": {"rule": "labelled"}}
    with pytest.raises(ConfigError, match="labelled pairs"):
        join_rule(labelled, results_root=tmp_path)
    assert join_rule(labelled, confirm_threshold=0.65) == cand.JoinRule("crop", 0.65)
    data = _dataset(tmp_path, chain=2)  # scene 0 in 5 of 17 images
    tight = {**CONFIG, "near_duplicates": {**CONFIG["near_duplicates"], "max_group_fraction": 0.1}}
    with pytest.raises(ChainingError, match="STOP"):
        prepare(data, tight, split_seed=3, close_bits=4, mask_threshold=4,
                rule=cand.JoinRule("full", 0.6), embedder=TinyEmbedder())  # fmt: skip


def _two_raters(embed_dir: Path, tmp_path: Path) -> tuple[list[int], list[str], list[str]]:
    """Rater 1 labels every pair as the scenes are; rater 2 says unsure on pairs 1-3 (outside
    the kappa pairs) and same_scene on one different pair."""
    pairs = lab.read_pairs(embed_dir / "label_pairs.csv")
    ids, first = [p.pair_id for p in pairs], [_same(p) for p in pairs]
    second = ["unsure"] * 3 + first[3:]
    k = next(j for j in range(3, len(ids)) if first[j] == "different")
    second[k] = "same_scene"
    _labels(tmp_path / "labels_rater_1.csv", ids, first, [""] * len(ids))
    _labels(tmp_path / "labels_rater_2.csv", ids, [""] * len(ids), second)
    return ids, first, second


def test_two_raters_without_a_third_member(tmp_path: Path) -> None:
    from medcompress.data.embed_step import embed_step, score_labels_step

    def quiet(_: str) -> None:
        return None

    data = _dataset(tmp_path, overlay=False, lookalike=True)
    out = tmp_path / "embed"
    embed_step(data, CONFIG, out, TinyEmbedder(), split_seed=3, close_bits=4, mask_threshold=4,
               log=quiet)  # fmt: skip
    ids, first, second = _two_raters(out, tmp_path)
    assert ids == list(range(1, len(ids) + 1))
    files = [tmp_path / "labels_rater_1.csv", tmp_path / "labels_rater_2.csv"]
    lines: list[str] = []
    scoring = score_labels_step(out, files, CONFIG, third_member=False,
                                kappa_pairs=(4, len(ids)), log=lines.append)  # fmt: skip
    assert scoring.adjudication is None and scoring.chosen is not None  # nobody adjudicates
    values = {r["metric"]: r["value"] for r in scoring.labels.rows}
    blind = lab.cohen_kappa(first[3:], second[3:], lab.LABELS)
    assert values["labels:kappa"] == pytest.approx(blind)  # on pairs 4-n only
    assert blind != pytest.approx(lab.cohen_kappa(first, second, lab.LABELS))
    assert (values["labels:kappa:first_pair"], values["labels:kappa:last_pair"],
            values["labels:kappa:pairs"]) == (4, len(ids), len(ids) - 3)  # fmt: skip
    assert values["labels:agreement"] == pytest.approx(
        sum(x == y for x, y in zip(first[3:], second[3:], strict=True)) / (len(ids) - 3)
    )
    assert values["labels:third_member"] == 0 and "labels:adjudicated" not in values
    both = np.array([x == y == "same_scene" for x, y in zip(first, second, strict=True)])
    either = np.array(["same_scene" in (x, y) for x, y in zip(first, second, strict=True)])
    assert values["labels:positives"] == both.sum() == values["labels:positives_both_same_scene"]
    assert values["labels:either_same_scene"] == either.sum() > both.sum()
    pairs = lab.read_pairs(out / "label_pairs.csv")
    primary = lab.score_rule(scoring.chosen, pairs, both)
    assert values["labels:chosen:sensitivity"] == primary.sensitivity[0]
    assert values["labels:chosen:precision"] == primary.precision[0]
    by_either = lab.score_rule(scoring.chosen, pairs, either)  # reported, not used
    assert values["labels:either_rater:chosen:sensitivity"] == pytest.approx(
        by_either.sensitivity[0]
    )
    assert values["labels:either_rater:chosen:precision"] == pytest.approx(by_either.precision[0])
    assert "labels:either_rater:same_rule" in values
    assert "labels:unsure_analysis:chosen:sensitivity" in values  # the unsure analysis stays
    assert values["labels:topup_required"] == int(scoring.topup_required)
    assert any("no third member" in line for line in lines)
    assert any(line.startswith("top-up: ") for line in lines)
    with pytest.raises(lab.LabelError, match="--kappa-pairs"):
        score_labels_step(out, files, CONFIG, third_member=False, log=quiet)
    with pytest.raises(lab.LabelError, match="numbered 400 to 500"):
        score_labels_step(out, files, CONFIG, third_member=False, kappa_pairs=(400, 500),
                          log=quiet)  # fmt: skip
    three = _third(tmp_path / "labels_rater_3.csv", {1: "same_scene"})
    with pytest.raises(lab.LabelError, match="no third member"):
        score_labels_step(out, [*files, three], CONFIG, third_member=False, kappa_pairs=(4, 9),
                          log=quiet)  # fmt: skip
    with pytest.raises(lab.LabelError, match="--private-dir"):  # without the flag: amendment 3
        score_labels_step(out, files, CONFIG, log=quiet)


def test_score_labels_without_a_third_member_on_the_command_line(tmp_path: Path) -> None:
    data = _dataset(tmp_path, overlay=False, lookalike=True)
    results, out = tmp_path / "results", tmp_path / "embed"
    done = _run("embed", "--data-root", str(data), "--out-dir", str(out), "--results-root",
                str(results), "--split-seed", "3", "--close-bits", "4", "--mask-threshold", "4",
                "--embedding-model", "tiny")  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    _pin(out)
    ids, _, _ = _two_raters(out, tmp_path)
    one, two = tmp_path / "labels_rater_1.csv", tmp_path / "labels_rater_2.csv"
    commitments = tmp_path / "commitments.csv"
    for path, role in ((one, "rater_1"), (two, "rater_2")):
        made = _run("commit-labels", "--labels", str(path), "--role", role, "--round", "first",
                    "--commitments", str(commitments))  # fmt: skip
        assert made.returncode == 0, made.stderr
    base = ("score-labels", "--embed-dir", str(out), "--results-root", str(results),
            "--commitments", str(commitments), "--labels", str(one), str(two))  # fmt: skip
    for half in (("--no-third-member",), ("--kappa-pairs", "4-9")):
        alone = _run(*base, *half)
        assert alone.returncode == 5 and "go together" in alone.stderr
    assert _run(*base, "--no-third-member", "--kappa-pairs", "nine").returncode == 2
    scored = _run(*base, "--no-third-member", "--kappa-pairs", f"4-{len(ids)}")
    assert scored.returncode == 0, scored.stdout + scored.stderr
    assert "matches the commitment of rater_2" in scored.stdout
    assert "no third member" in scored.stdout and "chosen rule" in scored.stdout
    assert "either-rater analysis" in scored.stdout  # every candidate is labelled here:
    assert "top-up: not possible, no candidate pair is left in the band" in scored.stdout
    assert "ADJUDICATION" not in scored.stdout
    from medcompress.data.prepare import labels_values

    values = labels_values(results)
    assert values["labels:kappa:first_pair"] == 4 and values["labels:third_member"] == 0
    assert "labels:either_rater:chosen:sensitivity" in values
    assert "labels:chosen:view_code" in values and values["labels:topup_required"] == 0
    assert (results / "data" / "near_duplicate_rules.csv").is_file()
    one.write_text(one.read_text().replace("different", "unsure", 1))  # changed after its commit
    refused = _run(*base, "--no-third-member", "--kappa-pairs", f"4-{len(ids)}")
    assert refused.returncode == 5 and "labels_rater_1.csv" in refused.stderr
    assert "is not in" in refused.stderr


def test_a_rule_reads_back_from_its_label() -> None:
    for rule in (cand.JoinRule("full", 0.88, 0.80), cand.JoinRule("centre", 0.9)):
        assert cand.JoinRule.from_label(rule.label) == rule
    for bad in ("full:0.9", "crop:0.65", "full:0.88:crop:0.80", "full", "full:x"):
        with pytest.raises(ValueError):
            cand.JoinRule.from_label(bad)


def test_the_rule_sweep_is_reproducible_and_marks_the_guard(tmp_path: Path) -> None:
    from medcompress.data.embed_step import embed_step
    from medcompress.data.prepare import prepare
    from medcompress.data.rule_sweep import rule_sweep, sweep_rows, sweep_rules

    def quiet(_: str) -> None:
        return None

    data = _dataset(tmp_path, overlay=False, lookalike=True)
    out = tmp_path / "embed"
    embed_step(data, CONFIG, out, TinyEmbedder(), split_seed=3, close_bits=4, mask_threshold=4,
               log=quiet)  # fmt: skip
    rules = sweep_rules([0.60, 0.95, 0.55, 0.70], 0.80)
    assert [r.label for r in rules] == [
        f"full:{t}:or_crop:0.80" for t in ("0.95", "0.70", "0.60", "0.55")
    ]
    common = {"split_seed": 3, "close_bits": 4, "mask_threshold": 4}
    first = rule_sweep(data, CONFIG, out, rules, log=quiet, **common)
    again = rule_sweep(data, CONFIG, out, rules, log=quiet, **common)
    guard = CONFIG["near_duplicates"]["max_group_fraction"]  # 0.5 of the 16 images
    rows = sweep_rows(first, label_rule=rules[0], max_group_fraction=guard, **common)
    assert rows == sweep_rows(again, label_rule=rules[0], max_group_fraction=guard, **common)
    assert [r.within_guard for r in first] == [True, True, True, False]
    assert [r.largest_group for r in first] == [2, 2, 5, 9]
    assert first[3].prepared is None and first[3].largest_fraction == 9 / 16
    assert [r.joined_pairs for r in first] == sorted(r.joined_pairs for r in first)
    for r in first[:3]:  # each split is the one prepare builds alone (run 4b)
        alone = prepare(data, CONFIG, rule=r.rule, embed_dir=out, **common)
        assert r.prepared is not None and alone.split_sha256 == r.prepared.split_sha256
        assert alone.rows == r.prepared.rows
    values = {row["metric"]: row["value"] for row in rows}
    assert values["sweep:full:0.55:or_crop:0.80:within_guard"] == 0
    assert "sweep:full:0.55:or_crop:0.80:val_test_confirmed_duplicates_of_train" not in values
    for label in ("full:0.95:or_crop:0.80", "full:0.70:or_crop:0.80", "full:0.60:or_crop:0.80"):
        assert values[f"sweep:{label}:within_guard"] == 1
        assert values[f"sweep:{label}:val_test_confirmed_duplicates_of_train"] == 0
        assert values[f"sweep:{label}:sheet_band_pairs:cosine"] > 0
    assert values["sweep:label_rule:threshold"] == 0.95 and values["sweep:rules"] == 4


def test_the_rule_sweep_and_run_4b_on_the_command_line(tmp_path: Path) -> None:
    from medcompress.data.prepare import _row, labels_values
    from medcompress.utils.io import append_results

    data = _dataset(tmp_path, overlay=False, lookalike=True)
    results, out = tmp_path / "results", tmp_path / "embed"
    done = _run("embed", "--data-root", str(data), "--out-dir", str(out), "--results-root",
                str(results), "--split-seed", "3", "--close-bits", "4", "--mask-threshold", "4",
                "--embedding-model", "tiny")  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    _pin(out)
    either = cand.JoinRule("full", 0.95, 0.80)  # as RC-D1-label's scoring writes it
    append_results([_row(f"labels:either_rater:rule:{k}", v) for k, v in either.code().items()],
                   results / "data" / "near_duplicate_labels.csv")  # fmt: skip
    sheets = tmp_path / "sweep_sheets"
    base = ("rule-sweep", "--data-root", str(data), "--embed-dir", str(out), "--results-root",
            str(results), "--crop", "0.80", "--split-seed", "3")  # fmt: skip
    wrong = _run(*base, "--out-dir", str(sheets), "--thresholds", "0.90", "0.70")
    assert wrong.returncode == 5 and "either-rater truth" in wrong.stderr  # must start at 0.95
    inside = _run(*base, "--out-dir", str(REPO / "tmp_sweep"), "--thresholds", "0.95")
    assert inside.returncode == 5 and "outside the repository" in inside.stderr
    swept = _run(*base, "--out-dir", str(sheets), "--thresholds", "0.60", "0.95", "0.70")
    assert swept.returncode == 0, swept.stdout + swept.stderr
    assert "from the committed labels file): full:0.95:or_crop:0.80" in swept.stdout
    assert "full:0.60:or_crop:0.80: joined pairs" in swept.stdout
    assert "OVER THE GUARD" in swept.stdout
    assert sorted(p.name for p in sheets.glob("*.png")) == [
        "rc_d1_sweep_full_0.70_or_crop_0.80.png",
        "rc_d1_sweep_full_0.95_or_crop_0.80.png",
    ]
    target = results / "data" / "near_duplicate_sweep.csv"
    counts = target.read_text()
    assert ".jpg" not in counts and "sweep:full:0.60:or_crop:0.80:within_guard" in counts
    again = _run(*base, "--out-dir", str(tmp_path / "sheets2"), "--thresholds", "0.95", "0.70",
                 "0.60")  # fmt: skip
    assert again.returncode == 0 and target.read_text() == counts  # the same counts again
    from medcompress.evaluation.aggregate import validate_results_tree

    assert validate_results_tree(results) == []
    hashes = {}  # rule label -> the split_sha256 the sweep printed
    for line in swept.stdout.splitlines():
        if "split_sha256" in line:
            label = line.split(": joined", 1)[0]
            hashes[label] = line.split("split_sha256 ", 1)[1].split(";")[0]
    prep = ("prepare", "--data-root", str(data), "--private-dir", str(tmp_path / "private"),
            "--results-root", str(results), "--split-seed", "3", "--close-bits", "4",
            "--mask-threshold", "4", "--embed-dir", str(out))  # fmt: skip
    for rule, why in (("full:0.88:or_crop:0.80", "not in the sweep"),
                      ("full:0.60:or_crop:0.80", "over the chaining guard"),
                      ("full:0.7:or_crop:0.80", "write the rule as")):  # fmt: skip
        refused = _run(*prep, "--rule", rule)
        assert refused.returncode == 5 and why in refused.stderr, refused.stderr
    both = _run(*prep, "--rule", "full:0.70:or_crop:0.80", "--confirm-threshold", "0.65")
    assert both.returncode == 5 and "exclude each other" in both.stderr
    run_4b = _run(*prep, "--rule", "full:0.70:or_crop:0.80", "--sheet-dir",
                  str(tmp_path / "sheets_4b"))  # fmt: skip
    assert run_4b.returncode == 0, run_4b.stdout + run_4b.stderr
    assert "joining rule: full:0.70:or_crop:0.80 (from the sweep" in run_4b.stdout
    assert hashes["full:0.70:or_crop:0.80"] in run_4b.stdout  # the swept split, unchanged
    summary = (results / "data" / "summary.csv").read_text()
    assert "near_duplicates:rule_from_sweep,1" in summary.replace(",1.0", ",1")
    assert labels_values(results)["labels:either_rater:rule:threshold"] == 0.95


def test_the_heldout_filter_keeps_train_and_moves_no_image(tmp_path: Path) -> None:
    from medcompress.data.dedup import cross_similarity_matrix, hamming_distance
    from medcompress.data.embed_step import embed_step
    from medcompress.data.embeddings import load_embeddings
    from medcompress.data.prepare import (
        HELDOUT_FILE,
        _features,
        crop_settings,
        prepare,
        write_private_outputs,
    )
    from medcompress.data.rule_sweep import filter_sweep, filter_sweep_rows, heldout_filters

    def quiet(_: str) -> None:
        return None

    data = _dataset(tmp_path, overlay=False, lookalike=True)
    out = tmp_path / "embed"
    embed_step(data, CONFIG, out, TinyEmbedder(), split_seed=3, close_bits=4, mask_threshold=4,
               log=quiet)  # fmt: skip
    rule = cand.JoinRule("full", 0.95, 0.80)
    filters = heldout_filters([0.40, 0.99, 0.62], crop=0.70, bits=4, bits_crop=0.50)
    assert [f.label for f in filters] == ["0.99", "0.62", "0.40"]  # the fewest removed first
    common = {"split_seed": 3, "close_bits": 4, "mask_threshold": 4}
    base, results = filter_sweep(data, CONFIG, out, rule, filters, log=quiet, **common)
    base2, results2 = filter_sweep(data, CONFIG, out, rule, filters, log=quiet, **common)
    rows = filter_sweep_rows(base, results, rule=rule, **common)
    assert rows == filter_sweep_rows(base2, results2, rule=rule, **common)  # reproducible

    def images(prepared):  # type: ignore[no-untyped-def]
        return {n: {s.image_path for s in m} for n, m in prepared.splits.items()}

    before = images(base)
    _, paths, fingerprints, signatures = _features(data, CONFIG, crop_settings(CONFIG))
    emb, _ = load_embeddings(out / "embeddings.npz", paths, SETTINGS)
    index = {p: i for i, p in enumerate(paths)}
    train = sorted(before["train"])
    for r in results:
        after = images(r.prepared)
        assert [s.sample_id for s in r.prepared.splits["train"]] == [
            s.sample_id for s in base.splits["train"]]  # train untouched  # fmt: skip
        for name in ("val", "test"):
            assert after[name] <= before[name]  # nothing added, nothing moved in
            gone = before[name] - after[name]
            assert {Path(p).stem for p in gone} == set(r.prepared.heldout_removed[name])
            for path in sorted(before[name]):  # removed exactly when a condition holds
                cos = max(float(emb["full"][index[path]] @ emb["full"][index[t]]) for t in train)
                crop = cross_similarity_matrix([signatures[path]],
                                               [signatures[t] for t in train],
                                               crop_settings(CONFIG))[0]  # fmt: skip
                close = [c for t, c in zip(train, crop, strict=True)
                         if hamming_distance(fingerprints[path], fingerprints[t]) <= 4]  # fmt: skip
                expected = (np.float32(cos) >= np.float32(r.heldout.threshold)
                            or crop.max() >= np.float32(0.70)
                            or any(c >= np.float32(0.50) for c in close))  # fmt: skip
                assert (path in gone) == bool(expected), (r.heldout.label, path)
        for name, members in after.items():  # no image changes split
            assert members <= before[name]
    values = {r.heldout.label: r.values for r in results}
    assert results[0].prepared.split_sha256 == base.split_sha256  # 0.99 removes nothing
    assert values["0.62"]["heldout_filter:removed:test"] == 1
    assert values["0.62"]["heldout_filter:kept_fraction:images:test"] == pytest.approx(2 / 3)
    assert values["0.62"]["heldout_filter:test_floor_met"] == 1 and results[1].floor_met
    assert values["0.40"]["heldout_filter:kept_fraction:images:test"] == 0.0
    assert values["0.40"]["heldout_filter:test_floor_met"] == 0 and not results[2].floor_met
    assert values["0.62"]["heldout_filter:unfiltered:images:test"] == len(before["test"])
    assert values["0.62"]["val_test_confirmed_duplicates_of_train"] == 0
    for r in results:  # each filtered split is the one prepare builds alone (run 4b)
        alone = prepare(data, CONFIG, rule=rule, embed_dir=out, heldout_filter=r.heldout,
                        **common)  # fmt: skip
        assert alone.split_sha256 == r.prepared.split_sha256 and alone.rows == r.prepared.rows
        for t in alone.top_cosine_cross_split:  # no val/test pair on the sheet
            assert "train" in (t.first_split, t.second_split)
    write_private_outputs(results[1].prepared, tmp_path / "private", data)
    removed = json.loads((tmp_path / "private" / HELDOUT_FILE).read_text())
    assert removed["removed"]["test"] == list(results[1].prepared.heldout_removed["test"])
    assert removed["filter"]["threshold"] == 0.62
    sweep_values = {row["metric"]: row["value"] for row in rows}
    assert sweep_values["filter_sweep:0.40:test_floor_met"] == 0
    assert sweep_values["filter_sweep:unfiltered:images:test"] == len(before["test"])
    assert sweep_values["filter_sweep:rule:threshold"] == 0.95


def test_the_filter_sweep_and_run_4b_with_the_filter_on_the_command_line(tmp_path: Path) -> None:
    from medcompress.data.prepare import _row
    from medcompress.utils.io import append_results

    data = _dataset(tmp_path, overlay=False, lookalike=True)
    results, out = tmp_path / "results", tmp_path / "embed"
    done = _run("embed", "--data-root", str(data), "--out-dir", str(out), "--results-root",
                str(results), "--split-seed", "3", "--close-bits", "4", "--mask-threshold", "4",
                "--embedding-model", "tiny")  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    _pin(out)
    sweep = results / "data" / "near_duplicate_sweep.csv"  # as rule-sweep writes it
    append_results([_row("sweep:full:0.95:or_crop:0.80:within_guard", 1),
                    _row("sweep:full:0.60:or_crop:0.80:within_guard", 0)], sweep)  # fmt: skip
    sheets = tmp_path / "filter_sheets"
    base = ("filter-sweep", "--data-root", str(data), "--embed-dir", str(out), "--results-root",
            str(results), "--split-seed", "3", "--thresholds", "0.40", "0.99", "0.62")  # fmt: skip
    for rule, why in (("full:0.60:or_crop:0.80", "over the chaining guard"),
                      ("full:0.90:or_crop:0.80", "not in the sweep")):  # fmt: skip
        refused = _run(*base, "--rule", rule, "--out-dir", str(sheets))
        assert refused.returncode == 5 and why in refused.stderr, refused.stderr
    inside = _run(*base, "--rule", "full:0.95:or_crop:0.80", "--out-dir", str(REPO / "tmp_f"))
    assert inside.returncode == 5 and "outside the repository" in inside.stderr
    swept = _run(*base, "--rule", "full:0.95:or_crop:0.80", "--out-dir", str(sheets))
    assert swept.returncode == 0, swept.stdout + swept.stderr
    assert "held-out filter F 0.40:" in swept.stdout
    assert "is BELOW the 50% floor" in swept.stdout and "meets the 50% floor" in swept.stdout
    assert "held-out filter F 0.62: val keeps" in swept.stdout
    assert sorted(p.name for p in sheets.glob("*.png")) == [
        "rc_d1_filter_0.40.png",
        "rc_d1_filter_0.62.png",
        "rc_d1_filter_0.99.png",
    ]
    target = results / "data" / "near_duplicate_filter_sweep.csv"
    counts = target.read_text()
    assert ".jpg" not in counts and "filter_sweep:0.62:kept_fraction:images:test" in counts
    again = _run(*base, "--rule", "full:0.95:or_crop:0.80", "--out-dir", str(tmp_path / "f2"))
    assert again.returncode == 0 and target.read_text() == counts  # the same counts again
    from medcompress.evaluation.aggregate import validate_results_tree

    assert validate_results_tree(results) == []
    hashes = {}  # F -> the split_sha256 the filter sweep printed
    for line in swept.stdout.splitlines():
        if line.startswith("held-out filter F "):
            hashes[line.split()[3].rstrip(":")] = line.split("split_sha256 ", 1)[1].split(";")[0]
    prep = ("prepare", "--data-root", str(data), "--private-dir", str(tmp_path / "private"),
            "--results-root", str(results), "--split-seed", "3", "--close-bits", "4",
            "--mask-threshold", "4", "--embed-dir", str(out))  # fmt: skip
    rule = ("--rule", "full:0.95:or_crop:0.80")
    for extra, why in ((("--heldout-filter", "0.85", *rule), "not in the filter sweep"),
                       (("--heldout-filter", "0.40", *rule), "keeps less than 50%"),
                       (("--heldout-filter", "0.625", *rule), "two decimals"),
                       (("--heldout-filter", "0.62"), "needs --rule")):  # fmt: skip
        refused = _run(*prep, *extra)
        assert refused.returncode == 5 and why in refused.stderr, (extra, refused.stderr)
    run_4b = _run(*prep, *rule, "--heldout-filter", "0.62", "--sheet-dir", str(tmp_path / "s4b"))
    assert run_4b.returncode == 0, run_4b.stdout + run_4b.stderr
    assert "held-out filter: F 0.62, crop 0.70, crop 0.50 within 4 bits" in run_4b.stdout
    assert "held-out filter F 0.62: val keeps" in run_4b.stdout
    assert hashes["0.62"] in run_4b.stdout  # the filtered split the sweep showed
    summary = (results / "data" / "summary.csv").read_text()
    assert "heldout_filter:threshold,0.62" in summary
    assert "near_duplicates:rule_from_sweep" in summary
    removed = json.loads((tmp_path / "private" / "heldout_filter_removed.json").read_text())
    assert removed["removed"]["test"] and removed["filter"]["threshold"] == 0.62
    assert validate_results_tree(results) == []  # summary.csv with the filter rows too
    append_results([_row("sweep:full:0.70:or_crop:0.80:within_guard", 1)], sweep)
    other = _run(*prep, "--rule", "full:0.70:or_crop:0.80", "--heldout-filter", "0.62")
    assert other.returncode == 5 and "used the rule full:0.95:or_crop:0.80" in other.stderr


def test_the_unfiltered_test_file_adds_the_removed_test_images_back(tmp_path: Path) -> None:
    from medcompress.config import ConfigError
    from medcompress.data.embed_step import embed_step
    from medcompress.data.heldout_filter import HeldoutFilter
    from medcompress.data.prepare import prepare, write_private_outputs
    from medcompress.data.prepared import UNFILTERED_TEST_FILE, load_prepared
    from medcompress.data.splits import ids_fingerprint, split_fingerprint

    def quiet(_: str) -> None:
        return None

    data = _dataset(tmp_path, overlay=False, lookalike=True)
    out = tmp_path / "embed"
    embed_step(data, CONFIG, out, TinyEmbedder(), split_seed=3, close_bits=4, mask_threshold=4,
               log=quiet)  # fmt: skip
    common = {"split_seed": 3, "close_bits": 4, "mask_threshold": 4, "embed_dir": out,
              "rule": cand.JoinRule("full", 0.95, 0.80)}  # fmt: skip
    base = prepare(data, CONFIG, **common)
    filtered = prepare(data, CONFIG, heldout_filter=HeldoutFilter(0.62), **common)
    assert filtered.heldout_removed["test"]  # s00_shift is removed (see the filter test)
    private = tmp_path / "private"
    hashes = write_private_outputs(filtered, private, data)
    ids = json.loads((private / UNFILTERED_TEST_FILE).read_text())
    unfiltered = sorted(s.sample_id for s in base.splits["test"])
    assert ids == unfiltered  # the filtered test with the removed test images added back
    assert {s.sample_id for s in filtered.splits["test"]} < set(ids)
    assert hashes["test_unfiltered_sha256"] == ids_fingerprint({"test_unfiltered": ids})
    # the split hash is of the filtered splits; the unfiltered test is added when asked
    loaded = load_prepared(data, CONFIG, private, expected_split_sha256=filtered.split_sha256,
                           unfiltered_test=True)  # fmt: skip
    assert sorted(s.sample_id for s in loaded["test_unfiltered"]) == unfiltered
    assert all(s.split == "test_unfiltered" for s in loaded["test_unfiltered"])
    assert [s.sample_id for s in loaded["test"]] == [s.sample_id for s in filtered.splits["test"]]
    stems = {Path(s.image_path).stem for s in loaded["test_unfiltered"]}
    assert set(filtered.heldout_removed["test"]) <= stems
    assert split_fingerprint({k: v for k, v in loaded.items() if k != "test_unfiltered"}) == (
        filtered.split_sha256)  # fmt: skip
    assert "test_unfiltered" not in load_prepared(data, CONFIG, private)  # only when asked
    write_private_outputs(base, tmp_path / "private_unfiltered", data)  # no filter: no file
    assert not (tmp_path / "private_unfiltered" / UNFILTERED_TEST_FILE).exists()
    with pytest.raises(ConfigError, match="held-out filter"):
        load_prepared(data, CONFIG, tmp_path / "private_unfiltered", unfiltered_test=True)


def test_the_strict_subset_restricts_the_comparisons() -> None:
    from medcompress.evaluation.compare_stage import Arm, ArmOutcomes, ProbeTable, restrict

    ids = np.array(["im1:a", "im1:b", "im2:a", "im3:a"])
    table = ProbeTable(
        keys=ids, images=ids, findings=("polyp",), codes=np.zeros(4, int),
        truth=np.array([True, False, True, False]), small=np.zeros(4, bool),
        rare=np.zeros(4, bool), audit_keys=np.array(["im2:a", "im3:a"]),
        audit_images=np.array(["im2:a", "im3:a"]),
    )  # fmt: skip
    arm = ArmOutcomes(Arm("E2"), (0,), np.arange(4.0), np.ones(4), np.ones(4, bool),
                      np.array([0.5, -0.5]))  # fmt: skip
    small, (out,) = restrict(table, [arm], {"im2", "im3"})
    assert small.keys.tolist() == ["im2:a", "im3:a"] and out.correct.tolist() == [2.0, 3.0]
    assert small.audit_keys.tolist() == ["im2:a", "im3:a"]
    assert out.audit_effect.tolist() == [0.5, -0.5]


def test_strict_comparisons_are_read_as_their_own_split(tmp_path: Path) -> None:
    from medcompress.evaluation.aggregate import classify_path

    path = tmp_path / "comparisons_strict" / "E7_vs_best" / "B2" / "comparison.csv"
    found = classify_path(path, tmp_path)
    assert found is not None and found.kind == "comparison" and found.split == "strict_test"
    main = classify_path(tmp_path / "comparisons" / "E7_vs_best" / "B2" / "comparison.csv",
                         tmp_path)  # fmt: skip
    assert main is not None and main.split == ""


def test_the_runner_plans_of_amendment_3(tmp_path: Path) -> None:
    from medcompress.cards import CardError, build_plan, card_file

    attached = build_plan("RC-D1-embed", data_root=Path("/kaggle/input/medcompress-data"))
    assert [s.label for s in attached.steps] == ["embed"] and attached.internet
    argv = attached.steps[0].argv
    assert argv[:2] == ("scripts/prepare_data.py", "embed") and "--out-dir" in argv
    fresh = build_plan("RC-D1-embed")
    assert [s.label for s in fresh.steps] == ["export", "copy Kvasir-SEG", "embed"]
    with pytest.raises(CardError, match="--embed-dir"):
        build_plan("RC-D1-label")
    files = [Path("/kaggle/input/l/a.csv"), Path("/kaggle/input/l/b.csv")]
    scored = build_plan("RC-D1-label", embed_dir=Path("/kaggle/input/e"), labels=files)
    assert scored.steps[0].argv[-3:] == ("--labels", *(str(f) for f in files))
    assert "--commitments" in scored.steps[0].argv and "--private-dir" in scored.steps[0].argv
    assert "--topup-dir" not in scored.steps[0].argv
    again = build_plan("RC-D1-label", embed_dir=Path("/kaggle/input/e"), labels=files,
                       topup_dir=Path("/kaggle/input/t"))  # fmt: skip
    assert "--topup-dir" in again.steps[0].argv
    run4 = build_plan("RC-D1", embed_dir=Path("/kaggle/input/e"))
    prepare_step = next(s for s in run4.steps if s.label == "prepare")
    assert "--embed-dir" in prepare_step.argv
    runs = REPO / "docs" / "runs"
    assert card_file("RC-D1-embed", runs).name.startswith("RC-D1-embed_")
    assert card_file("RC-D1-label", runs).name.startswith("RC-D1-label_")
    assert card_file("RC-D1", runs).name.startswith("RC-D1_")
