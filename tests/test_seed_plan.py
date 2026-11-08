"""The seed plan of 2026-11-08: tiers, the comparator ordering, the futility rule (CPU)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from medcompress import cards as cards_mod
from medcompress import seed_plan as sp
from medcompress.cards import CardError, build_plan
from medcompress.evaluation import probe_suites as ps

REPO = Path(__file__).resolve().parents[1]


def _score(label: str, small: float, mask: float = 0.9, margin: float = 1.0) -> sp.RankScore:
    return sp.RankScore(label, small, mask, margin)


def _scores(*items: sp.RankScore) -> dict[str, sp.RankScore]:
    return {s.label: s for s in items}


def test_the_ordering_breaks_a_13_of_13_tie_by_the_mask_recall_then_the_margin() -> None:
    tied = _scores(
        _score("E5:magnitude", 1.0, 0.95, 0.5),
        _score("E5:activation", 1.0, 0.97, 0.1),  # (ii) decides against the others
        _score("E5:taylor", 1.0, 0.95, 2.0),  # (iii) decides between taylor and E6
        _score("E6", 1.0, 0.95, 1.5),
    )
    best = sp.choose_comparator(tied)
    assert best.label == "E5:activation"
    order = [s.label for s in sp.rank(tied.values())]
    assert order == ["E5:activation", "E5:taylor", "E6", "E5:magnitude"]
    assert sp.decided_by(tied["E5:taylor"], tied["E6"]) == "(iii)"
    assert sp.decided_by(tied["E5:activation"], tied["E5:taylor"]) == "(ii)"
    # (i) first: a higher small-polyp recall wins whatever (ii) and (iii) say
    higher = {**tied, "E6": _score("E6", 1.0 + 1e-9, 0.0, -5.0)}
    assert sp.choose_comparator(higher).label == "E6"


def test_a_full_tie_goes_to_the_name_and_a_missing_criterion_is_refused() -> None:
    same = _scores(*(_score(c, 1.0, 0.9, 1.0) for c in sp.COMPARATOR_CANDIDATES))
    assert sp.choose_comparator(same).label == "E5:activation"  # (iv) the name
    assert sp.decided_by(same["E5:activation"], same["E5:magnitude"]) == "(iv) name"
    gap = {**same, "E6": _score("E6", 1.0, 0.9, math.nan)}
    with pytest.raises(sp.SeedPlanError, match="op_small_margin_mean missing"):
        sp.choose_comparator(gap)
    with pytest.raises(sp.SeedPlanError, match="no validation rows"):
        sp.choose_comparator({k: v for k, v in same.items() if k != "E6"})


def test_the_stopping_rule_reads_the_same_ordering() -> None:
    comparator = _score("E6", 1.0, 0.95, 1.5)
    assert sp.futility_stop(_score("E7", 0.92), comparator)  # below at (i): stop
    assert not sp.futility_stop(_score("E7", 1.0, 0.97, 0.0), comparator)  # above at (ii)
    assert sp.futility_stop(_score("E7", 1.0, 0.95, 1.0), comparator)  # below at (iii)
    assert sp.futility_stop(_score("E7", 1.0, 0.95, 1.5), comparator)  # full tie: the name
    for e7 in (_score("E7", 0.92), _score("E7", 1.0, 0.97, 0.0)):  # the same key both ways
        stop = sp.futility_stop(e7, comparator)
        assert stop == (sp.rank([e7, comparator])[0].label == "E6")


def test_two_seeds_are_refused_one_and_three_are_allowed() -> None:
    sp.check_seed_count("E6@B2", (0,))
    sp.check_seed_count("E7@B2", (0, 1, 2))
    with pytest.raises(sp.SeedPlanError, match="2 seeds"):
        sp.check_seed_count("E7@B2", (0, 2))


def test_the_scores_come_from_seed0_validation_rows() -> None:
    def row(exp: str, metric: str, value: float, seed: int = 0, split: str = "val") -> dict:
        return {"experiment": exp, "budget": "B2", "seed": seed, "metric": metric,
                "value": value, "split": split}  # fmt: skip

    rows = pd.DataFrame(
        [
            row("E6", sp.SMALL_RECALL, 1.0),
            row("E6", sp.SMALL_MARGIN, 2.5),
            # no op_mask_positive_recall row: from the strata, weighted by their counts
            row("E6", "op_recall_stratum:0", 1.0),
            row("E6", "op_recall_stratum:1", 0.5),
            row("E6", "n_stratum:0", 13),
            row("E6", "n_stratum:1", 20),
            row("E6", sp.SMALL_RECALL, 0.0, seed=1),  # another seed: not read
            row("E6", sp.SMALL_RECALL, 0.0, split="test"),  # the test split: never read
        ]
    )
    score = sp.scores_from_rows(rows, ["E6", "E5:taylor"])
    assert set(score) == {"E6"}
    assert score["E6"].small_recall == 1.0 and score["E6"].small_margin == 2.5
    assert score["E6"].mask_recall == pytest.approx((13 + 10) / 33)


def test_the_evaluation_writes_the_tie_break_rows_from_the_margins_and_thresholds() -> None:
    truth = np.array([True, True, True, False, False, True])
    out = ps.ProbeOutcomes(
        sample_ids=("a", "b", "c", "d", "e", "f"),
        findings=("polyp",) * 6,
        truth=truth,
        yes=np.zeros(6, dtype=bool),
        margin=np.array([2.0, 0.5, -3.0, -1.0, -2.0, 4.0]),
        area=np.array([0.004, 0.005, 0.02, np.nan, np.nan, 0.05]),
        has_mask=truth,
    )
    thresholds = {"polyp": -1.0}
    rows = {m.name: m.value for m in ps.operating_point_metrics(
        out, ["polyp"], thresholds, size_thresholds=[0.006, 0.013, 0.018],
        mask_findings=["polyp"], n_resamples=5,
    )}  # fmt: skip
    # small polyps (area < 0.006): a and b, margins 2.0 and 0.5 over the threshold -1.0
    assert rows[ps.OP_SMALL_MARGIN] == pytest.approx(((2.0 + 1.0) + (0.5 + 1.0)) / 2)
    # every mask-positive polyp (a, b, c, f): "yes" when the margin is above -1.0: a, b, f
    assert rows[ps.OP_MASK_RECALL] == pytest.approx(3 / 4)
    assert rows["op_small_finding_recall"] == 1.0


def test_seeds_1_and_2_wait_for_the_choice_and_stop_on_futility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cards_mod, "seed_plan_choice", lambda: None)
    assert build_plan("RC-E7", 0).steps  # seed 0 never waits
    for card in ("RC-E2", "RC-E5", "RC-E6", "RC-E7", "RC-A1"):  # A1: the secondary comparison
        with pytest.raises(CardError, match="wait for seed 0's comparator choice"):
            build_plan(card, 1)
    stop = {"comparator": "E6", "futility_stop": True}
    monkeypatch.setattr(cards_mod, "seed_plan_choice", lambda: stop)
    for card in ("RC-E2", "RC-E6", "RC-E7", "RC-A1"):
        with pytest.raises(CardError, match="futility rule stopped"):
            build_plan(card, 2)
    go = {"comparator": "E6", "futility_stop": False}
    monkeypatch.setattr(cards_mod, "seed_plan_choice", lambda: go)
    runs = [s.label for s in build_plan("RC-E6", 1).steps if s.label.startswith("run")]
    assert runs == ["run E6 B2 seed1"]
    runs = [s.label for s in build_plan("RC-A1", 2).steps if s.label.startswith("run")]
    assert runs == ["run A1 B2 seed2"]
    with pytest.raises(CardError, match="not 1"):
        build_plan("RC-A2", 1)  # the other ablations stay at seed 0
    with pytest.raises(CardError, match="the comparator is E6"):
        build_plan("RC-E5", 1)
    needs = [n.label for n in build_plan("RC-R1").needs]
    assert "E2 seed2 adapter" in needs
    monkeypatch.setattr(cards_mod, "seed_plan_choice", lambda: stop)
    assert "E2 seed2 adapter" not in [n.label for n in build_plan("RC-R1").needs]


def test_the_choice_script_writes_the_record_the_cards_read(tmp_path: Path) -> None:
    from medcompress.config import RunContext
    from medcompress.utils.io import append_results

    values = {"E5:magnitude": 0.8, "E5:activation": 0.85, "E5:taylor": 0.9, "E6": 0.7, "E7": 0.95}
    for label, value in values.items():
        exp, _, variant = label.partition(":")
        folder = RunContext(exp, seed=0, budget="B2", variant=variant or None).run_dir(tmp_path)
        rows = [{"experiment": label, "budget": "B2", "seed": 0, "metric": m, "value": value}
                for m in (sp.SMALL_RECALL, sp.MASK_RECALL, sp.SMALL_MARGIN)]  # fmt: skip
        append_results(rows, folder / "metrics.csv")
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "choose_comparator.py"),
         "--results-root", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO, check=False, timeout=120,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    record = json.loads((tmp_path / sp.CHOICE_FILE).read_text())
    assert record["comparator"] == "E5:taylor" and record["futility_stop"] is False
    assert [r["label"] for r in record["ranking"]][:2] == ["E5:taylor", "E5:activation"]
    assert sp.load_choice(tmp_path) == record
