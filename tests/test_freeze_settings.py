"""The evaluation settings frozen on 2026-10-31 (pre-registration amendment) follow from the
committed ``results/data/summary.csv`` (RC-D1 run 4b) by the pre-registered rules."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from medcompress.config import ConfigError, config_root, load_config, load_experiment
from medcompress.data.summary import (
    D04_RULE,
    answer_images,
    d04_max_new_tokens,
    read_summary,
    resolve_max_new_tokens,
    training_prevalence,
)

FROZEN = {"polyp": "polyp", "ulcerative colitis": "ulcerative_colitis",
          "oesophagitis": "oesophagitis"}  # fmt: skip
EXCLUDED = ("barretts", "short-segment barretts", "hemorrhoids")
MIN_TRAIN, MIN_HELD_OUT = 20, 10
"""Pre-registration section 2: at least 20 positive training images, 10 in validation and test."""


def _domain() -> dict:
    return load_config(config_root() / "domain" / "endoscopy.yaml")["domain"]


def test_d04_rule_gives_46_from_the_committed_summary() -> None:
    values = read_summary()
    assert values["answer_words:p99"] == 25.0
    assert d04_max_new_tokens(values) == math.ceil(1.5 * 25) + 8 == 46


def test_every_generation_limit_uses_the_rule_not_a_number() -> None:
    config, _ = load_experiment("E0")
    budgets = load_config(config_root() / "budgets.yaml")["budgets"]
    for value in (
        config["evaluation"]["max_new_tokens"],
        config["teacher_cache"]["max_new_tokens"],
        budgets["measurement"]["max_new_tokens"],
    ):
        assert value == D04_RULE
        assert resolve_max_new_tokens(value) == 46


def test_resolve_leaves_other_values_alone(tmp_path: Path) -> None:
    assert resolve_max_new_tokens(64) == 64
    assert resolve_max_new_tokens("TODO") == "TODO"
    assert resolve_max_new_tokens(None) is None
    summary = tmp_path / "summary.csv"
    summary.write_text("experiment,budget,seed,metric,value,ci_low,ci_high\n"
                       "DATA,,0,answer_words:p99,10.0,,\n")  # fmt: skip
    assert resolve_max_new_tokens(D04_RULE, summary) == 23
    summary.write_text("experiment,budget,seed,metric,value,ci_low,ci_high\n")
    with pytest.raises(ConfigError, match="answer_words:p99"):
        resolve_max_new_tokens(D04_RULE, summary)
    with pytest.raises(ConfigError, match="missing"):
        resolve_max_new_tokens(D04_RULE, tmp_path / "absent.csv")


def test_the_evaluation_reads_46() -> None:
    from medcompress.evaluation.runner import resolve_settings

    config, _ = load_experiment("E0")
    settings = resolve_settings(config)
    assert settings.max_new_tokens == 46
    assert settings.findings == ("polyp", "ulcerative_colitis", "oesophagitis")
    assert settings.mask_findings == ("polyp",)


def test_finding_list_follows_the_preregistered_rule() -> None:
    values = read_summary()
    dataset = load_config(config_root() / "data" / "kvasir_vqa_x1.yaml")["dataset"]
    assert dataset["label_mapping"]["rules"][0]["values"] == FROZEN
    assert _domain()["findings"] == list(FROZEN.values())
    for value in FROZEN:
        assert answer_images(values, value, "train") >= MIN_TRAIN
        assert answer_images(values, value, "val") >= MIN_HELD_OUT
        assert answer_images(values, value, "test") >= MIN_HELD_OUT
    for value in EXCLUDED:  # too few to test: a limitation
        assert answer_images(values, value, "test") < MIN_HELD_OUT
        assert value not in FROZEN
    counts = [(answer_images(values, v, "train"), answer_images(values, v, "test")) for v in FROZEN]
    assert counts == [(810, 106), (691, 98), (546, 25)]


def test_display_names_are_the_dataset_wording() -> None:
    names = _domain()["probe"]["display_names"]
    assert names == {finding: value for value, finding in FROZEN.items()}


def test_all_findings_are_common_and_no_rare_group_exists() -> None:
    values = read_summary()
    groups = _domain()["prevalence_groups"]
    assert groups["common"] == list(FROZEN.values())
    assert groups["medium"] == [] and groups["rare"] == []
    assert groups["thresholds"] is None  # no cut point is invented
    prevalence = [training_prevalence(values, v) for v in FROZEN]
    assert [round(p, 3) for p in prevalence] == [0.179, 0.153, 0.121]
    # the draft cut points (common >= 10 %, rare < 2 %) would give the same groups
    assert all(p >= 0.10 for p in prevalence)


def test_rare_finding_suite_is_dropped() -> None:
    config, _ = load_experiment("E0")
    assert "rare_findings" not in config["evaluation"]["suites"]
    assert "small_findings" in config["evaluation"]["suites"]


def test_small_finding_thresholds_are_the_committed_quartiles() -> None:
    values = read_summary()
    thresholds = _domain()["small_findings"]["thresholds"]
    assert thresholds == [values[f"lesion_area_threshold:q{k}"] for k in (1, 2, 3)]
    assert [round(t, 5) for t in thresholds] == [0.05676, 0.11336, 0.19932]
    assert _domain()["small_findings"]["small_stratum"] == 0  # small = below q1


def test_mask_findings_are_polyp() -> None:
    config, _ = load_experiment("E0")
    assert config["image_reliance_check"]["mask_findings"] == ["polyp"]
