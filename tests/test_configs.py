"""All YAML configs load, required keys are present and experiments match the registry."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from medcompress.budgets import load_budget_specs
from medcompress.config import (
    CANDIDATE_STATUS,
    ConfigError,
    PathsConfig,
    apply_overrides,
    config_root,
    deep_merge,
    find_todos,
    load_config,
    load_experiment,
)
from medcompress.data.probes import format_probe_question
from medcompress.registry import REGISTRY, SCORE_UNITS, TRAINING_SEEDS

CONFIG_ROOT = config_root()
DECIDED_STATUS = "DECIDED"
ALL_YAML = sorted(CONFIG_ROOT.rglob("*.yaml"))


def _rel(path: Path) -> str:
    return str(path.relative_to(CONFIG_ROOT))


def test_config_root_found() -> None:
    assert (CONFIG_ROOT / "base.yaml").is_file()
    assert len(ALL_YAML) > 30


@pytest.mark.parametrize("path", ALL_YAML, ids=_rel)
def test_every_config_loads(path: Path) -> None:
    config = load_config(path)
    assert isinstance(config, dict) and config
    assert "_base_" not in config


def test_base_config_sections() -> None:
    base = load_config(CONFIG_ROOT / "base.yaml")
    for key in ("paths", "components", "seeds", "logging", "hardware", "evaluation"):
        assert key in base, f"base.yaml missing {key}"
    PathsConfig.from_dict(base["paths"])
    assert base["seeds"]["training"] == list(TRAINING_SEEDS)
    for key in ("domain", "budgets", "finding_weights", "candidate_model_pairs"):
        assert (CONFIG_ROOT / base["components"][key]).is_file()


@pytest.mark.parametrize("exp_id", list(REGISTRY))
def test_experiment_config_matches_registry(exp_id: str) -> None:
    config, exp = load_experiment(exp_id)
    spec = REGISTRY[exp_id]
    assert exp.kind == spec.kind
    assert exp.parent == spec.parent
    assert exp.variant_of == spec.variant_of
    assert exp.requires == spec.requires
    assert exp.budgets == spec.budgets
    assert exp.variants == spec.variants
    assert exp.scripts == spec.scripts
    assert exp.outputs, f"{exp_id} lists no outputs"
    assert exp.seeds, f"{exp_id} lists no seeds"
    if spec.trains:
        assert exp.seeds == TRAINING_SEEDS, f"training arm {exp_id} must use 3 seeds"
    assert "paths" in config, f"{exp_id} does not inherit base.yaml"


def test_no_orphan_experiment_configs() -> None:
    for path in (CONFIG_ROOT / "experiments").rglob("*.yaml"):
        assert path.stem in REGISTRY, f"{_rel(path)} has no registry entry"


def test_recovery_is_identical_for_all_pruning_arms() -> None:
    pruning_arms = [eid for eid, spec in REGISTRY.items() if SCORE_UNITS in spec.scripts]
    assert {"E5", "E6", "E7", "A1", "A6"} <= set(pruning_arms)
    reference = load_experiment("E7")[0]["recovery"]
    for exp_id in pruning_arms:
        assert load_experiment(exp_id)[0]["recovery"] == reference, exp_id


def test_clinical_criterion_and_ablation_hooks() -> None:
    e7 = load_experiment("E7")[0]["pruning"]
    assert e7["criterion"] == "clinical"
    assert e7["clinical"]["delta"] == "logprob_drop"
    assert e7["clinical"]["weighting"] == "clinical"
    assert load_experiment("A1")[0]["pruning"]["clinical"]["weighting"] == "uniform"
    assert load_experiment("A2")[0]["pruning"]["clinical"]["weighting"] == "rarity_only"
    assert load_experiment("A3")[0]["pruning"]["criterion"] == "random"
    sizes = load_experiment("A4")[0]["pruning"]["calibration"]["size_by_variant"]
    assert sorted(sizes.values()) == [32, 64, 128, 256]
    assert set(load_experiment("A5")[0]["pruning"]["clinical"]["delta_by_variant"]) == {
        "logprob_drop",
        "kl",
    }
    assert load_experiment("E6")[0]["pruning"]["units"]["types"] == ["decoder_layer"]


@pytest.mark.parametrize(
    "path",
    sorted((CONFIG_ROOT / "models").glob("*.yaml")) + sorted((CONFIG_ROOT / "data").glob("*.yaml")),
    ids=_rel,
)
def test_candidate_configs_are_marked_and_not_guessed(path: Path) -> None:
    config = load_config(path)
    if "candidate_pairs" in config:
        assert config["status"] == CANDIDATE_STATUS
        return
    section = config.get("model") or config.get("dataset")
    assert section is not None
    assert section["status"] in (CANDIDATE_STATUS, DECIDED_STATUS)
    if section["status"] == DECIDED_STATUS:
        assert section.get("verified_on") and section.get("verified_from")
        assert not str(section.get("revision", "TODO")).startswith("TODO"), "pin a revision"
    verified = bool(section.get("verified_on")) and str(
        section.get("verified_from") or ""
    ).startswith("https://")
    if not verified:
        assert str(section["source"]).startswith("TODO"), "do not guess hub IDs / URLs"
        assert str(section["licence"]).startswith("TODO"), "do not guess licence terms"
    else:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(section["verified_on"]))
        assert not str(section["source"]).startswith("TODO")
        assert not str(section["licence"]).startswith("TODO")


@pytest.mark.parametrize("name", ["endoscopy", "retinal"])
def test_domain_configs(name: str) -> None:
    domain = load_config(CONFIG_ROOT / "domain" / f"{name}.yaml")["domain"]
    assert domain["status"].startswith((CANDIDATE_STATUS, DECIDED_STATUS))
    for key in ("findings", "prevalence_groups", "probe", "small_findings"):
        assert key in domain
    assert set(domain["prevalence_groups"]) >= {"common", "medium", "rare"}
    question = format_probe_question(domain["probe"]["template"], "lesion")
    assert question.startswith("Is there a lesion in this image?")


def test_budgets_config() -> None:
    specs = load_budget_specs(load_config(CONFIG_ROOT / "budgets.yaml"))
    assert list(specs) == ["B1", "B2", "B3"]
    assert all(spec.reference_experiment == "E2" for spec in specs.values())


def test_finding_weights_template() -> None:
    table = load_config(CONFIG_ROOT / "preregistration" / "finding_weights.yaml")
    assert "frozen_on" in table["preregistration"]
    row = table["findings"][0]
    assert set(row) == {"finding", "prevalence", "importance_tier", "cap", "final_weight"}


# ----------------------------------------------------------------- loader unit tests


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_base_inheritance_and_merge_order(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "x: 1\nnested: {k1: a, k2: a}\nlst: [1, 2]\n")
    _write(tmp_path / "b.yaml", "x: 2\nnested: {k2: b}\n")
    child = _write(
        tmp_path / "sub" / "child.yaml",
        "_base_: [../a.yaml, ../b.yaml]\nnested: {k3: c}\nlst: [9]\n",
    )
    config = load_config(child)
    assert config == {"x": 2, "nested": {"k1": "a", "k2": "b", "k3": "c"}, "lst": [9]}


def test_base_cycle_detected(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "_base_: b.yaml\n")
    _write(tmp_path / "b.yaml", "_base_: a.yaml\n")
    with pytest.raises(ConfigError, match="Cyclic"):
        load_config(tmp_path / "a.yaml")


def test_env_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path / "c.yaml", "a: ${env:MC_TEST_VAR,fallback}\nb: ${env:MC_TEST_SET}\n")
    monkeypatch.delenv("MC_TEST_VAR", raising=False)
    monkeypatch.setenv("MC_TEST_SET", "/kaggle/input/x")
    assert load_config(path) == {"a": "fallback", "b": "/kaggle/input/x"}
    monkeypatch.delenv("MC_TEST_SET")
    with pytest.raises(ConfigError, match="MC_TEST_SET"):
        load_config(path)


def test_overrides_and_todos() -> None:
    config = apply_overrides({"a": {"b": 1}}, ["a.b=2", "a.c=[1, 2]", "d.e=TODO"])
    assert config == {"a": {"b": 2, "c": [1, 2]}, "d": {"e": "TODO"}}
    assert find_todos(config) == ["d.e"]
    with pytest.raises(ConfigError):
        apply_overrides({}, ["no_equals_sign"])


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"b": 1}}
    merged = deep_merge(base, {"a": {"c": 2}})
    assert base == {"a": {"b": 1}}
    assert merged == {"a": {"b": 1, "c": 2}}


def test_apply_variant_picks_the_entry_of_every_by_variant_table() -> None:
    from medcompress.config import ConfigError, apply_variant

    config = {
        "pruning": {
            "criterion": "by_variant",
            "criterion_by_variant": {"a": "magnitude", "b": "taylor"},
            "calibration": {"size_by_variant": {"a": 32, "b": 64}, "split": "train"},
        },
        "other": {"keep": 1},
    }
    chosen = apply_variant(config, "b")
    assert chosen["pruning"] == {
        "criterion": "taylor",
        "calibration": {"size": 64, "split": "train"},
    }
    assert chosen["other"] == {"keep": 1}
    assert "criterion_by_variant" in config["pruning"]  # the input is left alone
    with pytest.raises(ConfigError, match="pruning.criterion_by_variant needs a variant"):
        apply_variant(config, None)
    with pytest.raises(ConfigError, match="no entry for variant 'c'"):
        apply_variant(config, "c")
    plain = {"x": {"y": 1}}
    assert apply_variant(plain, None) == plain and apply_variant(plain, "anything") == plain


def test_open_settings_of_several_readers_are_listed_in_one_error() -> None:
    from medcompress.config import collect_open_settings

    def reader(message: str) -> Callable[[], None]:
        def read() -> None:
            raise ConfigError(message)

        return read

    collect_open_settings(lambda: None)  # nothing open: no error
    with pytest.raises(ConfigError) as caught:
        collect_open_settings(
            reader("settings still open: a.b, c"),
            lambda: None,
            reader("settings still open: c, d"),
            reader("x must be positive"),
        )
    assert str(caught.value) == "settings still open: a.b, c, d; x must be positive"


def test_the_teacher_cache_holds_exactly_the_training_pool() -> None:
    """The E0 cache must cover the examples E2 trains on (one fixed pool, ``training_pool``)."""
    cache = load_experiment("E0")[0]["teacher_cache"]
    training = load_experiment("E2")[0]["training"]
    recovery = load_experiment("E7")[0]["recovery"]
    assert cache["pool_size"] == training["data_subset"] or (
        str(cache["pool_size"]).startswith("TODO")
        and str(training["data_subset"]).startswith("TODO")
    ), "teacher_cache.pool_size must equal training.data_subset"
    if not str(recovery["data_subset"]).startswith("TODO"):
        assert int(recovery["data_subset"]) <= int(training["data_subset"])
