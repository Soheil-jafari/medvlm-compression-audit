"""Ablation wiring and the reuse of stage-04 scores (light: no torch needed)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from medcompress.compression.pruning import stage
from medcompress.compression.pruning.criteria import CRITERIA
from medcompress.compression.pruning.criteria.common import removal_order
from medcompress.compression.pruning.units import PrunableUnit, UnitType
from medcompress.config import RunContext, apply_variant, load_experiment
from medcompress.registry import SAME_AS_E7

# Unit sizes of a 36-layer decoder like the student's (hidden 2048, MLP 11008, 2 KV groups).
LAYER, KV_GROUP, CHANNEL_GROUP = 77_000_000, 4_700_000, 786_000


def _units(n_layers: int = 36, kv_groups: int = 2, channel_groups: int = 86) -> list:
    units = []
    for i in range(n_layers):
        units.append(PrunableUnit(f"lm.layer{i}", UnitType.DECODER_LAYER, i, None, LAYER))
        units += [
            PrunableUnit(f"lm.layer{i}.head_group{g}", UnitType.ATTENTION_HEAD, i, g, KV_GROUP)
            for g in range(kv_groups)
        ]
        units += [
            PrunableUnit(f"lm.layer{i}.mlp{c}", UnitType.MLP_CHANNEL_GROUP, i, c, CHANNEL_GROUP)
            for c in range(channel_groups)
        ]
    return units


def test_every_pruning_arm_but_a3_ranks_by_the_score_per_parameter() -> None:
    for exp_id in ("E5", "E6", "E7", "A1", "A2", "A4", "A5", "A6"):
        removal = load_experiment(exp_id)[0]["pruning"]["removal"]
        assert removal["score_normalisation"] == "per_parameter", exp_id
    a3 = load_experiment("A3")[0]["pruning"]
    assert a3["criterion"] == "random" and a3["removal"]["score_normalisation"] == "none"


def _layer_share(order: list, fraction: float = 0.3) -> float:
    """Share of whole layers in the bytes of the first units that reach ``fraction``."""
    total = sum(u.num_params for u in order if u.unit_type is UnitType.DECODER_LAYER)
    taken = layers = 0
    for unit in order:
        if taken >= fraction * total:
            break
        taken += unit.num_params
        layers += unit.num_params if unit.unit_type is UnitType.DECODER_LAYER else 0
    return layers / taken


def test_a3_removes_random_units_not_random_layers() -> None:
    units = _units()
    scores = CRITERIA["random"]({}, seed=0).score(None, units, [])  # type: ignore[arg-type]
    # divided by the size, a random score sends whole layers first: at about B2 almost all
    # removed bytes would be whole layers
    per_weight = removal_order(scores, units, normalisation="per_parameter")
    assert _layer_share(per_weight) > 0.95
    # as A3 ranks them, every unit is equally likely to go next
    plain = removal_order(scores, units, normalisation="none")
    assert _layer_share(plain) < 0.8
    first = plain[:500]
    share = sum(u.unit_type is UnitType.MLP_CHANNEL_GROUP for u in first) / len(first)
    assert abs(share - 86 / 89) < 0.03


def _e7(tmp_path: Path) -> dict:
    config, _ = load_experiment("E7")
    config["paths"]["checkpoint_root"] = str(tmp_path)
    return config


def test_the_scores_fingerprint_follows_the_criterion_settings(tmp_path: Path) -> None:
    config = _e7(tmp_path)
    settings = stage.ScoringSettings.from_config(config)
    run = RunContext("E7", seed=0)
    base = stage._fingerprint(config, settings, run)
    assert json.loads(json.dumps(base)) == base  # stored and compared as JSON
    for key, value in (("ablation_method", "exact"), ("delta", "kl"), ("batch_size", 8)):
        changed = copy.deepcopy(config)
        changed["pruning"]["clinical"][key] = value
        assert stage._fingerprint(changed, settings, run) != base, key
    # the frozen weight table is part of it, except for the unweighted ablation (A1)
    assert "weights" in base["criterion_settings"]
    a1 = load_experiment("A1")[0]
    assert "weights" not in CRITERIA["clinical"].settings_fingerprint(a1)
    assert base["criterion_settings"]["probe"]["template"]


def test_the_scores_fingerprint_changes_when_e2_is_trained_again(tmp_path: Path) -> None:
    config = _e7(tmp_path)
    settings = stage.ScoringSettings.from_config(config)
    run = RunContext("E7", seed=1)
    assert stage._fingerprint(config, settings, run)["parent_adapter"] is None
    adapter = tmp_path / "E2" / "seed1" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "weights").write_bytes(b"first training")
    first = stage._fingerprint(config, settings, run)["parent_adapter"]
    assert first == stage.parent_adapter_digest(config, 1)
    (adapter / "weights").write_bytes(b"second training")
    assert stage._fingerprint(config, settings, run)["parent_adapter"] != first
    assert stage.parent_adapter_digest(config, 0) is None  # another seed, another adapter


def test_a_reused_scores_file_needs_the_same_fingerprint(tmp_path: Path) -> None:
    config = _e7(tmp_path)
    settings = stage.ScoringSettings.from_config(config)
    run = RunContext("E7", seed=0)
    target = tmp_path / "unit_scores.csv"
    target.write_text("unit_id,score\n")
    fingerprint = stage._fingerprint(config, settings, run)
    target.with_name(stage.SETTINGS_FILE).write_text(json.dumps(fingerprint))
    assert stage._reusable(target, fingerprint)
    changed = copy.deepcopy(config)
    changed["pruning"]["clinical"]["negative_delta"] = "clip0"
    assert not stage._reusable(target, stage._fingerprint(changed, settings, run))


def test_the_arms_declared_same_as_e7_resolve_to_e7s_settings() -> None:
    e7 = load_experiment("E7")[0]
    for arm in SAME_AS_E7:
        exp_id, variant = arm.split(":")
        resolved = apply_variant(load_experiment(exp_id)[0], variant)
        for section in ("pruning", "recovery", "training", "evaluation", "model", "seeds"):
            assert resolved.get(section) == e7.get(section), (arm, section)
    # and no other ablation arm is identical
    for exp_id, variant in (("A4", "n64"), ("A5", "kl"), ("A6", "heads_channels"), ("A1", None)):
        resolved = apply_variant(load_experiment(exp_id)[0], variant)
        assert resolved["pruning"] != e7["pruning"], (exp_id, variant)
