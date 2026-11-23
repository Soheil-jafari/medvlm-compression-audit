"""Finding-balanced recovery (amendment of 2026-11-22): the batch composition, the probe
schedule, the exact paired-margin gradient, the sampling-only control, the rebuild of the
pruned model from ``removal.json`` and the margin of the training pass."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from medcompress.compression import fbr
from medcompress.config import ConfigError, load_config
from medcompress.data.base import VQASample

FINDINGS = ("polyp", "oesophagitis", "ulcerative_colitis")


def _registered_cfg() -> Any:
    from medcompress.compression.distillation.offline_kd import TrainConfig

    config = load_config("experiments/E5.yaml")
    return TrainConfig.from_config({**fbr.recovery_config(config), "mode": "offline_kd"})


def test_the_fbr_values_are_the_amendments_and_leave_the_recovery_keys_alone() -> None:
    settings = fbr.FBRSettings.load()
    assert (settings.general_per_update, settings.probes_per_finding) == (10, 2)
    assert (settings.margin_coefficient, settings.huber_delta, settings.seed) == (0.25, 1.0, 0)
    assert settings.pilot_arms == ("E5:magnitude", "E5:activation", "E7")
    assert settings.control_arm == "E5:activation" and settings.gpu_hours_cap == 30
    section = load_config(fbr.CONFIG_FILE)["fbr"]
    recovery = load_config("recovery.yaml")["recovery"]
    assert not set(section) & set(recovery)  # nothing of the registered recovery is overridden
    cfg = _registered_cfg()
    assert (cfg.steps, cfg.batch_size, cfg.grad_accumulation) == (157, 1, 16)
    assert (cfg.kd_weight, cfg.kd_temperature, cfg.learning_rate) == (0.5, 2.0, 1e-4)
    settings.check_batch(cfg, FINDINGS)  # 10 + 2 x 3 = 16
    with pytest.raises(ConfigError, match="effective batch"):
        dataclasses.replace(settings, general_per_update=9).check_batch(cfg, FINDINGS)
    with pytest.raises(ConfigError, match="one positive and one negative"):
        dataclasses.replace(settings, probes_per_finding=4).check_batch(cfg, FINDINGS)


def _samples() -> list[VQASample]:
    out = []
    for i in range(12):
        findings = {"polyp": i % 2 == 0, "oesophagitis": i % 3 == 0}
        if i < 8:  # images 8-11 have no ulcerative-colitis label: never a negative for it
            findings["ulcerative_colitis"] = i % 4 == 1
        for suffix in ("", "b"):  # two QA rows per image: one probe per image
            out.append(VQASample(f"s{i}{suffix}", "d", f"img/{i}.png", "q", "a", findings=findings))
    return out


def test_the_probe_schedule_is_one_positive_and_one_negative_per_finding_per_update() -> None:
    probes = fbr.training_probes(_samples(), FINDINGS, "Is there {finding}?")
    assert len({(p.image_path, p.finding) for p in probes}) == len(probes)  # one per image
    uc = [p for p in probes if p.finding == "ulcerative_colitis"]
    assert {p.image_path for p in uc} == {f"img/{i}.png" for i in range(8)}  # labelled only
    schedule = fbr.probe_schedule(probes, FINDINGS, steps=20, seed=0)
    assert len(schedule.updates) == 20
    for update in schedule.updates:
        assert [(p.finding, p.present) for p in update] == [
            (f, v) for f in FINDINGS for v in (True, False)
        ]
    # unique images are used once before any repeats, in every stratum
    positives = [u[4].image_path for u in schedule.updates]  # ulcerative colitis, present
    n = schedule.log["ulcerative_colitis:positive"]["images_available"]
    assert n == 2 and len(set(positives[:n])) == n
    log = schedule.log["polyp:negative"]
    assert log["images_available"] == 6 and log["unique_images_used"] == 6
    assert log["uses"] == 20 and log["max_reuse"] == 4
    again = fbr.probe_schedule(probes, FINDINGS, steps=20, seed=0)
    assert again.fingerprint() == schedule.fingerprint()  # deterministic
    assert fbr.probe_schedule(probes, FINDINGS, steps=20, seed=1).fingerprint() != (
        schedule.fingerprint()
    )
    with pytest.raises(ValueError, match="no training image for ulcerative_colitis negative"):
        fbr.probe_schedule([p for p in probes if p.present or p.finding != "ulcerative_colitis"],
                           FINDINGS, steps=2, seed=0)  # fmt: skip


def test_the_ordinary_examples_follow_the_registered_order_ten_per_update() -> None:
    order = list(range(100, 125))
    assert fbr.general_indices(order, 0, 10) == order[:10]
    assert fbr.general_indices(order, 1, 10) == order[10:20]
    assert fbr.general_indices(order, 2, 10) == order[20:] + order[:5]  # cycles as registered


def test_huber_and_the_answer_token_rule() -> None:
    assert fbr.huber(0.5, 1.0) == 0.125 and fbr.huber(-3.0, 1.0) == 2.5
    assert fbr.huber_slope(0.5, 1.0) == 0.5 and fbr.huber_slope(-3.0, 1.0) == -1.0
    yes, no, info = fbr.choose_answer_ids({"a": 7, "b": 7, "c": 8, "d": 3, "e": 99}, [7, 8], [3])
    assert (yes, no, info["other_first_tokens"]) == (7, 3, 1)
    with pytest.raises(ValueError, match="never a single-token"):
        fbr.choose_answer_ids({"a": 7}, [7], [3])
