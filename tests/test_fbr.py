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


def test_huber_the_canonical_answer_tokens_and_the_first_token_report() -> None:
    assert fbr.huber(0.5, 1.0) == 0.125 and fbr.huber(-3.0, 1.0) == 2.5
    assert fbr.huber_slope(0.5, 1.0) == 0.5 and fbr.huber_slope(-3.0, 1.0) == -1.0

    class Tokenizer:
        vocab = {"yes": 7, "Yes": 8, "no": 3, "No": 4, "yes please": None}

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [self.vocab[text]] if self.vocab.get(text) else [1, 2]

        def decode(self, ids: list[int]) -> str:
            return {v: k for k, v in self.vocab.items() if v}.get(ids[0], "<other>")

    class Wrapper:
        tokenizer = Tokenizer()
        yes_ids, no_ids = (8, 7), (4, 3)

    answers = load_config("domain/endoscopy.yaml")["domain"]["probe"]["answers"]
    assert answers == {"positive": "yes", "negative": "no"}  # the template's canonical answers
    yes, no, report = fbr.canonical_answer_ids(Wrapper(), answers)
    assert (yes, no) == (7, 3) and report["positive"] == {"text": "yes", "token": 7}
    with pytest.raises(ValueError, match="stop and tell the research lead"):
        fbr.canonical_answer_ids(Wrapper(), {"positive": "yes please", "negative": "no"})
    with pytest.raises(ValueError, match="not one of"):
        fbr.canonical_answer_ids(Wrapper(), {"positive": "no", "negative": "no"})
    # E2 on 2026-11-24: "No" on some probes, other tokens on the rest, a yes spelling never
    first = {"a": 4, "b": 4, "c": 99, "d": 99, "e": 98}
    present = {"a": True, "b": False, "c": True, "d": False, "e": True}
    rep = fbr.first_token_report(first, present, Wrapper.tokenizer.decode, (8, 7), (4, 3))
    assert rep["by_label"]["positive"] == {"yes": 0, "no": 1, "other": 2}
    assert rep["by_label"]["negative"] == {"yes": 0, "no": 1, "other": 1}
    assert rep["yes_tokens"] == {} and rep["no_tokens"] == {"4": 2}
    assert rep["top_other_tokens"][0] == {"token": 99, "text": "<other>", "count": 2}


def test_only_a_scale_overflow_is_retried() -> None:
    """Note of 2026-11-26: an update is retried at half the scale only when every loss and
    margin is finite and only a gradient is not; a genuine non-finite loss is skipped."""
    assert fbr.scale_overflow([], gradients_finite=False)  # an overflow in the backward pass
    assert fbr.scale_overflow(["pair_gradient"], gradients_finite=False)
    assert not fbr.scale_overflow([], gradients_finite=True)  # a finite update
    for cause in ("ordinary_loss", "probe_ce", "student_margin", "e2_margin"):
        assert not fbr.scale_overflow([cause, "pair_gradient"], gradients_finite=False)
    assert fbr.SCALE_RETRIES == 3


def test_an_excluded_probe_is_replaced_by_the_next_unused_image_of_its_stratum() -> None:
    probes = fbr.training_probes(_samples(), FINDINGS, "Is there {finding}?")
    base = fbr.probe_schedule(probes, FINDINGS, steps=3, seed=0)
    target = base.updates[1][1]  # update 2's polyp negative
    replaced = fbr.probe_schedule(probes, FINDINGS, steps=3, seed=0, exclude={target.key})
    assert target.key not in replaced.unique_keys()
    assert replaced.updates[0] == base.updates[0]  # updates before it unchanged
    assert replaced.updates[1][1].key == base.updates[2][1].key  # the next one of its stratum
    assert [u[0] for u in replaced.updates] == [u[0] for u in base.updates]  # other strata too
    assert replaced.log["polyp:negative"]["excluded_skipped"] == 1
    assert fbr.probe_schedule(probes, FINDINGS, steps=3, seed=0).fingerprint() == base.fingerprint()
