"""RC-SMOKE: the job list, its report, the gate and the smoke-only settings (2026-11-13; CPU)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from medcompress import smoke
from medcompress.cards import build_plan
from medcompress.config import ConfigError, config_root, load_config
from medcompress.registry import get_experiment

REPO = Path(__file__).resolve().parents[1]


def test_every_gated_card_has_smoke_jobs_with_valid_runs_and_plans() -> None:
    jobs = smoke.smoke_jobs()
    assert set(smoke.GATED_CARDS.values()) <= {j.card for j in jobs}
    names = {j.name for j in jobs}
    for job in jobs:
        assert job.lane in ("A", "B", "both") and job.stages
        assert job.after is None or job.after in names
        card, seed, variants, budgets = job.plan
        build_plan(card, seed, variants=list(variants) or None, budgets=list(budgets) or None)
        for stage in job.stages:
            assert (REPO / stage.script).is_file(), stage.script
            if "--exp" in stage.args:
                exp = stage.args[stage.args.index("--exp") + 1]
                registered = get_experiment(exp).scripts
                checks = ("check_first_order.py", "check_gptq_inputs.py")  # diagnostics
                assert stage.script in registered or stage.script.endswith(checks)
    assert {j.lane for j in jobs if j.card in ("RC-E9",)} == {"both"}  # the teacher: both T4s
    e7 = [j.name for j in jobs if j.lane == "B"]
    assert e7.index("E7") < e7.index("E8 B2") and e7.index("E7") < e7.index("A1")


def test_the_report_parses_the_peaks_and_ranks_the_verdicts() -> None:
    output = ("noise\nMEDCOMPRESS_LOAD_PEAK Qwen2.5-VL-3B gpu0=8000000000\n"
              "MEDCOMPRESS_LOAD_PEAK Qwen2.5-VL-3B gpu0=7900000000\n"
              "MEDCOMPRESS_STAGE_PEAK 05_prune_and_recover gpu0=12000000000\n")  # fmt: skip
    loads, stage = smoke.parse_peaks(output)
    assert loads == [
        ("Qwen2.5-VL-3B", {"gpu0": 8000000000}),
        ("Qwen2.5-VL-3B", {"gpu0": 7900000000}),
    ]
    assert stage == {"gpu0": 12000000000}
    rows = [
        smoke.StageResult("RC-G2", "E7", "04 score", 0, 60.0, loads, stage),
        smoke.StageResult("RC-G2", "E7", "05 prune+recover", 6),
        smoke.StageResult("RC-R1", "E0 test", "06 test", 4),
        smoke.StageResult("RC-E3", "E3 w4", "03 quantize", 0),
        smoke.StageResult("RC-E8", "E8 B2", "03 quantize", None, note="skipped: E7 failed"),
    ]
    assert smoke.card_verdicts(rows) == {"RC-G2": "FAIL", "RC-R1": "WAIT", "RC-E3": "PASS",
                                         "RC-E8": "FAIL"}  # fmt: skip
    table = smoke.report_table(rows)
    assert "| RC-G2 | E7 | 05 prune+recover | FAIL (exit 6) |" in table
    assert "Qwen2.5-VL-3B: gpu0 7.45" in table and "| WAIT |" in table and "not run" in table


def test_the_smoke_budgets_change_only_the_measurement_counts() -> None:
    real = load_config(config_root() / "budgets.yaml")
    tiny = smoke.smoke_budgets(real)
    measurement = tiny["budgets"]["measurement"]
    assert (measurement["samples"], measurement["warmup_runs"], measurement["repeats"]) == (1, 0, 1)
    assert measurement["hardware"] == real["budgets"]["measurement"]["hardware"]
    assert tiny["budgets"]["levels"] == real["budgets"]["levels"]
    assert real["budgets"]["measurement"]["samples"] == 32  # the real file is untouched


def test_runs_wait_for_rc_smoke(tmp_path: Path) -> None:
    assert smoke.smoke_gate("RC-E7", {}) is not None  # the grid runs RC-G2's stages
    assert smoke.smoke_gate("RC-E7", {"RC-G2": {"commit": "x"}}) is None
    assert smoke.smoke_gate("RC-B0", {}) is None  # not gated
    assert smoke.passed_cards(tmp_path / "missing.yaml") == {}
    record = tmp_path / "passed.yaml"
    record.write_text("passed:\n  RC-E9: {commit: abc}\n")
    assert smoke.passed_cards(record) == {"RC-E9": {"commit": "abc"}}
    done = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_card.py"), "--card", "RC-G2", "--out",
         str(tmp_path), "--data-root", str(tmp_path)],
        capture_output=True, text=True, cwd=REPO, check=False, timeout=120,
    )  # fmt: skip
    if "RC-G2" not in smoke.passed_cards():
        assert done.returncode == 2 and "only after RC-SMOKE passed" in done.stderr


def test_the_search_stops_after_its_smoke_steps() -> None:
    from medcompress.budgets import smallest_prefix_meeting_target

    calls: list[int] = []

    def measure(k: int) -> float:
        calls.append(k)
        return 100.0 - k

    assert smallest_prefix_meeting_target(64, measure, 70.0, 0.0) == (30, 70.0)  # the real search
    calls.clear()
    count, value = smallest_prefix_meeting_target(64, measure, 70.0, 0.0, max_steps=2)
    assert calls == [0, 64, 32, 16] and (count, value) == (32, 68.0)  # ends, then 2 steps


def test_the_smoke_subset_gives_every_finding_both_labels() -> None:
    from medcompress.evaluation.runner import smoke_subset

    def row(image: str, polyp: bool, uc: bool, mask: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(image_path=image, findings={"polyp": polyp, "uc": uc},
                               mask_path=mask)  # fmt: skip

    samples = [
        row("a", True, False), row("a", True, False), row("b", False, False),
        row("c", True, True, "m.png"), row("d", False, True), row("e", True, False),
    ]  # fmt: skip
    chosen = smoke_subset(samples, ["polyp", "uc"], 1)
    assert [s.image_path for s in chosen] == ["a", "a", "b", "c"]  # both rows of a; e unneeded
    assert any(s.mask_path for s in chosen)


def test_smoke_settings_never_write_into_results(tmp_path: Path) -> None:
    from medcompress.cli import require_smoke_outside_results

    with pytest.raises(ConfigError, match="temporary root only"):
        require_smoke_outside_results({"paths": {"results_root": str(REPO / "results")}})
    require_smoke_outside_results({"paths": {"results_root": str(tmp_path)}})


def test_the_peaks_are_printed_only_for_rc_smoke(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from medcompress.models import loading

    monkeypatch.delenv(loading.PEAKS_ENV, raising=False)
    loading.peak_before_load()
    loading.peak_after_load("x")
    loading.print_stage_peak("s")
    assert capsys.readouterr().out == ""
