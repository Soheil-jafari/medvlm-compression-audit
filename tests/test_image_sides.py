"""RC-S0b: sides, image size, steps, child processes, LoRA-step rows and the settings (CPU)."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from medcompress.evaluation.aggregate import classify_path, validate_results_tree
from medcompress.models import image_sides as sides_mod
from medcompress.models.smoke import cuda_available

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "measure_image_sides.py"
LORA = {"rank": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["decoder_attention"]}


def _values(rows: list[dict]) -> dict[str, float]:
    return {r["metric"]: r["value"] for r in rows}


def _csv(path: Path) -> dict[tuple[str, str], float]:
    with path.open(newline="") as handle:
        return {(r["experiment"], r["metric"]): float(r["value"]) for r in csv.DictReader(handle)}


def test_sides_are_whole_visual_tokens_and_the_images_shrink_to_each_side() -> None:
    from medcompress.models.hf import _to_pil
    from medcompress.models.smoke import synthetic_images

    sides = sides_mod.check_sides([672, 448, 560, 448])
    assert sides == (448, 560, 672)
    for bad in ([500], [0], []):
        with pytest.raises(ValueError, match="multiples of 28"):
            sides_mod.check_sides(bad)
    assert sides_mod.source_size(sides) == (1344, 1008)
    image = synthetic_images(1, size=sides_mod.source_size(sides))[0]
    resized = {side: _to_pil(image, side).size for side in sides}
    assert resized == {448: (448, 336), 560: (560, 420), 672: (672, 504)}  # 448: as in RC-S0
    assert all(w % 28 == 0 and h % 28 == 0 for w, h in resized.values())
    # the RC-S0 images (448 x 336) are never enlarged, which is why RC-S0b draws larger ones
    assert _to_pil(synthetic_images(1)[0], 672).size == (448, 336)


def test_steps_and_their_row_names() -> None:
    assert sides_mod.STUDENT_STEPS == (
        "score", "lora_mb1", "lora_mb2", "lora_mb4", "lora_gc_mb1", "lora_gc_mb2", "lora_gc_mb4",
    )  # fmt: skip
    gc4 = sides_mod.parse_step("lora_gc_mb4")
    assert (gc4.kind, gc4.micro_batch, gc4.gradient_checkpointing) == ("lora", 4, True)
    for bad in ("lora_mb0", "lora", "train", "lora_gc_mb"):
        with pytest.raises(ValueError, match="unknown step"):
            sides_mod.parse_step(bad)
    assert sides_mod.check_steps(["score", "lora_mb1", "score"], "student") == ("score", "lora_mb1")
    with pytest.raises(ValueError, match="only the step 'score'"):
        sides_mod.check_steps(["score", "lora_mb1"], "teacher")
    names = {
        name: (
            sides_mod.ok_metric("student", sides_mod.parse_step(name)),
            sides_mod.peak_metric("student", sides_mod.parse_step(name)),
        )
        for name in ("score", "lora_mb4", "lora_gc_mb4")
    }
    assert names == {
        "score": ("student_scoring_ok", "student_peak_gpu_bytes_scoring"),
        "lora_mb4": ("student_lora_step_ok_mb4", "student_peak_gpu_bytes_lora_step_mb4"),
        "lora_gc_mb4": ("student_lora_gc_step_ok_mb4", "student_peak_gpu_bytes_lora_gc_step_mb4"),
    }
    owner = sides_mod.step_of_metric
    assert owner("student", "student_lora_step_first_seconds_mb2") == "lora_mb2"
    assert owner("student", "student_lora_gc_step_seconds_mb4") == "lora_gc_mb4"
    assert owner("student", "student_peak_gpu_bytes_lora_gc_step_mb1") == "lora_gc_mb1"
    assert owner("student", "student_visual_tokens") == "score"
    assert owner("student", "teacher_visual_tokens") is None


def _examples(n: int = 4) -> list[Any]:
    return [SimpleNamespace(sample_id=f"x{i}") for i in range(n)]


def test_a_lora_step_is_warmed_up_then_timed_with_its_micro_batch() -> None:
    calls: list[dict[str, Any]] = []

    def train(model: Any, examples: list, config: Any, *, seed: int) -> Any:
        calls.append(
            {"model": model, "n": len(examples), "batch": config.batch_size,
             "accum": config.grad_accumulation, "extra": dict(config.extra)}
        )  # fmt: skip
        return SimpleNamespace(losses=[1.25], seconds=0.5)

    rows = sides_mod.lora_step_rows(
        lambda: "fresh model", _examples(), 4, "student", lora=LORA, learning_rate=1e-4, train=train
    )
    values = _values(rows)
    one = {"model": "fresh model", "n": 4, "batch": 4, "accum": 1, "extra": {}}
    assert calls == [one, one]  # the warm-up step and the timed step, on the same model
    assert values["student_lora_step_ok_mb4"] == 1.0
    assert values["student_lora_step_first_seconds_mb4"] >= 0  # wall time of the whole call
    assert values["student_lora_step_seconds_mb4"] == 0.5  # the optimizer step, as in RC-S0
    calls.clear()
    gc_rows = sides_mod.lora_step_rows(
        lambda: "m", _examples(), 2, "student", lora=LORA, learning_rate=1e-4, train=train,
        gradient_checkpointing=True,
    )  # fmt: skip
    assert calls[0]["extra"] == {"gradient_checkpointing": True} and calls[0]["n"] == 2
    assert _values(gc_rows)["student_lora_gc_step_ok_mb2"] == 1.0
    assert "student_lora_step_ok_mb2" not in _values(gc_rows)  # the two variants never mix
    with pytest.raises(ValueError, match="micro-batch 5"):
        sides_mod.lora_step_rows(lambda: None, _examples(), 5, "s", lora=LORA, learning_rate=1e-4)


def test_an_out_of_memory_step_is_a_result_not_a_crash(capsys: pytest.CaptureFixture) -> None:
    def train(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    rows = sides_mod.lora_step_rows(
        lambda: object(), _examples(), 4, "student", lora=LORA, learning_rate=1e-4, train=train
    )
    values = _values(rows)
    peak = "student_peak_gpu_bytes_lora_step_mb4"
    assert values["student_lora_step_ok_mb4"] == 0.0
    assert not any("seconds" in name for name in values)  # no timing for a failed step
    # with CUDA the peak reached before the failure is written too; without CUDA it is unknown
    assert set(values) == {"student_lora_step_ok_mb4"} | ({peak} if cuda_available() else set())
    assert "out of memory" in capsys.readouterr().out

    def build() -> Any:
        raise RuntimeError("CUDA out of memory while loading")

    loaded = _values(
        sides_mod.lora_step_rows(build, _examples(), 1, "student", lora=LORA, learning_rate=1e-4)
    )
    assert loaded["student_lora_step_ok_mb1"] == 0.0


class FakeChildren:
    """Stands in for subprocess.run: each call 'is' one child that writes (or not) its rows."""

    def __init__(self, outcome: dict[tuple[int, str], str], target: Path | None = None) -> None:
        self.outcome = outcome
        self.target = target
        self.calls: list[tuple[int, str]] = []
        self.csv_lines_seen: list[int] = []

    def __call__(self, command: list[str], *, timeout: float | None, check: bool) -> Any:
        side = int(command[command.index("--side") + 1])
        name = command[command.index("--step") + 1]
        result = Path(command[command.index("--result-file") + 1])
        self.calls.append((side, name))
        if self.target is not None:
            lines = len(self.target.read_text().splitlines()) if self.target.exists() else 0
            self.csv_lines_seen.append(lines)
        step = sides_mod.parse_step(name)
        what = self.outcome.get((side, name), "ok")
        if what == "timeout":
            raise subprocess.TimeoutExpired(command, timeout or 0)
        if what == "killed":
            return SimpleNamespace(returncode=-9)  # wrote nothing
        if what == "oom":
            rows = sides_mod.failure_rows("student", step, 15_445_145_088.0)
        else:
            rows = [{"metric": sides_mod.ok_metric("student", step), "value": 1.0}]
            if step.kind == "lora":
                rows.append({"metric": sides_mod.seconds_metric("student", step), "value": 0.7})
        sides_mod.write_child_result(result, rows)
        return SimpleNamespace(returncode=0)


def _command(side: int, step: Any, result_file: Path) -> list[str]:
    return ["child", "--side", str(side), "--step", step.name, "--result-file", str(result_file)]


def test_one_child_per_step_and_every_failure_is_a_row(tmp_path: Path) -> None:
    target = tmp_path / "results" / "smoke" / "RC-S0b" / "student.csv"
    steps = ("score", "lora_mb1", "lora_mb4", "lora_gc_mb4")
    fake = FakeChildren(
        {(448, "lora_mb4"): "oom", (560, "score"): "killed", (672, "lora_gc_mb4"): "timeout"},
        target,
    )
    outcomes = sides_mod.measure_sides(
        "student", (448, 560, 672), steps, target, _command, tmp_path, run=fake
    )
    assert fake.calls == [(side, name) for side in (448, 560, 672) for name in steps]
    assert fake.csv_lines_seen == sorted(fake.csv_lines_seen)  # rows written as children end
    assert fake.csv_lines_seen[1] > 0
    failed = {(side, name) for side, name, ok in outcomes if not ok}
    assert failed == {(448, "lora_mb4"), (560, "score"), (672, "lora_gc_mb4")}
    rows = _csv(target)
    assert rows[("RC-S0b:side448", "student_lora_step_ok_mb4")] == 0.0
    assert rows[("RC-S0b:side448", "student_peak_gpu_bytes_lora_step_mb4")] == 15_445_145_088.0
    assert rows[("RC-S0b:side560", "student_scoring_ok")] == 0.0  # crashed: no peak known
    assert ("RC-S0b:side560", "student_peak_gpu_bytes_scoring") not in rows
    assert rows[("RC-S0b:side560", "student_lora_step_ok_mb1")] == 1.0  # the run went on
    assert rows[("RC-S0b:side672", "student_lora_gc_step_ok_mb4")] == 0.0
    assert validate_results_tree(tmp_path / "results") == []
    found = classify_path(target, tmp_path / "results")
    assert found is not None and found.kind == "smoke"


def test_a_rerun_replaces_the_rows_of_its_step_only(tmp_path: Path) -> None:
    target = tmp_path / "student.csv"
    sides_mod.measure_sides(
        "student", (448,), ("lora_mb2", "lora_mb4"), target, _command, tmp_path,
        run=FakeChildren({}),
    )  # fmt: skip
    assert _csv(target)[("RC-S0b:side448", "student_lora_step_seconds_mb2")] == 0.7
    sides_mod.measure_sides(
        "student", (448,), ("lora_mb2",), target, _command, tmp_path,
        run=FakeChildren({(448, "lora_mb2"): "oom"}),
    )  # fmt: skip
    rows = _csv(target)
    assert rows[("RC-S0b:side448", "student_lora_step_ok_mb2")] == 0.0
    assert ("RC-S0b:side448", "student_lora_step_seconds_mb2") not in rows  # no stale timing
    assert rows[("RC-S0b:side448", "student_lora_step_seconds_mb4")] == 0.7  # other step kept


def test_a_child_without_its_ok_row_is_a_failed_step(tmp_path: Path) -> None:
    step = sides_mod.parse_step("score")
    result = tmp_path / "r.json"

    def wrong_rows(command: list[str], *, timeout: float | None, check: bool) -> Any:
        result.write_text(json.dumps({"rows": [{"metric": "student_params", "value": 1.0}]}))
        return SimpleNamespace(returncode=0)

    def garbage(command: list[str], *, timeout: float | None, check: bool) -> Any:
        result.write_text("{not json")
        return SimpleNamespace(returncode=1)

    for run in (wrong_rows, garbage):
        rows = sides_mod.run_child(["x"], result, "student", step, run=run)
        assert rows == sides_mod.failure_rows("student", step, None)


def test_the_real_child_processes_record_a_failure_and_go_on(tmp_path: Path) -> None:
    """The script end to end on CPU: each step in its own process; the model cannot load."""
    config = tmp_path / "unloadable.yaml"
    config.write_text(
        "model:\n  name: unloadable\n  role: student\n  family: qwen2_5_vl\n"
        "  source: 'not a repo id!'\n  sha_seen: abc\n  status: CANDIDATE\n"
    )
    out = tmp_path / "results" / "smoke" / "RC-S0b"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "student", "--config", str(config), "--sides", "448",
         "--steps", "score", "lora_mb1", "--out-dir", str(out), "--allow-candidate"],
        capture_output=True, text=True, cwd=REPO, env=env, check=False, timeout=300,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert "done: 0 of 2 steps ok" in done.stdout and "Traceback" not in done.stderr
    rows = _csv(out / "student.csv")
    assert rows[("RC-S0b:side448", "student_scoring_ok")] == 0.0
    assert rows[("RC-S0b:side448", "student_lora_step_ok_mb1")] == 0.0


def test_the_settings_printout_quotes_lines_or_says_not_stated(tmp_path: Path) -> None:
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import measure_image_sides as script
    finally:
        sys.path.remove(str(SCRIPT.parent))
    text = "base_model: Qwen\nmax_pixels: 1003520\nImages were resized to 448 px\nlr: 2e-5"
    assert sides_mod.matching_lines(text) == [
        "max_pixels: 1003520",
        "Images were resized to 448 px",
    ]
    assert sides_mod.report_lines("README.md", []) == ["README.md: not stated"]
    assert sides_mod.report_lines("a.json", ["max_pixels: 1"]) == ["a.json: max_pixels: 1"]

    def offline(source: str) -> Any:
        raise OSError("offline")

    fallback = REPO / "configs" / "models" / "qwen3_vl_2b_student.yaml"
    spec, lines = script._revision_lines(fallback, offline)
    assert spec.revision is None  # the fallback stays unpinned
    assert "config revision: revision not pinned" in lines
    assert "config sha_seen: 89644892e4d85e24eaac8bacfd4f463576704203" in lines
    assert lines[-1] == "hub main now: could not read (OSError: offline)"
    student = REPO / "configs" / "models" / "qwen2_5_vl_3b_student.yaml"
    spec, lines = script._revision_lines(student, lambda _s: SimpleNamespace(sha="0123"))
    assert spec.revision == "66285546d2b821cf421d4f5eb2576359d3770cd3"  # pinned 2026-10-16
    assert lines[1:] == [
        "config revision: 66285546d2b821cf421d4f5eb2576359d3770cd3",
        "config sha_seen: 66285546d2b821cf421d4f5eb2576359d3770cd3",
        "hub main now: 0123",
    ]


def test_the_script_refuses_bad_sides_steps_and_unloadable_configs() -> None:
    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=REPO,
            check=False,
        )  # fmt: skip

    bad = run("student", "--config", "x.yaml", "--sides", "448", "500")
    assert bad.returncode == 2 and "multiples of 28" in bad.stderr and "Traceback" not in bad.stderr
    steps = run("teacher", "--config", "x.yaml", "--steps", "score", "lora_mb1")
    assert steps.returncode == 2 and "only the step 'score'" in steps.stderr
    candidate = run("student", "--config", "configs/models/qwen2_5_vl_3b_student.yaml")
    assert candidate.returncode == 2 and "candidate" in candidate.stderr
    helped = run("--help")
    assert helped.returncode == 0 and "settings" in helped.stdout and "child" in helped.stdout
