"""RC-S0b helpers: sides, synthetic image size, LoRA-step rows and the settings printout (CPU)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from medcompress.evaluation.aggregate import classify_path, validate_results_tree
from medcompress.models import image_sides as sides_mod
from medcompress.utils.io import upsert_results

REPO = Path(__file__).resolve().parents[1]
LORA = {"rank": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["decoder_attention"]}


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


def _examples(n: int = 4) -> list[Any]:
    return [SimpleNamespace(sample_id=f"x{i}") for i in range(n)]


def test_a_lora_step_is_recorded_with_its_micro_batch() -> None:
    seen: dict[str, Any] = {}

    def train(model: Any, examples: list, config: Any, *, seed: int) -> Any:
        seen.update(
            model=model, n=len(examples), batch=config.batch_size, accum=config.grad_accumulation
        )
        return SimpleNamespace(losses=[1.25])

    rows = sides_mod.lora_step_rows(
        lambda: "fresh model", _examples(), 4, "student", lora=LORA, learning_rate=1e-4, train=train
    )
    values = {r["metric"]: r["value"] for r in rows}
    assert seen == {"model": "fresh model", "n": 4, "batch": 4, "accum": 1}
    assert (
        values["student_lora_step_ok_mb4"] == 1.0 and values["student_lora_step_seconds_mb4"] >= 0
    )
    with pytest.raises(ValueError, match="micro-batch 5"):
        sides_mod.lora_step_rows(lambda: None, _examples(), 5, "s", lora=LORA, learning_rate=1e-4)


def test_an_out_of_memory_step_is_a_result_not_a_crash(capsys: pytest.CaptureFixture) -> None:
    def train(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    rows = sides_mod.lora_step_rows(
        lambda: object(), _examples(), 4, "student", lora=LORA, learning_rate=1e-4, train=train
    )
    values = {r["metric"]: r["value"] for r in rows}
    assert values == {"student_lora_step_ok_mb4": 0.0}  # no seconds; no GPU here, no peak
    assert "out of memory" in capsys.readouterr().out


def test_side_rows_are_a_valid_results_file_of_their_own(tmp_path: Path) -> None:
    rows = [{"experiment": "RC-M1", "metric": "student_visual_tokens", "value": 194.0}]
    target = tmp_path / "results" / "smoke" / "RC-S0b" / "student.csv"
    upsert_results(sides_mod.tag_rows(rows, 448), target)
    upsert_results(sides_mod.tag_rows(rows, 560), target)
    assert validate_results_tree(tmp_path / "results") == []
    found = classify_path(target, tmp_path / "results")
    assert found is not None and found.kind == "smoke"
    assert target.read_text().count("RC-S0b:side") == 2


def test_the_settings_printout_quotes_lines_or_says_not_stated() -> None:
    text = "base_model: Qwen\nmax_pixels: 1003520\nImages were resized to 448 px\nlr: 2e-5"
    assert sides_mod.matching_lines(text) == [
        "max_pixels: 1003520",
        "Images were resized to 448 px",
    ]
    assert sides_mod.report_lines("README.md", []) == ["README.md: not stated"]
    assert sides_mod.report_lines("a.json", ["max_pixels: 1"]) == ["a.json: max_pixels: 1"]


def test_the_script_refuses_sides_that_are_not_whole_tokens() -> None:
    script = REPO / "scripts" / "measure_image_sides.py"
    bad = subprocess.run(
        [sys.executable, str(script), "student", "--config", "x.yaml", "--sides", "448", "500"],
        capture_output=True, text=True, cwd=REPO, check=False,
    )  # fmt: skip
    assert bad.returncode == 2 and "multiples of 28" in bad.stderr and "Traceback" not in bad.stderr
    helped = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True, check=False
    )
    assert helped.returncode == 0 and "settings" in helped.stdout
