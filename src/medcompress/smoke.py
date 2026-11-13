"""RC-SMOKE: every card still to run, its real stages end to end with tiny settings (2026-11-13).

One Kaggle T4 x2 session runs the jobs below with ``scripts/smoke_cards.py``. Each stage runs in
its own process with the real stage script, as in production; only these settings shrink:

* calibration: 2 images (``pruning.calibration.size``, ``quantization.calibration.size``);
* the budget search: its two end points and at most 2 bisection steps
  (``smoke.budget_search_steps``);
* recovery and training: 2 updates (``recovery.steps``, ``training.steps``);
* evaluation: the first validation images that give every finding 2 positive and 2 negative
  images, plus one image with a mask (``smoke.eval_images_per_label``; about 12 probes: fewer
  cannot fit every finding's operating point), 4 free-text answers, 20 bootstrap resamples;
* the benchmark and the search's footprints: 1 sample, no warm-up, 1 timed pass (a copy of
  ``configs/budgets.yaml`` with that ``measurement`` block);
* RC-E3 also runs ``scripts/check_gptq_inputs.py`` on layer 0 for ``w4`` and ``w8`` with all 128
  real calibration examples (stage 03's own capture, then GPTQ on that layer): the tiny
  calibration of the other jobs could miss the example that made RC-E3 seed 0's Hessian NaN;
* first of all, the fp16 calibration check of seed 0 (``scripts/check_calibration.py``, 128
  images, the student) writes the smoke's own copy of ``results/calibration/seed0_replaced.csv``,
  which every later calibration set reads (amendment of 2026-11-13);
* the ablations whose scoring would take hours even on 2 images (A2, A4, A5, A6) score their
  first 300 units only (``smoke.max_scored_units``) and run stage 04 alone: their stages 05-07
  are E7's code, which the E7 job runs in full.

Nothing is written to ``results/``: the runs use a temporary copy of it (stage 05 reads RC-B0's
reference there), and the stage refuses smoke settings with the repository's results root.
Two lanes run at once, one per T4 (``CUDA_VISIBLE_DEVICES``), as each card uses one T4; the
teacher's jobs (E9, E0's test) run after them on both T4s. Each load prints its own peak GPU
memory (``MEDCOMPRESS_LOAD_PEAK``) and each stage its peak (``MEDCOMPRESS_STAGE_PEAK``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SMOKE_OVERRIDES: tuple[str, ...] = (
    "pruning.calibration.size=2",
    "quantization.calibration.size=2",
    "recovery.steps=2",
    "training.steps=2",
    "evaluation.bootstrap_resamples=20",
    "evaluation.generation_samples=4",
    "smoke.eval_images_per_label=2",
    "smoke.budget_search_steps=2",
)
SMOKE_MEASUREMENT: Mapping[str, int] = {"samples": 1, "warmup_runs": 0, "repeats": 1}
CAPPED_UNITS = 300
"""Units scored by the ablations whose full scoring would take hours (A2, A4, A5, A6)."""
GATED_CARDS: Mapping[str, str] = {
    "RC-G2": "RC-G2", "RC-E5": "RC-G2", "RC-E6": "RC-G2", "RC-E7": "RC-G2",
    "RC-E3": "RC-E3", "RC-E4": "RC-E4", "RC-E8": "RC-E8", "RC-E9": "RC-E9",
    "RC-E10": "RC-E10", "RC-R1": "RC-R1",
    **{f"RC-A{n}": f"RC-A{n}" for n in range(1, 7)},
}  # fmt: skip
"""The cards that run only after RC-SMOKE passed for them (the E5 / E6 / E7 grid runs the
stages of RC-G2's runs)."""

S = "scripts/"
SCORE, PRUNE, EVAL, BENCH = S + "04_score_pruning_units.py", S + "05_prune_and_recover.py", \
    S + "06_evaluate.py", S + "07_benchmark_efficiency.py"  # fmt: skip
QUANT, TRAIN, CHECK = S + "03_quantize.py", S + "02_train_student.py", S + "check_first_order.py"


@dataclass(frozen=True)
class SmokeStage:
    label: str
    script: str
    args: tuple[str, ...]
    overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class SmokeJob:
    """A card's part: its stages in order, the lane, and the plan whose hand-offs it attaches."""

    card: str
    name: str
    lane: str  # "pre" (T4 0, before the lanes), "A" (T4 0), "B" (T4 1) or "both"
    stages: tuple[SmokeStage, ...]
    plan: tuple[str, int | None, tuple[str, ...], tuple[str, ...]]  # card, seed, variants, budgets
    after: str | None = None  # a job whose outputs this one reads
    note: str = ""


def _run_args(
    exp: str, *, variant: str | None = None, budget: str | None = None
) -> tuple[str, ...]:
    out = ["--exp", exp, "--seed", "0"]
    if variant:
        out += ["--variant", variant]
    if budget:
        out += ["--budget", budget]
    return tuple(out)


def _pruning(exp: str, variant: str | None, *, score: bool = True) -> tuple[SmokeStage, ...]:
    args = _run_args(exp, variant=variant, budget="B2")
    stages = [SmokeStage("04 score", SCORE, args)] if score else []
    later = (SmokeStage("05 prune+recover", PRUNE, args), SmokeStage("06 validation", EVAL, args),
             SmokeStage("07 benchmark", BENCH, args))  # fmt: skip
    return (*stages, *later)


def _quantized(exp: str, variant: str | None, budget: str | None) -> tuple[SmokeStage, ...]:
    args = _run_args(exp, variant=variant, budget=budget)
    return (SmokeStage("03 quantize", QUANT, args), SmokeStage("06 validation", EVAL, args),
            SmokeStage("07 benchmark", BENCH, args))  # fmt: skip


GPTQ_CHECK = S + "check_gptq_inputs.py"
CAL_CHECK = S + "check_calibration.py"


def _gptq_check(variant: str) -> SmokeStage:
    """Layer 0 of E3 through stage 03's capture with ALL the real calibration examples (RC-E3
    seed 0 found a NaN Hessian at layer 0, 2026-11-13), then GPTQ on that layer."""
    args = (*_run_args("E3", variant=variant), "--layers", "0", "--gptq")
    return SmokeStage(
        "layer-0 GPTQ check", GPTQ_CHECK, args, ("quantization.calibration.size=128",)
    )


def _test(exp: str, variant: str | None = None) -> SmokeStage:
    return SmokeStage("06 test", EVAL, _run_args(exp, variant=variant), ("evaluation.split=test",))


def smoke_jobs() -> list[SmokeJob]:
    """Every card still to run, as smoke jobs (see the module docstring)."""
    g2 = ("RC-G2", None, (), ())
    cap = (f"smoke.max_scored_units={CAPPED_UNITS}",)
    jobs = [
        # first, on T4 0: the fp16 calibration check of seed 0 (amendment of 2026-11-13), which
        # every later job's calibration set reads (128: the GPTQ checks below use all of them)
        SmokeJob(
            "RC-CAL0",
            "calibration check seed 0 (student)",
            "pre",
            (SmokeStage("check", CAL_CHECK, ("--exp", "E3", "--seed", "0", "--size", "128")),),
            ("RC-E3", 0, ("w4",), ()),
            note="writes the smoke's own copy of the check",
        ),
        # lane A (T4 0)
        SmokeJob("RC-G2", "E5 magnitude", "A", _pruning("E5", "magnitude"), g2),
        SmokeJob(
            "RC-G2",
            "E5 activation",
            "A",
            (SmokeStage("04 score", SCORE, _run_args("E5", variant="activation")),),
            g2,
            note="stages 05-07 as E5 magnitude",
        ),
        SmokeJob(
            "RC-G2",
            "E5 taylor",
            "A",
            (SmokeStage("04 score", SCORE, _run_args("E5", variant="taylor")),),
            g2,
            note="stages 05-07 as E5 magnitude",
        ),
        SmokeJob("RC-G2", "E6", "A", _pruning("E6", None), g2),
        SmokeJob(
            "RC-E3",
            "E3 w4",
            "A",
            (*_quantized("E3", "w4", None), _test("E3", "w4")),
            ("RC-E3", 0, ("w4",), ()),
        ),
        SmokeJob(
            "RC-E3",
            "E3 w4 layer-0 GPTQ check",
            "A",
            (_gptq_check("w4"),),
            ("RC-E3", 0, ("w4",), ()),
            note="all 128 calibration examples, production path",
        ),
        SmokeJob(
            "RC-E3",
            "E3 w8 layer-0 GPTQ check",
            "A",
            (_gptq_check("w8"),),
            ("RC-E3", 0, ("w8",), ()),
            note="all 128 calibration examples; w8's other stages are w4's code",
        ),
        SmokeJob(
            "RC-E3", "E3 w8a8", "A", _quantized("E3", "w8a8", None), ("RC-E3", 0, ("w8a8",), ())
        ),
        SmokeJob(
            "RC-E4",
            "E4",
            "A",
            (
                SmokeStage("02 train", TRAIN, _run_args("E4")),
                SmokeStage("06 validation", EVAL, _run_args("E4")),
                SmokeStage("07 benchmark", BENCH, _run_args("E4")),
            ),
            ("RC-E4", 0, (), ()),
        ),
        # lane B (T4 1)
        SmokeJob("RC-G2", "E7", "B", _pruning("E7", None), g2),
        SmokeJob(
            "RC-G2",
            "E7 30-group check",
            "B",
            (SmokeStage("check", CHECK, ("--seed", "0")),),
            g2,
            after="E7",
        ),
        SmokeJob(
            "RC-E8",
            "E8 B2",
            "B",
            _quantized("E8", None, "B2"),
            ("RC-E8", 0, (), ("B2",)),
            after="E7",
            note="quantizes the smoke's own E7",
        ),
        SmokeJob(
            "RC-A1",
            "A1",
            "B",
            _pruning("A1", None),
            ("RC-A1", 0, (), ()),
            after="E7",
            note="re-weights the smoke's own E7 effects",
        ),
        SmokeJob(
            "RC-A3",
            "A3",
            "B",
            (SmokeStage("04 score", SCORE, _run_args("A3")),),
            ("RC-A3", 0, (), ()),
        ),
        SmokeJob(
            "RC-A2",
            "A2",
            "B",
            (SmokeStage("04 score", SCORE, _run_args("A2"), cap),),
            ("RC-A2", 0, (), ()),
            note=f"first {CAPPED_UNITS} units; 05-07 as E7",
        ),
        SmokeJob(
            "RC-A4",
            "A4 n32",
            "B",
            (SmokeStage("04 score", SCORE, _run_args("A4", variant="n32"), cap),),
            ("RC-A4", 0, ("n32",), ()),
            note=f"first {CAPPED_UNITS} units; 05-07 as E7",
        ),
        SmokeJob(
            "RC-A5",
            "A5 kl",
            "B",
            (SmokeStage("04 score", SCORE, _run_args("A5", variant="kl"), cap),),
            ("RC-A5", 0, ("kl",), ()),
            note=f"first {CAPPED_UNITS} units; 05-07 as E7",
        ),
        SmokeJob(
            "RC-A6",
            "A6 heads_channels",
            "B",
            (SmokeStage("04 score", SCORE, _run_args("A6", variant="heads_channels"), cap),),
            ("RC-A6", 0, ("heads_channels",), ()),
            note=f"first {CAPPED_UNITS} units; 05-07 as E7",
        ),
        SmokeJob(
            "RC-E10",
            "E10",
            "B",
            (
                SmokeStage("02 train", TRAIN, _run_args("E10")),
                SmokeStage("06 validation", EVAL, _run_args("E10")),
                SmokeStage("07 benchmark", BENCH, _run_args("E10")),
            ),
            ("RC-E10", None, (), ()),
        ),
        SmokeJob("RC-R1", "E2 test", "B", (_test("E2"),), ("RC-R1", None, (), ())),
        SmokeJob("RC-R1", "E1 test", "B", (_test("E1"),), ("RC-R1", None, (), ())),
        # both T4s: the teacher
        SmokeJob(
            "RC-E9",
            "E9 evaluation path",
            "both",
            (
                SmokeStage("06 validation", EVAL, _run_args("E9")),
                SmokeStage("07 benchmark", BENCH, _run_args("E9")),
            ),
            ("RC-E9", None, (), ()),
            note="the attached e9-seed0-quantized; no quantization",
        ),
        SmokeJob(
            "RC-R1",
            "E0 validation and test",
            "both",
            (SmokeStage("06 validation", EVAL, _run_args("E0")), _test("E0")),
            ("RC-R1", None, (), ()),
            note="the test reads the thresholds of this validation (E0 has none committed)",
        ),
    ]
    return jobs


def stage_status(code: int | None) -> str:
    """PASS (0), WAIT (4: a setting or an input is still open), FAIL (anything else)."""
    if code is None:
        return "not run"
    if code == 0:
        return "PASS"
    if code == 4:
        return "WAIT"
    return f"FAIL (exit {code})"


_LOAD = re.compile(r"^MEDCOMPRESS_LOAD_PEAK (\S+) (.*)$")
_STAGE = re.compile(r"^MEDCOMPRESS_STAGE_PEAK (\S+) (.*)$")


def _sizes(text: str) -> dict[str, int]:
    return {k: int(v) for k, v in (part.split("=") for part in text.split() if "=" in part)}


def parse_peaks(output: str) -> tuple[list[tuple[str, dict[str, int]]], dict[str, int]]:
    """``(loads, stage)``: each load's peak per GPU and the stage's peak, from a stage's output."""
    loads: list[tuple[str, dict[str, int]]] = []
    stage: dict[str, int] = {}
    for line in output.splitlines():
        if m := _LOAD.match(line.strip()):
            loads.append((m.group(1), _sizes(m.group(2))))
        elif m := _STAGE.match(line.strip()):
            stage = _sizes(m.group(2))
    return loads, stage


def _gib(sizes: Mapping[str, int]) -> str:
    return ", ".join(f"{k} {v / 2**30:.2f}" for k, v in sorted(sizes.items()) if v) or "-"


@dataclass
class StageResult:
    card: str
    job: str
    stage: str
    code: int | None
    seconds: float = 0.0
    loads: list[tuple[str, dict[str, int]]] = field(default_factory=list)
    peak: dict[str, int] = field(default_factory=dict)
    note: str = ""


def report_table(results: Sequence[StageResult]) -> str:
    """The PASS / WAIT / FAIL table per card and stage, with the peaks (GiB)."""
    lines = [
        "| Card | Job | Stage | Status | Minutes | Peak per load (GiB) | Stage peak (GiB) | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        loads = "; ".join(f"{name}: {_gib(s)}" for name, s in r.loads) or "-"
        lines.append(
            f"| {r.card} | {r.job} | {r.stage} | {stage_status(r.code)} | {r.seconds / 60:.1f} "
            f"| {loads} | {_gib(r.peak)} | {r.note} |"
        )
    return "\n".join(lines)


def card_verdicts(results: Sequence[StageResult]) -> dict[str, str]:
    """``{card: PASS | WAIT | FAIL}``, the worst of its stages: FAIL if any stage failed or did
    not run, WAIT if one waits on an open input, PASS otherwise."""
    rank = {"PASS": 0, "WAIT": 1, "FAIL": 2}
    out: dict[str, str] = {}
    for r in results:
        status = stage_status(r.code)
        verdict = status if status in ("PASS", "WAIT") else "FAIL"
        if rank[verdict] >= rank[out.get(r.card, "PASS")]:
            out[r.card] = verdict
    return out


def passed_cards(path: Any = None) -> dict[str, Any]:
    """``{card: record}`` of ``docs/runs/smoke_passed.yaml`` (the agent writes it from a reported
    ``smoke_report.md``)."""
    from pathlib import Path

    import yaml

    from medcompress.config import repo_root

    file = Path(path) if path is not None else repo_root() / "docs" / "runs" / "smoke_passed.yaml"
    if not file.is_file():
        return {}
    return dict((yaml.safe_load(file.read_text(encoding="utf-8")) or {}).get("passed") or {})


def smoke_gate(card: str, passed: Mapping[str, Any]) -> str | None:
    """Why ``card`` may not run yet (None if it may): RC-SMOKE has not passed for it."""
    needed = GATED_CARDS.get(card.upper())
    if needed is None or needed in passed:
        return None
    return (f"{card} runs only after RC-SMOKE passed for {needed} (docs/runs/smoke_passed.yaml; "
            "card RC-SMOKE)")  # fmt: skip


def smoke_budgets(source: Mapping[str, Any]) -> dict[str, Any]:
    """``configs/budgets.yaml`` with the smoke's measurement (1 sample, no warm-up, 1 pass)."""
    import copy

    out = copy.deepcopy(dict(source))
    out["budgets"]["measurement"].update(SMOKE_MEASUREMENT)
    return out
