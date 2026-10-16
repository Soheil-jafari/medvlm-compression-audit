"""The unattended run of one card (``scripts/run_card.py``, Kaggle "Save & Run All").

In order: refuse a card that is not ``ready``; check the libraries and the GPUs; verify the
attached shared data and hand-offs once and check that everything the card reads is there;
run each step as a child process (its output goes to the log and the screen); export the
hand-offs other accounts need; copy the results files to commit into one folder; and write
the text of the card's section 6 to ``<out>/run_report_<card>.md`` and the screen.

A step that waits on an open setting (exit code 4) stops the card unless the step is
optional; any other failure stops it. What finished before is kept and reported.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medcompress.cards import CardPlan, Need
from medcompress.handoffs import handoff_items
from medcompress.utils import bundles

CONFIG_INCOMPLETE = 4
INPUT_CHECK_FAILED = 5
ENVIRONMENT_FAILED = 6
RESULT_DIRS: tuple[str, ...] = ("results", "paper", "docs/results")
"""Committed outputs a card may write (only results files; never data or checkpoints)."""


@dataclass
class StepOutcome:
    label: str
    status: (
        str  # ok | skipped | waiting on a setting (optional step) | stopped: setting open | failed
    )
    code: int | None = None
    seconds: float = 0.0


@dataclass
class Session:
    """What one card run did, for its report."""

    plan: CardPlan
    out: Path
    started_utc: str
    commit: str | None = None
    environment: list[str] = field(default_factory=list)
    gpus: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    steps: list[StepOutcome] = field(default_factory=list)
    handoffs: list[str] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    unusual: list[str] = field(default_factory=list)
    peaks: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.status in ("ok", "skipped", "waiting on a setting") for s in self.steps)


def session_env(out: Path, gpus: int, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment of every step: path roots under ``out``, the Hugging Face cache, GPUs."""
    env = dict(os.environ if base is None else base)
    out = Path(out)
    env.setdefault("MEDCOMPRESS_CHECKPOINT_ROOT", str(out / "checkpoints"))
    env.setdefault("MEDCOMPRESS_CACHE_ROOT", str(out / "cache"))
    env.setdefault("MEDCOMPRESS_OUTPUT_ROOT", str(out / "outputs"))
    if Path("/kaggle").is_dir():
        env.setdefault("HF_HOME", "/kaggle/temp/hf")
    if gpus == 1:
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")  # a one-GPU card uses the first T4 only
    env["PYTHONUNBUFFERED"] = "1"
    return env


def need_path(need: Need, config: Mapping[str, Any]) -> Path | None:
    """Where a need must be after the hand-offs are linked (``None``: not one path)."""
    if need.text or need.kind == "outcomes":
        return None
    run = need.run.context if need.run is not None else None
    root, rel = handoff_items(need.kind, config, run)[0]
    return Path(str(config["paths"][root])) / rel


def missing_needs(plan: CardPlan, config: Mapping[str, Any]) -> list[str]:
    """The required inputs that are not there (attach them with ``--from``)."""
    missing = []
    for need in plan.needs:
        path = need_path(need, config)
        if path is not None and not need.optional and not path.exists():
            missing.append(f"{need.label} (expected at {path})")
    return missing


def made_by_g2(run: Any, config: Mapping[str, Any], repo: Path) -> bool:
    """True when the run's pruned model is attached and its results are committed (RC-G2)."""
    from medcompress.compression.recovery import pruned_dir

    metrics = Path(repo) / run.context.run_dir(config["paths"]["results_root"]) / "metrics.csv"
    if not pruned_dir(config, run.context).exists() or not metrics.is_file():
        return False
    return "budget_within_tolerance" in metrics.read_text(encoding="utf-8")


def step_command(step: Any, data_root: Path | None, handoffs: Sequence[Path]) -> list[str]:
    """The command of a step, with ``--data-root`` and ``--from`` when the step takes inputs."""
    command = [sys.executable, *step.argv]
    if step.inputs:
        if data_root is not None:
            command += ["--data-root", str(data_root)]
        for path in handoffs:
            command += ["--from", str(path)]
    return command


def default_runner(command: Sequence[str], env: Mapping[str, str], log: Path) -> int:
    """Run a command; its output goes to the screen and to ``log``."""
    with Path(log).open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            list(command), env=dict(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )  # fmt: skip
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        return process.wait()


def run_steps(
    session: Session,
    config: Mapping[str, Any],
    env: Mapping[str, str],
    *,
    repo: Path,
    data_root: Path | None,
    handoffs: Sequence[Path],
    runner: Callable[[Sequence[str], Mapping[str, str], Path], int] = default_runner,
) -> None:
    """Run the card's steps in order; stop at a failure or at an open setting (unless optional)."""
    log = session.out / "logs" / f"{session.plan.tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    for step in session.plan.steps:
        if step.skip_if_pruned is not None and made_by_g2(step.skip_if_pruned, config, repo):
            session.steps.append(StepOutcome(step.label, "skipped"))
            print(f"== {step.label}: skipped, RC-G2 made it (its pruned model is attached)")
            continue
        present = need_path(step.skip_if_present, config) if step.skip_if_present else None
        if present is not None and present.exists():
            session.steps.append(StepOutcome(step.label, "skipped"))
            print(f"== {step.label}: skipped, {step.skip_if_present.label} is already there")  # type: ignore[union-attr]
            continue
        print(f"== {step.label}", flush=True)
        start = time.perf_counter()
        code = runner(step_command(step, data_root, handoffs), env, log)
        seconds = time.perf_counter() - start
        if code == 0:
            session.steps.append(StepOutcome(step.label, "ok", code, seconds))
            continue
        if code == CONFIG_INCOMPLETE and step.optional:
            session.steps.append(StepOutcome(step.label, "waiting on a setting", code, seconds))
            session.unusual.append(f"{step.label}: waits on an open setting (see the log)")
            continue
        status = "stopped: setting open" if code == CONFIG_INCOMPLETE else "failed"
        session.steps.append(StepOutcome(step.label, status, code, seconds))
        session.unusual.append(f"{step.label}: exit code {code}; the card stopped here")
        break


def export_handoffs(session: Session, config: Mapping[str, Any]) -> None:
    """Write each hand-off of the card whose items exist; items that came from a dataset stay."""
    target_root = session.out / "handoff"
    for handoff in session.plan.handoffs:
        items, shared, missing = [], [], []
        for run in handoff.runs:
            context = None if handoff.kind == "teacher_cache" else run.context
            for root, rel in handoff_items(handoff.kind, config, context):
                source = Path(str(config["paths"][root])) / rel
                if source.is_symlink():
                    shared.append(rel)
                elif source.exists():
                    items.append((root, rel))
                else:
                    missing.append(rel)
        if missing:
            session.unusual.append(f"hand-off {handoff.name}: not written yet: {missing}")
        if shared:
            session.handoffs.append(f"{handoff.name}: {shared} came from an attached dataset")
        if not items:
            continue
        first = handoff.runs[0]
        meta = bundles.handoff_meta(
            config,
            experiment=first.exp,
            seed=None if handoff.kind == "teacher_cache" else first.seed,
            budget=first.budget if len(handoff.runs) == 1 else None,
            variant=first.variant if len(handoff.runs) == 1 else None,
            extra={"card": session.plan.card, "runs": [r.label for r in handoff.runs]},
        )
        try:
            made = bundles.export_handoff(
                config, items, target_root / handoff.name, kind=handoff.kind, meta=meta
            )
        except bundles.BundleError as exc:
            session.unusual.append(f"hand-off {handoff.name}: {exc}")
            continue
        size = sum(p.stat().st_size for p in made.root.rglob("*") if p.is_file())
        session.handoffs.append(
            f"{handoff.name}: {made.root} ({len(made.files)} files, {size / 1e9:.2f} GB); "
            "make it a private dataset"
        )


def collect_results(session: Session, repo: Path) -> None:
    """Copy the changed results files into ``<out>/results_to_commit/`` (same paths)."""
    try:
        listed = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all", "--",
             *RESULT_DIRS],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout.splitlines()  # fmt: skip
    except (OSError, subprocess.SubprocessError) as exc:
        session.unusual.append(f"could not list the changed results files: {exc}")
        return
    target = session.out / "results_to_commit"
    for line in listed:
        rel = line[3:].strip().strip('"')
        source = Path(repo) / rel
        if not source.is_file():
            continue
        session.results.append(rel)
        (target / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target / rel)
        if rel.endswith(".csv"):
            session.peaks += _peak_rows(source, rel)


def _peak_rows(path: Path, rel: str) -> list[str]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []
    return [
        f"{rel}: {r['metric']} = {float(r['value']):,.0f}"
        for r in rows
        if "peak" in r.get("metric", "") and r.get("value") not in (None, "")
    ]


def report_text(session: Session) -> str:
    """The card's section 6 as far as the run can fill it (the person adds name and notes)."""
    plan = session.plan
    lines = [
        f"### {plan.card} run report{f' (seed {plan.seed})' if plan.seed is not None else ''}",
        "",
        f"- Run by / date: <your name> / {session.started_utc} (UTC)",
        f"- Commit actually used: {session.commit or 'unknown'}",
        f"- Hardware: {', '.join(session.gpus) or 'no GPU visible'}",
        "- Library versions: " + ("; ".join(session.environment) or "not checked"),
        "- Inputs verified: " + ("; ".join(session.inputs) or "none"),
        "- Per step (wall time, status):",
    ]
    lines += [
        f"  - {s.label}: {s.seconds / 60:.1f} min, {s.status}"
        + (f" (exit code {s.code})" if s.code not in (0, None) else "")
        for s in session.steps
    ]
    lines.append("- Peak GPU memory (from the results files):")
    lines += [f"  - {p}" for p in session.peaks] or ["  - none recorded"]
    lines.append("- Hand-offs (each folder = ONE private Kaggle dataset; write its name here):")
    lines += [f"  - {h}" for h in session.handoffs] or ["  - none"]
    lines.append(
        f"- Results files to commit ({len(session.results)}; copies in "
        f"{session.out / 'results_to_commit'}):"
    )
    lines += [f"  - {r}" for r in session.results] or ["  - none"]
    lines.append("- Anything unusual: " + ("; ".join(session.unusual) or "none"))
    lines.append(f"- Outcome: {'all steps ok' if session.ok else 'stopped early (see above)'}")
    return "\n".join(lines) + "\n"


def write_report(session: Session) -> Path:
    """Save the report as ``<out>/run_report_<card>.md`` and return its path."""
    path = session.out / f"run_report_{session.plan.tag}.md"
    path.write_text(report_text(session), encoding="utf-8")
    return path


def now_utc() -> str:
    """The current time in UTC, to the minute (the report's date)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
