"""Gates, ranking files and the report of the student selection (decision D05).

Two kinds of input, two kinds of step:

* ``rank_selection`` and ``screen_feasibility`` read the per-probe answers that the round
  runs left under ``<output_root>/selection/`` (never committed) together with the
  committed ``metrics.csv`` files, and write ``results/selection/<round>/ranking.csv``.
  One person runs them after collecting every candidate's outputs in one place.
* ``render_report`` reads the committed CSVs only and writes ``docs/results/SELECTION.md``.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from medcompress.selection.ranking import (
    COMPONENTS,
    CandidateOutcomes,
    RoundRanking,
    rank_round,
    ranking_rows,
)
from medcompress.selection.rounds import (
    ROUNDS,
    candidate_name,
    is_open,
    round_eval_config,
)

ROUND_GATES: Mapping[str, tuple[tuple[str, str], ...]] = {
    "R0": (("G1_hardware", "gate_g1_hardware"),),
    "R1": (("G2_uses_image", "gate_g2_uses_image"),),
    "R2": (("tokenizer_match", "gate_tokenizer_match"),),
    "R3": (("tokenizer_match", "gate_tokenizer_match"),),
}
"""Gate name and the metric row that records it (1 = passed) for each round."""
PEAK_MEMORY_METRIC = "r0_peak_gpu_bytes_scoring"
"""fp16 peak memory of Round 0, the second tie-break of D05."""


def selection_results_dir(config: Mapping[str, Any], round_name: str) -> Path:
    """``results/selection/<round>`` (committed metrics and ranking)."""
    return Path(config["paths"]["results_root"]) / "selection" / round_name


def selection_output_dir(config: Mapping[str, Any], round_name: str) -> Path:
    """``<output_root>/selection/<round>`` (per-probe answers, never committed)."""
    return Path(config["paths"]["output_root"]) / "selection" / round_name


def pool(config: Mapping[str, Any]) -> list[str]:
    """Short names of the candidates in the pool, in config order."""
    return [candidate_name(p) for p in config["student_selection"]["candidates"]]


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Rows of a results CSV with ``seed`` and ``value`` as numbers (empty list if absent)."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        row["value"] = float(row["value"])
    return rows


def candidate_metrics(
    config: Mapping[str, Any], round_name: str, candidate: str
) -> dict[int, dict[str, float]]:
    """``{seed: {metric: value}}`` of one candidate in one round, from its ``metrics.csv``."""
    path = selection_results_dir(config, round_name) / candidate / "metrics.csv"
    out: dict[int, dict[str, float]] = {}
    for row in read_rows(path):
        out.setdefault(row["seed"], {})[row["metric"]] = row["value"]
    return out


def _failed_in_round(
    config: Mapping[str, Any], round_name: str, candidate: str, *, required: bool
) -> list[str]:
    seeds = candidate_metrics(config, round_name, candidate)
    failed: list[str] = []
    for gate, metric in ROUND_GATES[round_name]:
        values = [m[metric] for m in seeds.values() if metric in m]
        if not values:
            if required:  # a missing step, not a crash: the stage stops cleanly (exit code 4)
                from medcompress.config import ConfigError

                raise ConfigError(
                    f"{candidate} has no {metric!r} row in results/selection/{round_name}/"
                    f"{candidate}/metrics.csv; run the round before ranking it"
                )
            continue
        if any(v != 1.0 for v in values):
            failed.append(gate)
    return failed


def gate_failures(config: Mapping[str, Any], round_name: str, candidate: str) -> list[str]:
    """Gates a candidate has failed up to and including ``round_name``.

    The gate of the round itself must have been recorded. The hardware gate of Round 0 is
    carried into later rounds when its rows exist. Gates G3 to G5 are checked by the team
    and come from ``student_selection.manual_gate_failures``.
    """
    failed = list(_failed_in_round(config, round_name, candidate, required=True))
    if round_name != "R0":
        failed += _failed_in_round(config, "R0", candidate, required=False)
    manual = config["student_selection"].get("manual_gate_failures") or {}
    failed += [g for g in manual.get(candidate, []) if g not in failed]
    return failed


def previous_round(round_name: str) -> str:
    """The round before ``round_name``."""
    index = ROUNDS.index(round_name)
    if index == 0:
        raise ValueError("R0 is the first round")
    return ROUNDS[index - 1]


def expected_candidates(config: Mapping[str, Any], round_name: str) -> list[str]:
    """The candidates that should have been run in ``round_name``.

    R0 and R1 run the pool (R1 without the candidates that failed Round 0); each later round
    runs the advance list of the previous round's ``ranking.csv``.
    """
    names = pool(config)
    if round_name == "R0":
        return names
    if round_name == "R1":
        return [n for n in names if not _failed_in_round(config, "R0", n, required=False)]
    ranking = selection_results_dir(config, previous_round(round_name)) / "ranking.csv"
    rows = read_rows(ranking)
    if not rows:
        raise ValueError(f"rank {previous_round(round_name)} first: {ranking} does not exist")
    chosen = [
        str(r["experiment"]).split(":", 1)[1]
        for r in rows
        if r["metric"] == "advance" and r["value"] == 1.0
    ]
    if not chosen:
        raise ValueError(f"nobody advanced out of {previous_round(round_name)}")
    return chosen


def collect_outcomes(
    config: Mapping[str, Any], round_name: str, candidates: Sequence[str]
) -> dict[str, list[CandidateOutcomes]]:
    """Per-probe answers of every candidate and seed (a missing file is an error)."""
    seeds = int(config["student_selection"]["rounds"][round_name].get("seeds", 1))
    base = selection_output_dir(config, round_name)
    found: dict[str, list[CandidateOutcomes]] = {}
    missing: list[str] = []
    for name in candidates:
        files = [base / name / f"seed{k}" / "outcomes.npz" for k in range(seeds)]
        absent = [str(f) for f in files if not f.exists()]
        missing += absent
        if not absent:
            found[name] = [CandidateOutcomes.from_npz(f) for f in files]
    if missing:
        raise ValueError(
            "per-probe answers are missing (collect every candidate's outputs under "
            f"{base}): {', '.join(missing)}"
        )
    return found


def tie_break_values(
    config: Mapping[str, Any], candidates: Sequence[str]
) -> dict[str, tuple[float, ...]]:
    """Tie-break tuple per candidate: the fp16 peak memory measured in Round 0 (smaller wins).

    The tokenizer rule is the same for every candidate (decision C) and the licence
    preference is a team call, so neither adds a value here. A candidate without a Round 0
    measurement sorts last.
    """
    out: dict[str, tuple[float, ...]] = {}
    for name in candidates:
        seeds = candidate_metrics(config, "R0", name)
        values = [m[PEAK_MEMORY_METRIC] for m in seeds.values() if PEAK_MEMORY_METRIC in m]
        out[name] = (values[0] if values else math.inf,)
    return out


def _write_ranking(rows: Sequence[Mapping[str, Any]], target: Path) -> None:
    from medcompress.utils.io import append_results

    if target.exists():
        target.unlink()  # derived file: always rewritten from scratch
    append_results(rows, target)


def ranking_settings(config: Mapping[str, Any], round_name: str) -> dict[str, Any]:
    """Weights, reliance metric, places, finding count and resamples for a round's ranking.

    Raises:
        ConfigError: a setting the ranking needs is still ``TODO``.
    """
    from medcompress.config import ConfigError
    from medcompress.evaluation.runner import resolve_settings

    selection = config["student_selection"]
    weights = selection["score"]["weights"]
    wanted = [f"student_selection.score.weights.{c}" for c in COMPONENTS if is_open(weights.get(c))]
    if wanted:
        raise ConfigError(f"selection settings still open: {', '.join(wanted)}")
    settings = resolve_settings(round_eval_config(config, round_name))
    assert settings.n_resamples is not None
    return {
        "weights": tuple(float(weights[c]) for c in COMPONENTS),
        "reliance_metric": str(selection["score"]["image_reliance_metric"][round_name]),
        "keep": int(selection["rounds"][round_name]["keep"]),
        "n_findings": len(settings.findings),
        "n_resamples": int(settings.n_resamples),
        "alpha": float(settings.alpha),
    }


def rank_selection(config: Mapping[str, Any], round_name: str, *, seed: int = 0) -> RoundRanking:
    """Rank one round (R1 to R3) from the per-probe answers and write its ``ranking.csv``."""
    if round_name == "R0":
        raise ValueError("Round 0 has gates only; use screen_feasibility")
    settings = ranking_settings(config, round_name)
    names = expected_candidates(config, round_name)
    failures = {n: gate_failures(config, round_name, n) for n in names}
    # a candidate that failed a gate may have no per-probe answers (a stopped run)
    outcomes = collect_outcomes(config, round_name, [n for n in names if not failures[n]])
    result = rank_round(
        outcomes,
        gate_failures=failures,
        tie_break=tie_break_values(config, names),
        seed=seed,
        **settings,
    )
    _write_ranking(
        ranking_rows(result, round_name), selection_results_dir(config, round_name) / "ranking.csv"
    )
    return result


def screen_feasibility(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Round 0 verdicts: the gates each pool candidate failed (empty tuple = passed).

    Writes ``results/selection/R0/ranking.csv`` with an ``advance`` row per candidate and a
    ``gate_failed:<gate>`` row per failed gate. A candidate with no Round 0 rows is an error.
    """
    verdicts = {n: tuple(gate_failures(config, "R0", n)) for n in pool(config)}
    rows: list[dict[str, Any]] = []
    for name, failed in verdicts.items():
        base = {"experiment": f"R0:{name}", "budget": "", "seed": 0}
        rows.append({**base, "metric": "advance", "value": float(not failed)})
        rows += [{**base, "metric": f"gate_failed:{g}", "value": 1.0} for g in failed]
    _write_ranking(rows, selection_results_dir(config, "R0") / "ranking.csv")
    return verdicts


# --------------------------------------------------------------------- report


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "--" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def _interval(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "--"
    low, high = row.get("ci_low"), row.get("ci_high")
    text = _fmt(row["value"])
    if low in (None, "") or high in (None, ""):
        return text
    return f"{text} [{_fmt(float(low))}, {_fmt(float(high))}]"


def _by_candidate(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    out: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        candidate = str(row["experiment"]).split(":", 1)[1]
        out.setdefault(candidate, {})[str(row["metric"])] = row
    return out


def _feasibility_section(config: Mapping[str, Any]) -> str:
    rows = read_rows(selection_results_dir(config, "R0") / "ranking.csv")
    if not rows:
        return "Not run yet."
    verdict = _by_candidate(rows)
    table = []
    for name in pool(config):
        metrics = candidate_metrics(config, "R0", name).get(0, {})
        failed = [m.split(":", 1)[1] for m in verdict.get(name, {}) if m.startswith("gate_failed:")]
        peak = metrics.get(PEAK_MEMORY_METRIC)
        table.append(
            [
                name,
                "pass" if name in verdict and not failed else "FAIL: " + ", ".join(failed or ["?"]),
                _fmt(metrics.get("r0_params", math.nan) / 1e9, 2),
                _fmt(metrics.get("r0_visual_tokens"), 0),
                _fmt(metrics.get("r0_scoring_seconds_per_probe")),
                _fmt(peak / 2**30, 2) if peak is not None else "--",
                _fmt(metrics.get("r0_lora_step_ok"), 0),
            ]
        )
    header = ["Candidate", "G1", "Params (B)", "Visual tokens", "Scoring s/probe",
              "Peak scoring (GiB)", "LoRA step ok"]  # fmt: skip
    source = "`results/selection/R0/*/metrics.csv`, `results/selection/R0/ranking.csv`"
    return _table(header, table) + f"\n\nSource: {source}."


def _round_section(config: Mapping[str, Any], round_name: str) -> str:
    rows = read_rows(selection_results_dir(config, round_name) / "ranking.csv")
    if not rows:
        return "Not run yet."
    by = _by_candidate(rows)
    ranked = sorted((n for n, m in by.items() if "rank" in m), key=lambda n: by[n]["rank"]["value"])
    table = []
    for name in ranked:
        m = by[name]
        gap = m.get("score_gap_to_next")
        separated = "--"
        if gap is not None and gap.get("ci_low") not in (None, ""):
            separated = "yes" if float(gap["ci_low"]) > 0 else "no"
        table.append(
            [
                str(int(m["rank"]["value"])),
                name,
                _interval(m.get("score_S")),
                *[_fmt(m[f"raw:{c}"]["value"]) if f"raw:{c}" in m else "--" for c in COMPONENTS],
                _interval(gap),
                separated,
                "yes" if m["advance"]["value"] == 1.0 else "no",
            ]
        )
    header = ["Rank", "Candidate", "S [95% CI]", "Macro-F1", "Small-finding recall", "Reliance",
              "Gap to next [95% CI]", "Gap separated", "Advances"]  # fmt: skip
    text = _table(header, table)
    failed = {
        n: [k.split(":", 1)[1] for k in m if k.startswith("gate_failed:")]
        for n, m in by.items()
        if any(k.startswith("gate_failed:") for k in m)
    }
    if failed:
        text += "\n\nLeft the ranking by a gate: " + "; ".join(
            f"{n} ({', '.join(g)})" for n, g in sorted(failed.items())
        )
    metric = config["student_selection"]["score"]["image_reliance_metric"][round_name]
    text += (
        f"\n\nReliance component in this round: `{metric}`. Components are min-max scaled "
        "within the round before weighting; `S` and its interval are on that scale. "
        "`Gap separated` is yes when the interval of the score gap to the next rank "
        f"excludes zero. Source: `results/selection/{round_name}/ranking.csv`."
    )
    return text


def render_report(config: Mapping[str, Any]) -> str:
    """The text of ``docs/results/SELECTION.md``, written from the committed CSVs only."""
    selection = config["student_selection"]
    frozen = selection.get("frozen_on")
    intro = (
        "# Student selection (D05)\n\n"
        "Generated by `scripts/select_student.py --report` from `results/selection/`. "
        "Do not edit.\n\n"
        f"- Protocol: `configs/student_selection.yaml` (status {selection.get('status')}, "
        f"frozen on {'not frozen yet' if is_open(frozen) else frozen})\n"
        "- Split: validation only. Test images are never used in a selection round.\n"
        f"- Pool: {', '.join(pool(config))}\n"
    )
    parts = [intro, "## Round 0: feasibility on one T4", _feasibility_section(config)]
    titles = {
        "R1": "Round 1: simplified screen of the pool",
        "R2": "Round 2: full student recipe",
        "R3": "Round 3: the E2 recipe, two seeds",
    }
    for round_name, title in titles.items():
        parts += [f"## {title}", _round_section(config, round_name)]
    return "\n\n".join(parts) + "\n"


def write_report(config: Mapping[str, Any], path: Path | str = "docs/results/SELECTION.md") -> Path:
    """Write :func:`render_report` to ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(config), encoding="utf-8")
    return target
