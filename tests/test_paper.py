"""The paper takes every result from generated files; nothing is typed by hand.

Checks (``docs/FINAL_GOAL.md`` section 4):
* results-bearing sections contain no decimal numbers or percentages (other than the
  95% CI level) outside comments; numbers come from ``paper/tables/numbers.tex``;
* every ``\\input{tables/...}`` and ``\\includegraphics{figures/...}`` target exists;
* the committed tables and verdicts match what the aggregation step produces now.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from medcompress.evaluation.statistics import aggregate_results

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO / "paper"
RESULT_SECTIONS = ("00_abstract.tex", "05_results.tex", "06_discussion.tex", "07_conclusion.tex")
NUMBER = re.compile(r"(?<![\w\\])\d+\.\d+|\d+(\.\d+)?\s*\\%")
ALLOWED = ("95\\%",)


def _strip_comments(text: str) -> str:
    return "\n".join(re.split(r"(?<!\\)%", line)[0] for line in text.splitlines())


@pytest.mark.parametrize("name", RESULT_SECTIONS)
def test_no_hand_typed_numbers(name: str) -> None:
    text = _strip_comments((PAPER / "sections" / name).read_text(encoding="utf-8"))
    for allowed in ALLOWED:
        text = text.replace(allowed, "")
    text = re.sub(r"\[[^\]]*\]", "", text)  # layout options such as width=0.48\linewidth
    found = [m.group(0) for m in NUMBER.finditer(text)]
    assert not found, f"{name} has hand-typed numbers {found}; use numbers.tex macros"


def test_inputs_and_figures_exist() -> None:
    for tex in [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex"))]:
        text = _strip_comments(tex.read_text(encoding="utf-8"))
        for target in re.findall(r"\\input\{(tables/[^}]+)\}", text):
            assert (PAPER / f"{target}.tex").is_file() or (PAPER / target).is_file(), target
        for target in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
            assert (PAPER / target).is_file(), target


def test_committed_generated_files_are_current(tmp_path: Path) -> None:
    aggregate_results(
        REPO / "results",
        tmp_path / "summary",
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
        verdicts_path=tmp_path / "VERDICTS.md",
    )
    for fresh in sorted((tmp_path / "tables").glob("*.tex")):
        committed = PAPER / "tables" / fresh.name
        assert (
            committed.read_text() == fresh.read_text()
        ), f"{committed} is stale: run python scripts/08_aggregate_results.py"
    assert (REPO / "docs" / "results" / "VERDICTS.md").read_text() == (
        tmp_path / "VERDICTS.md"
    ).read_text()
    for name in ("all_rows.csv", "by_arm.csv"):
        assert (REPO / "results" / "summary" / name).read_text() == (
            tmp_path / "summary" / name
        ).read_text()
