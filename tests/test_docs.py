"""Consistency checks for the decision memos in docs/decisions/."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DECISIONS = REPO_ROOT / "docs" / "decisions"
REQUIRED_SECTIONS = ("## Question", "## Options", "## Recommendation", "## Risks")
MEMO_ROW = re.compile(r"^\|\s*(?:\[(D\d{2})\]\(([^)]+)\)|(D\d{2}))\s*\|.*\|\s*([^|]+?)\s*\|\s*$")


def _index_rows() -> dict[str, tuple[str | None, str]]:
    """Return memo ID -> (linked file or None, status) from the decisions index."""
    rows: dict[str, tuple[str | None, str]] = {}
    for line in (DECISIONS / "README.md").read_text(encoding="utf-8").splitlines():
        match = MEMO_ROW.match(line)
        if match:
            memo_id = match.group(1) or match.group(3)
            rows[memo_id] = (match.group(2), match.group(4))
    return rows


MEMO_FILES = sorted(DECISIONS.glob("D[0-9][0-9]_*.md"))


def test_index_parses() -> None:
    assert _index_rows(), "no memo rows found in docs/decisions/README.md"


@pytest.mark.parametrize("memo", MEMO_FILES, ids=lambda p: p.name)
def test_memo_is_indexed_and_linked(memo: Path) -> None:
    rows = _index_rows()
    memo_id = memo.name[:3]
    assert memo_id in rows, f"{memo_id} missing from the decisions index"
    link, status = rows[memo_id]
    assert link == memo.name, f"{memo_id} row should link {memo.name}"
    assert status != "not written yet"


@pytest.mark.parametrize("memo", MEMO_FILES, ids=lambda p: p.name)
def test_memo_has_required_sections(memo: Path) -> None:
    text = memo.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in text, f"{memo.name} lacks '{section}'"


def test_linked_memos_exist() -> None:
    for memo_id, (link, _status) in _index_rows().items():
        if link is not None:
            assert (DECISIONS / link).is_file(), f"{memo_id} links missing {link}"
