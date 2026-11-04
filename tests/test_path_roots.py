"""On Kaggle, the scripts warn when a path root lies inside the repository clone."""

from __future__ import annotations

from pathlib import Path

import pytest

from medcompress.cli import roots_inside_repo, warn_roots_inside_repo
from medcompress.config import repo_root

REPO = Path(__file__).resolve().parents[1]
CARDS = sorted((REPO / "docs" / "runs").glob("RC-*.md"))


def _config(root: Path) -> dict:
    return {"paths": {"output_root": "outputs", "checkpoint_root": str(root / "checkpoints"),
                      "cache_root": str(repo_root() / "cache")}}  # fmt: skip


def test_roots_inside_the_clone_are_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(repo_root())
    inside = {key for key, _, _ in roots_inside_repo(_config(tmp_path))}
    assert inside == {"output_root", "cache_root"}  # the checkpoint root is outside
    monkeypatch.setenv("MEDCOMPRESS_ON_KAGGLE", "1")
    warnings = warn_roots_inside_repo(_config(tmp_path))
    err = capsys.readouterr().err
    assert len(warnings) == 2 and "MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs" in err
    monkeypatch.setenv("MEDCOMPRESS_ON_KAGGLE", "0")
    assert warn_roots_inside_repo(_config(tmp_path)) == []  # only on Kaggle


@pytest.mark.parametrize("card", CARDS, ids=lambda p: p.name)
def test_every_kaggle_cell_sets_the_roots_outside_the_clone(card: Path) -> None:
    if card.name.startswith("RC-T1"):
        pytest.skip("RC-T1 runs on the rented GPU, not on Kaggle")
    lines = card.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("cd medvlm-compression-audit"):
            block = "\n".join(lines[i + 1 : i + 4])
            for export in ("MEDCOMPRESS_OUTPUT_ROOT=/kaggle/working/outputs",
                           "MEDCOMPRESS_CHECKPOINT_ROOT=/kaggle/working/checkpoints",
                           "MEDCOMPRESS_CACHE_ROOT=/kaggle/working/cache"):  # fmt: skip
                assert f"export {export}" in block, (card.name, i + 1)
