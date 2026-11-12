#!/usr/bin/env python
"""Does a student's tokenizer match the teacher cache's token ids? (RC-E10 step 1, CPU)

The offline distillation of E2 compares the student's logits with the teacher's cached top-20
token ids, so the two tokenizers must agree on every id the cache uses. For the student of
``--config`` (``experiments/E10.yaml``: Qwen3-VL-2B, the D05 fallback) against the teacher of
``components.teacher``:

1. **ids:** every token id in the cache (each answer's ids and its top-20 ids) must map to the
   same token string in both tokenizers;
2. **answers:** the student tokenizer must re-encode the cached answer texts into the cached
   ids (the final stop id compared by its string), for at least ``MIN_TOKENIZER_MATCH`` (99 %)
   of the answers, as the student selection rounds required.

Both pass: train with the E2 recipe (``training.mode: offline_kd``). Either fails: the E1
recipe (``sft``, answers only), stated in the report. Only the tokenizer files are
downloaded (Internet on); no model weights. Writes ``results/smoke/RC-E10/tokenizer_check.csv``
and prints the revisions it read (to pin).

Example (Kaggle CPU notebook, e0-teacher-cache attached):
    CACHE=$(python scripts/find_inputs.py handoff E0_teacher_cache) || exit 1
    python scripts/check_tokenizer_match.py --config experiments/E10.yaml --from $CACHE
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from medcompress.cli import build_parser, main_guard, resolve_config, run_stage  # noqa: E402
from medcompress.config import ConfigError  # noqa: E402

OUT = Path("results/smoke/RC-E10/tokenizer_check.csv")


@dataclass(frozen=True)
class TokenizerCheck:
    """The two parts of the check and what they decide."""

    ids_checked: int
    ids_differing: int
    answers_checked: int
    answers_matching: int
    min_share: float

    @property
    def share(self) -> float:
        return self.answers_matching / self.answers_checked if self.answers_checked else 0.0

    @property
    def passed(self) -> bool:
        return self.ids_checked > 0 and self.ids_differing == 0 and self.share >= self.min_share

    @property
    def recipe(self) -> str:
        """The training mode the pre-registered rule gives."""
        return "offline_kd" if self.passed else "sft"


def compare(
    teacher_tokens: Any,
    student_tokens: Any,
    student_encode: Any,
    records: Iterable[Any],
    *,
    min_share: float,
) -> TokenizerCheck:
    """The check on ``records`` (cache records: ``answer_text``, ``answer_token_ids``,
    ``topk_token_ids``). ``*_tokens(ids)`` give the token strings of ids,
    ``student_encode(text)`` the student's ids of a text."""
    used: set[int] = set()
    checked = matching = 0
    for record in records:
        ids = [int(i) for i in record.answer_token_ids]
        used.update(ids)
        used.update(int(i) for i in record.topk_token_ids.reshape(-1))
        checked += 1
        encoded = [int(i) for i in student_encode(record.answer_text.strip())]
        matching += int(ids[:-1] == encoded)
    ordered = sorted(used)
    differ = sum(
        a != b for a, b in zip(teacher_tokens(ordered), student_tokens(ordered), strict=True)
    )
    return TokenizerCheck(len(ordered), differ, checked, matching, min_share)


def _tokenizer(model_config: str) -> tuple[Any, str, str | None]:
    """The tokenizer of a model config (tokenizer files only) with its hub id and revision."""
    from transformers import AutoTokenizer

    from medcompress.config import load_config
    from medcompress.models.loading import ModelSpec

    spec = ModelSpec.from_config(load_config(model_config))
    revision = spec.resolved_revision  # the same revision the model loader uses
    kwargs = {"revision": revision} if revision else {}
    return AutoTokenizer.from_pretrained(spec.source, **kwargs), spec.source, revision


def rows(check: TokenizerCheck, exp: str) -> Sequence[dict[str, Any]]:
    """Result rows (``results/smoke`` schema)."""
    values = {
        "tokenizer_ids_checked": check.ids_checked,
        "tokenizer_ids_differing": check.ids_differing,
        "tokenizer_answers_checked": check.answers_checked,
        "tokenizer_answer_match_share": check.share,
        "tokenizer_check_passed": float(check.passed),
    }
    return [{"experiment": exp, "budget": "", "seed": 0, "metric": k, "value": float(v)}
            for k, v in values.items()]  # fmt: skip


def run_check(config: dict) -> None:
    from medcompress.config import student_config_path
    from medcompress.models.teacher_cache import load_pinned_teacher_cache
    from medcompress.selection.rounds import MIN_TOKENIZER_MATCH
    from medcompress.utils.io import upsert_results

    experiment = str((config.get("experiment") or {}).get("id") or "")
    if not experiment:
        raise ConfigError("--config must name an experiment config (experiments/E10.yaml)")
    student_config = student_config_path(config)
    teacher, t_source, t_rev = _tokenizer(config["components"]["teacher"])
    student, s_source, s_rev = _tokenizer(str(student_config))
    print(f"teacher tokenizer: {t_source} @ {t_rev}; student tokenizer: {s_source} @ {s_rev} "
          "(an unpinned revision is the hub's current one: pin what this prints)")  # fmt: skip
    cache = Path(config["paths"]["cache_root"]) / "teacher"
    check = compare(
        teacher.convert_ids_to_tokens,
        student.convert_ids_to_tokens,
        lambda text: student.encode(text, add_special_tokens=False),
        load_pinned_teacher_cache(config, cache),
        min_share=MIN_TOKENIZER_MATCH,
    )
    upsert_results(list(rows(check, experiment)), OUT)
    print(f"ids used by the cache: {check.ids_checked}, mapping to another token: "
          f"{check.ids_differing}; answers re-encoded exactly: {check.answers_matching} of "
          f"{check.answers_checked} ({check.share:.1%}, needed {check.min_share:.0%})")  # fmt: skip
    print(
        f"{'PASS' if check.passed else 'FAIL'}: set training.mode: {check.recipe} in "
        f"configs/experiments/{experiment}.yaml (pre-registration section 11 item 13)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser(__doc__.splitlines()[0], experiment_args=False)
    args = parser.parse_args(argv)
    if args.config is None:
        parser.error("--config is required (experiments/E10.yaml)")
    return run_stage("check_tokenizer_match", run_check, config=resolve_config(args))


if __name__ == "__main__":
    main_guard(main)
