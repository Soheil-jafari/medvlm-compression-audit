"""Free-text suites: answers to the dataset's own questions, language quality and error rates.

The model answers the original question of each sampled QA row by greedy generation.
Language metrics compare the answer text with the reference; the hallucination suite
looks for asserted-but-absent, missed and polarity-flipped findings with the rules of
``hallucination.py``. Both are secondary endpoints, computed on the same generations.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from medcompress.evaluation import hallucination, language
from medcompress.evaluation.probe_suites import Metric

if TYPE_CHECKING:
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper


def sample_qa_rows(samples: Sequence[VQASample], limit: int | None, seed: int) -> list[VQASample]:
    """At most ``limit`` QA rows by seeded sampling (all rows, in order, if ``limit`` is None).

    The same seed gives the same rows for every model, so arms are scored on the same questions.
    """
    rows = list(samples)
    if limit is None or limit >= len(rows):
        return rows
    rng = random.Random(f"qa:{seed}:{len(rows)}")
    chosen = sorted(rng.sample(range(len(rows)), limit))
    return [rows[i] for i in chosen]


def generate_answers(
    model: VLMWrapper, rows: Sequence[VQASample], data_root: Path, *, max_new_tokens: int
) -> list[str]:
    """Greedy answers to the question of each row."""
    outputs = model.generate(
        [Path(data_root) / r.image_path for r in rows],
        [r.question for r in rows],
        max_new_tokens=max_new_tokens,
    )
    return [o.text for o in outputs]


def language_metrics(predictions: Sequence[str], references: Sequence[str]) -> list[Metric]:
    """Exact match, token F1 and answer length."""
    lengths = language.answer_length_stats(predictions)
    return [
        Metric("n_answers", float(len(predictions))),
        Metric("exact_match", language.exact_match(predictions, references)),
        Metric("token_f1", language.token_f1(predictions, references)),
        Metric("answer_words_mean", lengths["mean"]),
        Metric("answer_words_median", lengths["median"]),
    ]


def hallucination_metrics(
    predictions: Sequence[str],
    rows: Sequence[VQASample],
    findings: Sequence[str],
    display_names: Mapping[str, str] | None = None,
) -> list[Metric]:
    """Rates of hallucinated, omitted and polarity-flipped findings over the answers.

    The labels of a row are its per-finding labels, keyed by the finding's display
    name (the wording that appears in answers).
    """
    names = dict(display_names or {})
    errors = []
    for prediction, row in zip(predictions, rows, strict=True):
        labels = {names.get(f, f): bool(row.findings[f]) for f in findings if f in row.findings}
        errors.append(hallucination.classify_answer_errors(prediction, row.answer, labels))
    return [
        Metric(f"{kind}_rate", rate) for kind, rate in hallucination.error_rates(errors).items()
    ]
