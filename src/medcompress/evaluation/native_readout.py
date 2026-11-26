"""The native-format readout of the probe answers (exploratory; note of 2026-11-25,
``docs/decisions/native_readout_2026-11-25.md``).

The students answer in the dataset's style ("evidence of polypoid lesion" / "No evidence of
polypoid lesion"), not with yes or no, so the pre-registered parser (``clinical.parse_yes_no``)
leaves most of their answers unparsed and the plain answer's sign is not a yes/no answer. This
parser reads the decoded free answer threshold-free:

* starts with the word "no" ("no evidence ...", "no ...", "no" alone) -> ``"no"``;
* starts with "evidence of", or with the word "yes" -> ``"yes"``;
* anything else -> ``None`` (unparsed: counted and reported, wrong for positives and negatives).

The pre-registered parser and metrics are unchanged; this readout is reported beside them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

import numpy as np

_NO = re.compile(r"^no\b")
_YES = re.compile(r"^(evidence\s+of\b|yes\b)")
_LEAD = re.compile(r"^[^a-z]+")
FIRST_TOKEN_CLASSES: tuple[str, ...] = ("yes spelling", "no spelling", "e", "other")


def normalise(answer: str) -> str:
    """Lower-cased, stripped, leading non-letters dropped."""
    return _LEAD.sub("", answer.strip().lower())


def parse_native(answer: str) -> str | None:
    """``"yes"``, ``"no"`` or ``None`` (unparsed), by the rule of the module docstring."""
    text = normalise(answer)
    if _NO.match(text):
        return "no"
    if _YES.match(text):
        return "yes"
    return None


def leading_pattern(answer: str, words: int = 2) -> str:
    """The first ``words`` words of the normalised answer (the training-pattern count)."""
    return " ".join(normalise(answer).split()[:words])


def native_rates(truth: np.ndarray, readings: Sequence[str | None]) -> dict[str, float]:
    """Yes-rate, sensitivity, specificity and balanced accuracy of native readings; unparsed
    answers count as wrong for positives and negatives."""
    yes = np.array([r == "yes" for r in readings], dtype=bool)
    no = np.array([r == "no" for r in readings], dtype=bool)
    pos, neg = truth, ~truth
    sens = float(yes[pos].mean()) if pos.any() else math.nan
    spec = float(no[neg].mean()) if neg.any() else math.nan
    return {
        "yes_rate": float(yes.mean()) if len(yes) else math.nan,
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": (sens + spec) / 2,
        "unparsed": int((~yes & ~no).sum()),
        "probes": int(len(yes)),
    }


def first_token_class(token: int, text: str, yes_ids: Sequence[int], no_ids: Sequence[int]) -> str:
    """The greedy first answer token as a yes spelling, a no spelling, "e" or other."""
    if int(token) in {int(i) for i in yes_ids}:
        return "yes spelling"
    if int(token) in {int(i) for i in no_ids}:
        return "no spelling"
    return "e" if text == "e" else "other"


def readout_rows(
    label: str, split: str, answers: Mapping[str, np.ndarray], order: Sequence[str]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """``(native readout rows, first-token rows)`` of one arm and split: per finding and macro
    (the mean over findings)."""
    findings = answers["probe_findings"]
    truth = answers["probe_truth"].astype(bool)
    readings = [parse_native(str(t)) for t in answers["probe_decoded_text"]]
    readout, tokens = [], []
    per = []
    for f in order:
        rows = findings == f
        if not rows.any():
            continue
        rates = native_rates(truth[rows], [r for r, k in zip(readings, rows, strict=True) if k])
        per.append(rates)
        readout.append({"run": label, "split": split, "finding": f, **rates})
    if per:
        keys = ("yes_rate", "sensitivity", "specificity", "balanced_accuracy")
        macro: dict[str, float] = {k: float(np.nanmean([p[k] for p in per])) for k in keys}
        macro["unparsed"] = sum(int(p["unparsed"]) for p in per)
        macro["probes"] = sum(int(p["probes"]) for p in per)
        readout.append({"run": label, "split": split, "finding": "macro", **macro})
    if "probe_first_token" in answers:
        yes_ids = answers.get("yes_ids", np.array([], dtype=np.int64))
        no_ids = answers.get("no_ids", np.array([], dtype=np.int64))
        classes = [first_token_class(t, str(x), yes_ids, no_ids)
                   for t, x in zip(answers["probe_first_token"],
                                   answers["probe_first_token_text"], strict=True)]  # fmt: skip
        for f in order:
            for present, name in ((True, "positive"), (False, "negative")):
                rows = (findings == f) & (truth == present)
                if not rows.any():
                    continue
                picked = [c for c, k in zip(classes, rows, strict=True) if k]
                tokens.append({"run": label, "split": split, "finding": f, "label": name,
                               **{c: picked.count(c) for c in FIRST_TOKEN_CLASSES}})  # fmt: skip
    return readout, tokens


def finding_questions(question: str, names: Sequence[str]) -> bool:
    """True for a question about the findings (it names one, or asks about abnormalities)."""
    q = question.lower()
    return any(n.lower() in q for n in names) or "abnormal" in q


def answer_patterns(
    answers: Sequence[str], *, top: int = 40
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """``(patterns, coverage)`` of training answers: the ``top`` most frequent leading
    two-word patterns with their counts and native reading, and how many answers the native
    parser reads as yes, no or unparsed (phrases and counts only; no sample id)."""
    from collections import Counter

    counts = Counter(leading_pattern(a) for a in answers)
    readings = Counter(parse_native(a) or "unparsed" for a in answers)
    rows = [{"pattern": p, "count": n, "native_reading": parse_native(p) or "unparsed"}
            for p, n in counts.most_common(top)]  # fmt: skip
    coverage = {"answers": len(answers), "yes": readings.get("yes", 0),
                "no": readings.get("no", 0), "unparsed": readings.get("unparsed", 0)}  # fmt: skip
    return rows, coverage
