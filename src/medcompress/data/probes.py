"""Fixed yes/no finding probes.

For every calibration or test image x and every finding c in the domain list,
one probe asks the domain template question (e.g. "Is there a polyp in this
image? Answer yes or no.") and stores the correct answer from the labels. The
probes are scored by one forward pass (``VLMWrapper.score_yes_no``); there is no
free generation. Used by the clinical criterion (E7, A1-A6), the lesion-mask
audit and per-finding recall.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medcompress.data.base import VQASample

FINDING_PLACEHOLDER = "{finding}"


@dataclass(frozen=True)
class FindingProbe:
    """One yes/no probe: image, finding, fixed question and correct answer."""

    sample_id: str
    image_path: str
    finding: str
    question: str
    answer: str  # "yes" or "no"


def format_probe_question(template: str, finding: str) -> str:
    """Fill the domain probe template with a finding name.

    >>> format_probe_question("Is there a {finding} in this image? Answer yes or no.", "polyp")
    'Is there a polyp in this image? Answer yes or no.'

    Raises ``ValueError`` if the template has no ``{finding}`` placeholder or the
    finding name is empty.
    """
    if FINDING_PLACEHOLDER not in template:
        raise ValueError(f"Probe template must contain {FINDING_PLACEHOLDER!r}: {template!r}")
    finding = finding.strip()
    if not finding:
        raise ValueError("Finding name must be non-empty.")
    return template.replace(FINDING_PLACEHOLDER, finding)


def build_probes(
    samples: Sequence[VQASample],
    findings: Sequence[str],
    template: str,
    *,
    display_names: Mapping[str, str] | None = None,
    answers: Mapping[str, str] | None = None,
) -> list[FindingProbe]:
    """Return one probe per (sample, finding) with the correct answer from ``sample.findings``.

    ``display_names`` maps finding IDs to the wording used in the question;
    ``answers`` maps ``positive``/``negative`` to the answer strings (default
    "yes"/"no"). Samples without a label for a finding get no probe for it.
    """
    raise NotImplementedError("TODO: cross product of samples and labelled findings.")
