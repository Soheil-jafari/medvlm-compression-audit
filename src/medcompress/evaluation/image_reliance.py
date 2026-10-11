"""Image-reliance stress tests and the lesion-mask audit.

Stress tests (same questions, modified image):
* blank image             -> a model that still answers "yes" ignores the image,
* lesion masked           -> answer should change for positive findings,
* area-matched random region masked (control) -> answer should mostly NOT change,
* another patient's image -> answer should follow the new image's labels.

Lesion-mask effect (PRIMARY ENDPOINT at B2):
    flip_rate(lesion masked) - flip_rate(area-matched random region masked)
computed on positive, mask-annotated findings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medcompress.config import RunContext
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper


class StressTest(str, Enum):
    """Image perturbations used by the reliance tests."""

    BLANK_IMAGE = "blank_image"
    LESION_MASKED = "lesion_masked"
    RANDOM_REGION_MASKED = "random_region_masked"
    OTHER_PATIENT_IMAGE = "other_patient_image"


def build_stress_inputs(
    samples: Sequence[VQASample], test: StressTest, *, seed: int
) -> list[VQASample]:
    """Return perturbed copies of samples for one stress test (images written to a temp dir).

    "Other patient" pairs must come from a different patient with different labels.
    """
    raise NotImplementedError("TODO: create perturbed inputs per stress test.")


def answer_flip_rate(original: Sequence[str], perturbed: Sequence[str]) -> float:
    """Fraction of answers that change under a perturbation."""
    raise NotImplementedError("TODO: fraction of changed answers.")


def lesion_mask_effect(
    lesion_masked_flips: Sequence[bool], random_masked_flips: Sequence[bool]
) -> float:
    """Return flip rate (lesion masked) - flip rate (random region masked). Primary endpoint."""
    raise NotImplementedError("TODO: lesion-mask effect (primary endpoint).")


def mask_audit(
    model: VLMWrapper, samples: Sequence[VQASample], *, seed: int
) -> Mapping[str, float]:
    """Post-pruning audit: does the model still depend on the lesion region?

    Runs the lesion-masked and random-region-masked tests with yes/no probes and
    returns flip rates and the lesion-mask effect.

    Used by: E5, E6, E7, A1-A6 (after recovery); E0, E2 for reference.
    """
    raise NotImplementedError("TODO: lesion vs random-region masking audit.")


def check_teacher_image_reliance(config: Mapping[str, Any], run: RunContext | None) -> None:
    """Stage entry point for ``scripts/00_check_teacher_image_reliance.py``.

    Go/no-go gate before caching: if the teacher answers almost as well with a
    blank image, the audit is meaningless for this model/dataset pair. Thresholds
    are pre-registered.

    Used by: E0.
    """
    raise NotImplementedError("TODO: blank-image and masking checks on the teacher (E0).")
