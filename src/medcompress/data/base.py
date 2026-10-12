"""Sample record shared by every stage, and the dataset interface.

A ``VQASample`` is one (image, question, answer) triple plus the metadata the
evaluation needs: patient ID (for patient-level splits), per-finding labels (for
probes and rare-finding recall) and an optional lesion mask (for small-finding
recall and the lesion-mask audit). Images are referenced by path, never embedded.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

SPLITS: tuple[str, ...] = ("train", "val", "test", "external_test")


@dataclass(frozen=True)
class VQASample:
    """One visual question-answering example.

    Fields
    ------
    sample_id     Stable unique ID within the dataset (used as a join key in results).
    dataset       Dataset config name, e.g. ``kvasir_vqa_x1`` (``configs/data/<name>.yaml``).
    image_path    Path relative to ``paths.data_root``; the image itself is never in git.
    question      Free-text question as given by the dataset.
    answer        Reference answer (free text, or "yes"/"no" for probe-style questions).
    patient_id    Patient / procedure ID; None only if the dataset has none (then
                  splits fall back to image-level and this is reported as a limitation).
    split         One of ``SPLITS`` once assigned by ``splits.make_patient_splits``.
    findings      Per-finding binary labels {finding_id: present}, using the domain
                  finding list (``configs/domain/<domain>.yaml``).
    mask_path     Optional lesion mask path (relative to ``paths.data_root``).
    lesion_area   Optional lesion area as a fraction of the image (from ``masks``).
    metadata      Anything else (question type, source subset, ...); never PHI.
    """

    sample_id: str
    dataset: str
    image_path: str
    question: str
    answer: str
    patient_id: str | None = None
    split: str | None = None
    findings: Mapping[str, bool] = field(default_factory=dict)
    mask_path: str | None = None
    lesion_area: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class VQADataset:
    """Interface for one dataset described by a ``configs/data/<name>.yaml`` file.

    Implementations read the dataset from ``paths.data_root`` (or a Kaggle input
    directory), map its labels onto the domain finding list and yield
    ``VQASample`` records. They must not download data or write anything into the
    repository.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config

    def load(self) -> list[VQASample]:
        """Read all samples (all splits) into memory as ``VQASample`` records."""
        raise NotImplementedError("TODO: dataset-specific reader (one per configs/data/*.yaml).")

    def __iter__(self) -> Iterator[VQASample]:
        """Iterate over ``load()``."""
        return iter(self.load())


def load_dataset(config: Mapping[str, Any]) -> VQADataset:
    """Return the reader for a dataset config (dispatch on ``dataset.name``).

    Used by: every stage that reads data (scripts 00-07).
    """
    raise NotImplementedError("TODO: registry of dataset readers keyed by dataset.name.")
