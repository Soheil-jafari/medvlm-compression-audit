"""File manifests: which files a run used, with content hashes.

A manifest lists every image / mask file of a dataset with its SHA-256 hash and
assigned split, so a run can prove which data it used without the data being in
git. Manifests are written under ``paths.data_root`` or ``paths.output_root``
(outside the repository); only their own hash is recorded in run cards and in
``docs/decisions_log.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestEntry:
    """One file in a manifest."""

    relative_path: str
    sha256: str
    split: str | None = None
    sample_id: str | None = None


@dataclass(frozen=True)
class DatasetManifest:
    """All files of one dataset version, plus the hash of the manifest itself."""

    dataset: str
    version: str
    entries: tuple[ManifestEntry, ...]
    manifest_sha256: str


def build_manifest(dataset: str, version: str, root: Path) -> DatasetManifest:
    """Hash every file under ``root`` and return a manifest (no files are copied)."""
    raise NotImplementedError("TODO: walk root, hash files, record relative paths.")


def write_manifest(manifest: DatasetManifest, path: Path) -> None:
    """Write a manifest as JSON lines to ``path`` (must be outside the repository)."""
    raise NotImplementedError("TODO: write JSONL; refuse paths inside the repository.")


def load_manifest(path: Path) -> DatasetManifest:
    """Read a manifest written by ``write_manifest``."""
    raise NotImplementedError("TODO: read JSONL manifest.")


def verify_manifest(manifest: DatasetManifest, root: Path) -> Sequence[str]:
    """Return relative paths whose current hash differs from the manifest (empty = OK)."""
    raise NotImplementedError("TODO: re-hash files and compare.")
