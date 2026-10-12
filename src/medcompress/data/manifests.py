"""File manifests: which files a run used, with content hashes.

A manifest lists every image / mask file of a dataset with its SHA-256 hash and
assigned split, so a run can prove which data it used without the data being in
git. Manifests are written under ``paths.data_root`` or ``paths.output_root``
(outside the repository); only their own hash is recorded in run cards and in
``docs/decisions_log.md``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

_CHUNK = 1 << 20


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


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_hash(entries: Sequence[ManifestEntry]) -> str:
    lines = "\n".join(f"{e.relative_path}\t{e.sha256}" for e in entries)
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def build_manifest(dataset: str, version: str, root: Path) -> DatasetManifest:
    """Hash every file under ``root`` and return a manifest (no files are copied)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries = tuple(
        ManifestEntry(path.relative_to(root).as_posix(), file_sha256(path))
        for path in sorted(p for p in root.rglob("*") if p.is_file())
    )
    return DatasetManifest(dataset, version, entries, _manifest_hash(entries))


def _repo_root() -> Path:
    from medcompress.config import repo_root

    return repo_root().resolve()


def write_manifest(manifest: DatasetManifest, path: Path) -> None:
    """Write a manifest as JSON lines to ``path`` (must be outside the repository)."""
    path = Path(path).resolve()
    if path.is_relative_to(_repo_root()):
        raise ValueError(f"Manifests must be written outside the repository: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "dataset": manifest.dataset,
        "version": manifest.version,
        "manifest_sha256": manifest.manifest_sha256,
    }
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for entry in manifest.entries:
            handle.write(json.dumps(asdict(entry), sort_keys=True) + "\n")


def load_manifest(path: Path) -> DatasetManifest:
    """Read a manifest written by ``write_manifest`` and check its own hash."""
    with Path(path).open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    if not lines:
        raise ValueError(f"Empty manifest: {path}")
    header, rows = lines[0], lines[1:]
    entries = tuple(ManifestEntry(**row) for row in rows)
    if _manifest_hash(entries) != header["manifest_sha256"]:
        raise ValueError(f"Manifest hash mismatch: {path}")
    return DatasetManifest(header["dataset"], header["version"], entries, header["manifest_sha256"])


def verify_manifest(manifest: DatasetManifest, root: Path) -> Sequence[str]:
    """Return relative paths whose current hash differs from the manifest (empty = OK).

    Missing files are reported too.
    """
    root = Path(root)
    bad = []
    for entry in manifest.entries:
        path = root / entry.relative_path
        if not path.is_file() or file_sha256(path) != entry.sha256:
            bad.append(entry.relative_path)
    return bad
