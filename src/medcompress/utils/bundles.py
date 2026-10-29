"""Folders that move between Kaggle accounts as private datasets, with a verified manifest.

Two kinds of folder travel between the team's accounts, both outside the repository:

* the **shared data** (kind ``data``): written once by RC-D1, the exported Kvasir-VQA-x1
  images and rows, the Kvasir-SEG images and masks, and the private RC-D1 files (splits,
  exclusions, mask links, file manifests), laid out as ``data/`` and ``private/``;
* a **hand-off** (every other kind): the output of one run that a later run needs, for example
  the teacher cache, an E2 adapter, a pruned or a quantized model, the evaluation outcomes. Its
  files keep their place under the path root they were written to, for example
  ``checkpoint_root/E2/seed0/adapter/...``.

Each folder carries :data:`MANIFEST_NAME`: every file with its SHA-256 and size, the commit
that wrote it, and what it holds (kind, experiment, seed, budget, variant, the hash of the
config of the run that wrote it, and the settings that must match in the run that reads it).
A run that reads a folder verifies every hash first and stops if one differs
(:class:`BundleError`, a :class:`~medcompress.config.ConfigError`). A Kaggle dataset is
mounted read-only (under ``/kaggle/input``), so a hand-off is used in place: its items are
linked into the reading run's path roots, never copied or changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medcompress.config import ConfigError

MANIFEST_NAME = "bundle_manifest.json"
VERIFIED_ENV = "MEDCOMPRESS_VERIFIED_BUNDLES"
"""Set by ``scripts/run_card.py`` after it hashed every file of a bundle in this session: the
steps it starts then check only the manifest of that bundle, not every file again (the
dataset is read-only)."""
FORMAT_VERSION = 1
DATA_KIND = "data"
EMBED_KIND = "near_duplicates"
"""RC-D1-embed's private output (embeddings, candidates, the labelling sheets), D02 amendment 3:
verified and read in place by the scoring step and RC-D1 run 4."""
TOPUP_KIND = "near_duplicate_topup"
"""The top-up's private output (the extra pairs, their sheets and template), D02 amendment 3."""
HANDOFF_KINDS: tuple[str, ...] = (
    "teacher_adapter",
    "teacher_cache",
    "adapter",
    "pruned",
    "quantized",
    "outcomes",
    "resume",
)
"""What a hand-off holds: our teacher adapter (RC-TT1), the teacher cache (RC-E0b), a trained
adapter (E1, E2, E4), a pruned
model (E5-E8, A1-A6), a quantized model (E3, E8, E9), the evaluation outcomes needed by the
comparison step, or the resume state of a training run that stopped."""
PATH_ROOTS: tuple[str, ...] = ("cache_root", "checkpoint_root", "output_root")
"""The path roots (``paths`` in ``configs/base.yaml``) a hand-off item may live under."""
DATA_FOLDERS: tuple[str, ...] = ("data", "private")
PRIVATE_FILES: tuple[str, ...] = (
    "splits.json",
    "excluded_images.json",
    "mask_links.json",
    "manifest_kvasir_vqa_x1.jsonl",
    "manifest_kvasir_seg.jsonl",
)
"""The RC-D1 files every later run needs (``data/prepare.py: write_private_outputs``)."""
DATA_FILES: tuple[str, ...] = (
    "kvasir_vqa_x1/train.jsonl",
    "kvasir_vqa_x1/test.jsonl",
)
DATA_DIRS: tuple[str, ...] = ("kvasir_vqa_x1/images", "kvasir_seg/images", "kvasir_seg/masks")
COMPAT_KEYS: tuple[str, ...] = (
    "inputs.image_longest_side",
    "seeds.data_split",
    "student",
    "student_revision",
    "teacher",
    "teacher_revision",
    "training.data_subset",
)
"""Settings a run that reads a hand-off must share with the run that wrote it."""
_CHUNK = 1 << 20


class BundleError(ConfigError):
    """A shared folder is missing, incomplete, changed, or does not fit the run that reads it."""


@dataclass(frozen=True)
class Bundle:
    """A verified folder: its root, what it holds and the hash of each file."""

    root: Path
    kind: str
    meta: Mapping[str, Any]
    files: Mapping[str, str]
    items: tuple[Mapping[str, str], ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        """``E2 seed 0 adapter`` (or ``shared data``) for messages."""
        if self.kind == DATA_KIND:
            return "shared data"
        parts = [str(self.meta.get("experiment") or "")]
        for key in ("variant", "budget"):
            if self.meta.get(key):
                parts.append(str(self.meta[key]))
        if self.meta.get("seed") is not None:
            parts.append(f"seed {self.meta['seed']}")
        return " ".join(p for p in parts if p) + f" {self.kind}"


# ------------------------------------------------------------------ hashes


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files_under(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.name != MANIFEST_NAME)


def _listing_sha256(files: Mapping[str, str]) -> str:
    lines = "\n".join(f"{name}\t{files[name]}" for name in sorted(files))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _without_paths(config: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in config.items() if k != "paths"}


def config_sha256(config: Mapping[str, Any]) -> str:
    """SHA-256 of the resolved config without ``paths`` (they differ from machine to machine)."""
    text = json.dumps(_without_paths(config), sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _model_revision(path: Any) -> str | None:
    if not path:
        return None
    try:
        from medcompress.config import load_config
        from medcompress.models.loading import ModelSpec

        return ModelSpec.from_config(load_config(str(path))).resolved_revision
    except Exception:  # noqa: BLE001 - a config that cannot be read has no revision to compare
        return None


def compat_values(config: Mapping[str, Any]) -> dict[str, Any]:
    """The settings of :data:`COMPAT_KEYS` in a resolved config (``None`` where absent)."""
    components = config.get("components") or {}
    student = (config.get("model") or {}).get("config") or components.get("student")
    teacher = components.get("teacher")
    return {
        "inputs.image_longest_side": (config.get("inputs") or {}).get("image_longest_side"),
        "seeds.data_split": (config.get("seeds") or {}).get("data_split"),
        "student": student,
        "student_revision": _model_revision(student),
        "teacher": teacher,
        "teacher_revision": _model_revision(teacher),
        "training.data_subset": (config.get("training") or {}).get("data_subset"),
    }


def git_commit(repo: Path | None = None) -> str | None:
    """The commit checked out in ``repo`` (the repository by default), or ``None``."""
    if repo is None:
        from medcompress.config import repo_root

        repo = repo_root()
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=30,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


# ------------------------------------------------------------------ writing


def write_bundle_manifest(
    root: Path,
    *,
    kind: str,
    meta: Mapping[str, Any] | None = None,
    items: Sequence[Mapping[str, str]] = (),
    commit: str | None = None,
) -> Bundle:
    """Hash every file under ``root`` and write :data:`MANIFEST_NAME` (atomically).

    ``commit`` defaults to the repository's current commit. The folder must be outside the
    repository: nothing in it may ever be committed.
    """
    from medcompress.config import repo_root

    root = Path(root).resolve()
    if root.is_relative_to(repo_root().resolve()):
        raise BundleError(f"a shared folder must be outside the repository: {root}")
    if kind not in (DATA_KIND, EMBED_KIND, TOPUP_KIND) and kind not in HANDOFF_KINDS:
        raise BundleError(
            f"unknown kind {kind!r}; use {DATA_KIND}, {EMBED_KIND}, {TOPUP_KIND} or one of "
            f"{HANDOFF_KINDS}"
        )
    files = {p.relative_to(root).as_posix(): file_sha256(p) for p in _files_under(root)}
    if not files:
        raise BundleError(f"{root} holds no files")
    sizes = {p.relative_to(root).as_posix(): p.stat().st_size for p in _files_under(root)}
    document = {
        "format_version": FORMAT_VERSION,
        "kind": kind,
        "commit": commit if commit is not None else git_commit(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **dict(meta or {}),
        "items": [dict(i) for i in items],
        "files": [{"path": n, "sha256": files[n], "bytes": sizes[n]} for n in sorted(files)],
        "listing_sha256": _listing_sha256(files),
    }
    tmp = root / f"{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(document, indent=1, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(root / MANIFEST_NAME)
    return _bundle(root, document)


def _bundle(root: Path, document: Mapping[str, Any]) -> Bundle:
    reserved = {"format_version", "kind", "items", "files", "listing_sha256"}
    return Bundle(
        root=Path(root),
        kind=str(document["kind"]),
        meta={k: v for k, v in document.items() if k not in reserved},
        files={f["path"]: f["sha256"] for f in document["files"]},
        items=tuple(dict(i) for i in document.get("items") or ()),
    )


# ------------------------------------------------------------------ reading and verifying


def find_bundle_root(path: Path, *, max_depth: int = 2) -> Path:
    """The folder holding :data:`MANIFEST_NAME`: ``path`` itself or a folder below it.

    Kaggle may mount an uploaded folder one or two levels below ``/kaggle/input/<dataset>``,
    so the manifest is looked for down to ``max_depth`` levels; exactly one must be found.
    """
    path = Path(path)
    if path.is_file() and path.name == MANIFEST_NAME:
        return path.parent
    if not path.is_dir():
        raise BundleError(f"{path} does not exist; attach the dataset and give its path")
    found: list[Path] = []
    level = [path]
    for _ in range(max_depth + 1):
        found += [p for p in level if (p / MANIFEST_NAME).is_file()]
        if found:
            break
        level = [c for p in level for c in sorted(p.iterdir()) if c.is_dir()]
    if not found:
        raise BundleError(f"no {MANIFEST_NAME} in {path} (or {max_depth} levels below it)")
    if len(found) > 1:
        raise BundleError(f"more than one {MANIFEST_NAME} under {path}: {found}; give one")
    return found[0]


def read_bundle(path: Path) -> Bundle:
    """The manifest of the folder at (or below) ``path``, without checking the files."""
    root = find_bundle_root(path)
    try:
        document = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BundleError(f"{root / MANIFEST_NAME} is not valid JSON: {exc}") from exc
    if document.get("format_version") != FORMAT_VERSION:
        raise BundleError(f"{root}: unsupported manifest format {document.get('format_version')}")
    files = {f["path"]: f["sha256"] for f in document.get("files") or ()}
    if _listing_sha256(files) != document.get("listing_sha256"):
        raise BundleError(f"{root}: the manifest was changed after it was written")
    return _bundle(root, document)


def verify_bundle(path: Path, *, kind: str | Iterable[str] | None = None) -> Bundle:
    """Read the manifest and check every file's SHA-256; raise :class:`BundleError` if not.

    ``kind`` (one or several) is the kind the reading run expects. Files that are present but
    not in the manifest are reported as a warning and not used by the check.
    """
    bundle = read_bundle(path)
    wanted = {kind} if isinstance(kind, str) else set(kind or ())
    if wanted and bundle.kind not in wanted:
        raise BundleError(f"{bundle.root} holds {bundle.kind!r}, this run needs {sorted(wanted)}")
    if verified_token(bundle) in os.environ.get(VERIFIED_ENV, "").split(os.pathsep):
        return bundle
    missing, changed = [], []
    for name, digest in bundle.files.items():
        target = bundle.root / name
        if not target.is_file():
            missing.append(name)
        elif file_sha256(target) != digest:
            changed.append(name)
    if missing or changed:
        detail = "; ".join(
            f"{label}: {len(names)} (first: {names[0]})"
            for label, names in (("missing", missing), ("hash differs", changed))
            if names
        )
        raise BundleError(f"{bundle.root} does not match its manifest ({detail}); stop here")
    extra = sorted(
        p.relative_to(bundle.root).as_posix()
        for p in _files_under(bundle.root)
        if p.relative_to(bundle.root).as_posix() not in bundle.files
    )
    if extra:
        print(f"note: {len(extra)} file(s) in {bundle.root} are not in its manifest: {extra[:3]}")
    return bundle


def verified_token(bundle: Bundle) -> str:
    """How :data:`VERIFIED_ENV` names a bundle verified in this session."""
    return f"{bundle.root.resolve()}|{_listing_sha256(dict(bundle.files))}"


def check_compat(bundle: Bundle, config: Mapping[str, Any]) -> None:
    """Stop if a setting of :data:`COMPAT_KEYS` differs between the writer and this run.

    A setting that one side does not have (``None``) is not compared.
    """
    theirs = bundle.meta.get("compat") or {}
    ours = compat_values(config)
    clashes = [
        f"{key}: {theirs.get(key)!r} there, {ours.get(key)!r} here"
        for key in COMPAT_KEYS
        if theirs.get(key) is not None
        and ours.get(key) is not None
        and str(theirs.get(key)) != str(ours.get(key))
    ]
    if clashes:
        raise BundleError(f"{bundle.label} was made with other settings: " + "; ".join(clashes))


# ------------------------------------------------------------------ the shared data


def check_data_layout(root: Path) -> list[str]:
    """What the shared-data folder lacks (empty if complete): ``data/`` and ``private/``."""
    root = Path(root)
    lacking = [
        f"private/{name}" for name in PRIVATE_FILES if not (root / "private" / name).is_file()
    ]
    lacking += [f"data/{name}" for name in DATA_FILES if not (root / "data" / name).is_file()]
    lacking += [
        f"data/{name}/"
        for name in DATA_DIRS
        if not (root / "data" / name).is_dir() or not any((root / "data" / name).iterdir())
    ]
    return lacking


def attach_data(config: dict[str, Any], data_root: Path) -> Bundle:
    """Verify the shared data; point ``paths.data_root`` and ``private_root`` at it.

    If ``config`` names the main dataset's config (``components.datasets.train_qa``) and that
    pins a ``split_sha256`` (RC-D1 run 4b), the shared data must hold that split."""
    bundle = verify_bundle(data_root, kind=DATA_KIND)
    lacking = check_data_layout(bundle.root)
    if lacking:
        raise BundleError(f"the shared data at {bundle.root} lack: {', '.join(lacking)}")
    pinned = _pinned_split(config)
    if pinned is not None and bundle.meta.get("split_sha256") != pinned:
        raise BundleError(
            f"the shared data at {bundle.root} hold split_sha256 "
            f"{bundle.meta.get('split_sha256')}, the config pins {pinned} (RC-D1 run 4b, D02 "
            "amendments 5-7): attach run 4b's shared data"
        )
    paths = config.setdefault("paths", {})
    paths["data_root"] = str(bundle.root / "data")
    paths["private_root"] = str(bundle.root / "private")
    return bundle


def _pinned_split(config: Mapping[str, Any]) -> str | None:
    """The split hash the main dataset's config pins, if ``config`` names that config."""
    name = ((config.get("components") or {}).get("datasets") or {}).get("train_qa")
    if not name:
        return None
    from medcompress.config import load_config
    from medcompress.data.prepared import pinned_split_sha256

    return pinned_split_sha256(load_config(name).get("dataset") or {})


# ------------------------------------------------------------------ hand-offs


def handoff_meta(
    config: Mapping[str, Any],
    *,
    experiment: str | None,
    seed: int | None,
    budget: str | None = None,
    variant: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """What a hand-off's manifest records about the run that wrote it."""
    return {
        "experiment": experiment,
        "seed": seed,
        "budget": budget,
        "variant": variant,
        "config_sha256": config_sha256(config),
        "compat": compat_values(config),
        **dict(extra or {}),
    }


def export_handoff(
    config: Mapping[str, Any],
    items: Sequence[tuple[str, str]],
    out_dir: Path,
    *,
    kind: str,
    meta: Mapping[str, Any],
    commit: str | None = None,
) -> Bundle:
    """Copy the items ``(path root, relative path)`` into ``out_dir`` and write its manifest.

    Each item keeps its place, ``<out_dir>/<path root>/<relative path>``, so the reading run
    links it back under its own path root. ``out_dir`` must be new or empty.
    """
    import shutil

    if kind not in HANDOFF_KINDS:
        raise BundleError(f"unknown hand-off kind {kind!r}; use one of {HANDOFF_KINDS}")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise BundleError(f"{out_dir} is not empty; give a new folder")
    paths = config.get("paths") or {}
    listed = []
    for root_key, rel in items:
        if root_key not in PATH_ROOTS:
            raise BundleError(f"unknown path root {root_key!r}; use one of {PATH_ROOTS}")
        source = Path(str(paths[root_key])) / rel
        if not source.exists():
            raise BundleError(f"{source} does not exist: has the run that writes it finished?")
        target = out_dir / root_key / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        listed.append({"root": root_key, "path": Path(rel).as_posix()})
    return write_bundle_manifest(out_dir, kind=kind, meta=meta, items=listed, commit=commit)


def attach_handoff(
    config: Mapping[str, Any], path: Path, *, kind: str | Iterable[str] | None = None
) -> Bundle:
    """Verify a hand-off and link each of its items into this run's path roots.

    The link ``<path root>/<relative path>`` points at the (read-only) item in the dataset.
    An existing path that is the same link is kept; anything else there stops the run, so a
    local result is never hidden by a hand-off. A ``resume`` hand-off is copied instead,
    because the training run writes into it.
    """
    bundle = verify_bundle(path, kind=kind or HANDOFF_KINDS)
    check_compat(bundle, config)
    paths = config.get("paths") or {}
    for item in bundle.items:
        root_key, rel = item["root"], item["path"]
        if root_key not in PATH_ROOTS or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise BundleError(f"{bundle.root}: bad item {item}")
        source = (bundle.root / root_key / rel).resolve()
        link = Path(str(paths[root_key])) / rel
        if link.is_symlink() and Path(os.readlink(link)) == source:
            continue
        if link.exists() or link.is_symlink():
            raise BundleError(f"{link} already exists here; remove it or run in a fresh session")
        link.parent.mkdir(parents=True, exist_ok=True)
        if bundle.kind == "resume":  # training writes into its resume state: a copy, not a link
            import shutil

            shutil.copytree(source, link)
        else:
            link.symlink_to(source, target_is_directory=source.is_dir())
    return bundle
