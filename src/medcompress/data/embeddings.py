"""Image embeddings for near-duplicate detection (D02 amendment 3 of 2026-10-19).

The crop correlation and the dHash both missed true copies in RC-D1 run 3 (the same frame at
another size or crop, smooth or blurred mucosa, moving highlights). The joining measure is
therefore the cosine similarity of embeddings from a small pretrained self-supervised vision
model (proposal: DINOv2-small from the Hugging Face hub; ``configs/data/kvasir_vqa_x1.yaml``,
``near_duplicates.embedding``). The model id, its revision and its licence are read from the
hub on Kaggle and printed by RC-D1-embed; nothing here is taken on trust.

Two views of every image, each resized to ``side`` x ``side`` (aspect ratio not kept, the same
for every image):

* ``full``: the whole image, black border and overlays included;
* ``centre``: the central ``centre_fraction`` of width and height (0.60, the crop of the earlier
  measure), which leaves out the corner text, the green box and the border.

Pixels are scaled to 0-1 and normalised with the mean and standard deviation of the model's
own image processor. The feature is the model's pooled output (the class token after the final
layer norm), L2-normalised, so the dot product of two features is their cosine similarity.
Which view (or the larger of the two cosines) joins images is decided on the labelled pairs.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

VIEWS: tuple[str, ...] = ("full", "centre")


@dataclass(frozen=True)
class EmbeddingSettings:
    """How the embeddings are made (pre-registered with D02 amendment 3)."""

    model: str
    revision: str | None = None
    side: int = 224
    centre_fraction: float = 0.60
    batch_size: int = 32

    @classmethod
    def from_config(cls, dataset_config: Mapping[str, Any]) -> EmbeddingSettings:
        """``near_duplicates.embedding`` of the dataset config (``TODO`` revision -> None)."""
        cfg = (dataset_config.get("near_duplicates") or {}).get("embedding") or {}
        revision = cfg.get("revision")
        return cls(
            model=str(cfg.get("model", "")),
            revision=None if revision in (None, "TODO") else str(revision),
            side=int(cfg.get("side", cls.side)),
            centre_fraction=float(cfg.get("centre_fraction", cls.centre_fraction)),
            batch_size=int(cfg.get("batch_size", cls.batch_size)),
        )


class Embedder(Protocol):
    """Maps a batch of normalised views ``(n, 3, side, side)`` to features ``(n, dim)``."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    info: dict[str, Any]

    def __call__(self, batch: np.ndarray) -> np.ndarray: ...


def view_images(image: Any, settings: EmbeddingSettings) -> dict[str, Any]:
    """The two views of one PIL image, RGB, each ``side`` x ``side``."""
    from PIL import Image

    rgb = image.convert("RGB")
    size = (settings.side, settings.side)
    width, height = rgb.size
    frac = settings.centre_fraction
    box = (
        width * (1 - frac) / 2, height * (1 - frac) / 2,
        width * (1 + frac) / 2, height * (1 + frac) / 2,
    )  # fmt: skip
    return {
        "full": rgb.resize(size, Image.Resampling.BICUBIC),
        "centre": rgb.resize(size, Image.Resampling.BICUBIC, box=box),
    }


def normalise(view: Any, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """``(3, side, side)`` float32: pixels scaled to 0-1, minus ``mean``, divided by ``std``."""
    pixels = np.asarray(view, dtype=np.float32) / 255.0
    pixels = (pixels - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1))


def l2_normalise(features: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length (a zero row stays zero)."""
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return np.divide(features, norms, out=np.zeros_like(features), where=norms > 0)


def embed_images(
    paths: Sequence[str | Path],
    embedder: Embedder,
    settings: EmbeddingSettings,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """``{view: (n, dim)}`` L2-normalised float32 embeddings, rows in the order of ``paths``."""
    from PIL import Image

    out: dict[str, list[np.ndarray]] = {view: [] for view in VIEWS}
    for start in range(0, len(paths), settings.batch_size):
        chunk = paths[start : start + settings.batch_size]
        batches: dict[str, list[np.ndarray]] = {view: [] for view in VIEWS}
        for path in chunk:
            with Image.open(path) as image:
                for view, picture in view_images(image, settings).items():
                    batches[view].append(normalise(picture, embedder.mean, embedder.std))
        for view in VIEWS:
            out[view].append(np.asarray(embedder(np.stack(batches[view])), dtype=np.float32))
        if progress is not None:
            progress(min(start + len(chunk), len(paths)), len(paths))
    return {view: l2_normalise(np.concatenate(parts)) for view, parts in out.items()}


# ------------------------------------------------------------------ the hub model (Kaggle)


@dataclass
class HubEmbedder:
    """A vision model from the Hugging Face hub, run with torch (CPU unless a GPU is there).

    ``info`` records what was used: the model id and the revision (the commit hash the hub
    resolved), the licence of the hub record, the feature size, and the processor's mean and
    standard deviation.
    """

    model_id: str
    revision: str | None = None
    device: str = "auto"
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    info: dict[str, Any] = field(default_factory=dict)
    _model: Any = None
    _torch: Any = None

    def load(self) -> HubEmbedder:
        """Resolve the hub record, then load the model at that exact revision."""
        import torch
        from huggingface_hub import model_info
        from transformers import AutoImageProcessor, AutoModel

        record = model_info(self.model_id, revision=self.revision)
        sha = record.sha
        card = getattr(record, "card_data", None)
        licence = getattr(card, "license", None) if card is not None else None
        if not licence:
            tags = [t for t in record.tags or () if t.startswith("license:")]
            licence = tags[0].split(":", 1)[1] if tags else None
        processor = AutoImageProcessor.from_pretrained(self.model_id, revision=sha)
        self.mean = tuple(float(v) for v in processor.image_mean)  # type: ignore[assignment]
        self.std = tuple(float(v) for v in processor.image_std)  # type: ignore[assignment]
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(self.model_id, revision=sha)
        self._model = model.float().to(device).eval()
        self._torch = torch
        self.device = device
        self.info = {
            "model": record.id, "revision": sha, "licence": licence or "not stated on the hub",
            "mean": list(self.mean), "std": list(self.std), "device": device,
            "feature": "pooled output (class token after the final layer norm)",
        }  # fmt: skip
        return self

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            pixels = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
            output = self._model(pixel_values=pixels)
            pooled = getattr(output, "pooler_output", None)
            if pooled is None:
                pooled = output.last_hidden_state[:, 0]
            features = pooled.float().cpu().numpy()
        self.info["dim"] = int(features.shape[1])
        return features


# ------------------------------------------------------------------ a stand-in for CPU tests


@dataclass
class TinyEmbedder:
    """A numpy stand-in for CPU tests and dry runs: a fixed random projection of the view
    shrunk to 16 x 16 grey. Never used for the study (``--embedding-model tiny``)."""

    dim: int = 64
    seed: int = 0
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.25, 0.25, 0.25)
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self._projection = rng.normal(size=(16 * 16, self.dim)).astype(np.float32)
        self.info = {"model": "tiny (test stand-in)", "revision": "none", "licence": "none",
                     "dim": self.dim, "mean": list(self.mean), "std": list(self.std)}  # fmt: skip

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        grey = np.asarray(batch, dtype=np.float32).mean(axis=1)  # (n, side, side)
        n, side, _ = grey.shape
        step = side // 16
        small = grey[:, : step * 16, : step * 16].reshape(n, 16, step, 16, step).mean(axis=(2, 4))
        flat = small.reshape(n, -1)
        flat = flat - flat.mean(axis=1, keepdims=True)
        return flat @ self._projection


def make_embedder(name: str, revision: str | None = None) -> Embedder:
    """``tiny`` (or ``tiny:<dim>``) for tests; anything else is a hub model id."""
    if name == "tiny" or name.startswith("tiny:"):
        dim = int(name.split(":", 1)[1]) if ":" in name else 64
        return TinyEmbedder(dim=dim)
    return HubEmbedder(name, revision).load()


# ------------------------------------------------------------------ similarities


def pair_cosines(embedding: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The cosine of each pair (rows ``a[k]`` and ``b[k]`` of unit-length embeddings)."""
    if len(a) == 0:
        return np.zeros(0, dtype=np.float32)
    return np.einsum("kd,kd->k", embedding[a], embedding[b]).astype(np.float32)


def knn_pairs(embedding: np.ndarray, k: int, *, block: int = 512) -> set[tuple[int, int]]:
    """Every pair (i, j), i < j, in which one image is among the other's ``k`` nearest
    neighbours by cosine (ties broken by index, so the result is deterministic)."""
    n = len(embedding)
    pairs: set[tuple[int, int]] = set()
    k = min(k, n - 1)
    if k <= 0:
        return pairs
    for start in range(0, n, block):
        sims = embedding[start : start + block] @ embedding.T
        rows = np.arange(start, min(start + block, n))
        sims[np.arange(len(rows)), rows] = -np.inf  # not its own neighbour
        order = np.argsort(-sims, axis=1, kind="stable")[:, :k]
        for i, neighbours in zip(rows, order, strict=True):
            for j in neighbours:
                pairs.add((int(min(i, j)), int(max(i, j))))
    return pairs


def pairs_at_or_above(
    embedding: np.ndarray, threshold: float, *, block: int = 512
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All pairs i < j whose cosine is at least ``threshold``: (i, j, cosine), exact."""
    n = len(embedding)
    a_out, b_out, c_out = [], [], []
    for start in range(0, n, block):
        sims = embedding[start : start + block] @ embedding.T
        rows, cols = np.nonzero(sims >= threshold)
        rows = rows + start
        keep = cols > rows
        a_out.append(rows[keep])
        b_out.append(cols[keep])
        c_out.append(sims[rows[keep] - start, cols[keep]])
    if not a_out:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, np.float32)
    return np.concatenate(a_out), np.concatenate(b_out), np.concatenate(c_out).astype(np.float32)


# ------------------------------------------------------------------ the self-check (Kaggle)


def synthetic_variants(seed: int = 7, size: tuple[int, int] = (320, 256)) -> dict[str, Any]:
    """An endoscopy-like synthetic scene, four copies of it (shifted by about 4 %, blurred,
    rescaled and cropped, brightened) and three different scenes, as PIL images."""
    from PIL import Image, ImageFilter

    def scene(s: int) -> Any:
        rng = np.random.default_rng(s)
        small = rng.integers(0, 255, size=(14, 16, 3), dtype=np.uint8)
        big = (size[0] + 40, size[1] + 40)
        base = np.asarray(Image.fromarray(small).resize(big, Image.Resampling.BICUBIC), float)
        yy, xx = np.mgrid[0 : big[1], 0 : big[0]]
        fall = 0.4 + 1.2 * np.hypot(xx - big[0] / 2, yy - big[1] / 2) / max(big)
        return Image.fromarray(np.clip(base * fall[..., None], 0, 255).astype(np.uint8))

    original = scene(seed)
    view = original.crop((20, 20, 20 + size[0], 20 + size[1]))
    return {
        "original": view,
        "copy:shifted": original.crop((33, 26, 33 + size[0], 26 + size[1])),
        "copy:blurred": view.filter(ImageFilter.GaussianBlur(2)),
        "copy:rescaled": original.crop((10, 10, 30 + size[0], 30 + size[1])).resize(
            (size[0] // 2, size[1] // 2), Image.Resampling.BILINEAR
        ),
        "copy:brightened": view.point(lambda v: min(255, int(v * 1.15) + 10)),
        **{
            f"different:{k}": scene(seed + 100 + k).crop((20, 20, 20 + size[0], 20 + size[1]))
            for k in range(3)
        },
    }


def self_check(embedder: Embedder, settings: EmbeddingSettings) -> dict[str, dict[str, float]]:
    """``{view: {variant: cosine with the original}}`` on :func:`synthetic_variants`."""
    images = synthetic_variants()
    out: dict[str, dict[str, float]] = {}
    for view in VIEWS:
        batch = np.stack(
            [normalise(view_images(img, settings)[view], embedder.mean, embedder.std)
             for img in images.values()]
        )  # fmt: skip
        features = l2_normalise(np.asarray(embedder(batch)))
        out[view] = {
            name: float(features[0] @ features[i])
            for i, name in enumerate(images)
            if name != "original"
        }
    return out


def self_check_passes(result: Mapping[str, Mapping[str, float]]) -> bool:
    """Every copy is nearer to the original than every different scene, in both views."""
    for scores in result.values():
        copies = [v for k, v in scores.items() if k.startswith("copy:")]
        others = [v for k, v in scores.items() if k.startswith("different:")]
        if min(copies) <= max(others):
            return False
    return True


# ------------------------------------------------------------------ saving (private)


def save_embeddings(
    path: Path, images: Sequence[str], embeddings: Mapping[str, np.ndarray], info: Mapping
) -> Path:
    """Write the embeddings and what made them (private: image names inside)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, images=np.array(list(images)), **{f"view_{v}": e for v, e in embeddings.items()})
    path.with_suffix(".json").write_text(json.dumps(dict(info), indent=1, default=str))
    return path


def load_embeddings(
    path: Path, images: Sequence[str], settings: EmbeddingSettings
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read saved embeddings; they must cover ``images`` in that order and fit ``settings``."""
    path = Path(path)
    info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as data:
        saved = [str(x) for x in data["images"]]
        if saved != list(images):
            raise ValueError(f"{path} was made for other images ({len(saved)} saved)")
        embeddings = {v: data[f"view_{v}"] for v in VIEWS}
    wanted = asdict(settings)
    recorded = info.get("settings") or {}
    for key in ("side", "centre_fraction"):
        if key in recorded and recorded[key] != wanted[key]:
            raise ValueError(f"{path}: {key} {recorded[key]} != {wanted[key]}")
    if settings.revision and info.get("revision") != settings.revision:
        raise ValueError(f"{path}: revision {info.get('revision')} != {settings.revision}")
    return embeddings, info
