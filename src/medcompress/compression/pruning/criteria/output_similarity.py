"""Output-similarity criterion: how much do the generated answers change without a layer."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from medcompress.compression.pruning import scoring
from medcompress.compression.pruning.criteria.base import PruningCriterion, register_criterion
from medcompress.compression.pruning.units import UnitType
from medcompress.config import ConfigError, find_todos
from medcompress.data.summary import resolve_max_new_tokens

if TYPE_CHECKING:
    import numpy as np

    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.data.base import VQASample
    from medcompress.models.base import VLMWrapper

Embedder = Callable[[Sequence[str]], "np.ndarray"]
"""Maps texts to unit-length sentence vectors, one row per text."""


class SentenceEmbedder:
    """Mean-pooled, length-normalised sentence vectors from a Hugging Face encoder.

    This is the pooling of the ``all-MiniLM-L6-v2`` sentence encoder used by the method this
    criterion follows. The encoder is pinned by ``revision`` and runs on the CPU.
    """

    def __init__(self, source: str, revision: str) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
        self.model = AutoModel.from_pretrained(source, revision=revision).eval()

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-length vectors for ``texts`` (an empty text gets a valid vector too)."""
        import torch

        out = []
        with torch.inference_mode():
            for start in range(0, len(texts), 64):
                batch = self.tokenizer(
                    list(texts[start : start + 64]),
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                out.append(torch.nn.functional.normalize(pooled, dim=-1))
        return torch.cat(out).numpy()


def protected_layer_indices(n_layers: int, first: int, last: int) -> list[int]:
    """Indices of the first ``first`` and last ``last`` layers, which the method keeps."""
    if first < 0 or last < 0 or first + last >= n_layers:
        raise ValueError("protected layers must leave at least one removable layer")
    return [*range(first), *range(n_layers - last, n_layers)]


@register_criterion("output_similarity")
class OutputSimilarityCriterion(PruningCriterion):
    """Layer importance as one minus the similarity of the answers with and without the layer.

    Follows Khan et al. (arXiv:2507.21976, section 3.2): each decoder layer is ablated on its
    own, the model answers the calibration questions, the answers with and without the layer
    are embedded with a sentence encoder and compared by cosine similarity, and the mean
    similarity over the calibration set is the layer's (un)importance. High similarity means
    the layer can go, so the importance returned here is ``1 - mean similarity``. Used by: E6.
    """

    def __init__(self, config: Any, *, seed: int = 0, embedder: Embedder | None = None) -> None:
        super().__init__(config, seed=seed)
        self._embedder = embedder

    @classmethod
    def check_settings(cls, config: Mapping[str, Any]) -> None:
        """The encoder (source and revision) and the generation length must be set."""
        embedder = ((config.get("pruning") or {}).get("output_similarity") or {}).get(
            "embedder"
        ) or {}
        wanted = {
            **{
                f"pruning.output_similarity.embedder.{k}": embedder.get(k)
                for k in ("source", "revision")
            },
            "evaluation.max_new_tokens": resolve_max_new_tokens(
                (config.get("evaluation") or {}).get("max_new_tokens")
            ),
        }
        open_keys = [k for k, v in wanted.items() if v is None or find_todos({"v": v})]
        if open_keys:
            raise ConfigError(f"settings still open: {', '.join(open_keys)}")

    def _embed(self) -> Embedder:
        if self._embedder is None:
            self.check_settings(self.config)
            settings = self.config["pruning"]["output_similarity"]["embedder"]
            self._embedder = SentenceEmbedder(settings["source"], settings["revision"])
        return self._embedder

    def score(
        self,
        model: VLMWrapper,
        units: Sequence[PrunableUnit],
        calibration: Sequence[VQASample],
    ) -> dict[str, float]:
        """Return ``{layer_id: 1 - mean cosine similarity of the answers}`` for whole layers."""
        import numpy as np

        if any(u.unit_type is not UnitType.DECODER_LAYER for u in units):
            raise ValueError("output similarity scores whole decoder layers only")
        if not calibration:
            raise ValueError("the output-similarity criterion needs calibration samples")
        evaluation = self.config.get("evaluation") or {}
        max_new = resolve_max_new_tokens(evaluation.get("max_new_tokens"))
        if max_new is None or find_todos({"v": max_new}):
            raise ConfigError("settings still open: evaluation.max_new_tokens")
        embed = self._embed()
        images = [self.data_root / Path(s.image_path) for s in calibration]
        questions = [s.question for s in calibration]
        reference = embed(
            [o.text for o in model.generate(images, questions, max_new_tokens=int(max_new))]
        )
        out: dict[str, float] = {}
        for unit in units:
            with scoring.ablate(model, unit):
                answers = model.generate(images, questions, max_new_tokens=int(max_new))
            similarity = (np.asarray(embed([o.text for o in answers])) * reference).sum(axis=1)
            out[unit.unit_id] = float(1.0 - similarity.mean())
        return out
