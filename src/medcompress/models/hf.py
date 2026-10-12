"""Hugging Face implementation of :class:`VLMWrapper` (chat-template vision-language models).

One generic path serves the teacher family and the student candidates: the chat
template builds the prompt, the processor turns images and text into tensors, and the
language model is the single decoder stack found by
``compression/pruning/surgery.py``. Padding is on the left, so the last position of
every row is the answer position. Heavy libraries are imported inside the methods.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from medcompress.models.base import GenerationOutput, VLMWrapper, YesNoScore
from medcompress.models.tokens import fingerprint_tokenizer, yes_no_token_ids

if TYPE_CHECKING:
    from medcompress.compression.pruning.units import PrunableUnit
    from medcompress.models.loading import ModelSpec


def _to_pil(image: Path | str | Any, longest_side: int | None) -> Any:
    """Open or copy an image as RGB; shrink it so its longest side is at most ``longest_side``."""
    from PIL import Image

    if isinstance(image, str | Path):
        with Image.open(image) as handle:
            pil = handle.convert("RGB")
    else:
        pil = image.convert("RGB")
    if longest_side is not None and max(pil.size) > longest_side:
        scale = longest_side / max(pil.size)
        size = (max(1, round(pil.width * scale)), max(1, round(pil.height * scale)))
        pil = pil.resize(size, Image.BICUBIC)
    return pil


class HFVLMWrapper(VLMWrapper):
    """Wrap a transformers vision-language model and its processor.

    Attributes:
        module: the underlying ``PreTrainedModel`` (replaced when an adapter is merged).
        processor: the matching processor (tokenizer, image processor, chat template).
    """

    def __init__(
        self,
        module: Any,
        processor: Any,
        spec: ModelSpec | None = None,
        *,
        image_longest_side: int | None = None,
        batch_size: int = 8,
        mlp_channel_group_size: int = 128,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.module = module
        self.processor = processor
        self.spec = spec
        self.image_longest_side = image_longest_side
        self.batch_size = batch_size
        self.mlp_channel_group_size = mlp_channel_group_size
        self.tokenizer = processor.tokenizer
        self.tokenizer.padding_side = "left"
        self.yes_ids, self.no_ids = yes_no_token_ids(self.tokenizer)

    # ------------------------------------------------------------------ inputs

    def build_inputs(self, images: Sequence[Path | Any], questions: Sequence[str]) -> Any:
        """Tokenise ``questions`` with their ``images`` (one image per question)."""
        if len(images) != len(questions):
            raise ValueError("need exactly one image per question")
        texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for q in questions
        ]
        pil = [_to_pil(image, self.image_longest_side) for image in images]
        inputs = self.processor(text=texts, images=pil, padding=True, return_tensors="pt")
        return inputs.to(self.module.device)

    def _batches(self, n: int) -> list[range]:
        return [range(i, min(i + self.batch_size, n)) for i in range(0, n, self.batch_size)]

    # ---------------------------------------------------------------- scoring

    def score_yes_no(
        self, images: Sequence[Path | Any], questions: Sequence[str]
    ) -> list[YesNoScore]:
        """Log-probabilities of the yes and no answers at the answer position.

        Each value is the log of the summed probability of the single-token spellings
        of that answer, under the full vocabulary (the two values need not add to one).
        """
        import torch

        scores: list[YesNoScore] = []
        for rows in self._batches(len(questions)):
            inputs = self.build_inputs([images[i] for i in rows], [questions[i] for i in rows])
            with torch.inference_mode():
                logits = self.module(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :]
                logp = torch.log_softmax(logits.float(), dim=-1)
                yes = torch.logsumexp(logp[:, list(self.yes_ids)], dim=-1)
                no = torch.logsumexp(logp[:, list(self.no_ids)], dim=-1)
            scores.extend(YesNoScore(float(y), float(n)) for y, n in zip(yes, no, strict=True))
        return scores

    # -------------------------------------------------------------- generation

    def _stop_ids(self) -> set[int]:
        config = getattr(self.module, "generation_config", None)
        eos = getattr(config, "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        if eos is None:
            return set()
        return {int(e) for e in (eos if isinstance(eos, list | tuple) else [eos])}

    def generate(
        self,
        images: Sequence[Path | Any],
        questions: Sequence[str],
        *,
        max_new_tokens: int,
        return_topk: int | None = None,
    ) -> list[GenerationOutput]:
        """Greedy answers. With ``return_topk`` the top-k log-probabilities of every
        generated token (before any logits processor) are returned as well; the stop
        token is kept in ``token_ids`` so a student can learn to stop."""
        import torch

        stop = self._stop_ids()
        pad_id = self.tokenizer.pad_token_id
        outputs: list[GenerationOutput] = []
        for rows in self._batches(len(questions)):
            inputs = self.build_inputs([images[i] for i in rows], [questions[i] for i in rows])
            prompt_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                result = self.module.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_logits=return_topk is not None,
                    pad_token_id=pad_id,
                )
                new_tokens = result.sequences[:, prompt_len:]
                for b in range(new_tokens.shape[0]):
                    ids = [int(t) for t in new_tokens[b]]
                    end = next((k for k, t in enumerate(ids) if t in stop), None)
                    kept = ids if end is None else ids[: end + 1]
                    text_ids = [t for t in kept if t not in stop]
                    topk_ids = topk_lp = None
                    if return_topk is not None:
                        step_logits = torch.stack(
                            [result.logits[k][b] for k in range(len(kept))]
                        ).float()
                        top = torch.log_softmax(step_logits, dim=-1).topk(return_topk, dim=-1)
                        topk_ids = [[int(i) for i in row] for row in top.indices]
                        topk_lp = [[float(v) for v in row] for row in top.values]
                    outputs.append(
                        GenerationOutput(
                            text=self.tokenizer.decode(text_ids, skip_special_tokens=True).strip(),
                            token_ids=kept,
                            topk_token_ids=topk_ids,
                            topk_logprobs=topk_lp,
                        )
                    )
        return outputs

    # -------------------------------------------------------- units and sizes

    def list_prunable_units(self, *, include_vision: bool = False) -> list[PrunableUnit]:
        from medcompress.compression.pruning.units import UnitType, enumerate_units

        return enumerate_units(
            self.module,
            [UnitType.DECODER_LAYER, UnitType.ATTENTION_HEAD, UnitType.MLP_CHANNEL_GROUP],
            include_vision=include_vision,
            mlp_channel_group_size=self.mlp_channel_group_size,
        )

    def num_parameters(self) -> int:
        from medcompress.compression.pruning.surgery import count_parameters

        return count_parameters(self.module)

    def tokenizer_fingerprint(self) -> str:
        return fingerprint_tokenizer(self.tokenizer)
