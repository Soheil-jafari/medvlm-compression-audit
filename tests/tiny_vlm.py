"""A tiny offline Qwen2.5-VL (random weights), word-level tokenizer and real processor classes.

Shared by the heavy tests. Needs torch, transformers, torchvision, tokenizers and
Pillow; import it only after the ``importorskip`` calls of the test module.
"""

from __future__ import annotations

import torch
import transformers
from PIL import Image as PIL_Image

SPECIALS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
]
WORDS = (
    "Is there a polyp in this image ? Answer yes or no . Yes No yes no What do you see "
    "user assistant system"
).split()
TEMPLATE = (
    "{% for m in messages %}<|im_start|> {{ m['role'] }} "
    "{% for c in m['content'] %}"
    "{% if c['type'] == 'image' %}<|vision_start|> <|image_pad|> <|vision_end|> "
    "{% elif c['type'] == 'text' %}{{ c['text'] }} {% endif %}{% endfor %}"
    "<|im_end|> {% endfor %}{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
)
PROBE = "Is there a polyp in this image ? Answer yes or no ."


def tiny_processor(extra_words: tuple[str, ...] = ()) -> object:
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2_5_VLProcessor,
        Qwen2VLImageProcessor,
        Qwen2VLVideoProcessor,
    )

    vocab = {t: i for i, t in enumerate(["<unk>", *SPECIALS, *WORDS, *extra_words])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIALS[1:]}, replace_extra_special_tokens=False
    )
    return Qwen2_5_VLProcessor(
        image_processor=Qwen2VLImageProcessor(min_pixels=28 * 28 * 4, max_pixels=28 * 28 * 16),
        tokenizer=tokenizer,
        video_processor=Qwen2VLVideoProcessor(),
        chat_template=TEMPLATE,
    )


def tiny_model(processor: object, seed: int = 0) -> object:
    torch.manual_seed(seed)
    ids = processor.tokenizer.convert_tokens_to_ids
    text = dict(
        vocab_size=48,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
    )
    config = transformers.Qwen2_5_VLConfig(
        text_config=text,
        vision_config=dict(
            depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32
        ),
        image_token_id=ids("<|image_pad|>"),
        video_token_id=ids("<|video_pad|>"),
        vision_start_token_id=ids("<|vision_start|>"),
        vision_end_token_id=ids("<|vision_end|>"),
    )
    model = transformers.Qwen2_5_VLForConditionalGeneration(config).eval()
    model.generation_config.eos_token_id = processor.tokenizer.eos_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    return model


def tiny_image(seed: int, size: tuple[int, int] = (84, 56)) -> object:
    gen = torch.Generator().manual_seed(seed)
    array = (torch.rand(size[1], size[0], 3, generator=gen) * 255).to(torch.uint8).numpy()
    return PIL_Image.fromarray(array)


def tiny_wrapper(seed: int = 0, *, widen_logits: bool = True) -> object:
    """A wrapper around a fresh tiny model; ``widen_logits`` scales the output layer so that
    training can lower the loss (random logits are almost flat)."""
    from medcompress.models.hf import HFVLMWrapper

    processor = tiny_processor()
    model = tiny_model(processor, seed)
    if widen_logits:
        with torch.no_grad():
            model.lm_head.weight.mul_(20.0)
    return HFVLMWrapper(model, processor, batch_size=2, mlp_channel_group_size=16)
