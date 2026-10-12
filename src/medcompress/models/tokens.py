"""Tokenizer helpers that need no torch: yes/no answer tokens and tokenizer fingerprints.

The yes/no probe is scored from the log-probabilities of the first answer token.
Qwen-family tokenizers hold ``Yes``, ``yes``, ``No`` and ``no`` as single tokens, and
the score of an answer is the log of the total probability of its spellings, so the
result does not depend on which spelling the model prefers. A spelling that is
several tokens long is skipped; if none is left for an answer the model cannot be
scored this way and loading fails loudly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

YES_SPELLINGS: tuple[str, ...] = ("Yes", "yes", "YES", " Yes", " yes")
NO_SPELLINGS: tuple[str, ...] = ("No", "no", "NO", " No", " no")


def single_token_ids(
    encode: Callable[[str], Sequence[int]], spellings: Sequence[str]
) -> tuple[int, ...]:
    """Return the distinct token ids of the spellings that encode to exactly one token."""
    found: list[int] = []
    for spelling in spellings:
        ids = list(encode(spelling))
        if len(ids) == 1 and ids[0] not in found:
            found.append(int(ids[0]))
    return tuple(found)


def yes_no_token_ids(tokenizer: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(yes_ids, no_ids)`` for a Hugging Face tokenizer.

    Raises:
        ValueError: if either answer has no single-token spelling.
    """

    unknown = getattr(tokenizer, "unk_token_id", None)

    def encode(text: str) -> list[int]:
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        # a spelling that falls back to the unknown token is not a real spelling
        return [] if unknown is not None and unknown in ids else ids

    yes_ids = single_token_ids(encode, YES_SPELLINGS)
    no_ids = single_token_ids(encode, NO_SPELLINGS)
    if not yes_ids or not no_ids:
        raise ValueError(
            "no single-token spelling of 'yes' or 'no' in this tokenizer; "
            "yes/no probes cannot be scored from one forward pass"
        )
    if set(yes_ids) & set(no_ids):
        raise ValueError("a token id is shared between the yes and no spellings")
    return yes_ids, no_ids


def fingerprint_vocabulary(vocab: Mapping[str, int], special_tokens: Sequence[str]) -> str:
    """Hash a vocabulary (token string to id) together with the special tokens."""
    payload = json.dumps(
        {"vocab": sorted(vocab.items()), "special": sorted(special_tokens)},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return "vocab:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_tokenizer(tokenizer: Any) -> str:
    """Return a stable hash of everything that decides how text becomes token ids.

    A fast tokenizer is hashed through its serialisation (vocabulary, merges,
    normaliser, pre-tokenizer, added tokens), without its padding and truncation
    settings. Otherwise the vocabulary and special
    tokens are hashed. Both start with a prefix naming the method, so hashes of
    different kinds never compare equal by accident.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        # padding and truncation are switched on by the first batched call, so they
        # are not part of the tokenization itself and must not change the hash
        state = json.loads(backend.to_str())
        state.pop("padding", None)
        state.pop("truncation", None)
        canonical = json.dumps(state, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return "fast:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return fingerprint_vocabulary(tokenizer.get_vocab(), list(tokenizer.all_special_tokens))
