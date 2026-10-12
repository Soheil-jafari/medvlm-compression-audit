"""The integer grid of group-wise symmetric weight quantization and its storage (NumPy only).

A weight matrix ``W`` of shape ``(out_features, in_features)`` is split along the input
dimension into groups of ``group_size`` columns. Every (row, group) pair gets one scale ``s``;
a weight is stored as an unsigned code ``q`` in ``0 .. 2**bits - 1`` and read back as
``(q - z) * s`` with the fixed zero point ``z = 2**(bits - 1)``. This is the symmetric
convention of the GPTQ reference implementation (Frantar et al., arXiv:2210.17323): the scale
of a group is ``2 * max|w| / (2**bits - 1)``, so the codes run from ``-z * s`` to
``(2**bits - 1 - z) * s``.

The functions here are the single definition of that grid. The GPTQ rounding
(``gptq.py``), the quantization-aware training of E4 (``qakd.py``) and the checkpoint format
(``checkpoint.py``) all use them, and the kernels of the inference library are checked
against :func:`dequantize` before a model is reported as quantized.
"""

from __future__ import annotations

import numpy as np

SUPPORTED_BITS: tuple[int, ...] = (4, 8)


def check_shape(in_features: int, bits: int, group_size: int) -> int:
    """Return the number of groups; raise for an unsupported bit width or a ragged split."""
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")
    if group_size < 1 or in_features % group_size:
        raise ValueError(f"in_features {in_features} is not a multiple of group_size {group_size}")
    return in_features // group_size


def zero_point(bits: int) -> int:
    """The fixed zero point of the symmetric grid, ``2**(bits - 1)``."""
    return 1 << (bits - 1)


def max_code(bits: int) -> int:
    """The largest code, ``2**bits - 1``."""
    return (1 << bits) - 1


def symmetric_scales(weight: np.ndarray, bits: int, group_size: int) -> np.ndarray:
    """Scales ``(out_features, n_groups)``: ``2 * max|w| / (2**bits - 1)`` per row and group.

    A group of zeros gets the scale 1 (any scale reproduces it exactly).
    """
    w = np.asarray(weight, dtype=np.float64)
    out_features, in_features = w.shape
    groups = check_shape(in_features, bits, group_size)
    absmax = np.abs(w).reshape(out_features, groups, group_size).max(axis=2)
    scales = 2.0 * absmax / max_code(bits)
    scales[scales == 0] = 1.0
    return scales


def expand_groups(per_group: np.ndarray, group_size: int) -> np.ndarray:
    """Repeat a ``(out, n_groups)`` array over the columns of each group: ``(out, in)``."""
    return np.repeat(np.asarray(per_group), group_size, axis=1)


def quantize_codes(
    weight: np.ndarray, scales: np.ndarray, bits: int, group_size: int
) -> np.ndarray:
    """Codes ``clamp(round(w / s) + z, 0, 2**bits - 1)`` as ``uint8``, shape ``(out, in)``.

    Rounding is half to even (NumPy and PyTorch both round this way).
    """
    w = np.asarray(weight, dtype=np.float64)
    check_shape(w.shape[1], bits, group_size)
    s = expand_groups(scales, group_size)
    codes = np.rint(w / s) + zero_point(bits)
    return np.clip(codes, 0, max_code(bits)).astype(np.uint8)


def dequantize(codes: np.ndarray, scales: np.ndarray, bits: int, group_size: int) -> np.ndarray:
    """Weights ``(q - z) * s`` (float64), shape ``(out, in)``."""
    q = np.asarray(codes, dtype=np.float64)
    check_shape(q.shape[1], bits, group_size)
    return (q - zero_point(bits)) * expand_groups(scales, group_size)


def round_to_nearest(
    weight: np.ndarray, bits: int, group_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Plain rounding (no calibration): ``(codes, scales)``. The baseline GPTQ must beat."""
    scales = symmetric_scales(weight, bits, group_size)
    return quantize_codes(weight, scales, bits, group_size), scales


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """Pack codes ``(out, in)`` into ``int32`` words along the input dimension.

    Word ``k`` of output row ``r`` holds the codes of columns ``k * n .. k * n + n - 1``
    with ``n = 32 / bits``, the first column in the lowest bits. Shape
    ``(in * bits / 32, out)``, the layout of GPTQ's ``qweight``.
    """
    q = np.asarray(codes, dtype=np.uint32)
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")
    per_word = 32 // bits
    out_features, in_features = q.shape
    if in_features % per_word:
        raise ValueError(f"in_features {in_features} is not a multiple of {per_word}")
    if q.size and int(q.max()) > max_code(bits):
        raise ValueError("a code does not fit in the bit width")
    words = q.T.reshape(in_features // per_word, per_word, out_features)
    packed = np.zeros((in_features // per_word, out_features), dtype=np.uint32)
    for j in range(per_word):
        packed |= words[:, j, :] << np.uint32(bits * j)
    return packed.view(np.int32)


def unpack_codes(packed: np.ndarray, bits: int, in_features: int) -> np.ndarray:
    """Inverse of :func:`pack_codes`: codes ``(out, in)`` as ``uint8``."""
    words = np.asarray(packed).view(np.uint32)
    per_word = 32 // bits
    if words.shape[0] * per_word != in_features:
        raise ValueError("packed shape does not match in_features")
    mask = np.uint32(max_code(bits))
    columns = [(words >> np.uint32(bits * j)) & mask for j in range(per_word)]
    stacked = np.stack(columns, axis=1).reshape(in_features, words.shape[1])
    return stacked.T.astype(np.uint8)


def packed_bytes(out_features: int, in_features: int, bits: int, group_size: int) -> int:
    """Bytes of one quantized matrix: packed codes plus 16-bit scales (no zero points stored)."""
    groups = check_shape(in_features, bits, group_size)
    return out_features * in_features * bits // 8 + out_features * groups * 2
