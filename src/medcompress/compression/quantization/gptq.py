"""GPTQ: rounding a weight matrix to the integer grid using calibration inputs.

Reference: Frantar, Ashkboos, Hoefler, Alistarh, *GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers*, arXiv:2210.17323. For one linear layer with weights ``W``
(``out x in``) and calibration inputs ``X`` (``tokens x in``), GPTQ rounds the columns of ``W``
one after another and spreads the rounding error of each column over the columns not yet
rounded, so that the layer's outputs ``X W^T`` change as little as possible. It needs only the
Hessian ``H = 2 X^T X / n`` of the layer's inputs.

This module holds the algorithm twice with the same steps: :func:`gptq_numpy` (float64, the
reference that the tests check here) and :func:`gptq_torch` (the same steps on the GPU, checked
against the reference by the torch tests). Settings follow the reference implementation:
dampening of 1 % of the mean diagonal, blocks of 128 columns, no reordering of the columns
("act order" off), symmetric grid (``grid.py``). The scale of a group is taken when the
algorithm reaches the group's first column, from the weights as updated so far.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from medcompress.compression.quantization import grid

DEFAULT_DAMPING = 0.01
DEFAULT_BLOCK_SIZE = 128


@dataclass(frozen=True)
class GPTQResult:
    """Codes ``(out, in)``, scales ``(out, n_groups)`` and the layer's output error."""

    codes: np.ndarray
    scales: np.ndarray
    bits: int
    group_size: int
    output_error: float  # sum over columns of squared rounding error weighted by the Hessian

    def dequantized(self) -> np.ndarray:
        """The rounded weights ``(q - z) * s``."""
        return grid.dequantize(self.codes, self.scales, self.bits, self.group_size)


class HessianAccumulator:
    """Running ``H = 2 / n * sum_t x_t x_t^T`` over calibration tokens (NumPy, float64).

    Mirrors the update of the reference implementation: each batch rescales the old sum, so
    the result does not depend on how the tokens were split into batches.
    """

    def __init__(self, in_features: int) -> None:
        self.hessian = np.zeros((in_features, in_features), dtype=np.float64)
        self.tokens = 0

    def add(self, inputs: np.ndarray) -> None:
        """Add a ``(tokens, in_features)`` block of layer inputs (padding already removed)."""
        x = np.asarray(inputs, dtype=np.float64).reshape(-1, self.hessian.shape[0])
        n_new = x.shape[0]
        if n_new == 0:
            return
        total = self.tokens + n_new
        self.hessian *= self.tokens / total
        x = np.sqrt(2.0 / total) * x
        self.hessian += x.T @ x
        self.tokens = total


def _prepare_hessian(
    weight: np.ndarray, hessian: np.ndarray, damping: float
) -> tuple[np.ndarray, np.ndarray]:
    """Dead inputs (never active) get a unit diagonal and zero weights; add the dampening."""
    h = np.array(hessian, dtype=np.float64, copy=True)
    w = np.array(weight, dtype=np.float64, copy=True)
    dead = np.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    h[np.diag_indices_from(h)] += damping * float(np.mean(np.diag(h)))
    return w, h


def _upper_cholesky_of_inverse(h: np.ndarray) -> np.ndarray:
    """Upper-triangular ``U`` with ``U^T U = H^-1`` (the ``Hinv`` of the reference code)."""
    lower = np.linalg.cholesky(h)
    inverse = np.linalg.inv(lower)
    h_inv = inverse.T @ inverse
    return np.linalg.cholesky(h_inv).T


def gptq_numpy(
    weight: np.ndarray,
    hessian: np.ndarray,
    *,
    bits: int,
    group_size: int,
    damping: float = DEFAULT_DAMPING,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> GPTQResult:
    """Round ``weight`` with GPTQ (reference implementation in float64).

    Args:
        weight: ``(out_features, in_features)``.
        hessian: ``(in_features, in_features)``, e.g. from :class:`HessianAccumulator`.
        bits: 4 or 8. group_size: columns per scale; must divide ``in_features``.
    """
    w, h = _prepare_hessian(weight, hessian, damping)
    rows, columns = w.shape
    groups = grid.check_shape(columns, bits, group_size)
    h_inv = _upper_cholesky_of_inverse(h)
    zero, top = grid.zero_point(bits), grid.max_code(bits)
    codes = np.zeros((rows, columns), dtype=np.uint8)
    scales = np.ones((rows, groups), dtype=np.float64)
    loss = 0.0
    for start in range(0, columns, block_size):
        stop = min(start + block_size, columns)
        block = w[:, start:stop].copy()
        errors = np.zeros_like(block)
        h_block = h_inv[start:stop, start:stop]
        for i in range(stop - start):
            column = start + i
            if column % group_size == 0:
                g = column // group_size
                scales[:, g] = grid.symmetric_scales(
                    w[:, column : column + group_size], bits, group_size
                )[:, 0]
            s = scales[:, column // group_size]
            q = np.clip(np.rint(block[:, i] / s) + zero, 0, top)
            codes[:, column] = q.astype(np.uint8)
            rounded = (q - zero) * s
            d = h_block[i, i]
            err = (block[:, i] - rounded) / d
            loss += float(np.sum((block[:, i] - rounded) ** 2) / d**2) / 2
            block[:, i:] -= np.outer(err, h_block[i, i:])
            errors[:, i] = err
        w[:, stop:] -= errors @ h_inv[start:stop, stop:]
    return GPTQResult(codes, scales, bits, group_size, loss)


def gptq_torch(
    weight: Any,
    hessian: Any,
    *,
    bits: int,
    group_size: int,
    damping: float = DEFAULT_DAMPING,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> GPTQResult:
    """The same algorithm as :func:`gptq_numpy` with torch tensors (float32 on their device).

    ``weight`` and ``hessian`` are torch tensors; the result holds NumPy arrays. Used by the
    quantization stage on the GPU (E3, E8, E9); the torch tests compare it with the reference.
    """
    import torch

    w = weight.detach().to(torch.float32).clone()
    h = hessian.detach().to(device=w.device, dtype=torch.float32).clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    h += damping * torch.mean(torch.diag(h)) * torch.eye(h.shape[0], device=h.device)
    lower = torch.linalg.cholesky(h)
    h_inv = torch.linalg.cholesky(torch.cholesky_inverse(lower), upper=True)
    rows, columns = w.shape
    groups = grid.check_shape(columns, bits, group_size)
    zero, top = grid.zero_point(bits), grid.max_code(bits)
    codes = torch.zeros((rows, columns), dtype=torch.uint8, device=w.device)
    scales = torch.ones((rows, groups), dtype=torch.float32, device=w.device)
    loss = torch.zeros((), dtype=torch.float32, device=w.device)
    for start in range(0, columns, block_size):
        stop = min(start + block_size, columns)
        block = w[:, start:stop].clone()
        errors = torch.zeros_like(block)
        h_block = h_inv[start:stop, start:stop]
        for i in range(stop - start):
            column = start + i
            if column % group_size == 0:
                absmax = w[:, column : column + group_size].abs().amax(dim=1)
                scale = 2.0 * absmax / top
                scales[:, column // group_size] = torch.where(
                    scale == 0, torch.ones_like(scale), scale
                )
            s = scales[:, column // group_size]
            q = torch.clamp(torch.round(block[:, i] / s) + zero, 0, top)
            codes[:, column] = q.to(torch.uint8)
            rounded = (q - zero) * s
            d = h_block[i, i]
            err = (block[:, i] - rounded) / d
            loss += torch.sum((block[:, i] - rounded) ** 2) / d**2 / 2
            block[:, i:] -= err[:, None] * h_block[i, i:][None, :]
            errors[:, i] = err
        w[:, stop:] -= errors @ h_inv[start:stop, stop:]
    return GPTQResult(
        codes.cpu().numpy(), scales.double().cpu().numpy(), bits, group_size, float(loss)
    )


def output_error(weight: np.ndarray, rounded: np.ndarray, hessian: np.ndarray) -> float:
    """``trace((W - Q) H (W - Q)^T) / 2``: the squared change of the layer's outputs on the
    calibration inputs (up to a constant), the quantity GPTQ minimises."""
    diff = np.asarray(weight, dtype=np.float64) - np.asarray(rounded, dtype=np.float64)
    return float(np.einsum("ij,jk,ik->", diff, np.asarray(hessian, dtype=np.float64), diff) / 2)
