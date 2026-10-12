"""E4 references in NumPy: the confidence gate, decoupled distillation and the LSQ quantizer."""

from __future__ import annotations

import math

import numpy as np
import pytest

from medcompress.compression.quantization import grid
from medcompress.compression.quantization.qakd import (
    confidence_gate_numpy,
    dkd_per_token_numpy,
    gated_dkd_numpy,
    lsq_reference,
)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))


def test_the_gate_is_one_for_a_certain_teacher_and_lower_for_an_unsure_one() -> None:
    sure = np.array([[0.0, -1e9, -1e9]])  # all mass on one token
    unsure = np.log(np.array([[0.4, 0.3, 0.2]]))  # 0.1 outside the top 3
    gate_sure = confidence_gate_numpy(sure, vocab_size=1000)
    gate_unsure = confidence_gate_numpy(unsure, vocab_size=1000)
    assert gate_sure[0] == pytest.approx(1.0)
    assert math.exp(-1) <= gate_unsure[0] < gate_sure[0]
    # the tail is one extra outcome: entropy of (0.4, 0.3, 0.2, 0.1)
    p = np.array([0.4, 0.3, 0.2, 0.1])
    entropy = float(-(p * np.log(p)).sum())
    assert gate_unsure[0] == pytest.approx(math.exp(-entropy / math.log(1000)))


def test_decoupled_distillation_adds_up_to_the_kl_divergence() -> None:
    # Zhao et al.: KL(t || s) = TCKD + (1 - p_target^t) * NCKD, at temperature 1
    rng = np.random.default_rng(0)
    s, t = rng.normal(size=(5, 6)), _log_softmax(rng.normal(size=(5, 6)))
    target = rng.integers(0, 6, size=5)
    tckd = dkd_per_token_numpy(s, t, target, alpha=1.0, beta=0.0, temperature=1.0)
    nckd = dkd_per_token_numpy(s, t, target, alpha=0.0, beta=1.0, temperature=1.0)
    log_ps, log_pt = _log_softmax(s), _log_softmax(t)
    kl = (np.exp(log_pt) * (log_pt - log_ps)).sum(-1)
    pt_target = np.exp(np.take_along_axis(log_pt, target[:, None], -1))[:, 0]
    assert np.allclose(kl, tckd + (1 - pt_target) * nckd, atol=1e-6)
    same = dkd_per_token_numpy(t, t, target)
    assert np.allclose(same, 0.0, atol=1e-6)


def test_the_gated_loss_ignores_masked_positions_and_targets_outside_the_top_k() -> None:
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(1, 3, 10))
    ids = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    logprobs = np.log(np.full((1, 3, 3), 0.3))
    targets = np.array([[1, 0, 7]])  # position 1: token 0 is not cached -> left out
    full = gated_dkd_numpy(logits, ids, logprobs, targets, np.array([[1, 1, 1]]))
    only_first = gated_dkd_numpy(logits, ids, logprobs, targets, np.array([[1, 0, 0]]))
    per_token = dkd_per_token_numpy(
        np.take_along_axis(logits, ids, -1)[0], logprobs[0], np.array([0, 0, 0])
    )
    assert only_first == pytest.approx(per_token[0])
    assert full == pytest.approx((per_token[0] + per_token[2]) / 2)  # equal gates
    assert gated_dkd_numpy(logits, ids, logprobs, targets, np.zeros((1, 3))) == 0.0


def test_the_lsq_forward_is_the_grid_and_its_step_gradient_follows_lsq() -> None:
    w = np.array([[0.26, -0.95, 0.10, 2.00]])
    scales = np.array([[0.2]])
    fake, grad = lsq_reference(w, scales, 4, 4)
    codes = grid.quantize_codes(w, scales, 4, 4)
    assert np.allclose(fake, grid.dequantize(codes, scales, 4, 4))
    # inside the grid: round(w/s) - w/s; clipped at the top (2.0 / 0.2 = 10 > 7): Qp = 7
    assert grad[0, 0] == pytest.approx(1 - 1.3)  # 0.26 / 0.2 = 1.3 rounds to 1
    assert grad[0, 1] == pytest.approx(round(-4.75) + 4.75)
    assert grad[0, 3] == 7.0
    assert lsq_reference(np.array([[-5.0, 0, 0, 0]]), scales, 4, 4)[1][0, 0] == -8.0
