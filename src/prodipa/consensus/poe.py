"""Full-vocabulary product-of-experts reference numerics."""

from __future__ import annotations

import numpy as np


def log_softmax(values, axis=-1):
    values = np.asarray(values, dtype=np.float32)
    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def full_vocab_poe(teacher_logits, weights=None, temperatures=None):
    """Normalize a weighted geometric teacher consensus across the full vocab."""
    logits = np.asarray(teacher_logits)
    if logits.ndim < 2:
        raise ValueError("teacher_logits must have shape [teachers, ..., vocabulary]")
    teachers = logits.shape[0]
    weights = np.ones(teachers, dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
    temperatures = np.ones(teachers, dtype=np.float32) if temperatures is None else np.asarray(temperatures, dtype=np.float32)
    if weights.shape != (teachers,) or temperatures.shape != (teachers,):
        raise ValueError("weights and temperatures must have one value per teacher")
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative, and not all zero")
    if np.any(temperatures <= 0) or not np.isfinite(temperatures).all():
        raise ValueError("temperatures must be finite and positive")
    weights = weights / weights.sum()
    reshape = (teachers,) + (1,) * (logits.ndim - 1)
    teacher_logp = log_softmax(logits.astype(np.float32) / temperatures.reshape(reshape))
    combined = np.tensordot(weights, teacher_logp, axes=(0, 0)).astype(np.float32)
    return log_softmax(combined, axis=-1)

