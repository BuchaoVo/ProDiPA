"""Detached realized-token scalar-advantage objective."""

from __future__ import annotations

import numpy as np

from prodipa.consensus.poe import log_softmax


def scalar_advantage(student_logits, poe_logprobs, sampled_token_ids, clip=None):
    """Return detached numeric A=log q(a|h)-log pi(a|h), preserving sign."""
    student_logp = log_softmax(student_logits, axis=-1)
    poe_logp = np.asarray(poe_logprobs, dtype=np.float32)
    token_ids = np.asarray(sampled_token_ids, dtype=np.int64)
    if student_logp.shape != poe_logp.shape or token_ids.shape != student_logp.shape[:-1]:
        raise ValueError("student/PoE/token shapes do not align")
    if np.any(token_ids < 0) or np.any(token_ids >= student_logp.shape[-1]):
        raise ValueError("sampled token id outside vocabulary")
    selected = np.expand_dims(token_ids, axis=-1)
    advantage = np.take_along_axis(poe_logp, selected, axis=-1)[..., 0]
    advantage -= np.take_along_axis(student_logp, selected, axis=-1)[..., 0]
    if clip is not None:
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("clip must be finite and positive")
        advantage = np.clip(advantage, -clip, clip)
    return advantage.astype(np.float32, copy=True)


def scalar_preference_loss_and_grad(student_logits, cached_alpha, token_ids, preference_mask, clip=None):
    """Return loss, detached advantage, and dL/dlogits for a frozen advantage.

    This NumPy reference makes the stop-gradient semantics explicit.  It is
    used for parity tests; a framework trainer must detach the returned
    advantage before multiplying it by student log-probability.
    """
    logits = np.asarray(student_logits, dtype=np.float32)
    alpha = np.asarray(cached_alpha, dtype=np.float32)
    token_ids = np.asarray(token_ids, dtype=np.int64)
    mask = np.asarray(preference_mask, dtype=bool)
    if alpha.shape != token_ids.shape or mask.shape != token_ids.shape or logits.shape[:-1] != token_ids.shape:
        raise ValueError("logits, alpha, token ids, and preference mask do not align")
    student_logp = log_softmax(logits)
    selected = np.take_along_axis(student_logp, token_ids[..., None], axis=-1)[..., 0]
    advantage = alpha - selected
    if clip is not None:
        advantage = np.clip(advantage, -clip, clip)
    normalizer = max(int(mask.sum()), 1)
    loss = -float((advantage * selected * mask).sum() / normalizer)
    gradient = np.zeros_like(logits, dtype=np.float32)
    if mask.any():
        probabilities = np.exp(student_logp)
        gradient = probabilities * ((advantage * mask) / normalizer)[..., None]
        chosen = np.indices(token_ids.shape)
        gradient[(*chosen, token_ids)] -= advantage * mask / normalizer
    return loss, advantage.astype(np.float32, copy=True), gradient

