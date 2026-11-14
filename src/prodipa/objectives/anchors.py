"""Mask rules for full LM and observed-EOS anchors."""

from __future__ import annotations

import numpy as np


def validate_loss_masks(response_mask, preference_mask, lm_mask):
    response = np.asarray(response_mask, dtype=bool)
    preference = np.asarray(preference_mask, dtype=bool)
    lm = np.asarray(lm_mask, dtype=bool)
    if response.shape != preference.shape or response.shape != lm.shape:
        raise ValueError("response, preference, and LM masks must share shape")
    if np.any(preference & ~response):
        raise ValueError("preference mask cannot include prompt/padding tokens")
    if np.any(response & ~lm):
        raise ValueError("LM anchor must cover every real response token")
    return True


def eos_mask(sequence_ids, response_mask, eos_token_id, observed_eos, truncated, eos_position):
    sequence_ids = np.asarray(sequence_ids)
    response = np.asarray(response_mask, dtype=bool)
    if sequence_ids.shape != response.shape:
        raise ValueError("sequence ids and response mask must share shape")
    result = np.zeros_like(response)
    if not observed_eos:
        return result
    if truncated:
        raise ValueError("a truncated sample cannot claim observed EOS")
    if eos_position is None or eos_position < 0 or eos_position >= sequence_ids.shape[-1]:
        raise ValueError("observed EOS requires a valid eos_position")
    if not response[..., eos_position].all() or not np.all(sequence_ids[..., eos_position] == eos_token_id):
        raise ValueError("eos_position does not point to a real response EOS token")
    result[..., eos_position] = True
    return result

