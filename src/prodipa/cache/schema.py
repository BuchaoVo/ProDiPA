"""Versioned cache contract shared by builders, readers, and trainers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CacheCompatibility:
    source_dataset_version: str
    split_version: str
    tokenizer_hash: str
    prompt_hash: str
    teacher_versions: tuple[str, ...]

    def validate(self):
        scalar_values = (self.source_dataset_version, self.split_version, self.tokenizer_hash, self.prompt_hash)
        if any(not value for value in scalar_values) or not self.teacher_versions or any(not value for value in self.teacher_versions):
            raise ValueError("cache compatibility fields must be non-empty")


@dataclass(frozen=True)
class CacheRecord:
    cache_id: str
    parent_sequence_id: str
    parent_split: str
    condition_id: str
    prompt_ids: tuple[int, ...]
    sequence_ids: tuple[int, ...]
    response_start: int
    response_mask: tuple[bool, ...]
    observed_eos: bool
    eos_position: int | None
    truncated: bool
    behavior_policy_version: str
    behavior_logprobs: tuple[float, ...]
    teacher_weight_config: tuple[float, ...]
    poe_scalar_logprob: tuple[float, ...]
    reference_logprobs: tuple[float, ...]
    preference_mask: tuple[bool, ...]
    annotation_dtype: str
    full_vocab_normalization: bool
    cache_round: int
    acquisition_rule: str
    acquisition_feature_versions: tuple[str, ...]
    cost_record_id: str
    compatibility: CacheCompatibility

    def validate_for_training(self):
        self.compatibility.validate()
        if self.parent_split != "train":
            raise ValueError("cache training/refresh accepts Train records only")
        if not self.parent_sequence_id or not self.cache_id:
            raise ValueError("cache and parent sequence IDs are required")
        length = len(self.sequence_ids)
        aligned = (
            len(self.response_mask), len(self.behavior_logprobs),
            len(self.poe_scalar_logprob), len(self.reference_logprobs),
            len(self.preference_mask),
        )
        if any(size != length for size in aligned):
            raise ValueError("all token-aligned cache arrays must match sequence_ids")
        if not 0 <= self.response_start <= length:
            raise ValueError("response_start outside sequence_ids")
        if self.response_start != len(self.prompt_ids) or self.sequence_ids[: self.response_start] != self.prompt_ids:
            raise ValueError("prompt_ids and response_start do not match sequence_ids")
        if any(self.response_mask[: self.response_start]):
            raise ValueError("prompt token marked as response")
        if any(pref and not response for pref, response in zip(self.preference_mask, self.response_mask)):
            raise ValueError("preference mask includes prompt/padding")
        if not self.full_vocab_normalization:
            raise ValueError("exact_scalar cache requires full-vocabulary normalization")
        if self.annotation_dtype != "float32":
            raise ValueError("exact_scalar annotations must be float32")
        if not self.teacher_weight_config or len(self.teacher_weight_config) != len(self.compatibility.teacher_versions):
            raise ValueError("teacher weights must align with teacher versions")
        if any(weight < 0 for weight in self.teacher_weight_config) or sum(self.teacher_weight_config) <= 0:
            raise ValueError("teacher weights must be nonnegative and not all zero")
        if not self.acquisition_feature_versions or any(not value for value in self.acquisition_feature_versions):
            raise ValueError("acquisition feature versions are required")
        if self.observed_eos:
            if self.truncated or self.eos_position is None:
                raise ValueError("observed EOS cannot be truncated or positionless")
            if not 0 <= self.eos_position < length or not self.response_mask[self.eos_position]:
                raise ValueError("EOS position is not a response token")
        elif self.eos_position is not None:
            raise ValueError("unobserved EOS must not have an eos_position")
        return True


def require_compatible(expected: CacheCompatibility, actual: CacheCompatibility):
    expected.validate()
    actual.validate()
    mismatches = [name for name in expected.__dict__ if getattr(expected, name) != getattr(actual, name)]
    if mismatches:
        raise ValueError(f"incompatible cache metadata: {', '.join(mismatches)}")
    return True
