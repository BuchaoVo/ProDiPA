import pytest

from prodipa.cache.schema import CacheCompatibility, CacheRecord, require_compatible
from prodipa.cache.writer import write_jsonl_shard
from prodipa.models.prompt import PromptSpec


def compatibility(**changes):
    values = {
        "source_dataset_version": "prodipa_v1",
        "split_version": "split_v1",
        "tokenizer_hash": "tok123",
        "prompt_hash": "prompt123",
        "teacher_versions": ("fold@a", "sol@b", "thermo@c"),
    }
    values.update(changes)
    return CacheCompatibility(**values)


def record(**changes):
    values = {
        "cache_id": "cache-1",
        "parent_sequence_id": "seq-1",
        "parent_split": "train",
        "condition_id": "cond-1",
        "prompt_ids": (10, 11),
        "sequence_ids": (10, 11, 7, 2),
        "response_start": 2,
        "response_mask": (False, False, True, True),
        "observed_eos": True,
        "eos_position": 3,
        "truncated": False,
        "behavior_policy_version": "student@0",
        "behavior_logprobs": (0.0, 0.0, -1.0, -0.2),
        "teacher_weight_config": (1 / 3, 1 / 3, 1 / 3),
        "poe_scalar_logprob": (0.0, 0.0, -0.8, -0.1),
        "reference_logprobs": (0.0, 0.0, -1.2, -0.3),
        "preference_mask": (False, False, True, True),
        "annotation_dtype": "float32",
        "full_vocab_normalization": True,
        "cache_round": 0,
        "acquisition_rule": "initial",
        "acquisition_feature_versions": ("coverage@v1", "drift@v1"),
        "cost_record_id": "cost-1",
        "compatibility": compatibility(),
    }
    values.update(changes)
    return CacheRecord(**values)


def test_num_07_prompt_condition_and_response_offset_are_exact():
    prompt = PromptSpec("v1", "[Generate] Condition=<{condition}>\n")
    assert prompt.render("family-a").endswith("<family-a>\n")
    assert record().validate_for_training()
    with pytest.raises(ValueError, match="prompt_ids"):
        record(prompt_ids=(99, 11)).validate_for_training()


@pytest.mark.parametrize("field,value", [
    ("tokenizer_hash", "other"),
    ("prompt_hash", "other"),
    ("teacher_versions", ("fold@new", "sol@b", "thermo@c")),
])
def test_num_08_version_or_hash_mismatch_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        require_compatible(compatibility(), compatibility(**{field: value}))


def test_num_09_cache_rejects_val_or_test_parent():
    with pytest.raises(ValueError, match="Train"):
        record(parent_split="val").validate_for_training()


def test_num_09_writer_is_train_only_and_immutable(tmp_path):
    path = tmp_path / "shard-000.jsonl"
    assert write_jsonl_shard(path, [record()]) == path
    with pytest.raises(FileExistsError, match="immutable"):
        write_jsonl_shard(path, [record()])
