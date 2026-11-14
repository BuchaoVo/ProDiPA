import importlib.util
from pathlib import Path

import pytest


pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("datasets")


SCRIPT = Path(__file__).parents[2] / "scripts" / "teacher_sft" / "build_instruction_dataset.py"
SPEC = importlib.util.spec_from_file_location("prodipa_teacher_sft_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

GENERATION_SCRIPT = Path(__file__).parents[2] / "scripts" / "teacher_sft" / "generate_sequences.py"
GENERATION_SPEC = importlib.util.spec_from_file_location("prodipa_teacher_generation", GENERATION_SCRIPT)
GENERATION = importlib.util.module_from_spec(GENERATION_SPEC)
GENERATION_SPEC.loader.exec_module(GENERATION)


class CharacterTokenizer:
    bos_token = "^"
    eos_token = "!"
    pad_token_id = 0

    def __call__(self, texts, **_kwargs):
        if isinstance(texts, str):
            return {"input_ids": [ord(char) for char in texts]}
        return {"input_ids": [[ord(char) for char in text] for text in texts]}


def test_response_only_labels_include_real_eos():
    examples = {"instruction": ["generate"], "input": ["high"], "output": ["Seq=<ACDE>"]}
    result = MODULE.tokenize_instruction_batch(examples, CharacterTokenizer(), max_seq_length=768)
    source = "generate high"
    target = "Seq=<ACDE>!"
    assert result["input_ids"][0].tolist() == [ord(char) for char in source + target]
    assert result["labels"][0].tolist() == [MODULE.IGNORE_INDEX] * len(source) + [
        ord(char) for char in target
    ]


def test_sequence_only_uses_same_jsonl_sequence_without_prompt_wrapper():
    examples = {"instruction": ["generate"], "input": ["high"], "output": ["Seq=<ACDE>"]}
    result = MODULE.tokenize_instruction_batch(
        examples, CharacterTokenizer(), max_seq_length=768, training_format="sequence_only"
    )
    source = "^"
    target = "ACDE!"
    assert result["input_ids"][0].tolist() == [ord(char) for char in source + target]
    assert result["labels"][0].tolist() == [MODULE.IGNORE_INDEX] + [
        ord(char) for char in target
    ]


def test_sequence_only_rejects_malformed_output():
    examples = {"instruction": ["generate"], "input": ["high"], "output": ["ACDE"]}
    with pytest.raises(ValueError, match="Expected"):
        MODULE.tokenize_instruction_batch(
            examples, CharacterTokenizer(), max_seq_length=768, training_format="sequence_only"
        )


def test_direct_dpo_uses_one_prompt_and_response_only_labels():
    examples = {
        "instruction": ["generate"], "input": ["shared"],
        "chosen": ["Seq=<ACDE>"], "rejected": ["Seq=<AAAA>"],
    }
    result = MODULE.tokenize_preference_batch(examples, CharacterTokenizer(), max_seq_length=768)
    prompt = "generate shared"
    for side, response in (("chosen", "Seq=<ACDE>!"), ("rejected", "Seq=<AAAA>!")):
        assert result[f"{side}_input_ids"][0].tolist() == [ord(char) for char in prompt + response]
        assert result[f"{side}_labels"][0].tolist() == [MODULE.IGNORE_INDEX] * len(prompt) + [
            ord(char) for char in response
        ]


def test_generation_extracts_dpo_missing_opening_delimiter_without_hiding_format_shift():
    completion = "Seq=MACDE>"
    assert GENERATION.extract_sequence(completion, "prollama") == "MACDE"
    assert GENERATION.sequence_format(completion, "prollama") == "missing_opening_delimiter"
