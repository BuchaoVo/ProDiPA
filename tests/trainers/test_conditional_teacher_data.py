import pytest
import torch

from prodipa.conditional.teacher_data import (
    DataCollatorForSupervisedDataset,
    tokenize_instruction_batch,
)


class CharacterTokenizer:
    eos_token = "#"
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, texts, return_attention_mask=False, add_special_tokens=True):
        assert return_attention_mask is False
        offset = 100 if add_special_tokens else 0
        return {"input_ids": [[offset + index + 1 for index, _ in enumerate(text)] for text in texts]}

    def decode(self, token_ids, skip_special_tokens=False):
        return ">" if token_ids == [9] else "token"


def test_instruction_prompt_is_masked_and_response_includes_eos():
    result = tokenize_instruction_batch(
        {"instruction": ["generate"], "input": ["family"], "output": ["Seq=<AC>"]},
        CharacterTokenizer(),
        max_seq_length=64,
    )
    prompt_length = len("generate family")
    assert result["labels"][0][:prompt_length] == [-100] * prompt_length
    assert result["labels"][0][prompt_length:] == result["input_ids"][0][prompt_length:]
    assert len(result["input_ids"][0]) == len(result["labels"][0])


def test_instruction_tokenizer_refuses_truncation():
    with pytest.raises(ValueError, match="truncation is forbidden"):
        tokenize_instruction_batch(
            {"instruction": ["long prompt"], "input": ["x"], "output": ["Seq=<AC>"]},
            CharacterTokenizer(),
            max_seq_length=4,
        )


def test_conditional_collator_masks_padding():
    collator = DataCollatorForSupervisedDataset(CharacterTokenizer())
    batch = collator(
        [
            {"input_ids": torch.tensor([1, 9, 0]), "labels": torch.tensor([-100, 9, 0])},
            {"input_ids": torch.tensor([1, 2, 9, 0]), "labels": torch.tensor([-100, 2, 9, 0])},
        ]
    )
    assert batch["input_ids"].tolist() == [[1, 9, 0, 0], [1, 2, 9, 0]]
    assert batch["labels"].tolist() == [[-100, 9, 0, -100], [-100, 2, 9, 0]]
    assert batch["attention_mask"].tolist() == [[True, True, True, False], [True, True, True, True]]
    assert batch["close_token_mask"].tolist() == [
        [False, True, False, False],
        [False, False, True, False],
    ]
