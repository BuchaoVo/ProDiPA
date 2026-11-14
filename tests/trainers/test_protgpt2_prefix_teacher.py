import pytest
import torch

from prodipa.unconditional.generate import _postprocess_sequence
from prodipa.unconditional.train_teacher import (
    CausalProteinCollator,
    encode_sequence,
    _causal_cross_entropy,
)
from prodipa.unconditional.teacher_sequence_protocol import PROTGPT2_FASTA60_V1


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, sequence, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": list(range(1, len(sequence) + 1))}

    def decode(self, token_ids, **_kwargs):
        return "".join("ACDE"[token_id - 1] for token_id in token_ids)


def test_encode_and_collate_supervise_eos_without_padding():
    tokenizer = FakeTokenizer()
    short = encode_sequence(tokenizer, "AC", max_length=6)
    long = encode_sequence(tokenizer, "ACD", max_length=6)
    batch = CausalProteinCollator(pad_token_id=0)(
        [{"input_ids": short}, {"input_ids": long}]
    )

    assert short == [0, 1, 2, 0]
    assert long == [0, 1, 2, 3, 0]
    assert batch["input_ids"].shape == torch.Size([2, 5])
    assert batch["labels"][0].tolist() == [0, 1, 2, 0, -100]
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 0]
    assert batch["close_token_mask"].tolist() == [[False, False, False, True, False], [False, False, False, False, True]]


def test_terminal_weight_matches_explicit_ce_and_keeps_padding_gradient_zero():
    batch = CausalProteinCollator(0)([{"input_ids": [0, 1, 0]}, {"input_ids": [0, 2, 3, 0]}])
    logits = torch.randn(2, 4, 5, requires_grad=True)
    loss, normalizer = _causal_cross_entropy(logits, batch['labels'], batch['close_token_mask'], 4.)
    labels = batch['labels'][:, 1:]
    ce = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1,5), labels.reshape(-1), ignore_index=-100, reduction='none').reshape(2,3)
    weights = torch.tensor([[1.,4.,0.],[1.,1.,4.]])
    assert normalizer == 11
    torch.testing.assert_close(loss, (ce * weights).sum()/11)
    loss.backward()
    assert logits.grad[0,2:].eq(0).all()
    assert logits.grad[1,3].eq(0).all()


def test_unit_weight_is_ordinary_ce():
    labels = torch.tensor([[0,1,0,-100]])
    logits = torch.randn(1,4,5)
    loss, n = _causal_cross_entropy(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,5),labels[:,1:].reshape(-1),ignore_index=-100)
    torch.testing.assert_close(loss,expected)
    assert n == 2


@pytest.mark.parametrize('mask', [[[True,False,False,False]], [[False,True,False,False]], [[False,False,True,True]], [[False,False,False,False]]])
def test_weighted_ce_rejects_wrong_terminal_mask(mask):
    with pytest.raises(ValueError):
        _causal_cross_entropy(torch.randn(1,4,5),torch.tensor([[0,1,0,-100]]),torch.tensor(mask),4.)


def test_encode_refuses_truncation():
    with pytest.raises(ValueError, match="truncation is forbidden"):
        encode_sequence(FakeTokenizer(), "ACDEF", max_length=6)


def test_pretraining_compatible_protocol_preserves_wrapping_and_true_eos():
    tokenizer = FakeTokenizer()
    sequence = "A" * 61
    encoded = encode_sequence(
        tokenizer,
        sequence,
        max_length=68,
        sequence_protocol=PROTGPT2_FASTA60_V1,
    )
    # BOS/EOS share id 0. Text is: leading newline + 60 aa + newline + 1 aa
    # + trailing newline, followed by one true special-token EOS.
    assert encoded[0] == 0
    assert encoded[-1] == 0
    assert len(encoded) == 2 + len(sequence) + 3


def test_generation_parser_keeps_failure_in_record():
    tokenizer = FakeTokenizer()
    complete = _postprocess_sequence([0, 1, 2, 3, 0, 0], tokenizer, 3, 4)
    unfinished = _postprocess_sequence([0, 1, 2, 3], tokenizer, 3, 4)

    assert complete["sequence"] == "ACD"
    assert complete["valid"] is True
    assert complete["eos_reached"] is True
    assert unfinished["sequence"] == "ACD"
    assert unfinished["valid"] is False
    assert unfinished["termination_status"] == "max_new_tokens"
