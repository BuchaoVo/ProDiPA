from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from experiments.prodipa.conditional_evaluation.score_protgpt2_ppl import (
    archive_stale_output,
    format_model_card_text,
    shifted_mean_nll,
)


@pytest.mark.parametrize("length,chunk_lengths", [
    (60, [60]),
    (61, [60, 1]),
    (120, [60, 60]),
    (121, [60, 60, 1]),
])
def test_model_card_wrapping_boundaries(length: int, chunk_lengths: list[int]) -> None:
    eos = "<|endoftext|>"
    lines = format_model_card_text("A" * length, eos).split("\n")
    assert lines[0] == eos
    assert lines[-1] == eos
    assert [len(chunk) for chunk in lines[1:-1]] == chunk_lengths


def test_real_tokenizer_adds_exactly_one_boundary_eos() -> None:
    model_path = Path(os.environ.get("PROTGPT2_MODEL", "__PRODIPA_MODELS__/ProtGPT2"))
    if not model_path.is_dir():
        pytest.skip(f"ProtGPT2 tokenizer is unavailable: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    text = format_model_card_text("A" * 121, tokenizer.eos_token)
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    assert token_ids[0] == tokenizer.eos_token_id
    assert token_ids[-1] == tokenizer.eos_token_id
    assert token_ids.count(tokenizer.eos_token_id) == 2


def test_shifted_ce_matches_hugging_face_loss_and_repeats_in_fp32() -> None:
    torch.manual_seed(7)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=31,
        n_positions=32,
        n_embd=16,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )).float().eval()
    input_ids = torch.randint(0, 31, (1, 17), dtype=torch.long)
    with torch.inference_mode():
        first = model(input_ids=input_ids, labels=input_ids)
        second = model(input_ids=input_ids, labels=input_ids)
        manual = shifted_mean_nll(first.logits, input_ids)
    torch.testing.assert_close(manual, first.loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(first.loss, second.loss, rtol=0.0, atol=0.0)


def test_stale_ppl_output_is_archived(tmp_path: Path) -> None:
    output = tmp_path / "scores" / "protgpt2_ppl.tsv"
    manifest = output.with_suffix(".manifest.json")
    output.parent.mkdir(parents=True)
    output.write_text("old score\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    archive_stale_output(output, manifest)
    archived = list((output.parent / "history").iterdir())
    assert not output.exists()
    assert not manifest.exists()
    assert {path.name.split("_", 1)[1] for path in archived} == {
        output.name,
        manifest.name,
    }


def test_legacy_done_marker_cannot_skip_ppl_manifest_validation() -> None:
    runner = Path(
        "experiments/prodipa/conditional_evaluation/"
        "run_separate_paradigm_scoring_queue.sh"
    ).read_text(encoding="utf-8")
    assert 'echo protgpt2_ppl >> "$QUEUE"' in runner
    assert '[[ -s "$OUT/.done/temberture" ]]' in runner
