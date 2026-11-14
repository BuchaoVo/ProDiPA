#!/usr/bin/env python3
"""Compare Base and one conditional LoRA teacher on teacher-forced examples.

This diagnostic performs no generation, scoring, training, or weight updates.
It measures response-body likelihood, the context-specific closing-marker
probability, and the EOS probability immediately after the closing marker.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def forward_stats(
    model,
    ids: list[int],
    close_idx: int,
    eos_idx: int,
    body_start_idx: int,
    aa_length: int,
    base: bool,
) -> dict:
    device = model.get_input_embeddings().weight.device
    inputs = torch.tensor([ids], device=device, dtype=torch.long)
    adapter_context = model.disable_adapter() if base else contextlib.nullcontext()
    with adapter_context, torch.inference_mode():
        logits = model(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            use_cache=False,
        ).logits[0].float()
        close_logp = torch.log_softmax(logits[close_idx - 1], dim=-1)[ids[close_idx]].item()
        eos_logp = torch.log_softmax(logits[eos_idx - 1], dim=-1)[ids[eos_idx]].item()
        target_positions = torch.arange(body_start_idx, close_idx, device=device)
        if target_positions.numel() < 1:
            raise ValueError("Missing residue tokens")
        body_logits = logits[target_positions - 1]
        body_log_probs = torch.log_softmax(body_logits, dim=-1)
        body_logp = body_log_probs.gather(
            -1, inputs[0, target_positions, None]
        ).sum().item()
        close_token_id = ids[close_idx]
        premature_close_probabilities = body_log_probs[:, close_token_id].exp()
    close_probability = math.exp(close_logp)
    return {
        "body_logp_per_aa": body_logp / aa_length,
        "raw_close_probability": close_probability,
        "close_log_odds": close_logp - math.log1p(-close_probability),
        "close_nll": -close_logp,
        "eos_after_close_probability": math.exp(eos_logp),
        "eos_nll": -eos_logp,
        "mean_nonterminal_close_probability": float(
            premature_close_probabilities.mean().item()
        ),
        "max_nonterminal_close_probability": float(
            premature_close_probabilities.max().item()
        ),
        "nonterminal_close_argmax_fraction": float(
            body_logits.argmax(dim=-1).eq(close_token_id).float().mean().item()
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    source = Path(args.data).expanduser().resolve()
    adapter_dir = Path(args.adapter).expanduser().resolve()
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Expected a JSON list")

    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer EOS is not configured")
    wrapper_ids = tokenizer("Seq=<", add_special_tokens=False).input_ids
    if not wrapper_ids:
        raise ValueError("Missing response wrapper tokens")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=dtype,
        device_map={"": 0},
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval()

    rows = []
    close_token_ids = set()
    for index, record in enumerate(records[: args.limit]):
        prompt = record["instruction"]
        if record.get("input"):
            prompt += " " + str(record["input"])
        output = record["output"]
        if not (output.startswith("Seq=<") and output.endswith(">")):
            raise ValueError(f"Row {index}: unexpected response format")
        sequence = output[len("Seq=<") : -1]
        prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
        target_ids = tokenizer(
            output + tokenizer.eos_token, add_special_tokens=False
        ).input_ids
        if len(prompt_ids) + len(target_ids) > args.max_seq_length:
            raise ValueError(f"Row {index}: exceeds token budget")
        if target_ids[-1] != tokenizer.eos_token_id:
            raise ValueError(f"Row {index}: target does not end in EOS")
        # SentencePiece may assign a different token ID to an isolated '>' than
        # to the same character after a protein sequence.  Locate it in the
        # complete target and validate its decoded value instead.
        close_token_id = target_ids[-2]
        if tokenizer.decode([close_token_id], skip_special_tokens=False) != ">":
            raise ValueError(
                f"Row {index}: penultimate target token does not decode to '>': "
                f"{close_token_id}"
            )
        close_token_ids.add(close_token_id)
        if target_ids[: len(wrapper_ids)] != wrapper_ids:
            raise ValueError(f"Row {index}: response wrapper tokenization changed")

        ids = prompt_ids + target_ids
        close_idx = len(ids) - 2
        eos_idx = len(ids) - 1
        body_start = len(prompt_ids) + len(wrapper_ids)
        base_stats = forward_stats(
            model, ids, close_idx, eos_idx, body_start, len(sequence), base=True
        )
        lora_stats = forward_stats(
            model, ids, close_idx, eos_idx, body_start, len(sequence), base=False
        )
        row = {
            "index": index,
            "length_aa": len(sequence),
            "close_token_id": close_token_id,
            "body_gain_logp_per_aa": (
                lora_stats["body_logp_per_aa"] - base_stats["body_logp_per_aa"]
            ),
            "close_log_odds_shift": (
                lora_stats["close_log_odds"] - base_stats["close_log_odds"]
            ),
            "base": base_stats,
            "lora": lora_stats,
        }
        rows.append(row)
        print(
            f"[{index + 1}/{min(args.limit, len(records))}] "
            f"AA delta_logP/aa={row['body_gain_logp_per_aa']:+.4f}, "
            f"close odds shift={row['close_log_odds_shift']:+.3f}",
            flush=True,
        )

    summary = {
        "records": len(rows),
        "source_json": str(source),
        "source_json_sha256": sha256_file(source),
        "adapter_dir": str(adapter_dir),
        "adapter_config_sha256": sha256_file(adapter_dir / "adapter_config.json"),
        "adapter_weights_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
        "close_token_ids_in_context": sorted(close_token_ids),
        "body_gain_logp_per_aa_mean": mean(row["body_gain_logp_per_aa"] for row in rows),
        "body_gain_logp_per_aa_median": median(row["body_gain_logp_per_aa"] for row in rows),
        "close_odds_ratio_geometric_lora_over_base": math.exp(
            mean(row["close_log_odds_shift"] for row in rows)
        ),
        "mean_raw_close_prob_base": mean(
            row["base"]["raw_close_probability"] for row in rows
        ),
        "mean_raw_close_prob_lora": mean(
            row["lora"]["raw_close_probability"] for row in rows
        ),
        "mean_eos_after_close_prob_base": mean(
            row["base"]["eos_after_close_probability"] for row in rows
        ),
        "mean_eos_after_close_prob_lora": mean(
            row["lora"]["eos_after_close_probability"] for row in rows
        ),
        "mean_nonterminal_close_prob_base": mean(
            row["base"]["mean_nonterminal_close_probability"] for row in rows
        ),
        "mean_nonterminal_close_prob_lora": mean(
            row["lora"]["mean_nonterminal_close_probability"] for row in rows
        ),
        "max_nonterminal_close_prob_lora": max(
            row["lora"]["max_nonterminal_close_probability"] for row in rows
        ),
        "max_nonterminal_close_prob_base": max(
            row["base"]["max_nonterminal_close_probability"] for row in rows
        ),
        "nonterminal_close_argmax_fraction_base": mean(
            row["base"]["nonterminal_close_argmax_fraction"] for row in rows
        ),
        "nonterminal_close_argmax_fraction_lora": mean(
            row["lora"]["nonterminal_close_argmax_fraction"] for row in rows
        ),
    }
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"summary": summary, "per_sequence": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
