#!/usr/bin/env python3
"""Measure Sol-teacher EOS probability and sampling reachability at true endpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from prodipa.unconditional.teacher_sequence_protocol import (
    PLAIN_DIRECT_EOS_V1,
    PROTGPT2_FASTA60_V1,
    encode_teacher_sequence,
)


def read_sequences(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["sequence"].strip().upper() for row in csv.DictReader(handle)]


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys(), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def measure_model(
    label: str,
    adapter: Path,
    base_model: Path,
    tokenizer,
    sequences: list[str],
    device: torch.device,
) -> list[dict]:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base, adapter, is_trainable=False).to(device).eval()
    eos = tokenizer.eos_token_id
    processors = (
        RepetitionPenaltyLogitsProcessor(1.2),
        TemperatureLogitsWarper(0.7),
        TopKLogitsWarper(200),
        TopPLogitsWarper(0.9),
    )
    rows = []
    for protocol in (PLAIN_DIRECT_EOS_V1, PROTGPT2_FASTA60_V1):
        for sequence_index, sequence in enumerate(sequences):
            full_ids = encode_teacher_sequence(tokenizer, sequence, protocol)
            prefix_ids = full_ids[:-1]
            input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            with torch.inference_mode():
                logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits[0, -1].float()
            probability = float(torch.softmax(logits, dim=-1)[eos].cpu())
            eos_logit = logits[eos]
            rank = int((logits > eos_logit).sum().item()) + 1
            scores = logits.unsqueeze(0)
            for processor in processors:
                scores = processor(input_ids, scores)
            reachable = bool(torch.isfinite(scores[0, eos]).item())
            filtered_probability = (
                float(torch.softmax(scores, dim=-1)[0, eos].cpu()) if reachable else 0.0
            )
            rows.append({
                "model": label,
                "adapter": str(adapter.resolve()),
                "protocol": protocol,
                "sequence_index": sequence_index,
                "length_aa": len(sequence),
                "prefix_tokens": len(prefix_ids),
                "terminal_predecessor_id": prefix_ids[-1],
                "terminal_predecessor_token": tokenizer.convert_ids_to_tokens(prefix_ids[-1]),
                "eos_probability": probability,
                "eos_rank": rank,
                "eos_in_top_k_200": rank <= 200,
                "eos_reachable_top_k_200_top_p_0.9": reachable,
                "eos_filtered_probability": filtered_probability,
            })
    del model, base
    torch.cuda.empty_cache()
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    keys = sorted({(row["model"], row["protocol"]) for row in rows})
    native = {
        "old_sol_teacher": PLAIN_DIRECT_EOS_V1,
        "new_sol_teacher": PROTGPT2_FASTA60_V1,
    }
    for model, protocol in keys:
        subset = [row for row in rows if row["model"] == model and row["protocol"] == protocol]
        probabilities = [float(row["eos_probability"]) for row in subset]
        ranks = [int(row["eos_rank"]) for row in subset]
        filtered = [float(row["eos_filtered_probability"]) for row in subset]
        result.append({
            "model": model,
            "protocol": protocol,
            "native_protocol": protocol == native[model],
            "n": len(subset),
            "eos_probability_mean": statistics.fmean(probabilities),
            "eos_probability_median": statistics.median(probabilities),
            "eos_probability_q05": quantile(probabilities, 0.05),
            "eos_rank_median": statistics.median(ranks),
            "eos_rank_q95": quantile([float(value) for value in ranks], 0.95),
            "eos_top_k_200_rate": statistics.fmean(float(row["eos_in_top_k_200"]) for row in subset),
            "eos_reachable_rate": statistics.fmean(float(row["eos_reachable_top_k_200_top_p_0.9"]) for row in subset),
            "eos_filtered_probability_mean": statistics.fmean(filtered),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--old-adapter", type=Path, required=True)
    parser.add_argument("--new-adapter", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    sequences = read_sequences(args.validation)
    device = torch.device("cuda")
    rows = []
    rows.extend(measure_model("old_sol_teacher", args.old_adapter, args.base_model, tokenizer, sequences, device))
    rows.extend(measure_model("new_sol_teacher", args.new_adapter, args.base_model, tokenizer, sequences, device))
    summaries = summarize(rows)
    write_tsv(args.out_dir / "endpoint_eos_per_sequence.tsv", rows)
    write_tsv(args.out_dir / "endpoint_eos_summary.tsv", summaries)
    payload = {
        "status": "COMPLETE",
        "validation": str(args.validation.resolve()),
        "n_sequences": len(sequences),
        "sampling_filter": {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 200,
            "repetition_penalty": 1.2,
        },
        "definition": (
            "EOS is reachable when its logit remains finite after repetition penalty, "
            "temperature, top-k and top-p filtering at the true held-out endpoint."
        ),
        "summary": summaries,
    }
    (args.out_dir / "ENDPOINT_EOS.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
