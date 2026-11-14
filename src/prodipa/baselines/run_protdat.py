#!/usr/bin/env python3
"""Run official ProtDAT text-conditioned generation on frozen TPR lengths."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    from .audit_resources import audit_config
    from .common import (
        atomic_write_json,
        atomic_write_jsonl,
        build_generation_record,
        expand_path,
        load_config,
        read_jsonl,
        seed_everything,
    )
except ImportError:
    from audit_resources import audit_config
    from common import (
        atomic_write_json,
        atomic_write_jsonl,
        build_generation_record,
        expand_path,
        load_config,
        read_jsonl,
        seed_everything,
    )


def resolved(audit: dict[str, Any], name: str) -> Path:
    for item in audit["resources"]:
        if item["name"] == name and item["status"] == "ready":
            return Path(item["resolved_path"])
    raise RuntimeError(f"resource is not ready: {name}")


def attention_masks(description_mask, sequence, heads: int, cross_vocab: int, device: str):
    import torch

    description_count = description_mask.ne(0).sum(dim=1)
    batch, description_length = description_mask.shape
    description = torch.ones(
        (batch, description_length, description_length), dtype=torch.bool, device=device
    )
    cross = torch.ones((batch, cross_vocab, description_length), dtype=torch.bool, device=device)
    for index, count in enumerate(description_count.tolist()):
        description[index, :, count:] = False
        cross[index, :, count:] = False
    description = description.unsqueeze(1).expand(-1, heads, -1, -1)
    cross = cross.unsqueeze(1).expand(-1, heads, -1, -1)
    sequence_mask = torch.ones(
        (batch, sequence.shape[1], cross_vocab + sequence.shape[1]), dtype=torch.bool, device=device
    )
    sequence_mask[:, :, cross_vocab:] = torch.tril(sequence_mask[:, :, cross_vocab:])
    sequence_mask = sequence_mask.unsqueeze(1).expand(-1, heads, -1, -1)
    return [description, cross, sequence_mask]


def nucleus_sample(logits, previous, top_p: float, repetition_penalty: float, temperature: float):
    import torch
    import torch.nn.functional as functional

    logits = logits / temperature
    selected = []
    for index in range(logits.shape[0]):
        if previous is not None:
            token = int(previous[index])
            logits[index, token] = (
                logits[index, token] * repetition_penalty
                if logits[index, token] <= 0
                else logits[index, token] / repetition_penalty
            )
        probabilities = functional.softmax(logits[index], dim=-1)
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        sorted_probabilities[cumulative - sorted_probabilities > top_p] = 0
        sorted_probabilities.div_(sorted_probabilities.sum())
        choice = torch.multinomial(sorted_probabilities, 1)
        selected.append(sorted_indices.gather(0, choice))
    return torch.stack(selected).reshape(-1, 1)


def generate(model, description, description_mask, tokenizer, length: int, protocol, device: str):
    import torch

    batch = description.shape[0]
    start = tokenizer.cls_token_id
    if start is None:
        start = 0
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    sequence = torch.full((batch, 1), start, dtype=torch.long, device=device)
    previous = None
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(length):
            masks = attention_masks(
                description_mask, sequence, int(protocol["num_heads"]),
                int(protocol["cross_vocab_size"]), device,
            )
            hidden = model(des=description, seq=sequence, mask=masks)[2][:, -1, :]
            logits = model.emb2seq(hidden)
            next_token = nucleus_sample(
                logits, previous, float(protocol["top_p"]),
                float(protocol["repetition_penalty"]), float(protocol["temperature"]),
            ).to(device)
            previous = next_token.squeeze(1)
            sequence = torch.cat((sequence, next_token), dim=1)

    results = []
    for row in sequence.detach().cpu().tolist():
        eos_position = row.index(eos) if eos in row else len(row)
        decoded = tokenizer.decode(row[:eos_position], skip_special_tokens=True)
        results.append(("".join(decoded.split()), row, eos in row))
    return results


def run(config_path: Path, project_root: Path, device: str) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    config = load_config(config_path)
    audit = audit_config(config_path, project_root)
    if audit["status"] != "ready":
        raise RuntimeError(f"ProtDAT resources are incomplete: {audit['blockers']}")
    repo = resolved(audit, "official_repo")
    checkpoint = resolved(audit, "checkpoint_weights")
    text_model_path = resolved(audit, "text_model")
    sequence_tokenizer_path = resolved(audit, "sequence_tokenizer")
    manifest_path = resolved(audit, "length_manifest")
    expected_bytes = int(config["resources"]["checkpoint_weights"]["expected_bytes"])
    if checkpoint.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"ProtDAT checkpoint size is {checkpoint.stat().st_size}, expected {expected_bytes}"
        )

    sys.path.insert(0, str(repo))
    from Decoder import ProtDAT_Decoder

    protocol = config["protocol"]
    total = int(protocol["requests"])
    seed = int(protocol["seed"])
    batch_size = int(protocol["batch_size"])
    prompt = str(protocol["prompt"])
    output = expand_path(protocol["output"], base=project_root)
    manifest = read_jsonl(manifest_path)
    if len(manifest) != total:
        raise ValueError(f"length manifest has {len(manifest)} rows, expected {total}")
    records = {row["request_id"]: row for row in (read_jsonl(output) if output.exists() else [])}
    if len(records) == total:
        print(f"already complete: {output}")
        return

    text_tokenizer = AutoTokenizer.from_pretrained(str(text_model_path), local_files_only=True)
    sequence_tokenizer = AutoTokenizer.from_pretrained(
        str(sequence_tokenizer_path), local_files_only=True
    )
    model = ProtDAT_Decoder(
        d_model=int(protocol["d_model"]),
        des_vocab_size=text_tokenizer.vocab_size,
        seq_vocab_size=sequence_tokenizer.vocab_size,
        cross_vocab_size=int(protocol["cross_vocab_size"]),
        layer=int(protocol["layers"]),
        head_num=int(protocol["num_heads"]),
        dropout=0.0,
        device=device,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device)
    text_model = AutoModel.from_pretrained(
        str(text_model_path), local_files_only=True, torch_dtype=torch.bfloat16
    ).eval().to(device)
    encoded = text_tokenizer(
        prompt, max_length=512, padding=False, truncation=True,
        return_tensors="pt", add_special_tokens=True,
    )
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        description_single = text_model(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        ).last_hidden_state
    mask_single = encoded["attention_mask"].to(device)

    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, source in enumerate(manifest, start=1):
        request_id = f"protdat_tpr_text_{index:06d}"
        if request_id not in records:
            grouped[int(source["target_length"])].append((index, source))
    for length in sorted(grouped):
        for offset in range(0, len(grouped[length]), batch_size):
            batch = grouped[length][offset : offset + batch_size]
            batch_seed = seed + batch[0][0] - 1
            seed_everything(batch_seed)
            count = len(batch)
            generated = generate(
                model,
                description_single.repeat(count, 1, 1),
                mask_single.repeat(count, 1),
                sequence_tokenizer,
                length,
                protocol,
                device,
            )
            for (index, source), (sequence, token_ids, eos_reached) in zip(
                batch, generated, strict=True
            ):
                request_id = f"protdat_tpr_text_{index:06d}"
                records[request_id] = build_generation_record(
                    request_id=request_id,
                    benchmark="conditional_tpr",
                    model="protdat_tpr_text",
                    seed=seed + index - 1,
                    condition=prompt,
                    raw_output=sequence,
                    sequence=sequence,
                    terminated=eos_reached,
                    termination_reason="eos" if eos_reached else "target_length_reached",
                    generation_method="text_conditioned_autoregressive",
                    generation_config_id=str(config["generation_config_id"]),
                    method_metadata={
                        "target_length": length,
                        "actual_length": len(sequence),
                        "top_p": protocol["top_p"],
                        "temperature": protocol["temperature"],
                        "repetition_penalty": protocol["repetition_penalty"],
                        "max_new_tokens": length,
                        "batch_seed": batch_seed,
                        "raw_token_ids": token_ids,
                        "length_source_id": source.get("length_source_id"),
                        "length_source_sequence_sha256": source.get(
                            "length_source_sequence_sha256"
                        ),
                    },
                )
            ordered = [f"protdat_tpr_text_{index:06d}" for index in range(1, total + 1)]
            atomic_write_jsonl(output, [records[key] for key in ordered if key in records])
            print(f"ProtDAT requests complete: {len(records)}/{total}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, default=Path("configs/baselines/conditional_protdat.json"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = audit_config(args.config, args.project_root)
    if args.report:
        atomic_write_json(args.report, report)
    print(f"ProtDAT preflight: {report['status']} blockers={report['blockers']}")
    if args.action == "run":
        run(args.config, args.project_root, args.device)
    elif report["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
