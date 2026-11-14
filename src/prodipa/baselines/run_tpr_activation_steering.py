#!/usr/bin/env python3
"""Extract and apply multi-property activation directions for TPR generation.

The bundled research source expects a private ``llama`` module exposing a
``head_out`` hook that is not shipped with that source.  This executable keeps
the same positive-minus-negative construction and layer-wise norm-preserving
injection, while using the output of the standard Hugging Face LLaMA
attention ``o_proj`` as the auditable compatibility point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FAMILY_PROMPT = (
    "[Generate by superfamily]"
    "Superfamily=<Tetratricopeptide-like helical domain superfamily>"
    "Seq=<"
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"Expected a non-empty JSON array: {path}")
    return value


def prompt_for(record: dict[str, Any]) -> str:
    if all(key in record for key in ("instruction", "input", "output")):
        output = str(record["output"])
        if output.endswith(">"):
            output = output[:-1]
        return str(record["instruction"]) + str(record["input"]) + output
    sequence = str(record.get("sequence", "")).strip().upper()
    if not sequence:
        raise ValueError("Record has neither instruction/input/output nor sequence")
    return FAMILY_PROMPT + sequence


def load_model(model_path: Path, device: torch.device) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def cohort_mean(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    layers = model.model.layers
    attention_outputs: list[torch.Tensor | None] = [None] * len(layers)
    mlp_outputs: list[torch.Tensor | None] = [None] * len(layers)
    hooks = []

    def capture(storage: list[torch.Tensor | None], index: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            storage[index] = output[0] if isinstance(output, tuple) else output
        return hook

    for index, layer in enumerate(layers):
        hooks.append(layer.self_attn.o_proj.register_forward_hook(capture(attention_outputs, index)))
        hooks.append(layer.mlp.register_forward_hook(capture(mlp_outputs, index)))

    attention_sum = torch.zeros(len(layers), model.config.hidden_size, dtype=torch.float64)
    mlp_sum = torch.zeros_like(attention_sum)
    token_lengths: list[int] = []
    processed = 0
    try:
        for start in range(0, len(records), batch_size):
            prompts = [prompt_for(item) for item in records[start : start + batch_size]]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
            lengths = encoded["attention_mask"].sum(dim=1)
            maximum = int(lengths.max().item())
            if maximum > model.config.max_position_embeddings:
                raise ValueError(
                    f"{label} contains {maximum} tokens, exceeding model capacity "
                    f"{model.config.max_position_embeddings}; truncation is disabled"
                )
            token_lengths.extend(int(value) for value in lengths.tolist())
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(**encoded, use_cache=False, return_dict=True)
            row = torch.arange(len(prompts), device=device)
            last = lengths.to(device=device, dtype=torch.long) - 1
            for index in range(len(layers)):
                attention = attention_outputs[index]
                mlp = mlp_outputs[index]
                if attention is None or mlp is None:
                    raise RuntimeError(f"Missing layer activation at index {index}")
                attention_sum[index] += attention[row, last].float().sum(dim=0).cpu().double()
                mlp_sum[index] += mlp[row, last].float().sum(dim=0).cpu().double()
                attention_outputs[index] = None
                mlp_outputs[index] = None
            processed += len(prompts)
            print(
                f"cohort={label} processed={processed}/{len(records)} "
                f"batch_max_tokens={maximum}",
                flush=True,
            )
    finally:
        for hook in hooks:
            hook.remove()

    metadata = {
        "records": processed,
        "token_length_min": min(token_lengths),
        "token_length_max": max(token_lengths),
        "token_length_mean": sum(token_lengths) / len(token_lengths),
    }
    return (attention_sum / processed).float(), (mlp_sum / processed).float(), metadata


def extract(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for vector extraction")
    device = torch.device(args.device)
    positive_path = Path(args.positive).resolve()
    negative_path = Path(args.negative).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    status = output / "STATUS.json"
    atomic_json(status, {
        "state": "RUNNING", "property": args.property,
        "started_at": datetime.now().astimezone().isoformat(),
    })
    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        positive = load_records(positive_path)
        negative = load_records(negative_path)
        if len(positive) != len(negative):
            raise ValueError(f"Cohort size mismatch: positive={len(positive)} negative={len(negative)}")
        model, tokenizer = load_model(Path(args.model).resolve(), device)
        pos_attention, pos_mlp, pos_meta = cohort_mean(
            model, tokenizer, positive, args.batch_size, device, "positive"
        )
        neg_attention, neg_mlp, neg_meta = cohort_mean(
            model, tokenizer, negative, args.batch_size, device, "negative"
        )
        attention_path = output / "attention_vectors.pt"
        mlp_path = output / "mlp_vectors.pt"
        torch.save((pos_attention, neg_attention), attention_path)
        torch.save((pos_mlp, neg_mlp), mlp_path)
        manifest = {
            "state": "COMPLETE",
            "property": args.property,
            "seed": args.seed,
            "model": str(Path(args.model).resolve()),
            "hook_compatibility_point": "model.layers.*.self_attn.o_proj output",
            "mlp_hook": "model.layers.*.mlp output",
            "construction": "positive cohort mean minus negative cohort mean",
            "positive_path": str(positive_path),
            "positive_sha256": sha256(positive_path),
            "negative_path": str(negative_path),
            "negative_sha256": sha256(negative_path),
            "positive": pos_meta,
            "negative": neg_meta,
            "attention_vectors": str(attention_path),
            "mlp_vectors": str(mlp_path),
            "completed_at": datetime.now().astimezone().isoformat(),
        }
        atomic_json(output / "MANIFEST.json", manifest)
        atomic_json(status, manifest)
    except BaseException as error:
        atomic_json(status, {
            "state": "FAILED", "property": args.property,
            "error": f"{type(error).__name__}: {error}",
            "failed_at": datetime.now().astimezone().isoformat(),
        })
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("--property", choices=("fold", "sol", "thermo"), required=True)
    extract_parser.add_argument("--model", required=True)
    extract_parser.add_argument("--positive", required=True)
    extract_parser.add_argument("--negative", required=True)
    extract_parser.add_argument("--output", required=True)
    extract_parser.add_argument("--device", default="cuda:0")
    extract_parser.add_argument("--batch-size", type=int, default=2)
    extract_parser.add_argument("--seed", type=int, default=42)
    extract_parser.add_argument("--resume", action="store_true")
    extract_parser.set_defaults(function=extract)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
