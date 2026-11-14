#!/usr/bin/env python3
"""Resumable fixed-budget generation for the eight formal teacher evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
FIELDS = (
    "paradigm", "model", "sample_id", "seed", "sequence", "length_aa",
    "valid_sequence", "eos_reached", "truncated", "valid_complete",
    "wrapper_closed", "invalid_reason", "raw_completion", "temperature",
    "top_p", "top_k", "repetition_penalty", "max_new_tokens",
)


def adapter_weights_file(adapter: Path) -> Path:
    """Return a supported PEFT adapter weight file, preferring safetensors."""
    candidates = (
        adapter / "adapter_model.safetensors",
        adapter / "adapter_model.bin",
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    expected = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(
        f"No PEFT adapter weights found in {adapter}; expected one of: {expected}"
    )


def read_existing(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if tuple(rows[0].keys()) != FIELDS if rows else False:
        raise ValueError(f"Unexpected schema in {path}")
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_completion(paradigm: str, text: str) -> tuple[str, bool]:
    if paradigm == "unconditional":
        return "".join(text.split()).upper(), True
    strict = re.search(r"Seq\s*=\s*<([^>]*)>", text)
    if strict:
        return "".join(strict.group(1).split()).upper(), True
    wrapper_closed = ">" in text
    payload = text.split(">", 1)[0]
    payload = re.sub(r"^\s*Seq\s*=\s*<?", "", payload)
    return "".join(payload.split()).upper(), wrapper_closed


def main() -> None:
    global FIELDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paradigm", choices=("unconditional", "conditional"), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, default=512)
    parser.add_argument("--sample-id-start", type=int, default=0)
    parser.add_argument("--sample-id-stop", type=int)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--generation-max-aa", type=int, default=0,
                        help="Opt-in ProtGPT2 residue cap; 0 preserves the historical protocol")
    parser.add_argument('--sequence-format', choices=('auto', 'legacy', 'unwrapped'), default='auto',
                        help='ProtGPT2 only: auto follows checkpoint SEQUENCE_PROTOCOL.json')
    parser.add_argument(
        "--superfamily",
        default="Tetratricopeptide-like helical domain superfamily",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.paradigm == 'unconditional':
        from prodipa.unconditional.sequence_format import resolve_sequence_format
        args.sequence_format = resolve_sequence_format(args.adapter, args.sequence_format)
        if args.sequence_format == 'unwrapped':
            FIELDS += ('sequence_format',)
    elif args.sequence_format != 'auto':
        raise ValueError('sequence-format is a ProtGPT2-only option')
    if args.generation_max_aa < 0 or (args.generation_max_aa and args.paradigm != "unconditional"):
        raise ValueError("generation-max-aa is a non-negative ProtGPT2-only option")
    if args.generation_max_aa:
        FIELDS += ("generation_max_aa", "length_forced_eos", "natural_eos_reached")

    sample_id_stop = args.num_sequences if args.sample_id_stop is None else args.sample_id_stop
    if not 0 <= args.sample_id_start < sample_id_stop <= args.num_sequences:
        raise ValueError(
            "Expected 0 <= sample-id-start < sample-id-stop <= num-sequences, "
            f"got {args.sample_id_start}, {sample_id_stop}, {args.num_sequences}"
        )
    expected_ids = set(range(args.sample_id_start, sample_id_stop))

    if not args.base_model.is_dir():
        raise FileNotFoundError(args.base_model)
    adapter_weights = adapter_weights_file(args.adapter) if args.adapter else None

    rows = read_existing(args.out)
    if args.sequence_format == 'unwrapped' and any(row['sequence_format'] != 'unwrapped' for row in rows):
        raise ValueError('Existing generation uses a different sequence format')
    if args.generation_max_aa and any(int(row["generation_max_aa"]) != args.generation_max_aa for row in rows):
        raise ValueError("Existing generation uses a different residue cap")
    done = {int(row["sample_id"]) for row in rows}
    if not done <= expected_ids:
        raise ValueError(f"Existing sample IDs fall outside the requested shard in {args.out}")
    if done == expected_ids:
        print(f"[OK] already complete: {args.out}")
        return
    if any(row["model"] != args.model_name or row["paradigm"] != args.paradigm for row in rows):
        raise ValueError(f"Model/paradigm mismatch in {args.out}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval().to(args.device)

    if args.paradigm == "conditional":
        prompt = (
            "[Generate by superfamily] "
            f"Superfamily=<{args.superfamily}>"
        )
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(args.device)
        attention_mask = encoded["attention_mask"].to(args.device)
    else:
        start_token = tokenizer.bos_token_id
        if start_token is None:
            start_token = tokenizer.eos_token_id
        if start_token is None:
            raise ValueError("ProtGPT2 tokenizer has neither BOS nor EOS token")
        input_ids = torch.tensor([[start_token]], dtype=torch.long, device=args.device)
        attention_mask = torch.ones_like(input_ids)
    prompt_length = input_ids.shape[1]
    format_processor = None
    if args.paradigm == 'unconditional' and args.sequence_format == 'unwrapped':
        from prodipa.unconditional.sequence_format import UnwrappedProteinLogitsProcessor
        format_processor = UnwrappedProteinLogitsProcessor(tokenizer)

    for sample_id in range(args.sample_id_start, sample_id_stop):
        if sample_id in done:
            continue
        seed = args.base_seed + sample_id * 1009
        set_seed(seed)
        length_processor = None
        extra_kwargs = {}
        processors = [format_processor] if format_processor is not None else []
        if args.generation_max_aa:
            from prodipa.unconditional.length_control import ProteinLengthLogitsProcessor
            length_processor = ProteinLengthLogitsProcessor(tokenizer, args.generation_max_aa, prompt_length)
            processors.append(length_processor)
        if processors:
            extra_kwargs['logits_processor'] = processors
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                **extra_kwargs,
            )[0, prompt_length:].detach().cpu().tolist()
        eos_reached = tokenizer.eos_token_id in generated
        realized = generated[:generated.index(tokenizer.eos_token_id)] if eos_reached else generated
        truncated = not eos_reached and len(generated) >= args.max_new_tokens
        forced = length_processor.forced_flags()[0] if length_processor is not None else False
        truncated = truncated or forced
        completion = tokenizer.decode(
            realized, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        sequence, wrapper_closed = parse_completion(args.paradigm, completion)
        reasons = []
        if not sequence:
            reasons.append("empty_or_unparsed")
        elif set(sequence) - AA:
            reasons.append("noncanonical_amino_acid")
        if sequence and not 50 <= len(sequence) <= 512:
            reasons.append("length_outside_50_512")
        valid_sequence = not reasons
        if not eos_reached:
            reasons.append("eos_not_reached")
        if forced:
            reasons.append("max_length_aa_reached")
        elif truncated:
            reasons.append("max_new_tokens_reached")
        rows.append({
            "paradigm": args.paradigm,
            "model": args.model_name,
            "sample_id": sample_id,
            "seed": seed,
            "sequence": sequence,
            "length_aa": len(sequence),
            "valid_sequence": str(valid_sequence).lower(),
            "eos_reached": str(eos_reached).lower(),
            "truncated": str(truncated).lower(),
            "valid_complete": str(valid_sequence and eos_reached and not truncated).lower(),
            "wrapper_closed": str(wrapper_closed).lower(),
            "invalid_reason": ";".join(reasons),
            "raw_completion": completion.replace("\t", " ").replace("\n", "\\n"),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            **({'sequence_format': 'unwrapped'} if args.sequence_format == 'unwrapped' else {}),
            **({"generation_max_aa": args.generation_max_aa,
                "length_forced_eos": str(forced).lower(),
                "natural_eos_reached": str(eos_reached and not forced).lower()}
               if args.generation_max_aa else {}),
        })
        rows.sort(key=lambda row: int(row["sample_id"]))
        write_rows(args.out, rows)
        print(
            f"{args.paradigm}/{args.model_name} {len(rows)}/{len(expected_ids)} "
            f"length={len(sequence)} eos={eos_reached} valid={valid_sequence}",
            flush=True,
        )

    metadata = {
        "paradigm": args.paradigm,
        "model": args.model_name,
        "base_model": str(args.base_model.resolve()),
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "adapter_weights": str(adapter_weights.resolve()) if adapter_weights else None,
        "num_requests": args.num_sequences,
        "sample_id_start": args.sample_id_start,
        "sample_id_stop": sample_id_stop,
        "base_seed": args.base_seed,
        "decoding": {
            **({'sequence_format': args.sequence_format} if args.paradigm == 'unconditional' else {}),
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            "generation_max_aa": args.generation_max_aa,
        },
        "superfamily": args.superfamily if args.paradigm == "conditional" else None,
    }
    args.out.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
