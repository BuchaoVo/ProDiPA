from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None
    set_seed = None


@dataclass
class GenerateConfig:
    base_model_path: str
    adapter_path: str
    output_json_path: str
    num_sequences: int = 1000
    batch_size: int = 8
    max_new_tokens: int = 512
    do_sample: bool = True
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 500
    repetition_penalty: float = 1.2
    seed: int = 42
    use_bf16_if_available: bool = True
    trust_remote_code: bool = True
    overwrite_output: bool = True
    model_label: str = "teacher"
    min_length_aa: int = 50
    max_length_aa: int = 512
    enforce_aa_limit: bool = False
    sequence_format: str = 'auto'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate unconditional ProtGPT2 protein sequences.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("generate.yaml")),
        help="YAML generation config path.",
    )
    bool_fields = {"do_sample", "use_bf16_if_available", "trust_remote_code", "overwrite_output", "enforce_aa_limit"}
    for field in fields(GenerateConfig):
        arg_name = "--" + field.name
        if field.name in bool_fields:
            parser.add_argument(arg_name, default=None, action=argparse.BooleanOptionalAction)
        else:
            parser.add_argument(arg_name, default=None)
    return parser.parse_args()


def _cast_value(field_name: str, value: Any) -> Any:
    int_fields = {
        "num_sequences",
        "batch_size",
        "max_new_tokens",
        "top_k",
        "seed",
        "min_length_aa",
        "max_length_aa",
    }
    float_fields = {"temperature", "top_p", "repetition_penalty"}
    bool_fields = {"do_sample", "use_bf16_if_available", "trust_remote_code", "overwrite_output", "enforce_aa_limit"}
    if field_name in int_fields:
        return int(value)
    if field_name in float_fields:
        return float(value)
    if field_name in bool_fields:
        return bool(value)
    return value


def _load_config(config_path: str, overrides: argparse.Namespace) -> GenerateConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError("Generation config must be a YAML mapping.")
    values: dict[str, Any] = dict(payload)
    for field in fields(GenerateConfig):
        override_value = getattr(overrides, field.name)
        if override_value is not None:
            values[field.name] = _cast_value(field.name, override_value)
    config = GenerateConfig(**values)
    if not config.base_model_path:
        raise ValueError("`base_model_path` must be set.")
    if not config.adapter_path:
        raise ValueError("`adapter_path` must be set.")
    if config.num_sequences <= 0:
        raise ValueError("`num_sequences` must be > 0.")
    if config.batch_size <= 0:
        raise ValueError("`batch_size` must be > 0.")
    if config.max_new_tokens <= 0:
        raise ValueError("`max_new_tokens` must be > 0.")
    return config


def _resolve_torch_dtype(config: GenerateConfig) -> torch.dtype:
    require_runtime_dependencies()
    if not torch.cuda.is_available():
        return torch.float32
    if config.use_bf16_if_available and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _postprocess_sequence(
    token_ids: list[int], tokenizer, min_length_aa: int, max_length_aa: int
) -> dict[str, Any]:
    eos_token_id = tokenizer.eos_token_id
    generated_ids = list(token_ids[1:])
    eos_position = None
    if eos_token_id is not None and eos_token_id in generated_ids:
        eos_position = generated_ids.index(eos_token_id)
    cleaned_ids = generated_ids if eos_position is None else generated_ids[:eos_position]
    seq = tokenizer.decode(cleaned_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    sequence = "".join(seq.split()).upper()
    canonical = set("ACDEFGHIKLMNPQRSTVWY")
    canonical_only = bool(sequence) and set(sequence).issubset(canonical)
    eos_reached = eos_position is not None
    valid = (
        canonical_only
        and eos_reached
        and min_length_aa <= len(sequence) <= max_length_aa
    )
    return {
        "sequence": sequence,
        "length_aa": len(sequence),
        "eos_reached": eos_reached,
        "termination_status": "eos" if eos_reached else "max_new_tokens",
        "canonical_only": canonical_only,
        "valid": valid,
        "generated_token_count": len(cleaned_ids),
        "raw_token_ids": token_ids,
    }


def _save_results(output_path: Path, records: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _base_nll_per_sequence(model, rows: list[list[int]], eos_token_id: int) -> list[float]:
    realized_rows = []
    for row in rows:
        generated = list(row[1:])
        if eos_token_id in generated:
            generated = generated[: generated.index(eos_token_id) + 1]
        realized_rows.append([eos_token_id, *generated])
    width = max(len(row) for row in realized_rows)
    labels, masks = [], []
    for row in realized_rows:
        pad = width - len(row)
        labels.append(row + [-100] * pad)
        masks.append([1] * len(row) + [0] * pad)
    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [[token if token != -100 else eos_token_id for token in row] for row in labels],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    adapter_context = (
        model.disable_adapter() if isinstance(model, PeftModel) else contextlib.nullcontext()
    )
    with adapter_context, torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    shifted_labels = labels_tensor[:, 1:]
    valid = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    realized = (
        logits[:, :-1].float().log_softmax(-1)
        .gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    )
    nll = -(realized * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    return [float(value) for value in nll.cpu()]


def require_runtime_dependencies() -> None:
    missing = []
    if torch is None:
        missing.append("torch")
    if PeftModel is None:
        missing.append("peft")
    if AutoModelForCausalLM is None or AutoTokenizer is None or set_seed is None:
        missing.append("transformers")
    if missing:
        raise ImportError("Missing runtime dependencies required for generation: " + ", ".join(missing))


def main() -> None:
    args = parse_args()
    config = _load_config(args.config, args)
    require_runtime_dependencies()
    output_path = Path(config.output_json_path)
    if output_path.suffix.lower() != ".json":
        raise ValueError("`output_json_path` must end with `.json`.")
    if output_path.exists() and not config.overwrite_output:
        raise FileExistsError(f"Output already exists: {output_path}")

    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_path,
        trust_remote_code=config.trust_remote_code,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must have eos_token when pad_token is missing.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_path,
        dtype=_resolve_torch_dtype(config),
        trust_remote_code=config.trust_remote_code,
        local_files_only=True,
    )
    if config.adapter_path.upper() == "BASE":
        model = base_model
    else:
        model = PeftModel.from_pretrained(base_model, config.adapter_path, is_trainable=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for unconditional generation.")

    records: list[dict[str, Any]] = []
    from .sequence_format import resolve_sequence_format, UnwrappedProteinLogitsProcessor
    config.sequence_format = resolve_sequence_format(config.adapter_path, config.sequence_format)
    format_processor = UnwrappedProteinLogitsProcessor(tokenizer) if config.sequence_format == 'unwrapped' else None
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    while len(records) < config.num_sequences:
        current_batch = min(config.batch_size, config.num_sequences - len(records))
        prompt_ids = torch.full((current_batch, 1), eos_token_id, dtype=torch.long, device=device)
        prompt_attention_mask = torch.ones_like(prompt_ids)
        length_processor = None
        extra_kwargs = {}
        processors = [format_processor] if format_processor is not None else []
        if config.enforce_aa_limit:
            from .length_control import ProteinLengthLogitsProcessor
            length_processor = ProteinLengthLogitsProcessor(tokenizer, config.max_length_aa)
            processors.append(length_processor)
        if processors:
            extra_kwargs['logits_processor'] = processors
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                use_cache=True,
                **extra_kwargs,
            )
        raw_rows = generated_ids.detach().cpu().tolist()
        base_nlls = _base_nll_per_sequence(model, raw_rows, eos_token_id)
        forced_flags = length_processor.forced_flags() if length_processor is not None else [False] * len(raw_rows)
        for row, base_nll, forced in zip(raw_rows, base_nlls, forced_flags):
            parsed = _postprocess_sequence(
                row,
                tokenizer,
                config.min_length_aa,
                config.max_length_aa,
            )
            if config.enforce_aa_limit:
                parsed["length_forced_eos"] = forced
                parsed["natural_eos_reached"] = parsed["eos_reached"] and not forced
                if forced:
                    parsed["termination_status"] = "max_length_aa"
                    parsed["valid"] = False
            records.append(
                {
                    "sample_id": len(records),
                    "model": config.model_label,
                    "seed": config.seed,
                    "instruction": "N/A",
                    "reference": "N/A",
                    "response#1": parsed["sequence"],
                    "base_nll_per_token": base_nll,
                    "base_ppl": math.exp(min(base_nll, 80.0)),
                    **parsed,
                }
            )
        print(f"Generated {len(records)}/{config.num_sequences} sequences.")

    _save_results(output_path, records)
    metadata = {
        **config.__dict__,
        "runtime_seconds": time.time() - started,
        "peak_gpu_memory_gb": (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        ),
        "valid_sequences": sum(bool(record["valid"]) for record in records),
        "eos_sequences": sum(bool(record["eos_reached"]) for record in records),
    }
    with output_path.with_suffix(".metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} sequences to: {output_path}")


if __name__ == "__main__":
    main()
