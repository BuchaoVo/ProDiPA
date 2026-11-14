"""Generate a frozen TPR cohort with homolog-context autoregressive prompting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .audit_resources import audit_config
    from .common import atomic_write_json, atomic_write_jsonl, build_generation_record, load_config
except ImportError:
    from audit_resources import audit_config
    from common import atomic_write_json, atomic_write_jsonl, build_generation_record, load_config


def resource_path(report: dict[str, Any], name: str) -> Path:
    for item in report["resources"]:
        if item["name"] == name and item["status"] == "ready":
            return Path(item["resolved_path"])
    raise FileNotFoundError(f"resource is not ready: {name}")


def read_contexts(path: Path, requests: int, depth: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    errors = []
    if len(rows) != requests:
        errors.append(f"requests={len(rows)} expected={requests}")
    request_ids = [str(row.get("request_id", "")) for row in rows]
    if not all(request_ids) or len(request_ids) != len(set(request_ids)):
        errors.append("request IDs are missing or duplicated")
    for index, row in enumerate(rows):
        sequences = row.get("context_sequences", [])
        groups = row.get("context_homology_groups", [])
        if len(sequences) != depth or len(groups) != depth or len(set(groups)) != depth:
            errors.append(f"row {index} does not contain {depth} independent homologs")
        if row.get("family_id") != "G3DSA:1.25.40.10":
            errors.append(f"row {index} has the wrong family")
    if errors:
        raise ValueError("; ".join(errors[:20]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/baselines/conditional_protgpt3_msa.json")
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_config(config_path)
    resources = audit_config(config_path, root)
    context_report: dict[str, Any] = {"status": "blocked"}
    if resources["status"] == "ready":
        contexts = read_contexts(
            resource_path(resources, "context_manifest"),
            int(config["protocol"]["requests"]),
            int(config["method"]["context_depth"]),
        )
        context_report = {
            "status": "ready", "requests": len(contexts),
            "context_depth": config["method"]["context_depth"],
        }
    report = {
        "baseline": config["baseline"],
        "status": "ready" if resources["status"] == "ready" and context_report["status"] == "ready" else "blocked",
        "resources": resources,
        "context_audit": context_report,
    }
    atomic_write_json(args.report, report)
    print(f"{config['baseline']} preflight: {report['status']} report={args.report}")
    if report["status"] != "ready":
        raise SystemExit(2)
    if args.action == "preflight":
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = resource_path(resources, "model")
    contexts = read_contexts(
        resource_path(resources, "context_manifest"),
        int(config["protocol"]["requests"]),
        int(config["method"]["context_depth"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        add_bos_token=False,
        add_eos_token=False,
        padding_side="left",
    )
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(args.device)
    model.eval()
    separator = "<s>"
    mode_token = "<no_gap>"
    protocol = config["protocol"]
    output_path = root / protocol["output"]
    existing = []
    if output_path.exists():
        existing = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line]
    if len(existing) > len(contexts):
        raise ValueError(f"existing output has {len(existing)} rows, above request budget {len(contexts)}")
    records = list(existing)
    for row in contexts[len(records):]:
        seed = int(row["seed"])
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Match the released model card exactly: manual BOS, N-to-C direction,
        # unaligned mode, one separator before every homolog and a final
        # separator before continuation.  The WordLevel tokenizer requires
        # every control marker and amino acid to be whitespace-separated.
        prompt_tokens = ["<|bos|>", "1", mode_token]
        for sequence in row["context_sequences"]:
            prompt_tokens.append(separator)
            prompt_tokens.extend(sequence)
        prompt_tokens.append(separator)
        prompt = " ".join(prompt_tokens)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        model_limit = int(getattr(model.config, "max_position_embeddings", 0) or 0)
        if model_limit and encoded["input_ids"].shape[1] + int(protocol["max_new_tokens"]) > model_limit:
            raise ValueError(
                f"request {row['request_id']} needs {encoded['input_ids'].shape[1]} context tokens "
                f"plus {protocol['max_new_tokens']} generated tokens, above model limit {model_limit}"
            )
        encoded = {key: value.to(args.device) for key, value in encoded.items()}
        generation_kwargs: dict[str, Any] = {
            "do_sample": True,
            "temperature": float(protocol["temperature"]),
            "top_p": float(protocol["top_p"]),
            "top_k": int(protocol["top_k"]),
            "max_new_tokens": int(protocol["max_new_tokens"]),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        with torch.inference_mode():
            output = model.generate(**encoded, **generation_kwargs)[0]
        generated_ids = output[encoded["input_ids"].shape[1]:].tolist()
        eos_at = generated_ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in generated_ids else None
        terminated = eos_at is not None
        sequence_ids = generated_ids if eos_at is None else generated_ids[:eos_at]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
        sequence = "".join(tokenizer.decode(sequence_ids, skip_special_tokens=True).split()).upper()
        records.append(build_generation_record(
            request_id=row["request_id"], benchmark="conditional_tpr",
            model=config["baseline"], seed=seed, condition=config["condition"]["family_name"],
            raw_output=raw, sequence=sequence, terminated=terminated,
            termination_reason="eos" if terminated else "max_new_tokens",
            generation_method=config["method"]["mechanism"],
            generation_config_id="protgpt3_msa_tpr_frozen_contexts",
            method_metadata={
                "mode": row["mode"],
                "context_sequence_ids": row["context_sequence_ids"],
                "context_homology_groups": row["context_homology_groups"],
                "temperature": float(protocol["temperature"]),
                "top_p": float(protocol["top_p"]),
                "top_k": int(protocol["top_k"]),
                "max_new_tokens": int(protocol["max_new_tokens"]),
            },
        ))
        atomic_write_jsonl(output_path, records)
        print(f"{config['baseline']} {len(records)}/{len(contexts)} length={len(sequence)} terminated={terminated}", flush=True)
    print(f"wrote {len(records)} requests to {output_path}")


if __name__ == "__main__":
    main()
