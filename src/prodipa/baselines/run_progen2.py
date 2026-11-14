from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    from .audit_resources import audit_config
    from .common import atomic_write_json, atomic_write_jsonl, build_generation_record, expand_path, load_config, read_jsonl, seed_everything
except ImportError:  # direct script execution
    from audit_resources import audit_config
    from common import atomic_write_json, atomic_write_jsonl, build_generation_record, expand_path, load_config, read_jsonl, seed_everything


def _resolved(audit: dict[str, Any], name: str) -> Path:
    for item in audit["resources"]:
        if item["name"] == name and item["status"] == "ready":
            return Path(item["resolved_path"])
    raise RuntimeError(f"resource is not ready: {name}")


def _split_completion(decoded: str, context: str) -> tuple[str, bool, str]:
    payload = decoded[len(context) :] if decoded.startswith(context) else decoded
    positions = [(payload.find(token), token) for token in ("1", "2") if payload.find(token) >= 0]
    if positions:
        position, terminal = min(positions)
        return payload[:position], True, f"terminal_{terminal}"
    return payload, False, "max_token_budget"


def run(config_path: Path, project_root: Path, device: str) -> None:
    config = load_config(config_path)
    audit = audit_config(config_path, project_root)
    if audit["status"] != "ready":
        raise RuntimeError(f"ProGen2 resources are incomplete: {audit['blockers']}")

    repo = _resolved(audit, "official_repo")
    checkpoint = _resolved(audit, "checkpoint")
    tokenizer_path = _resolved(audit, "tokenizer")
    source_root = repo / "progen2" if (repo / "progen2" / "sample.py").is_file() else repo
    sys.path.insert(0, str(source_root))

    import torch
    from tokenizers import Tokenizer
    from models.progen.modeling_progen import ProGenForCausalLM

    protocol = config["protocol"]
    output = expand_path(protocol["output"], base=project_root)
    total = int(protocol["requests"])
    batch_size = int(protocol["batch_size"])
    seed = int(protocol["seed"])
    context = str(protocol["context"])
    max_total_tokens = int(protocol["max_total_tokens"])

    existing = read_jsonl(output) if output.exists() else []
    if len(existing) > total or len(existing) % batch_size != 0:
        raise RuntimeError("partial ProGen2 output is not aligned to a completed batch")
    if len(existing) == total:
        print(f"already complete: {output}")
        return

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = ProGenForCausalLM.from_pretrained(
        str(checkpoint), torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(torch.device(device)).eval()
    pad_token_id = tokenizer.encode("<|pad|>").ids[0]
    prompt_ids = tokenizer.encode(context).ids
    records = list(existing)

    for start in range(len(existing), total, batch_size):
        count = min(batch_size, total - start)
        batch_seed = seed + start
        seed_everything(batch_seed)
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).view(1, -1)
        with torch.no_grad():
            token_batch = model.generate(
                input_ids,
                do_sample=True,
                temperature=float(protocol["temperature"]),
                top_p=float(protocol["top_p"]),
                max_length=max_total_tokens,
                num_return_sequences=count,
                pad_token_id=pad_token_id,
            )
        token_lists = token_batch.detach().cpu().tolist()
        decoded_batch = tokenizer.decode_batch(token_lists)
        for offset, (token_ids, decoded) in enumerate(zip(token_lists, decoded_batch), start=1):
            request_number = start + offset
            sequence, terminated, reason = _split_completion(decoded, context)
            records.append(
                build_generation_record(
                    request_id=f"progen2_base_{request_number:06d}",
                    benchmark="unconditional",
                    model="progen2_base",
                    seed=seed,
                    condition="",
                    raw_output=decoded,
                    sequence=sequence,
                    terminated=terminated,
                    termination_reason=reason,
                    generation_method="autoregressive_sampling",
                    generation_config_id=str(config["generation_config_id"]),
                    method_metadata={
                        "raw_token_ids": token_ids,
                        "control_token": context,
                        "batch_seed": batch_seed,
                        "max_total_tokens_including_control": max_total_tokens,
                        "temperature": float(protocol["temperature"]),
                        "top_p": float(protocol["top_p"]),
                        "top_k": None,
                        "repetition_penalty": None,
                    },
                )
            )
        atomic_write_jsonl(output, records)
        print(f"ProGen2 requests complete: {len(records)}/{total}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="ProGen2-base adapter for ProDiPA")
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, default=Path("configs/baselines/progen2.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = audit_config(args.config, args.project_root)
    if args.report:
        atomic_write_json(args.report, report)
    print(f"ProGen2 preflight: {report['status']} blockers={report['blockers']}")
    if args.action == "run":
        run(args.config, args.project_root, args.device)
    elif report["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
