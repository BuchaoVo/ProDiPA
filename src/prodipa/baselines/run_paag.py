#!/usr/bin/env python3
"""Run official PAAG annotation-conditioned generation on frozen TPR lengths."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types
from collections import defaultdict
from pathlib import Path
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


def install_optional_import_shims() -> None:
    import yaml as pyyaml

    ruamel = types.ModuleType("ruamel")
    ruamel.__spec__ = importlib.util.spec_from_loader("ruamel", loader=None)
    yaml_module = types.ModuleType("ruamel.yaml")
    yaml_module.__spec__ = importlib.util.spec_from_loader("ruamel.yaml", loader=None)
    yaml_module.load = pyyaml.load
    yaml_module.Loader = pyyaml.Loader
    ruamel.yaml = yaml_module
    wandb = types.ModuleType("wandb")
    wandb.__spec__ = importlib.util.spec_from_loader("wandb", loader=None)
    sys.modules.setdefault("ruamel", ruamel)
    sys.modules.setdefault("ruamel.yaml", yaml_module)
    sys.modules.setdefault("wandb", wandb)


def run(config_path: Path, project_root: Path, device: str) -> None:
    import torch

    config = load_config(config_path)
    audit = audit_config(config_path, project_root)
    if audit["status"] != "ready":
        raise RuntimeError(f"PAAG resources are incomplete: {audit['blockers']}")
    repo = resolved(audit, "official_repo")
    checkpoint = resolved(audit, "checkpoint")
    manifest_path = resolved(audit, "length_manifest")
    auxiliary = {
        name: resolved(audit, name)
        for name in (
            "protbert_tokenizer", "protbert_encoder", "protbert_decoder",
            "scibert_tokenizer", "scibert_encoder",
        )
    }
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

    install_optional_import_shims()
    sys.path.insert(0, str(repo))
    original_argv = sys.argv
    sys.argv = [
        str(repo / "paag_downstream_pl.py"),
        "--config", str(repo / "configs" / "pretrain_full_dataset.yaml"),
    ]
    from models.paag_pretrain_pl import PAAG_Pretrain

    original_load = torch.load
    by_name = {
        "ProtBert_tokenizer.pt": auxiliary["protbert_tokenizer"],
        "ProtBert.pt": auxiliary["protbert_encoder"],
        "ProtDecoder.pt": auxiliary["protbert_decoder"],
        "scibert_tokenizer.pt": auxiliary["scibert_tokenizer"],
        "scibert.pt": auxiliary["scibert_encoder"],
    }

    def trusted_official_load(path, *args, **kwargs):
        if isinstance(path, (str, bytes, os.PathLike)):
            candidate = Path(os.fsdecode(os.fspath(path)))
            if not candidate.exists() and candidate.name in by_name:
                path = by_name[candidate.name]
        kwargs.setdefault("weights_only", False)
        return original_load(path, *args, **kwargs)

    torch.load = trusted_official_load
    try:
        model = PAAG_Pretrain.load_from_checkpoint(
            str(checkpoint),
            map_location="cpu",
            strict=False,
            seq_config=str(repo / "configs" / "config_seq.json"),
            text_config=str(repo / "configs" / "config_text.json"),
        )
    finally:
        sys.argv = original_argv
        torch.load = original_load
    model = model.eval().to(device)

    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, source in enumerate(manifest, start=1):
        request_id = f"paag_tpr_annotation_{index:06d}"
        if request_id not in records:
            grouped[int(source["target_length"])].append((index, source))
    for length in sorted(grouped):
        for offset in range(0, len(grouped[length]), batch_size):
            batch = grouped[length][offset : offset + batch_size]
            batch_seed = seed + batch[0][0] - 1
            seed_everything(batch_seed)
            texts = [prompt] * len(batch)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                sequences = model.generate(
                    texts,
                    [""] * len(batch),
                    len(batch),
                    sample=True,
                    max_length=length,
                    min_length=length,
                    top_p=float(protocol["top_p"]),
                )
            for (index, source), sequence in zip(batch, sequences, strict=True):
                normalized = "".join(sequence.split())
                request_id = f"paag_tpr_annotation_{index:06d}"
                records[request_id] = build_generation_record(
                    request_id=request_id,
                    benchmark="conditional_tpr",
                    model="paag_tpr_annotation",
                    seed=seed + index - 1,
                    condition=prompt,
                    raw_output=sequence,
                    sequence=normalized,
                    terminated=True,
                    termination_reason="fixed_length_generation_complete",
                    generation_method="annotation_conditioned_autoregressive",
                    generation_config_id=str(config["generation_config_id"]),
                    method_metadata={
                        "target_length": length,
                        "actual_length": len(normalized),
                        "top_p": protocol["top_p"],
                        "repetition_penalty": protocol["repetition_penalty"],
                        "max_new_tokens": length,
                        "batch_seed": batch_seed,
                        "length_source_id": source.get("length_source_id"),
                        "length_source_sequence_sha256": source.get(
                            "length_source_sequence_sha256"
                        ),
                    },
                )
            ordered = [f"paag_tpr_annotation_{index:06d}" for index in range(1, total + 1)]
            atomic_write_jsonl(output, [records[key] for key in ordered if key in records])
            print(f"PAAG requests complete: {len(records)}/{total}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, default=Path("configs/baselines/conditional_paag.json"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = audit_config(args.config, args.project_root)
    if args.report:
        atomic_write_json(args.report, report)
    print(f"PAAG preflight: {report['status']} blockers={report['blockers']}")
    if args.action == "run":
        run(args.config, args.project_root, args.device)
    elif report["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
