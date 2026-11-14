#!/usr/bin/env python3
"""Audit fixed full-PoE cache shards and publish an immutable cache manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import yaml

from prodipa.conditional.offline_opd import cache_record_path, load_cache_record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_weight(path: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"adapter weights not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--theta-zero", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()
    reports = [
        json.loads((args.cache_dir / f"shard_{rank}.json").read_text())
        for rank in range(args.shards)
    ]
    if any(item.get("status") != "COMPLETE" for item in reports):
        raise ValueError("one or more cache shards are incomplete")
    if any(item.get("consistency_audit", {}).get("status") != "PASS" for item in reports):
        raise ValueError("online/offline consistency audit failed")

    lengths: list[int] = []
    bytes_total = 0
    prompt_lengths: set[int] = set()
    vocab_sizes: set[int] = set()
    for sample_id in range(args.sample_count):
        path = cache_record_path(args.cache_dir, sample_id)
        record = load_cache_record(path)
        if int(record["sample_id"]) != sample_id:
            raise ValueError(f"cache sample id mismatch: {path}")
        ids = record["input_ids"]
        target = record["target_log_probs"]
        prompt_length = int(record["prompt_length"])
        if target.dtype != torch.bfloat16:
            raise ValueError(f"cache target dtype is not bfloat16: {path}")
        if target.ndim != 2 or target.shape[0] != ids.numel() - prompt_length:
            raise ValueError(f"cache target alignment failure: {path}")
        if not torch.isfinite(target).all():
            raise ValueError(f"non-finite cache target: {path}")
        lengths.append(int(target.shape[0]))
        prompt_lengths.add(prompt_length)
        vocab_sizes.add(int(target.shape[1]))
        bytes_total += path.stat().st_size
    if len(prompt_lengths) != 1 or len(vocab_sizes) != 1:
        raise ValueError("cache prompt length or vocabulary is inconsistent")

    config = yaml.safe_load(args.config.read_text())
    teachers = []
    for item in config["teachers"]["adapters"]:
        path = Path(item["adapter_path"])
        weights = adapter_weight(path)
        teachers.append({
            "name": item["name"],
            "weight": float(item["weight"]),
            "temperature": float(item["temperature"]),
            "adapter_path": str(path.resolve()),
            "adapter_weights_sha256": sha256(weights),
        })
    theta_weight = adapter_weight(args.theta_zero)
    audits = [item["consistency_audit"] for item in reports]
    consistency = {
        "status": "PASS",
        "definition": "same theta-zero rollout and student logits; live teacher PoE versus serialized full-vocabulary PoE",
        "max_abs_target_logprob_diff": max(
            float(item["max_abs_target_logprob_diff"]) for item in audits
        ),
        "max_absolute_loss_diff": max(float(item["absolute_loss_diff"]) for item in audits),
        "shard_audits": audits,
    }
    (args.cache_dir / "CONSISTENCY_AUDIT.json").write_text(
        json.dumps(consistency, indent=2) + "\n"
    )
    manifest = {
        "status": "COMPLETE",
        "schema_version": 1,
        "cache_type": "fixed_rollout_full_vocabulary_poe_log_probabilities",
        "sample_count": args.sample_count,
        "reuse_epochs": 4,
        "seed": 42,
        "annotation_dtype": "bfloat16",
        "full_vocabulary": True,
        "prompt_length_tokens": prompt_lengths.pop(),
        "vocab_size": vocab_sizes.pop(),
        "completion_tokens": {
            "minimum": min(lengths),
            "maximum": max(lengths),
            "total": sum(lengths),
        },
        "cache_bytes": bytes_total,
        "theta_zero": {
            "path": str(args.theta_zero.resolve()),
            "adapter_weights_sha256": sha256(theta_weight),
            "adapter_config_sha256": sha256(args.theta_zero / "adapter_config.json"),
        },
        "teachers": teachers,
        "opd_config": str(args.config.resolve()),
        "opd_config_sha256": sha256(args.config),
        "generation": config["generation"],
        "distill": config["distill"],
        "shard_reports": reports,
        "consistency_audit": consistency,
    }
    target = args.cache_dir / "CACHE_MANIFEST.json"
    temporary = target.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, target)
    print(f"published offline cache manifest: {target}")


if __name__ == "__main__":
    main()
