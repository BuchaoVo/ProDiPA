#!/usr/bin/env python3
"""Audit and publish the fixed unconditional full-PoE cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from prodipa.unconditional.offline_opd import cache_record_path, load_cache_record
from prodipa.unconditional.train import _load_teacher_config


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
    raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--theta-zero", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=512)
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

    config = _load_teacher_config(str(args.teacher_config), 1.0)
    teachers = []
    for item in config.teachers:
        path = Path(item.adapter_path)
        teachers.append(
            {
                "name": item.name,
                "weight": item.weight,
                "temperature": item.temperature,
                "adapter_path": str(path.resolve()),
                "adapter_weights_sha256": sha256(adapter_weight(path)),
            }
        )
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
        "reuse_epochs_to_formal_endpoint": 3.5,
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
            "adapter_weights_sha256": sha256(adapter_weight(args.theta_zero)),
            "adapter_config_sha256": sha256(args.theta_zero / "adapter_config.json"),
        },
        "base_model": str(args.base_model.resolve()),
        "teacher_config": str(args.teacher_config.resolve()),
        "teacher_config_sha256": sha256(args.teacher_config),
        "teachers": teachers,
        "generation": {
            "max_new_tokens": 512,
            "temperature": 1.0,
            "repetition_penalty": 1.2,
            "top_p": 0.95,
            "top_k": 500,
        },
        "distill": {"loss_type": "jsd", "beta": 0.5},
        "shard_reports": reports,
        "consistency_audit": consistency,
    }
    target = args.cache_dir / "CACHE_MANIFEST.json"
    temporary = target.with_suffix(f".json.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, target)


if __name__ == "__main__":
    main()
