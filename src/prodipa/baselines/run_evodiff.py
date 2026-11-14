#!/usr/bin/env python3
"""Run official EvoDiff OA-DM-640M generation on frozen target lengths."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from .audit_resources import audit_config
    from .common import atomic_write_json, atomic_write_jsonl, build_generation_record, expand_path, load_config, read_jsonl, seed_everything
except ImportError:
    from audit_resources import audit_config
    from common import atomic_write_json, atomic_write_jsonl, build_generation_record, expand_path, load_config, read_jsonl, seed_everything


def resolved(audit: dict[str, Any], name: str) -> Path:
    for item in audit["resources"]:
        if item["name"] == name and item["status"] == "ready":
            return Path(item["resolved_path"])
    raise RuntimeError(f"resource is not ready: {name}")


def generate_batch(model, tokenizer, length: int, batch_size: int, device: str):
    import numpy as np
    import torch

    mask = tokenizer.mask_id
    sample = torch.full((batch_size, length), mask, dtype=torch.long, device=device)
    positions = np.arange(length)
    np.random.shuffle(positions)
    with torch.inference_mode():
        for position in positions:
            timestep = torch.zeros(batch_size, dtype=torch.long, device=device)
            logits = model(sample, timestep)
            probabilities = torch.softmax(logits[:, position, : len(tokenizer.all_aas) - 6], dim=1)
            sample[:, position] = torch.multinomial(probabilities, num_samples=1).squeeze(1)
    return sample, [tokenizer.untokenize(row) for row in sample]


def run(config_path: Path, project_root: Path, device: str) -> None:
    config = load_config(config_path)
    audit = audit_config(config_path, project_root)
    if audit["status"] != "ready":
        raise RuntimeError(f"EvoDiff resources are incomplete: {audit['blockers']}")
    repo = resolved(audit, "official_repo")
    checkpoint = resolved(audit, "checkpoint_weights")
    manifest_path = resolved(audit, "length_manifest")
    expected_bytes = int(config["resources"]["checkpoint_weights"]["expected_bytes"])
    if checkpoint.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"EvoDiff checkpoint size is {checkpoint.stat().st_size}, expected {expected_bytes}; download is incomplete"
        )

    import sys

    sys.path.insert(0, str(repo))
    from evodiff.pretrained import OA_DM_640M

    protocol = config["protocol"]
    total = int(protocol["requests"])
    batch_size = int(protocol["batch_size"])
    seed = int(protocol["seed"])
    output = expand_path(protocol["output"], base=project_root)
    manifest = read_jsonl(manifest_path)
    if len(manifest) != total:
        raise ValueError(f"length manifest has {len(manifest)} rows, expected {total}")
    records = {row["request_id"]: row for row in (read_jsonl(output) if output.exists() else [])}
    if len(records) == total:
        print(f"already complete: {output}")
        return

    model, _collater, tokenizer, scheme = OA_DM_640M(path_to_checkpoints=str(checkpoint))
    if scheme != "mask":
        raise RuntimeError(f"unexpected EvoDiff scheme: {scheme}")
    model = model.eval().to(device)

    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(manifest, start=1):
        request_id = f"evodiff_oa_dm_640m_{index:06d}"
        if request_id not in records:
            grouped[int(row["target_length"])].append((index, row))

    for length in sorted(grouped):
        items = grouped[length]
        for offset in range(0, len(items), batch_size):
            batch = items[offset : offset + batch_size]
            batch_seed = seed + batch[0][0]
            seed_everything(batch_seed)
            token_rows, sequences = generate_batch(model, tokenizer, length, len(batch), device)
            for (index, source), token_ids, sequence in zip(batch, token_rows.cpu().tolist(), sequences, strict=True):
                request_id = f"evodiff_oa_dm_640m_{index:06d}"
                records[request_id] = build_generation_record(
                    request_id=request_id,
                    benchmark="unconditional",
                    model="evodiff_oa_dm_640m",
                    seed=seed + index - 1,
                    condition=f"target_length={length}",
                    raw_output=sequence,
                    sequence=sequence,
                    terminated=True,
                    termination_reason="all_positions_unmasked",
                    generation_method="order_agnostic_discrete_diffusion",
                    generation_config_id=str(config["generation_config_id"]),
                    method_metadata={
                        "target_length": length,
                        "actual_length": len(sequence),
                        "unmask_order": "uniform_random_per_batch",
                        "batch_seed": batch_seed,
                        "raw_token_ids": token_ids,
                        "length_source_id": source.get("length_source_id"),
                        "length_source_sequence_sha256": source.get("length_source_sequence_sha256"),
                    },
                )
            ordered = [f"evodiff_oa_dm_640m_{index:06d}" for index in range(1, total + 1)]
            atomic_write_jsonl(output, [records[x] for x in ordered if x in records])
            print(f"EvoDiff requests complete: {len(records)}/{total}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, default=Path("configs/baselines/evodiff.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = audit_config(args.config, args.project_root)
    if args.report:
        atomic_write_json(args.report, report)
    print(f"EvoDiff preflight: {report['status']} blockers={report['blockers']}")
    if args.action == "run":
        run(args.config, args.project_root, args.device)
    elif report["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
