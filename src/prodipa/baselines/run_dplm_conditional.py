from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from .audit_resources import audit_config
    from .common import (
        atomic_write_json, atomic_write_jsonl, build_generation_record, expand_path,
        load_config, read_jsonl, seed_everything,
    )
    from .run_dplm import _load_official_generation_class, _resolved
except ImportError:  # direct script execution
    from audit_resources import audit_config
    from common import (
        atomic_write_json, atomic_write_jsonl, build_generation_record, expand_path,
        load_config, read_jsonl, seed_everything,
    )
    from run_dplm import _load_official_generation_class, _resolved


def masked_template(row: dict[str, Any]) -> str:
    length = int(row["target_length"])
    start = int(row["motif_start_0based"])
    end = int(row["motif_end_exclusive"])
    motif = str(row["motif_sequence"])
    if end - start != len(motif) or not (0 <= start < end <= length):
        raise ValueError(f"invalid motif coordinates: {row['request_id']}")
    return "<mask>" * start + motif + "<mask>" * (length - end)


def run(config_path: Path, project_root: Path, device: str) -> None:
    import torch

    config = load_config(config_path)
    audit = audit_config(config_path, project_root)
    if audit["status"] != "ready":
        raise RuntimeError(f"DPLM resources are incomplete: {audit['blockers']}")
    repo = _resolved(audit, "official_repo")
    model_path = _resolved(audit, "checkpoint")
    manifest_path = _resolved(audit, "condition_manifest")
    protocol = config["protocol"]
    total = int(protocol["requests"])
    batch_size = int(protocol["batch_size"])
    seed = int(protocol["seed"])
    max_iter = int(protocol["max_iter"])
    sampling_strategy = str(protocol["sampling_strategy"])
    output = expand_path(protocol["output"], base=project_root)
    manifest = read_jsonl(manifest_path)
    if len(manifest) != total:
        raise ValueError(f"condition manifest has {len(manifest)} rows, expected {total}")
    if [row.get("sample_id") for row in manifest] != list(range(total)):
        raise ValueError("condition manifest sample_id values are not ordered 0..requests-1")

    existing = read_jsonl(output) if output.exists() else []
    if len(existing) > total or len(existing) % batch_size != 0:
        raise RuntimeError("partial DPLM output is not aligned to a completed batch")
    for index, row in enumerate(existing):
        if row.get("request_id") != manifest[index]["request_id"]:
            raise RuntimeError("existing output does not match the frozen condition manifest")
    if len(existing) == total:
        print(f"already complete: {output}")
        return

    DiffusionProteinLanguageModel = _load_official_generation_class(repo)
    model = DiffusionProteinLanguageModel.from_pretrained(str(model_path)).eval().to(device)
    tokenizer = model.tokenizer
    records = list(existing)
    for start_index in range(len(existing), total, batch_size):
        batch_manifest = manifest[start_index : start_index + batch_size]
        batch_seed = seed + start_index
        seed_everything(batch_seed)
        encoded = tokenizer.batch_encode_plus(
            [masked_template(row) for row in batch_manifest],
            add_special_tokens=True, padding="longest", return_tensors="pt",
        )
        input_tokens = encoded["input_ids"].to(device)
        partial_masks = input_tokens.ne(model.mask_id)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output_tokens = model.generate(
                input_tokens=input_tokens,
                tokenizer=tokenizer,
                max_iter=max_iter,
                temperature=1.0,
                sampling_strategy=sampling_strategy,
                partial_masks=partial_masks,
            )
        token_lists = output_tokens.detach().cpu().tolist()
        decoded = [
            "".join(value.split())
            for value in tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        ]
        for row, token_ids, sequence in zip(batch_manifest, token_lists, decoded):
            motif_start = int(row["motif_start_0based"])
            motif_end = int(row["motif_end_exclusive"])
            motif = str(row["motif_sequence"])
            if len(sequence) != int(row["target_length"]):
                raise RuntimeError(
                    f"generated length mismatch for {row['request_id']}: "
                    f"{len(sequence)} != {row['target_length']}"
                )
            if sequence[motif_start:motif_end] != motif:
                raise RuntimeError(f"DPLM changed the frozen motif: {row['request_id']}")
            records.append(build_generation_record(
                request_id=str(row["request_id"]),
                benchmark="conditional_tpr",
                model="dplm_650m_tpr_infilling",
                seed=seed,
                condition=(
                    "family=G3DSA:1.25.40.10;"
                    f"motif_start={motif_start};motif={motif}"
                ),
                raw_output=sequence,
                sequence=sequence,
                terminated=True,
                termination_reason="diffusion_iterations_complete",
                generation_method="masked_diffusion_tpr_motif_infilling",
                generation_config_id=str(config["generation_config_id"]),
                method_metadata={
                    "raw_token_ids": token_ids,
                    "target_length": int(row["target_length"]),
                    "actual_length": len(sequence),
                    "motif_start_0based": motif_start,
                    "motif_end_exclusive": motif_end,
                    "motif_sequence": motif,
                    "source_sequence_sha256": row["source_sequence_sha256"],
                    "diffusion_iterations": max_iter,
                    "sampling_strategy": sampling_strategy,
                    "temperature_cli_effective": False,
                    "batch_seed": batch_seed,
                },
            ))
        atomic_write_jsonl(output, records)
        print(f"Conditional DPLM requests complete: {len(records)}/{total}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Conditional DPLM-650M TPR motif infilling")
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/baselines/conditional_dplm.json")
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = audit_config(args.config, args.project_root)
    if args.report:
        atomic_write_json(args.report, report)
    print(f"Conditional DPLM preflight: {report['status']} blockers={report['blockers']}")
    if args.action == "run":
        run(args.config, args.project_root, args.device)
    elif report["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
