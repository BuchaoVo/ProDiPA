#!/usr/bin/env python3
"""Run official DPLM-2 sequence/structure co-generation on frozen lengths."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import logging
import sys
import types
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


def load_official_model_class(repo: Path):
    source = repo / "src"
    byprot_root = source / "byprot"
    sys.path.insert(0, str(source))

    byprot = types.ModuleType("byprot")
    byprot.__path__ = [str(byprot_root)]
    utils = types.ModuleType("byprot.utils")
    utils.__path__ = [str(byprot_root / "utils")]
    utils.get_logger = logging.getLogger

    def training_only_yaml(*_args, **_kwargs):
        raise RuntimeError("training-only YAML loader was called during DPLM-2 inference")

    utils.load_yaml_config = training_only_yaml
    byprot.utils = utils
    models = types.ModuleType("byprot.models")
    models.__path__ = [str(byprot_root / "models")]
    models.MODEL_REGISTRY = {}

    def register_model(name):
        def decorator(cls):
            models.MODEL_REGISTRY[name] = cls
            return cls
        return decorator

    models.register_model = register_model
    datamodules = types.ModuleType("byprot.datamodules")
    datamodules.__path__ = [str(byprot_root / "datamodules")]
    dataset = types.ModuleType("byprot.datamodules.dataset")
    dataset.__path__ = [str(byprot_root / "datamodules" / "dataset")]
    dplm2 = types.ModuleType("byprot.models.dplm2")
    dplm2.__path__ = [str(byprot_root / "models" / "dplm2")]
    modules = types.ModuleType("byprot.models.dplm2.modules")
    modules.__path__ = [str(byprot_root / "models" / "dplm2" / "modules")]
    sys.modules.update({
        "byprot": byprot,
        "byprot.utils": utils,
        "byprot.models": models,
        "byprot.datamodules": datamodules,
        "byprot.datamodules.dataset": dataset,
        "byprot.models.dplm2": dplm2,
        "byprot.models.dplm2.modules": modules,
    })

    esm_modeling = importlib.import_module("transformers.models.esm.modeling_esm")
    exports = list(getattr(esm_modeling, "__all__", ()))
    for name in dir(esm_modeling):
        if not name.startswith("_") and name not in exports:
            exports.append(name)
    esm_modeling.__all__ = exports

    importlib.import_module("byprot.datamodules.dataset.tokenized_protein")
    importlib.import_module("byprot.models.utils")
    importlib.import_module("byprot.models.dplm2.modules.dplm2_modeling_esm")
    module = importlib.import_module("byprot.models.dplm2.dplm2")
    return module.MultimodalDiffusionProteinLanguageModel


def initial_tokens(tokenizer, lengths: list[int], device: str):
    structure_inputs = []
    sequence_inputs = []
    for length in lengths:
        structure = tokenizer.all_tokens[50] * length
        structure_inputs.append(tokenizer.struct_cls_token + structure + tokenizer.struct_eos_token)
        sequence_inputs.append(tokenizer.aa_cls_token + ("A" * length) + tokenizer.aa_eos_token)
    structures = tokenizer.batch_encode_plus(
        structure_inputs, add_special_tokens=False, padding="longest", return_tensors="pt"
    )["input_ids"]
    sequences = tokenizer.batch_encode_plus(
        sequence_inputs, add_special_tokens=False, padding="longest", return_tensors="pt"
    )["input_ids"]
    import torch

    return torch.cat([structures, sequences], dim=1).to(device)


def run(config_path: Path, project_root: Path, device: str) -> None:
    import torch

    config = load_config(config_path)
    audit = audit_config(config_path, project_root)
    if audit["status"] != "ready":
        raise RuntimeError(f"DPLM-2 resources are incomplete: {audit['blockers']}")
    repo = resolved(audit, "official_repo")
    checkpoint = resolved(audit, "checkpoint")
    weights = resolved(audit, "checkpoint_weights")
    manifest_path = resolved(audit, "length_manifest")
    expected_bytes = int(config["resources"]["checkpoint_weights"].get("expected_bytes", 0))
    if expected_bytes and weights.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"DPLM-2 checkpoint size is {weights.stat().st_size}, expected {expected_bytes}; download is incomplete"
        )

    protocol = config["protocol"]
    total = int(protocol["requests"])
    batch_size = int(protocol["batch_size"])
    seed = int(protocol["seed"])
    max_iter = int(protocol["max_iter"])
    output = expand_path(protocol["output"], base=project_root)
    structure_output = expand_path(protocol["structure_output"], base=project_root)
    manifest = read_jsonl(manifest_path)
    if len(manifest) != total:
        raise ValueError(f"length manifest has {len(manifest)} rows, expected {total}")

    existing = read_jsonl(output) if output.exists() else []
    existing_structures = read_jsonl(structure_output) if structure_output.exists() else []
    if len(existing) != len(existing_structures):
        raise RuntimeError("sequence and structure resumable outputs are not aligned")
    completed = {row["request_id"]: row for row in existing}
    completed_structures = {row["request_id"]: row for row in existing_structures}
    if len(completed) == total:
        print(f"already complete: {output}")
        return

    Model = load_official_model_class(repo)
    model = Model.from_pretrained(
        str(checkpoint),
        cfg_override={
            "tokenizer": {"vocab_file": str(checkpoint)},
            "struct_tokenizer": {"enable": False},
        },
    ).eval().to(device)
    tokenizer = model.tokenizer

    for start in range(0, total, batch_size):
        rows = manifest[start : start + batch_size]
        ids = [f"dplm2_650m_{index:06d}" for index in range(start + 1, start + 1 + len(rows))]
        if all(request_id in completed for request_id in ids):
            continue
        if any(request_id in completed for request_id in ids):
            raise RuntimeError("partial DPLM-2 output is not aligned to a completed batch")
        lengths = [int(row["target_length"]) for row in rows]
        batch_seed = seed + start
        seed_everything(batch_seed)
        input_tokens = initial_tokens(tokenizer, lengths, device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            generated = model.generate(
                input_tokens=input_tokens,
                max_iter=max_iter,
                temperature=float(protocol["temperature"]),
                unmasking_strategy=str(protocol["unmasking_strategy"]),
                sampling_strategy=str(protocol["sampling_strategy"]),
            )
        structure_tokens, sequence_tokens = generated["output_tokens"].chunk(2, dim=1)
        sequences = [
            "".join(value.split())
            for value in tokenizer.batch_decode(sequence_tokens, skip_special_tokens=True)
        ]
        structures = [
            ",".join(value.split())
            for value in tokenizer.batch_decode(structure_tokens, skip_special_tokens=True)
        ]
        for request_id, source, sequence, structure in zip(ids, rows, sequences, structures, strict=True):
            completed[request_id] = build_generation_record(
                request_id=request_id,
                benchmark="unconditional",
                model="dplm2_650m",
                seed=seed + int(request_id.rsplit("_", 1)[1]) - 1,
                condition=f"target_length={source['target_length']}",
                raw_output=sequence,
                sequence=sequence,
                terminated=True,
                termination_reason="diffusion_iterations_complete",
                generation_method="sequence_structure_co_diffusion",
                generation_config_id=str(config["generation_config_id"]),
                method_metadata={
                    "target_length": int(source["target_length"]),
                    "actual_length": len(sequence),
                    "diffusion_iterations": max_iter,
                    "sampling_strategy": protocol["sampling_strategy"],
                    "unmasking_strategy": protocol["unmasking_strategy"],
                    "temperature": protocol["temperature"],
                    "batch_seed": batch_seed,
                    "length_source_id": source.get("length_source_id"),
                    "length_source_sequence_sha256": source.get("length_source_sequence_sha256"),
                    "structure_token_sha256": hashlib.sha256(structure.encode()).hexdigest(),
                },
            )
            completed_structures[request_id] = {
                "request_id": request_id,
                "target_length": int(source["target_length"]),
                "structure_tokens": structure,
                "structure_token_sha256": hashlib.sha256(structure.encode()).hexdigest(),
            }
        ordered_ids = [f"dplm2_650m_{index:06d}" for index in range(1, total + 1)]
        atomic_write_jsonl(output, [completed[x] for x in ordered_ids if x in completed])
        atomic_write_jsonl(structure_output, [completed_structures[x] for x in ordered_ids if x in completed_structures])
        print(f"DPLM-2 requests complete: {len(completed)}/{total}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, default=Path("configs/baselines/dplm2.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = audit_config(args.config, args.project_root)
    if args.report:
        atomic_write_json(args.report, report)
    print(f"DPLM-2 preflight: {report['status']} blockers={report['blockers']}")
    if args.action == "run":
        run(args.config, args.project_root, args.device)
    elif report["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
