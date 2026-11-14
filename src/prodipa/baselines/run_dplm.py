from __future__ import annotations

import argparse
import importlib
import sys
import types
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


def _load_official_generation_class(repo: Path):
    """Load only DPLM's official inference modules, not unrelated training plugins.

    The upstream package initializer eagerly imports every datamodule/task and therefore
    requires the full training environment. Formal sequence generation only needs the
    registry, model utilities, ESM-DPLM network, and diffusion wrapper. Namespace stubs
    keep the checked-out source untouched while loading exactly those upstream modules.
    """
    source = repo / "src"
    byprot_root = source / "byprot"
    sys.path.insert(0, str(source))

    byprot = types.ModuleType("byprot")
    byprot.__path__ = [str(byprot_root)]
    models = types.ModuleType("byprot.models")
    models.__path__ = [str(byprot_root / "models")]
    models.MODEL_REGISTRY = {}

    def register_model(name):
        def decorator(cls):
            models.MODEL_REGISTRY[name] = cls
            return cls
        return decorator

    models.register_model = register_model
    utils = types.ModuleType("byprot.utils")
    utils.__path__ = [str(byprot_root / "utils")]

    def unavailable_training_yaml(*_args, **_kwargs):
        raise RuntimeError("training-only YAML loader was invoked during DPLM inference")

    utils.load_yaml_config = unavailable_training_yaml
    dplm_package = types.ModuleType("byprot.models.dplm")
    dplm_package.__path__ = [str(byprot_root / "models" / "dplm")]
    modules_package = types.ModuleType("byprot.models.dplm.modules")
    modules_package.__path__ = [str(byprot_root / "models" / "dplm" / "modules")]
    sys.modules.update({
        "byprot": byprot,
        "byprot.models": models,
        "byprot.utils": utils,
        "byprot.models.dplm": dplm_package,
        "byprot.models.dplm.modules": modules_package,
    })
    # DPLM pins transformers 4.39.2 and its ESM module imports internal classes
    # with ``from ...modeling_esm import *``.  Newer Transformers releases keep
    # those classes available but omit them from ``__all__``.  Restore the
    # upstream-visible export surface at runtime so the checked-out DPLM source
    # remains untouched and the rest of ProDiPA can use its pinned environment.
    esm_modeling = importlib.import_module("transformers.models.esm.modeling_esm")
    esm_exports = list(getattr(esm_modeling, "__all__", ()))
    for name in (
        "EsmAttention",
        "EsmContactPredictionHead",
        "EsmEmbeddings",
        "EsmEncoder",
        "EsmForMaskedLM",
        "EsmIntermediate",
        "EsmLMHead",
        "EsmLayer",
        "EsmModel",
        "EsmOutput",
        "EsmPooler",
        "EsmPreTrainedModel",
        "EsmSelfAttention",
        "EsmSelfOutput",
    ):
        if hasattr(esm_modeling, name) and name not in esm_exports:
            esm_exports.append(name)
    esm_modeling.__all__ = esm_exports
    importlib.import_module("byprot.models.utils")
    importlib.import_module("byprot.models.dplm.modules.dplm_modeling_esm")
    module = importlib.import_module("byprot.models.dplm.dplm")
    return module.DiffusionProteinLanguageModel


def run(config_path: Path, project_root: Path, device: str) -> None:
    config = load_config(config_path)
    audit = audit_config(config_path, project_root)
    if audit["status"] != "ready":
        raise RuntimeError(f"DPLM resources are incomplete: {audit['blockers']}")
    repo = _resolved(audit, "official_repo")
    model_path = _resolved(audit, "checkpoint")
    manifest_path = _resolved(audit, "length_manifest")
    import torch
    DiffusionProteinLanguageModel = _load_official_generation_class(repo)

    protocol = config["protocol"]
    total = int(protocol["requests"])
    batch_size = int(protocol["batch_size"])
    seed = int(protocol["seed"])
    max_iter = int(protocol["max_iter"])
    sampling_strategy = str(protocol["sampling_strategy"])
    output = expand_path(protocol["output"], base=project_root)
    manifest = read_jsonl(manifest_path)
    if len(manifest) != total:
        raise ValueError(f"length manifest has {len(manifest)} rows, expected {total}")

    existing = read_jsonl(output) if output.exists() else []
    if len(existing) > total or len(existing) % batch_size != 0:
        raise RuntimeError("partial DPLM output is not aligned to a completed batch")
    if len(existing) == total:
        print(f"already complete: {output}")
        return

    model = DiffusionProteinLanguageModel.from_pretrained(str(model_path)).eval().to(device)
    tokenizer = model.tokenizer
    records = list(existing)
    for start in range(len(existing), total, batch_size):
        batch_manifest = manifest[start : start + batch_size]
        batch_seed = seed + start
        seed_everything(batch_seed)
        masked = ["<mask>" * int(row["target_length"]) for row in batch_manifest]
        encoded = tokenizer.batch_encode_plus(
            masked, add_special_tokens=True, padding="longest", return_tensors="pt"
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
            records.append(
                build_generation_record(
                    request_id=str(row["request_id"]),
                    benchmark="unconditional",
                    model="dplm_650m",
                    seed=seed,
                    condition=f"target_length={row['target_length']}",
                    raw_output=sequence,
                    sequence=sequence,
                    terminated=True,
                    termination_reason="diffusion_iterations_complete",
                    generation_method="masked_diffusion",
                    generation_config_id=str(config["generation_config_id"]),
                    method_metadata={
                        "raw_token_ids": token_ids,
                        "target_length": int(row["target_length"]),
                        "actual_length": len(sequence),
                        "diffusion_iterations": max_iter,
                        "sampling_strategy": sampling_strategy,
                        "temperature_cli_effective": False,
                        "batch_seed": batch_seed,
                        "length_source_id": row.get("length_source_id"),
                        "length_source_sequence_sha256": row.get("length_source_sequence_sha256"),
                    },
                )
            )
        atomic_write_jsonl(output, records)
        print(f"DPLM requests complete: {len(records)}/{total}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="DPLM-650M adapter for ProDiPA")
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, default=Path("configs/baselines/dplm.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = audit_config(args.config, args.project_root)
    if args.report:
        atomic_write_json(args.report, report)
    print(f"DPLM preflight: {report['status']} blockers={report['blockers']}")
    if args.action == "run":
        run(args.config, args.project_root, args.device)
    elif report["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
