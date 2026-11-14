#!/usr/bin/env python3
"""Freeze configs for the canonical Sol V3 three-teacher training run."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from prodipa.unconditional.prefix_data import scan_dataset


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/unconditional"
OUT = ROOT / "outputs/experiments/unconditional_teacher_sol_v3_fasta60_v1"
MODEL = Path("__PRODIPA_MODELS__/ProtGPT2")
TRAINER = ROOT / "src/prodipa/unconditional/train_teacher.py"
FILES = {
    "fold": (DATA / "fold/train_plddt_300.csv", DATA / "fold/test_plddt_100.csv"),
    "sol": (DATA / "sol/train_sol_300.csv", DATA / "sol/test_sol_100.csv"),
    "thermo": (DATA / "thermo/train_thermo_300.csv", DATA / "thermo/test_thermo_100.csv"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    audit = json.loads((DATA / "FINAL_AUDIT.json").read_text(encoding="utf-8"))
    release = json.loads(
        (ROOT / "outputs/unconditional/protgpt2/CANONICAL_RELEASE.json").read_text(encoding="utf-8")
    )
    if audit.get("status") != "PASS_PUBLISHED_SOL_V3":
        raise RuntimeError("canonical Unconditional data audit is not PASS_PUBLISHED_SOL_V3")
    if release.get("sol_data_revision") != "Sol-Designable-V3-Strict70-15":
        raise RuntimeError("canonical release does not identify Sol V3 data")
    preflight = scan_dataset(DATA)
    if preflight["split_homology_overlap"] != 0:
        raise RuntimeError("canonical data has train/validation homology leakage")

    OUT.mkdir(parents=True, exist_ok=True)
    config_dir = OUT / "teachers/configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    source_revision = "sha256:" + sha256(TRAINER)
    previous_config = ROOT / "outputs/unconditional/protgpt2/fold/training_config.yaml"
    base_sha = "5474ac7a495a424e04b54b32c6a39d13713c4413d5d01f6145e84de65f6818a3"
    if previous_config.is_file():
        previous = yaml.safe_load(previous_config.read_text(encoding="utf-8"))
        base_sha = previous.get("base_model_sha256", base_sha)

    configs = {}
    for attribute, (train_path, validation_path) in FILES.items():
        payload = {
            "property_name": attribute,
            "model_name_or_path": str(MODEL),
            "tokenizer_name_or_path": str(MODEL),
            "train_path": str(train_path),
            "validation_path": str(validation_path),
            "output_dir": str(OUT / "teachers" / attribute),
            "text_column": "sequence",
            "id_column": "sequence_id",
            "num_virtual_tokens": 100,
            "max_length": 512,
            "learning_rate": 0.005,
            "lr_end": 1.0e-7,
            "lr_power": 3.0,
            "max_epochs": 50,
            "micro_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "seed": 42,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.0001,
            "dtype": "bfloat16",
            "num_workers": 0,
            "log_every_optimizer_steps": 10,
            "resume": True,
            "base_model_sha256": base_sha,
            "source_code_revision": source_revision,
            "generation_samples_per_epoch": 0,
            "sequence_protocol": "protgpt2_fasta60_v1",
        }
        config_path = config_dir / f"{attribute}.yaml"
        serialized = yaml.safe_dump(payload, sort_keys=False)
        if config_path.exists() and config_path.read_text(encoding="utf-8") != serialized:
            if (OUT / "teachers" / attribute / "latest_resume/training_state.pt").exists():
                raise RuntimeError(f"refusing to change a resumable config: {config_path}")
        config_path.write_text(serialized, encoding="utf-8")
        configs[attribute] = {
            "config": str(config_path),
            "train_sha256": sha256(train_path),
            "validation_sha256": sha256(validation_path),
            "output_dir": payload["output_dir"],
        }

    protocol = {
        "status": "READY",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "name": "unconditional_three_prefix_teachers_sol_v3_fasta60_v1",
        "data_revision": "Sol-Designable-V3-Strict70-15",
        "canonical_data_audit_sha256": sha256(DATA / "FINAL_AUDIT.json"),
        "canonical_manifest_sha256": sha256(DATA / "selected_sequence_manifest.tsv"),
        "split_homology_overlap": 0,
        "sequence_protocol": "protgpt2_fasta60_v1",
        "base_model": str(MODEL),
        "base_model_sha256": base_sha,
        "trainer_sha256": sha256(TRAINER),
        "training": {
            "parameterization": "Prefix-Tuning",
            "num_virtual_tokens": 100,
            "max_epochs": 50,
            "early_stopping_patience": 5,
            "effective_batch_size": 8,
            "seed": 42,
            "gpu_assignment": {"fold": 0, "sol": 1, "thermo": 2},
            "gpu_3_role": "reserved_for_post_training_generation_and_evaluation",
        },
        "configs": configs,
        "canonical_preflight": preflight,
        "current_canonical_teacher_weights_unchanged": True,
    }
    (OUT / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (OUT / "STATUS.json").write_text(json.dumps({
        "state": "READY", "stage": "teacher_training",
        "detail": "configs frozen; current canonical teachers unchanged",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "READY", "output": str(OUT), "configs": configs}, indent=2))


if __name__ == "__main__":
    main()
