#!/usr/bin/env python3
"""Fail-closed audit of models, adapters, data, evaluators, and external references."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--proteinopd-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--uniprotkb-fasta", type=Path)
    parser.add_argument("--protrek-root", type=Path)
    parser.add_argument("--protrek-model", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    required = [
        root / "tools/mmseqs/bin/mmseqs",
        root / "scripts/run_esmfold_properties.py",
        root / "scripts/run_protein_sol.py",
        root / "scripts/run_temBERTure_cls.py",
        Path("__PRODIPA_MODELS__/ProtGPT2"),
        Path("__PRODIPA_MODELS__/ProLLaMA"),
        Path("__PRODIPA_MODELS__/esmfold_v1_hf"),
    ]
    for paradigm, backbone, leaf in (
        ("unconditional", "protgpt2", "best_adapter"),
        ("conditional", "prollama", "final_adapter"),
    ):
        for prop in ("fold", "sol", "thermo"):
            required.append(root / "outputs" / paradigm / backbone / prop / leaf / "adapter_model.safetensors")
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    proteinopd_root = args.proteinopd_root.resolve()
    if not proteinopd_root.is_dir():
        raise FileNotFoundError(proteinopd_root)
    try:
        proteinopd_commit = subprocess.check_output(
            ["git", "-C", str(proteinopd_root), "rev-parse", "HEAD"], text=True,
        ).strip()
    except subprocess.CalledProcessError:
        proteinopd_commit = "UNKNOWN"

    mmseqs = root / "tools/mmseqs/bin/mmseqs"
    uniprot_ready = bool(args.uniprotkb_fasta and args.uniprotkb_fasta.is_file())
    protrek_ready = bool(
        args.protrek_root and args.protrek_root.is_dir()
        and args.protrek_model and args.protrek_model.exists()
    )
    report = {
        "status": "READY_FOR_AVAILABLE_STAGES",
        "proteinopd_reference": {
            "path": str(proteinopd_root),
            "git_commit": proteinopd_commit,
            "note": (
                "The checked repository exposes generation/training and result-table code. "
                "The frozen pooled min-max protocol provides the authoritative HV definition."
            ),
        },
        "mmseqs": {
            "path": str(mmseqs.resolve()),
            "version": subprocess.check_output([str(mmseqs), "version"], text=True).strip(),
        },
        "external_dependencies": {
            "novelty_u": {
                "ready": uniprot_ready,
                "uniprotkb_fasta": str(args.uniprotkb_fasta.resolve()) if uniprot_ready else None,
                "reason_if_missing": None if uniprot_ready else "No frozen full UniProtKB reference FASTA supplied.",
            },
            "protrek": {
                "ready": protrek_ready,
                "root": str(args.protrek_root.resolve()) if args.protrek_root and args.protrek_root.exists() else None,
                "model": str(args.protrek_model.resolve()) if args.protrek_model and args.protrek_model.exists() else None,
                "reason_if_missing": None if protrek_ready else "No local ProTrek code/checkpoint supplied.",
            },
            "hypervolume": {
                "ready": True,
                "definition": (
                    "Unconditional objectives: -log(1+PPL), Novelty-U, pLDDT, Sol, Thermo; "
                    "pooled per-sequence min-max normalization and zero reference point."
                ),
            },
        },
        "required_paths": [str(path.resolve()) for path in required],
        "adapter_sha256": {
            str(path.resolve()): sha256(path)
            for path in required if path.name == "adapter_model.safetensors"
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
