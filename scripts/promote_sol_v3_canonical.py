#!/usr/bin/env python3
"""Promote audited Sol V3 data into the canonical Unconditional release."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from prodipa.unconditional.prefix_data import run_preflight
from prodipa.unconditional.teacher_sequence_protocol import (
    PROTGPT2_FASTA60_V1,
    encode_teacher_sequence,
    format_teacher_sequence,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/rebuild/sol_designable_v3_strict70_15"
TARGET = ROOT / "data/unconditional"
REFERENCES = ROOT / "outputs/evaluation/references"
TOKENIZER = Path("__PRODIPA_MODELS__/ProtGPT2")
RELEASE = ROOT / "outputs/unconditional/protgpt2/CANONICAL_RELEASE.json"
AA = set("ACDEFGHIKLMNPQRSTVWY")
BIN_NAMES = ("50-149", "150-249", "250-349", "350-512")
MANIFEST_FIELDS = [
    "dataset", "attribute", "sequence_id", "representative_accession", "cluster_name",
    "sequence_sha256", "homology_group", "source_split", "length_aa", "length_bin",
    "starts_with_m", "global_entropy_bits", "max_single_residue_fraction",
    "max_homopolymer_run", "low_complexity_window_fraction", "unique_residues",
    "fold_score", "fold_mean_pae", "fold_ptm_01", "sol_score", "thermo_score",
    "sol_structure_qualified",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    name = ""
    parts: list[str] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith(">"):
            if name:
                records[name] = "".join(parts)
            name, parts = line[1:], []
        else:
            parts.append(line.strip())
    if name:
        records[name] = "".join(parts)
    return records


def joint_group_by_hash() -> dict[str, str]:
    work = SOURCE / "P3_homology_work"
    sequences = read_fasta(work / "combined_candidates_fold_thermo.fasta")
    representatives: dict[str, str] = {}
    with (work / "clusters_cluster.tsv").open(encoding="utf-8") as handle:
        for representative, member in csv.reader(handle, delimiter="\t"):
            representatives[member] = representative
    if set(representatives) != set(sequences):
        raise RuntimeError("P3 joint-cluster membership is incomplete")
    return {
        hashlib.sha256(sequence.encode()).hexdigest():
        "SOLV3P3HG_" + hashlib.sha256(sequences[representatives[member]].encode()).hexdigest()[:16]
        for member, sequence in sequences.items()
    }


def build_manifest(stage: Path) -> None:
    old_rows = read_csv(TARGET / "selected_sequence_manifest.tsv", "\t")
    sol_rows = read_csv(SOURCE / "selected_sequence_manifest.tsv", "\t")
    group_by_hash = joint_group_by_hash()
    merged: list[dict[str, str]] = []
    for row in old_rows:
        if row["attribute"] == "sol":
            continue
        item = {field: row.get(field, "") for field in MANIFEST_FIELDS}
        item["homology_group"] = group_by_hash.get(row["sequence_sha256"], "")
        if not item["homology_group"]:
            raise RuntimeError(f"Fold/Thermo row missing from P3 joint clustering: {row['sequence_sha256']}")
        merged.append(item)
    for row in sol_rows:
        dataset = row["dataset"]
        item = {
            "dataset": dataset,
            "attribute": "sol",
            "sequence_id": row["sequence_id"],
            "representative_accession": row["representative_accession"],
            "cluster_name": row["cluster_name"],
            "sequence_sha256": row["sequence_sha256"],
            "homology_group": row["p3_homology_group"],
            "source_split": "train" if dataset == "train" else "validation",
            "length_aa": row["length_aa"],
            "length_bin": row["length_bin"],
            "starts_with_m": row["starts_with_m"],
            "global_entropy_bits": row["global_entropy_bits"],
            "max_single_residue_fraction": row["max_single_residue_fraction"],
            "max_homopolymer_run": row["max_homopolymer_run"],
            "low_complexity_window_fraction": row["low_complexity_window_fraction"],
            "unique_residues": row["unique_residues"],
            "fold_score": row["fold_plddt_100"],
            "fold_mean_pae": row["fold_mean_pae"],
            "fold_ptm_01": row["fold_ptm_01"],
            "sol_score": row["sol_score"],
            "thermo_score": "",
            "sol_structure_qualified": "true",
        }
        merged.append(item)
    order = {"test": 0, "train": 1}
    attributes = {"plddt": 0, "sol": 1, "thermo": 2}
    merged.sort(key=lambda row: (order[row["dataset"]], attributes[row["attribute"]], row["sequence_id"]))
    if len(merged) != 1200:
        raise RuntimeError(f"canonical manifest should have 1,200 rows, found {len(merged)}")
    with (stage / "selected_sequence_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, MANIFEST_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(merged)


def update_build_report(stage: Path) -> None:
    report = json.loads((TARGET / "build_report.json").read_text(encoding="utf-8"))
    p3 = json.loads((SOURCE / "P3_BUILD_REPORT.json").read_text(encoding="utf-8"))
    manifest = read_csv(SOURCE / "selected_sequence_manifest.tsv", "\t")
    by_split = {
        "train": [row for row in manifest if row["dataset"] == "train"],
        "test": [row for row in manifest if row["dataset"] == "test"],
    }
    for split, rows in by_split.items():
        lengths = [int(row["length_aa"]) for row in rows]
        scores = [float(row["sol_score"]) for row in rows]
        fold = [float(row["fold_plddt_100"]) for row in rows]
        pae = [float(row["fold_mean_pae"]) for row in rows]
        summary = {
            "n": len(rows),
            "length_mean": sum(lengths) / len(lengths),
            "length_median": sorted(lengths)[len(lengths) // 2],
            "length_min": min(lengths), "length_max": max(lengths),
            "length_bins": [sum(row["length_bin"] == name for row in rows) for name in BIN_NAMES],
            "attribute_mean_source_scale": sum(scores) / len(scores),
            "attribute_min_source_scale": min(scores),
            "attribute_max_source_scale": max(scores),
            "starts_with_m": sum(row["sequence"].startswith("M") for row in rows),
            "complexity_gate_failures": 0,
            "sol_structure_qualified": len(rows),
            "fold_score_mean": sum(fold) / len(fold),
            "fold_mean_pae_mean": sum(pae) / len(pae),
        }
        if split == "train":
            summary["length_ks_vs_test"] = p3["train_validation_length_ks"]
        report["datasets"][f"{split}_sol"] = summary
    report.update({
        "status": "PUBLISHED",
        "selection_protocol": "sol_designable_v3_strict70_15_p3_joint_homology_split",
        "published_sol_revision": "Sol-Designable-V3-Strict70-15",
        "published_sol_at": datetime.now(timezone.utc).isoformat(),
        "sol_structure_qualified_definition": {"min_fold_score": 70.0, "max_fold_mean_pae": 15.0},
        "cross_attribute_train_test_homology_overlap": 0,
        "pairwise_all_attribute_cross_split_homology_hits": 0,
        "sol_v3_p3_report_sha256": sha256(SOURCE / "P3_BUILD_REPORT.json"),
    })
    report["quotas"]["sol"] = {
        "train": [45, 132, 65, 58], "test": [15, 44, 22, 19],
    }
    report["candidate_capacities"]["sol"] = {
        "train": [p3["capacity_before_selection"][f"train:{name}"] for name in BIN_NAMES],
        "test": [p3["capacity_before_selection"][f"validation:{name}"] for name in BIN_NAMES],
    }
    (stage / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def write_tokenization_audit(stage: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)
    datasets: dict[str, dict[str, object]] = {}
    for path in sorted(stage.glob("*/*.csv")):
        rows = read_csv(path)
        lengths, newlines = [], []
        mismatch = 0
        for row in rows:
            sequence = row["sequence"].strip().upper()
            token_ids = encode_teacher_sequence(tokenizer, sequence, PROTGPT2_FASTA60_V1)
            text = format_teacher_sequence(sequence, PROTGPT2_FASTA60_V1)
            lengths.append(len(token_ids)); newlines.append(text.count("\n"))
            decoded = tokenizer.decode(token_ids[1:-1], clean_up_tokenization_spaces=False)
            mismatch += decoded != text
        if mismatch or any(value > 512 for value in lengths):
            raise RuntimeError(f"tokenization audit failed for {path}")
        datasets[path.name] = {
            "n": len(rows), "min_tokens_including_bos_eos": min(lengths),
            "mean_tokens_including_bos_eos": sum(lengths) / len(lengths),
            "max_tokens_including_bos_eos": max(lengths),
            "over_512_tokens": sum(value > 512 for value in lengths),
            "roundtrip_mismatch": mismatch, "min_text_newlines": min(newlines),
            "max_text_newlines": max(newlines),
        }
    payload = {
        "status": "PASS", "protocol": PROTGPT2_FASTA60_V1,
        "tokenizer": str(TOKENIZER), "eos_token_id": tokenizer.eos_token_id,
        "datasets": datasets,
    }
    (stage / "TOKENIZATION_AUDIT.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_readme(stage: Path) -> None:
    historical = (TARGET / "README.md").read_text(encoding="utf-8")
    marker = "## Previous balanced release notes"
    if marker in historical:
        historical = historical.split(marker, 1)[1].lstrip()
    text = f"""# Canonical Unconditional teacher data

Status: **PASS, PUBLISHED**.

## Sol-Designable-V3-Strict70-15

The canonical Sol Train/Validation data were replaced by the P3 joint-homology
split on {datetime.now(timezone.utc).date().isoformat()}.  The 300/100 rows use
length-bin quotas `[45, 132, 65, 58]` and `[15, 44, 22, 19]`; all satisfy
Sol>=0.70, pLDDT>=70 and mean pAE<=15.  Explicit Fragment records were excluded.

Sol, Fold and Thermo were jointly audited at 30% identity and 80% bidirectional
coverage.  There are zero direct Train–Validation homology hits across the three
attributes.  FASTA-60 remains the teacher input protocol.  The existing Sol
Teacher adapter predates this data revision and must be retrained before it can
be described as the Sol V3 Teacher.

Final Sol V3 provenance and audits are retained under `sol/SOL_V3_P3_*`.

## Previous balanced release notes

{historical}"""
    (stage / "README.md").write_text(text, encoding="utf-8")


def write_final_audit(stage: Path, preflight: dict[str, object]) -> None:
    files = {}
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "FINAL_AUDIT.json":
            files[str(path.relative_to(stage))] = sha256(path)
    audit = {
        "status": "PASS_PUBLISHED_SOL_V3",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "sol_revision": "Sol-Designable-V3-Strict70-15",
        "files": files,
        "preflight": preflight,
        "cross_attribute_train_test_homology_overlap": 0,
        "pairwise_all_attribute_cross_split_homology_hits": 0,
        "old_sol_v2_replaced": True,
    }
    (stage / "FINAL_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


def write_references(stage: Path) -> None:
    records: list[str] = []
    for path in (
        stage / "fold/train_plddt_300.csv", stage / "sol/train_sol_300.csv",
        stage / "thermo/train_thermo_300.csv",
    ):
        records.extend(row["sequence"].strip().upper() for row in read_csv(path))
    unique = sorted(set(records))
    fasta = REFERENCES.with_name(REFERENCES.name + f".next.{os.getpid()}")
    if fasta.exists():
        shutil.rmtree(fasta)
    shutil.copytree(REFERENCES, fasta)
    with (fasta / "unconditional_train_union.fasta").open("w", encoding="ascii") as handle:
        for sequence in unique:
            identifier = hashlib.sha256(sequence.encode("ascii")).hexdigest()
            handle.write(f">sha256:{identifier}\n{sequence}\n")
    manifest = json.loads((fasta / "novelty_t_reference_manifest.json").read_text(encoding="utf-8"))
    path = fasta / "unconditional_train_union.fasta"
    manifest["unconditional"] = {
        "path": str((REFERENCES / path.name).resolve()), "input_records": len(records),
        "unique_sequences": len(unique), "sha256": sha256(path),
        "data_revision": "Sol-Designable-V3-Strict70-15",
    }
    (fasta / "novelty_t_reference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def atomic_replace_dir(staged: Path, target: Path) -> None:
    old = target.with_name(target.name + f".replaced.{os.getpid()}")
    if old.exists():
        shutil.rmtree(old)
    os.replace(target, old)
    try:
        os.replace(staged, target)
    except BaseException:
        os.replace(old, target)
        raise
    shutil.rmtree(old)


def main() -> None:
    source_audit = json.loads((SOURCE / "FINAL_AUDIT.json").read_text(encoding="utf-8"))
    p3_report = json.loads((SOURCE / "P3_BUILD_REPORT.json").read_text(encoding="utf-8"))
    if source_audit.get("status") != "PASS_TRAINING_READY_NOT_STARTED" or p3_report.get("status") != "TRAINING_READY":
        raise RuntimeError("Sol V3 P3 source audit is not ready for publication")
    if p3_report["leakage_audit"].get("all_attribute_pairwise_cross_split_homology_hits") != 0:
        raise RuntimeError("Sol V3 P3 pairwise leakage audit is not zero")
    stage = TARGET.with_name(TARGET.name + f".next.{os.getpid()}")
    refs_stage = REFERENCES.with_name(REFERENCES.name + f".next.{os.getpid()}")
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(TARGET, stage)
    try:
        shutil.copy2(SOURCE / "train_sol_300.csv", stage / "sol/train_sol_300.csv")
        shutil.copy2(SOURCE / "test_sol_100.csv", stage / "sol/test_sol_100.csv")
        evidence = {
            "P3_BUILD_REPORT.json": "SOL_V3_P3_BUILD_REPORT.json",
            "P3_CROSS_ATTRIBUTE_HOMOLOGY_AUDIT.tsv": "SOL_V3_P3_CROSS_ATTRIBUTE_HOMOLOGY_AUDIT.tsv",
            "P3_PAIRWISE_CROSS_SPLIT_HITS.tsv": "SOL_V3_P3_PAIRWISE_CROSS_SPLIT_HITS.tsv",
            "selected_sequence_manifest.tsv": "SOL_V3_P3_SELECTED_MANIFEST.tsv",
            "FINAL_AUDIT.json": "SOL_V3_P3_SOURCE_FINAL_AUDIT.json",
        }
        for source_name, target_name in evidence.items():
            shutil.copy2(SOURCE / source_name, stage / "sol" / target_name)
        build_manifest(stage)
        update_build_report(stage)
        write_tokenization_audit(stage)
        write_readme(stage)
        preflight_path = stage / "FINAL_PREFLIGHT.json"
        preflight = run_preflight(stage, preflight_path)
        preflight["data_dir"] = str(TARGET.resolve())
        preflight["provenance_path"] = str((TARGET / "selected_sequence_manifest.tsv").resolve())
        preflight_path.write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
        write_final_audit(stage, preflight)
        write_references(stage)

        release = json.loads(RELEASE.read_text(encoding="utf-8"))
        release.update({
            "status": "PUBLISHED_DATA_UPDATED_TEACHER_RETRAIN_REQUIRED",
            "data_protocol": "sol_designable_v3_strict70_15_p3_joint_homology_split",
            "data_updated_at": datetime.now(timezone.utc).isoformat(),
            "sol_data_revision": "Sol-Designable-V3-Strict70-15",
            "sol_teacher_status": "EXISTING_ADAPTER_PREDATES_SOL_V3_RETRAIN_REQUIRED",
            "known_formal_evaluation_scope": "historical_pre_sol_v3_data",
        })
        release["data_sha256"] = {
            str(path.relative_to(stage)): sha256(path)
            for path in sorted(stage.glob("*/*.csv"))
        }
        release_tmp = RELEASE.with_name(RELEASE.name + f".next.{os.getpid()}")
        release_tmp.write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")

        atomic_replace_dir(stage, TARGET)
        atomic_replace_dir(refs_stage, REFERENCES)
        os.replace(release_tmp, RELEASE)

        # The user requested permanent removal rather than archival after the
        # final evidence was promoted into the canonical Sol directory.
        shutil.rmtree(SOURCE)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        if refs_stage.exists():
            shutil.rmtree(refs_stage)
        raise

    result = {
        "status": "PASS_PUBLISHED_AND_REBUILD_REMOVED",
        "canonical_sol_train_sha256": sha256(TARGET / "sol/train_sol_300.csv"),
        "canonical_sol_test_sha256": sha256(TARGET / "sol/test_sol_100.csv"),
        "canonical_manifest_sha256": sha256(TARGET / "selected_sequence_manifest.tsv"),
        "novelty_reference_sha256": sha256(REFERENCES / "unconditional_train_union.fasta"),
        "source_rebuild_directory_exists": SOURCE.exists(),
        "teacher_retraining_required": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
