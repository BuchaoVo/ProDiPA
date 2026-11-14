#!/usr/bin/env python3
"""Rebuild compact canonical provenance after intermediate Sol artifacts are removed."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from prodipa.unconditional.prefix_data import run_preflight


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/unconditional"
WORK = ROOT / "outputs/experiments/unconditional_teacher_sol_v3_fasta60_v1/canonical_preflight_work"
MMSEQS = ROOT / "tools/mmseqs/bin/mmseqs"
FILES = {
    ("train", "plddt"): DATA / "fold/train_plddt_300.csv",
    ("train", "sol"): DATA / "sol/train_sol_300.csv",
    ("train", "thermo"): DATA / "thermo/train_thermo_300.csv",
    ("test", "plddt"): DATA / "fold/test_plddt_100.csv",
    ("test", "sol"): DATA / "sol/test_sol_100.csv",
    ("test", "thermo"): DATA / "thermo/test_thermo_100.csv",
}
EXPECTED_SOL_HASHES = {
    "train": "32eb2851d84019b536a20a7aaca0171b54173439950057dba8120a51f48aabba",
    "test": "41f344ec5bc5c5066d2b3bd6827bdb98a7c2342b9abe7e6010094e19aaa0b761",
}
AA = set("ACDEFGHIKLMNPQRSTVWY")
BINS = ((50, 149), (150, 249), (250, 349), (350, 512))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load() -> dict[tuple[str, str], list[dict[str, str]]]:
    datasets = {}
    for key, path in FILES.items():
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = 300 if key[0] == "train" else 100
        if len(rows) != required:
            raise RuntimeError(f"{path}: expected {required} rows, found {len(rows)}")
        seen = set()
        for row in rows:
            sequence = row["sequence"].strip().upper()
            if sequence in seen or not set(sequence) <= AA or not 50 <= len(sequence) <= 512:
                raise RuntimeError(f"invalid or duplicate sequence in {path}")
            seen.add(sequence); row["sequence"] = sequence
        datasets[key] = rows
    if sha256(FILES[("train", "sol")]) != EXPECTED_SOL_HASHES["train"]:
        raise RuntimeError("canonical Sol V3 train hash changed")
    if sha256(FILES[("test", "sol")]) != EXPECTED_SOL_HASHES["test"]:
        raise RuntimeError("canonical Sol V3 validation hash changed")
    return datasets


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="ascii") as handle:
        for identifier, sequence in records:
            handle.write(f">{identifier}\n{sequence}\n")


def main() -> None:
    datasets = load()
    WORK.mkdir(parents=True, exist_ok=True)
    all_records = []
    id_meta = {}
    for (split, attribute), rows in datasets.items():
        for index, row in enumerate(rows):
            identifier = f"{split.upper()}_{attribute}_{index:04d}"
            sequence = row["sequence"]
            all_records.append((identifier, sequence))
            id_meta[identifier] = (split, attribute, sequence)
    combined = WORK / "all_final_datasets.fasta"
    write_fasta(combined, all_records)
    prefix = WORK / "joint"
    cluster_tsv = WORK / "joint_cluster.tsv"
    subprocess.run([
        str(MMSEQS), "easy-cluster", str(combined), str(prefix), str(WORK / "cluster_tmp"),
        "--min-seq-id", "0.30", "-c", "0.80", "--cov-mode", "0",
        "--cluster-mode", "0", "-s", "7.5", "--threads", "32",
    ], check=True)
    representative = {}
    with cluster_tsv.open(encoding="utf-8") as handle:
        for rep, member in csv.reader(handle, delimiter="\t"):
            representative[member] = rep
    if set(representative) != set(id_meta):
        raise RuntimeError("joint homology clustering did not cover every row")

    # A direct search is stricter than relying only on set-cover membership.
    unique_by_split: dict[str, dict[str, set[str]]] = {
        "train": defaultdict(set), "test": defaultdict(set),
    }
    for identifier, (split, attribute, sequence) in id_meta.items():
        unique_by_split[split][sequence].add(attribute)
    train_records = [(f"Q{i:04d}", sequence) for i, sequence in enumerate(sorted(unique_by_split["train"]))]
    test_records = [(f"T{i:04d}", sequence) for i, sequence in enumerate(sorted(unique_by_split["test"]))]
    write_fasta(WORK / "train_unique.fasta", train_records)
    write_fasta(WORK / "test_unique.fasta", test_records)
    hits = WORK / "direct_cross_split_hits.tsv"
    if hits.exists():
        hits.unlink()
    subprocess.run([
        str(MMSEQS), "easy-search", str(WORK / "train_unique.fasta"),
        str(WORK / "test_unique.fasta"), str(hits), str(WORK / "search_tmp"),
        "--min-seq-id", "0.30", "-c", "0.80", "--cov-mode", "0", "-s", "7.5",
        "--max-seqs", "100000", "--threads", "32",
        "--format-output", "query,target,pident,qcov,tcov,evalue,alnlen",
    ], check=True)
    hit_rows = [line for line in hits.read_text(encoding="utf-8").splitlines() if line]
    if hit_rows:
        raise RuntimeError(f"direct Train/Validation homology audit found {len(hit_rows)} hits")

    manifest = []
    split_groups = {"train": set(), "test": set()}
    for identifier, (split, attribute, sequence) in id_meta.items():
        rep = representative[identifier]
        rep_sequence = id_meta[rep][2]
        group = "UCG_" + hashlib.sha256(rep_sequence.encode()).hexdigest()[:16]
        sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
        manifest.append({
            "dataset": split, "attribute": attribute,
            "sequence_sha256": sequence_hash, "homology_group": group,
        })
        split_groups[split].add(group)
    overlap = split_groups["train"] & split_groups["test"]
    if overlap:
        raise RuntimeError(f"cluster-level Train/Validation overlap: {len(overlap)}")
    manifest.sort(key=lambda row: (row["dataset"], row["attribute"], row["sequence_sha256"]))
    with (DATA / "selected_sequence_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, ["dataset", "attribute", "sequence_sha256", "homology_group"],
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader(); writer.writerows(manifest)

    quotas = {}
    summaries = {}
    for (split, attribute), rows in datasets.items():
        counts = [sum(lo <= len(row["sequence"]) <= hi for row in rows) for lo, hi in BINS]
        quotas.setdefault(attribute, {})[split] = counts
        property_column = {"plddt": "pLDDT", "sol": "sol", "thermo": "thermo"}[attribute]
        values = [float(row[property_column]) for row in rows]
        summaries[f"{split}_{attribute}"] = {
            "n": len(rows), "length_bins": counts,
            "mean_property": sum(values) / len(values), "sha256": sha256(FILES[(split, attribute)]),
        }
    report = {
        "status": "PUBLISHED",
        "selection_protocol": "sol_designable_v3_strict70_15_p3_joint_homology_split",
        "teacher_sequence_protocol": "protgpt2_fasta60_v1",
        "thresholds": {"plddt": 85.0, "sol": 0.70, "thermo": 0.60},
        "quotas": quotas, "datasets": summaries,
        "cross_attribute_train_test_homology_overlap": 0,
        "pairwise_all_attribute_cross_split_homology_hits": 0,
        "pairwise_train_unique_sequences": len(unique_by_split["train"]),
        "pairwise_validation_unique_sequences": len(unique_by_split["test"]),
        "metadata_rebuilt_after_intermediate_artifact_cleanup": True,
    }
    (DATA / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    preflight = run_preflight(DATA, DATA / "FINAL_PREFLIGHT.json")
    audit = {
        "status": "PASS_PUBLISHED_SOL_V3",
        "rebuilt_at": datetime.now(timezone.utc).isoformat(),
        "sol_revision": "Sol-Designable-V3-Strict70-15",
        "sol_train_sha256": EXPECTED_SOL_HASHES["train"],
        "sol_validation_sha256": EXPECTED_SOL_HASHES["test"],
        "manifest_sha256": sha256(DATA / "selected_sequence_manifest.tsv"),
        "split_homology_overlap": preflight["split_homology_overlap"],
        "pairwise_all_attribute_cross_split_homology_hits": 0,
        "train_unique_sequences": len(unique_by_split["train"]),
        "validation_unique_sequences": len(unique_by_split["test"]),
        "intermediate_sol_rebuild_artifacts_removed": True,
    }
    (DATA / "FINAL_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
