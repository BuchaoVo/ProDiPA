#!/usr/bin/env python3
"""Assign TPR candidates to project-aware 30%/80% homology groups."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import shutil
import subprocess
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--membership", type=Path, default=Path("data/provenance/split_membership.tsv.gz"))
    parser.add_argument("--mmseqs", type=Path, default=Path("tools/mmseqs/bin/mmseqs"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    candidates = read_tsv(args.candidates)
    membership = read_tsv(args.membership)
    by_sequence: dict[str, dict] = {}
    for row in membership:
        item = by_sequence.setdefault(row["sequence"], {"project": [], "candidates": []})
        item["project"].append(row)
    for row in candidates:
        by_sequence.setdefault(row["sequence"], {"project": [], "candidates": []})["candidates"].append(row)
    unique = [(f"SEQ{index:07d}", sequence, meta)
              for index, (sequence, meta) in enumerate(sorted(by_sequence.items()), 1)]
    args.work.mkdir(parents=True, exist_ok=True)
    fasta = args.work / "combined.fasta"
    with fasta.open("w", encoding="ascii") as handle:
        for sequence_id, sequence, _ in unique:
            handle.write(f">{sequence_id}\n{sequence}\n")
    prefix = args.work / "clusters"
    temporary = args.work / "tmp"
    if (prefix.parent / f"{prefix.name}_cluster.tsv").exists():
        (prefix.parent / f"{prefix.name}_cluster.tsv").unlink()
    subprocess.run([
        str(args.mmseqs), "easy-cluster", str(fasta), str(prefix), str(temporary),
        "--min-seq-id", "0.30", "-c", "0.80", "--cov-mode", "0",
        "--cluster-mode", "0", "--threads", str(args.threads),
    ], check=True)
    cluster_file = prefix.parent / f"{prefix.name}_cluster.tsv"
    cluster_by_member: dict[str, str] = {}
    with cluster_file.open(encoding="utf-8") as handle:
        for representative, member in csv.reader(handle, delimiter="\t"):
            cluster_by_member[member] = representative
    meta_by_id = {sequence_id: (sequence, meta) for sequence_id, sequence, meta in unique}
    cluster_context: dict[str, dict[str, set[str]]] = {}
    for sequence_id, (_, meta) in meta_by_id.items():
        cluster = cluster_by_member[sequence_id]
        context = cluster_context.setdefault(cluster, {"project_splits": set(), "legacy_splits": set()})
        context["project_splits"].update(row["split"] for row in meta["project"])
        context["legacy_splits"].update(row["legacy_split"] for row in meta["candidates"] if row.get("legacy_split"))

    output = []
    for sequence_id, (sequence, meta) in meta_by_id.items():
        if not meta["candidates"]:
            continue
        cluster = cluster_by_member[sequence_id]
        context = cluster_context[cluster]
        project_splits = context["project_splits"]
        legacy_splits = context["legacy_splits"]
        unusable = len(project_splits) > 1 or len(legacy_splits) > 1
        if project_splits:
            inherited = next(iter(project_splits)) if len(project_splits) == 1 else "excluded"
            split = inherited if inherited in {"train", "validation"} else "excluded"
            evidence = "frozen_project_split"
        elif legacy_splits:
            split = next(iter(legacy_splits)) if len(legacy_splits) == 1 else "excluded"
            evidence = "retained_legacy_split"
        else:
            digest = hashlib.sha256(cluster.encode()).hexdigest()
            split = "test" if int(digest[:8], 16) % 4 == 0 else "train"
            evidence = "deterministic_preselection_25pct_test"
        if unusable:
            split, evidence = "excluded", "cross_split_cluster_conflict"
        group = "TPRHG_" + hashlib.sha256(cluster.encode()).hexdigest()[:16]
        for row in meta["candidates"]:
            output.append(row | {
                "homology_group": group, "split": split,
                "split_evidence": evidence, "cluster_representative": cluster,
            })
    fields = list(output[0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(output)
    print(f"candidates={len(output)} groups={len({r['homology_group'] for r in output})} path={args.out}")
    shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
