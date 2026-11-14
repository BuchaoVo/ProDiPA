#!/usr/bin/env python3
"""Remove existing homology and select one P2 candidate per new 30/80 group."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/P2_high_sol_reviewed_candidates.tsv"),
    )
    parser.add_argument(
        "--existing", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/strict_candidate_manifest_expanded.tsv"),
    )
    parser.add_argument("--mmseqs", type=Path, default=Path("tools/mmseqs/bin/mmseqs"))
    parser.add_argument(
        "--work", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/P2_homology_work"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/P2_structure_queue.tsv"),
    )
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    existing = read_tsv(args.existing)
    candidates = read_tsv(args.candidates)
    args.work.mkdir(parents=True, exist_ok=True)
    fasta = args.work / "combined.fasta"
    sequence_by_internal: dict[str, str] = {}
    source_by_internal: dict[str, str] = {}
    candidate_by_internal: dict[str, dict[str, str]] = {}
    with fasta.open("w", encoding="ascii") as handle:
        for index, row in enumerate(existing):
            internal = f"OLD{index:07d}"
            sequence_by_internal[internal] = row["sequence"]
            source_by_internal[internal] = "existing"
            handle.write(f">{internal}\n{row['sequence']}\n")
        for index, row in enumerate(candidates):
            internal = f"NEW{index:07d}"
            sequence_by_internal[internal] = row["sequence"]
            source_by_internal[internal] = "new"
            candidate_by_internal[internal] = row
            handle.write(f">{internal}\n{row['sequence']}\n")

    prefix = args.work / "clusters"
    temporary = args.work / "tmp"
    cluster_file = args.work / "clusters_cluster.tsv"
    if not cluster_file.exists():
        subprocess.run([
            str(args.mmseqs), "easy-cluster", str(fasta), str(prefix), str(temporary),
            "--min-seq-id", "0.30", "-c", "0.80", "--cov-mode", "0",
            "--cluster-mode", "0", "-s", "7.5", "--threads", str(args.threads),
        ], check=True)

    cluster_by_member = {}
    with cluster_file.open(encoding="utf-8") as handle:
        for representative, member in csv.reader(handle, delimiter="\t"):
            cluster_by_member[member] = representative
    missing = set(sequence_by_internal) - set(cluster_by_member)
    if missing:
        raise RuntimeError(f"MMseqs output misses {len(missing)} inputs")

    members: dict[str, list[str]] = defaultdict(list)
    for member, representative in cluster_by_member.items():
        members[representative].append(member)
    overlapping_clusters = {
        representative for representative, cluster_members in members.items()
        if any(source_by_internal[member] == "existing" for member in cluster_members)
        and any(source_by_internal[member] == "new" for member in cluster_members)
    }
    new_only_clusters = {
        representative: [member for member in cluster_members if source_by_internal[member] == "new"]
        for representative, cluster_members in members.items()
        if all(source_by_internal[member] == "new" for member in cluster_members)
    }
    selected = []
    for representative, cluster_members in new_only_clusters.items():
        member = min(cluster_members, key=lambda internal: (
            -float(candidate_by_internal[internal]["solubility_scaled_01"]),
            int(candidate_by_internal[internal]["stable_rank"]),
            candidate_by_internal[internal]["sequence_id"],
        ))
        row = candidate_by_internal[member]
        representative_sequence = sequence_by_internal[representative]
        group = "SOLV3P2HG_" + hashlib.sha256(representative_sequence.encode()).hexdigest()[:16]
        selected.append(row | {
            "homology_group": group,
            "cluster_representative": representative,
            "cluster_member_count": str(len(cluster_members)),
            "homology_protocol": "MMseqs2_30pct_identity_80pct_bidirectional_coverage_s7.5",
        })
    selected.sort(key=lambda row: (-float(row["solubility_scaled_01"]), int(row["stable_rank"])))
    fields = list(selected[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)

    new_in_existing_clusters = sum(
        source_by_internal[member] == "new"
        for representative in overlapping_clusters for member in members[representative]
    )
    status = {
        "status": "READY_FOR_STRICT_ESMFOLD",
        "homology_protocol": {
            "min_sequence_identity": 0.30,
            "min_bidirectional_coverage": 0.80,
            "sensitivity": 7.5,
            "cluster_mode": 0,
        },
        "existing_strict_sequences": len(existing),
        "new_high_sol_sequences": len(candidates),
        "new_sequences_excluded_in_existing_clusters": new_in_existing_clusters,
        "new_only_homology_groups": len(new_only_clusters),
        "selected_one_per_new_group": len(selected),
        "new_redundant_sequences_not_selected": (
            len(candidates) - new_in_existing_clusters - len(selected)
        ),
        "output": str(args.output.resolve()),
        "training_started": False,
    }
    (args.output.parent / "P2_HOMOLOGY_STATUS.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
