#!/usr/bin/env python3
"""Select Sol V3 expansion candidates for strict ESMFold scoring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/P1_prefiltered_candidates.tsv"),
    )
    parser.add_argument(
        "--sol",
        type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/P1_protein_sol.tsv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/P1_high_sol_candidates.tsv"),
    )
    parser.add_argument("--threshold", type=float, default=0.70)
    args = parser.parse_args()

    rows = read_tsv(args.candidates)
    scores = {row["sequence_id"]: row for row in read_tsv(args.sol)}
    missing = [row["sequence_id"] for row in rows if row["sequence_id"] not in scores]
    failed = [
        row["sequence_id"]
        for row in rows
        if row["sequence_id"] in scores and scores[row["sequence_id"]]["status"] != "ok"
    ]
    if missing or failed:
        raise RuntimeError(f"incomplete Protein-Sol scores: missing={len(missing)} failed={len(failed)}")

    selected = []
    for row in rows:
        score = float(scores[row["sequence_id"]]["solubility_scaled_01"])
        if score >= args.threshold:
            selected.append(row | {"solubility_scaled_01": f"{score:.9g}"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) + ["solubility_scaled_01"]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)

    bins = Counter(row["length_bin"] for row in selected)
    # External expansion cohorts are clustered after the cheap Sol screen and
    # therefore do not have a homology-group label yet.  Treat each row as
    # provisionally distinct in that case; the status explicitly records this.
    has_homology_groups = all(row.get("homology_group") for row in selected)
    groups = Counter(
        row["homology_group"] if has_homology_groups else row["sequence_id"]
        for row in selected
    )
    status = {
        "status": "READY_FOR_STRICT_ESMFOLD" if selected else "NO_HIGH_SOL_CANDIDATES",
        "sol_threshold": args.threshold,
        "prefiltered_candidates": len(rows),
        "protein_sol_complete": len(rows),
        "selected_for_structure": len(selected),
        "selected_fraction": len(selected) / len(rows) if rows else None,
        "length_bins": dict(sorted(bins.items())),
        "unique_homology_groups": len(groups),
        "duplicate_candidates_beyond_one_per_group": sum(count - 1 for count in groups.values()),
        "homology_groups_provisional": not has_homology_groups,
        "inputs": {
            "candidates": str(args.candidates.resolve()),
            "candidates_sha256": sha256_file(args.candidates),
            "protein_sol": str(args.sol.resolve()),
            "protein_sol_sha256": sha256_file(args.sol),
        },
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "training_started": False,
    }
    status_path = args.output.parent / "P1_STRUCTURE_QUEUE_STATUS.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
