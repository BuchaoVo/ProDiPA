#!/usr/bin/env python3
"""Prepare auditable candidate tables for the stricter teacher-data rebuild."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
ARCHIVE_PREFIX = "ProDiPA_frozen_dataset_2026-11-02"


def read_release_tsv(archive: Path, name: str) -> list[dict[str, str]]:
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.extractfile(f"{ARCHIVE_PREFIX}/{name}")
        if member is None:
            raise FileNotFoundError(name)
        with gzip.GzipFile(fileobj=member) as compressed:
            text = io.TextIOWrapper(compressed, encoding="utf-8")
            return list(csv.DictReader(text, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty candidate table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def scoring_view(row: dict[str, str]) -> dict[str, str]:
    return {
        "sequence_id": row["sequence_id"],
        "sequence": row["sequence"],
        "length_aa": row.get("length_aa", str(len(row["sequence"]))),
        "homology_group": row.get("homology_group", ""),
        "split": row.get("split", ""),
        "candidate_source": row.get("candidate_source", "frozen_split_membership"),
        "valid_complete": "true",
    }


def prepare_unconditional(args: argparse.Namespace) -> dict[str, int]:
    train25 = read_release_tsv(args.release_archive, "train_25000_complete.tsv.gz")
    val1 = read_release_tsv(args.release_archive, "validation_1000_complete.tsv.gz")
    scored_ids = {row["sequence_id"] for row in train25 + val1}
    membership: list[dict[str, str]] = []
    with gzip.open(args.split_membership, "rt", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["sequence_id"] in scored_ids:
                continue
            if not (50 <= int(row["length_aa"]) <= 512 and set(row["sequence"]) <= AA):
                continue
            membership.append(row)
    validation = [scoring_view(row | {"candidate_source": "frozen_validation_outside_1k"})
                  for row in membership if row["split"] == "validation"]
    train_short = [scoring_view(row | {"candidate_source": "frozen_train_outside_25k_short"})
                   for row in membership
                   if row["split"] == "train" and int(row["length_aa"]) <= 149]
    write_tsv(args.out / "unconditional_validation_unscored.tsv", validation)
    write_tsv(args.out / "unconditional_train_short_unscored.tsv", train_short)
    return {"validation_unscored": len(validation), "train_short_unscored": len(train_short)}


def prepare_tpr_seeds(args: argparse.Namespace) -> dict[str, int]:
    rows_by_sequence: dict[str, dict[str, str]] = {}
    conflicts: list[str] = []
    for path in sorted(args.conditional_data.glob("*/*.json")):
        split = "train" if path.name.startswith("train_") else "test"
        attribute = "fold" if "plddt" in path.name else ("sol" if "/sol/" in str(path) else "thermo")
        for item in json.loads(path.read_text(encoding="utf-8")):
            sequence = item["output"].removeprefix("Seq=<").removesuffix(">")
            digest = hashlib.sha256(sequence.encode("ascii")).hexdigest()
            score_key = {"fold": "fold_score", "sol": "sol_score", "thermo": "thermo_score"}[attribute]
            raw_score = item["plddt" if attribute == "fold" else attribute]
            score = float(raw_score) * (100.0 if attribute == "fold" else 1.0)
            row = rows_by_sequence.setdefault(sequence, {
                "sequence_id": f"TPRLEGACY_{digest[:16]}",
                "sequence": sequence,
                "length_aa": str(len(sequence)),
                "legacy_split": split,
                "legacy_attributes": attribute,
                "cath_superfamily": "G3DSA:1.25.40.10",
                "family_evidence": "retained previously verified TPR teacher record",
                "tpr_coverage_union": "",
                "fold_score": "",
                "sol_score": "",
                "thermo_score": "",
                "valid_complete": "true",
            })
            if row["legacy_split"] != split:
                conflicts.append(row["sequence_id"])
            if attribute not in row["legacy_attributes"].split(";"):
                row["legacy_attributes"] += ";" + attribute
            existing = row[score_key]
            if existing and abs(float(existing) - score) > 1e-8:
                raise ValueError(f"Conflicting legacy score for {row['sequence_id']} / {attribute}")
            row[score_key] = f"{score:.9g}"
    if conflicts:
        raise ValueError(f"Legacy TPR sequence appears in train and test: {sorted(set(conflicts))[:5]}")
    rows = sorted(rows_by_sequence.values(), key=lambda row: row["sequence_id"])
    write_tsv(args.out / "tpr_legacy_candidates.tsv", rows)
    return {"tpr_legacy_unique": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-archive", type=Path,
                        default=Path("releases/zenodo/ProDiPA_frozen_dataset_2026-11-02.tar.gz"))
    parser.add_argument("--split-membership", type=Path,
                        default=Path("data/provenance/split_membership.tsv.gz"))
    parser.add_argument("--conditional-data", type=Path, default=Path("data/conditional"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = prepare_unconditional(args) | prepare_tpr_seeds(args)
    (args.out / "candidate_preparation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
