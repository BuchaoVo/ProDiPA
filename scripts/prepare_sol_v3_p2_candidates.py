#!/usr/bin/env python3
"""Prepare complete reviewed 350–512-aa Sol V3 P2 candidates."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

from build_unconditional_teacher_datasets_balanced import AA, complexity_metrics


def fasta_records(path: Path):
    header = None
    pieces: list[str] = []
    with path.open(encoding="ascii") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(pieces)
                header, pieces = line[1:], []
            else:
                pieces.append(line.strip())
    if header is not None:
        yield header, "".join(pieces)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fasta", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/P2_uniprot_reviewed.fasta"),
    )
    parser.add_argument(
        "--keys", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/P2_uniprot_reviewed_keys.tsv"),
    )
    parser.add_argument(
        "--membership", type=Path, default=Path("data/provenance/split_membership.tsv.gz"),
    )
    parser.add_argument(
        "--expanded-manifest", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/strict_candidate_manifest_expanded.tsv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/P2_prefiltered_reviewed_candidates.tsv"),
    )
    args = parser.parse_args()

    key_rows = read_tsv(args.keys)
    by_accession = {row["accession"]: row for row in key_rows}
    existing_sequences = set()
    with gzip.open(args.membership, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            existing_sequences.add(row["sequence"].strip().upper())
    for row in read_tsv(args.expanded_manifest):
        existing_sequences.add(row["sequence"].strip().upper())

    stages = {
        "fasta_records": 0,
        "reviewed_header": 0,
        "not_fragment_annotated": 0,
        "length_350_512": 0,
        "canonical": 0,
        "not_existing_exact_sequence": 0,
        "complexity_pass": 0,
        "unique_new_sequence": 0,
    }
    by_sequence: dict[str, dict[str, str]] = {}
    for header, raw_sequence in fasta_records(args.fasta):
        stages["fasta_records"] += 1
        match = re.match(r"sp\|([^|]+)\|", header)
        if not match:
            continue
        stages["reviewed_header"] += 1
        accession = match.group(1)
        if re.search(r"\bfragments?\b", header, flags=re.IGNORECASE):
            continue
        stages["not_fragment_annotated"] += 1
        sequence = raw_sequence.strip().upper()
        if not 350 <= len(sequence) <= 512:
            continue
        stages["length_350_512"] += 1
        if not set(sequence) <= AA:
            continue
        stages["canonical"] += 1
        if sequence in existing_sequences:
            continue
        stages["not_existing_exact_sequence"] += 1
        metrics = complexity_metrics(sequence)
        if not metrics["passes_complexity_gate"]:
            continue
        stages["complexity_pass"] += 1
        key = by_accession.get(accession)
        if key is None:
            raise RuntimeError(f"FASTA accession absent from key manifest: {accession}")
        row = {
            "sequence_id": f"P2SP_{accession}",
            "sequence": sequence,
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "length_aa": str(len(sequence)),
            "length_bin": "350-512",
            "accession": accession,
            "reviewed": "true",
            "fragment_annotation": "false",
            "uniprot_header": header,
            "mmseqs_key": key["mmseqs_key"],
            "stable_rank": key["stable_rank"],
            "source_release": "UniProtKB_2026_03_Swiss-Prot",
            "global_entropy_bits": f"{metrics['global_entropy_bits']:.9g}",
            "max_single_residue_fraction": f"{metrics['max_single_residue_fraction']:.9g}",
            "max_homopolymer_run": str(metrics["max_homopolymer_run"]),
            "low_complexity_window_fraction": f"{metrics['low_complexity_window_fraction']:.9g}",
            "unique_residues": str(metrics["unique_residues"]),
            "valid_complete": "true",
        }
        previous = by_sequence.get(sequence)
        if previous is None or int(row["stable_rank"]) < int(previous["stable_rank"]):
            by_sequence[sequence] = row

    rows = sorted(by_sequence.values(), key=lambda row: int(row["stable_rank"]))
    stages["unique_new_sequence"] = len(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    status = {
        "status": "READY_FOR_PROTEIN_SOL",
        "scope": "reviewed Swiss-Prot, complete-header, 350-512 aa only",
        "stages": stages,
        "candidates": len(rows),
        "existing_exact_sequences_excluded_against": len(existing_sequences),
        "rules": {
            "fragment_annotation_excluded": True,
            "canonical_only": True,
            "complexity_gate_relaxed": False,
            "length_range_aa": [350, 512],
            "include_250_349": False,
            "training_started": False,
        },
        "output": str(args.output.resolve()),
    }
    (args.output.parent / "P2_PREFILTER_STATUS.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
