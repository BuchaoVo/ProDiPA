#!/usr/bin/env python3
"""Select a deterministic reviewed UniProtKB 350–512-aa expansion cohort.

The MMseqs lookup source column is joined with the sequence index in one
streaming pass.  Source 0 is verified against the adjacent ``.source`` file as
``uniprot_sprot.fasta.gz``.  MMseqs sequence-record lengths include the trailing
newline and NUL bytes, hence the requested amino-acid range maps to 352–514.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from pathlib import Path


def stable_rank(accession: str, key: str, seed: int) -> int:
    value = f"Sol-Designable-V3-P2|{seed}|{accession}|{key}".encode()
    return int.from_bytes(hashlib.sha256(value).digest(), "big")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("__PRODIPA_DATABASES__/uniprotkb/2026_03/mmseqs/uniprotkb_2026_03"),
    )
    parser.add_argument("--source-id", type=int, default=0)
    parser.add_argument("--min-aa", type=int, default=350)
    parser.add_argument("--max-aa", type=int, default=512)
    parser.add_argument("--max-selected", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rebuild/sol_designable_v3_strict70_15/P2_uniprot_reviewed_keys.tsv"),
    )
    args = parser.parse_args()

    source_lines = args.db.with_suffix(".source").read_text(encoding="utf-8").splitlines()
    sources = {}
    for line in source_lines:
        source, name = line.split("\t", 1)
        sources[int(source)] = name
    if args.source_id not in sources:
        raise ValueError(f"source id {args.source_id} absent from {args.db}.source")
    source_name = sources[args.source_id]
    if args.source_id == 0 and source_name != "uniprot_sprot.fasta.gz":
        raise ValueError(f"source 0 is not reviewed Swiss-Prot: {source_name}")

    lookup_path = args.db.with_suffix(".lookup")
    index_path = args.db.with_suffix(".index")
    heap: list[tuple[int, str, str, int]] = []
    scanned = source_rows = eligible = 0
    with lookup_path.open(encoding="utf-8") as lookup, index_path.open(encoding="utf-8") as index:
        for lookup_line, index_line in zip(lookup, index, strict=True):
            scanned += 1
            key, accession, source_text = lookup_line.rstrip("\n").split("\t")
            index_key, _offset, record_length_text = index_line.rstrip("\n").split("\t")
            if key != index_key:
                raise RuntimeError(f"lookup/index key mismatch at row {scanned}: {key} != {index_key}")
            if int(source_text) != args.source_id:
                continue
            source_rows += 1
            length_aa = int(record_length_text) - 2
            if not args.min_aa <= length_aa <= args.max_aa:
                continue
            eligible += 1
            rank = stable_rank(accession, key, args.seed)
            item = (-rank, key, accession, length_aa)
            if len(heap) < args.max_selected:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)

    selected = sorted(
        ((-negative_rank, key, accession, length) for negative_rank, key, accession, length in heap),
        key=lambda row: (row[0], int(row[1])),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["mmseqs_key", "accession", "length_aa", "stable_rank"])
        for rank, key, accession, length in selected:
            writer.writerow([key, accession, length, rank])
    key_path = args.output.with_suffix(".keys")
    key_path.write_text("".join(f"{key}\n" for _, key, _, _ in selected), encoding="ascii")
    status = {
        "status": "READY_FOR_MMSEQS_EXTRACTION",
        "database": str(args.db),
        "database_source_id": args.source_id,
        "database_source_name": source_name,
        "scanned_database_rows": scanned,
        "reviewed_source_rows": source_rows,
        "eligible_reviewed_350_512": eligible,
        "selected": len(selected),
        "selection": "smallest seeded SHA256 ranks without replacement",
        "seed": args.seed,
        "length_range_aa": [args.min_aa, args.max_aa],
        "mmseqs_record_length_range": [args.min_aa + 2, args.max_aa + 2],
        "output": str(args.output.resolve()),
        "keys": str(key_path.resolve()),
        "training_started": False,
    }
    status_path = args.output.parent / "P2_KEY_SELECTION_STATUS.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
