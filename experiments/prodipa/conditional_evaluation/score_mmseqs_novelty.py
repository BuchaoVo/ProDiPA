#!/usr/bin/env python3
"""Compute frozen nearest-neighbour novelty with the bundled MMseqs2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


FIELDS = (
    "sequence_id", "best_reference_id", "max_identity_01", "novelty_01",
    "novelty_percent", "alignment_length", "bits", "evalue", "status",
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    reference_group = parser.add_mutually_exclusive_group(required=True)
    reference_group.add_argument("--reference", type=Path)
    reference_group.add_argument(
        "--reference-db", type=Path,
        help="Existing MMseqs2 sequence database prefix. It is reused without rebuilding.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--split-memory-limit",
        help="MMseqs2 split-memory-limit (for example 300G); keeps full UniProtKB searches bounded",
    )
    parser.add_argument("--tmp-dir", type=Path)
    parser.add_argument("--reference-label", required=True)
    args = parser.parse_args()

    for path in (args.input, args.mmseqs):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.reference is not None and not args.reference.exists():
        raise FileNotFoundError(args.reference)
    if args.reference_db is not None and not args.reference_db.exists():
        raise FileNotFoundError(args.reference_db)
    rows = read_tsv(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.tmp_dir is not None:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prodipa_mmseqs_", dir=args.tmp_dir) as temporary:
        tmp = Path(temporary)
        query_fasta = tmp / "query.fasta"
        with query_fasta.open("w", encoding="ascii") as handle:
            for row in rows:
                handle.write(f">{row['sequence_id']}\n{row['sequence']}\n")
        query_db = tmp / "query_db"
        reference_db = args.reference_db.resolve() if args.reference_db is not None else tmp / "reference_db"
        result_db, work = tmp / "result_db", tmp / "work"
        result_tsv = tmp / "results.tsv"
        exe = str(args.mmseqs.resolve())
        run([exe, "createdb", str(query_fasta), str(query_db)])
        if args.reference is not None:
            run([exe, "createdb", str(args.reference.resolve()), str(reference_db)])
        search_command = [
            exe, "search", str(query_db), str(reference_db), str(result_db), str(work),
            "--min-seq-id", "0", "--threads", str(args.threads),
        ]
        if args.split_memory_limit:
            search_command.extend(["--split-memory-limit", args.split_memory_limit])
        run(search_command)
        run([
            exe, "convertalis", str(query_db), str(reference_db), str(result_db),
            str(result_tsv), "--format-output",
            "query,target,pident,alnlen,evalue,bits", "--threads", str(args.threads),
        ])
        best: dict[str, tuple] = {}
        if result_tsv.exists():
            with result_tsv.open(newline="", encoding="utf-8") as handle:
                for query, target, pident, alnlen, evalue, bits in csv.reader(handle, delimiter="\t"):
                    candidate = (float(pident), float(bits), target, int(alnlen), float(evalue))
                    if query not in best or candidate[:2] > best[query][:2]:
                        best[query] = candidate

    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            hit = best.get(row["sequence_id"])
            if hit is None:
                writer.writerow({
                    "sequence_id": row["sequence_id"], "best_reference_id": "",
                    "max_identity_01": 0, "novelty_01": 1, "novelty_percent": 100,
                    "alignment_length": "", "bits": "", "evalue": "", "status": "no_hit",
                })
                continue
            identity = hit[0] / 100.0
            writer.writerow({
                "sequence_id": row["sequence_id"], "best_reference_id": hit[2],
                "max_identity_01": f"{identity:.10g}", "novelty_01": f"{1-identity:.10g}",
                "novelty_percent": f"{100*(1-identity):.10g}",
                "alignment_length": hit[3], "bits": f"{hit[1]:.10g}",
                "evalue": f"{hit[4]:.10g}", "status": "ok",
            })
    manifest = {
        "reference_label": args.reference_label,
        "reference_path": str((args.reference or args.reference_db).resolve()),
        "reference_kind": "mmseqs_database" if args.reference_db is not None else "fasta",
        "mmseqs_path": str(args.mmseqs.resolve()),
        "mmseqs_version": subprocess.check_output([str(args.mmseqs), "version"], text=True).strip(),
        "search": {
            "min_seq_id": 0,
            "threads": args.threads,
            "split_memory_limit": args.split_memory_limit,
        },
        "temporary_root": str(args.tmp_dir.resolve()) if args.tmp_dir is not None else None,
        "definition": "100 * (1 - maximum MMseqs2 pident / 100) among returned search hits",
        "queries": len(rows),
        "query_input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
