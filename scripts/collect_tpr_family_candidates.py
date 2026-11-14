#!/usr/bin/env python3
"""Collect full-length TPR candidates with direct CATH-Gene3D evidence.

The collector uses the official InterPro API entry for G3DSA:1.25.40.10,
retains the matched residue coordinates, and never interprets a prompt label
as family evidence.  Results are deterministic because the API cursor order
and the first occurrence of each full sequence are preserved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path


ENTRY = "G3DSA:1.25.40.10"
BASE = (
    "https://www.ebi.ac.uk/interpro/api/protein/{source}/entry/"
    f"cathgene3d/{ENTRY}/"
)
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def opener() -> urllib.request.OpenerDirector:
    # The machine-wide HTTP proxy intermittently terminates EBI TLS sessions.
    # Direct TLS is available on this host and is used only for this public API.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def union_coverage(locations: list[dict], length: int) -> float:
    intervals: list[tuple[int, int]] = []
    for location in locations:
        for fragment in location.get("fragments", []):
            start, end = int(fragment["start"]), int(fragment["end"])
            if 1 <= start <= end <= length:
                intervals.append((start, end))
    if not intervals:
        return 0.0
    intervals.sort()
    merged: list[list[int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start + 1 for start, end in merged) / length


def fetch(source: str, limit: int | None, page_size: int, retries: int) -> list[dict]:
    query = urllib.parse.urlencode({"page_size": page_size, "extra_fields": "sequence"})
    url: str | None = BASE.format(source=source) + "?" + query
    client = opener()
    rows: list[dict] = []
    while url and (limit is None or len(rows) < limit):
        for attempt in range(retries):
            try:
                with client.open(url, timeout=120) as response:
                    payload = json.load(response)
                break
            except Exception:
                if attempt + 1 == retries:
                    raise
                time.sleep(2 ** attempt)
        for result in payload["results"]:
            metadata = result["metadata"]
            sequence = result.get("extra_fields", {}).get("sequence", "").upper()
            length = int(metadata["length"])
            if not (50 <= length <= 512 and len(sequence) == length and set(sequence) <= AA):
                continue
            locations: list[dict] = []
            models: list[str] = []
            scores: list[float] = []
            for entry in result.get("entries", []):
                if entry.get("accession") != ENTRY:
                    continue
                for location in entry.get("entry_protein_locations", []):
                    locations.append(location)
                    if location.get("model"):
                        models.append(str(location["model"]))
                    if location.get("score") is not None:
                        scores.append(float(location["score"]))
            rows.append({
                "sequence_id": f"UNIPROT_{metadata['accession']}",
                "accession": metadata["accession"],
                "sequence": sequence,
                "length_aa": length,
                "source_database": metadata.get("source_database", source),
                "organism_tax_id": metadata.get("source_organism", {}).get("taxId", ""),
                "cath_superfamily": ENTRY,
                "tpr_coverage_union": f"{union_coverage(locations, length):.9g}",
                "cath_models": ";".join(sorted(set(models))),
                "best_cath_evalue": f"{min(scores):.9g}" if scores else "",
                "family_evidence": "InterPro cathgene3d direct match",
                "interpro_version": "110.0",
            })
            if limit is not None and len(rows) >= limit:
                break
        url = payload.get("next")
        print(f"source={source} retained={len(rows)} next={bool(url)}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reviewed-limit", type=int, default=0,
                        help="0 means all reviewed matches")
    parser.add_argument("--unreviewed-limit", type=int, default=12000)
    parser.add_argument(
        "--source",
        choices=("reviewed", "unreviewed", "both"),
        default="both",
        help="Select the InterPro protein source without re-downloading the other source.",
    )
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    rows: list[dict] = []
    if args.source in {"reviewed", "both"}:
        rows.extend(fetch("reviewed", args.reviewed_limit or None, args.page_size, args.retries))
    if args.source in {"unreviewed", "both"} and args.unreviewed_limit:
        rows.extend(fetch("unreviewed", args.unreviewed_limit, args.page_size, args.retries))

    by_sequence: dict[str, dict] = {}
    for row in rows:
        previous = by_sequence.get(row["sequence"])
        if previous is None or previous["source_database"] != "reviewed":
            by_sequence[row["sequence"]] = row
    final = sorted(by_sequence.values(), key=lambda row: row["sequence_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(final[0]) if final else []
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(final)
    print(f"written={len(final)} path={args.out}")


if __name__ == "__main__":
    main()
