#!/usr/bin/env python3
"""Validate and atomically merge fixed-budget generation shards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    fields: list[str] | None = None
    rows: list[dict[str, str]] = []
    metadata: list[dict] = []
    for shard in range(args.num_shards):
        path = args.input_dir / f"shard_{shard}.tsv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            current_fields = list(reader.fieldnames or [])
            if fields is None:
                fields = current_fields
            elif current_fields != fields:
                raise ValueError(f"Schema mismatch in {path}")
            rows.extend(reader)
        metadata.append(json.loads(path.with_suffix(".metadata.json").read_text()))

    ids = [int(row["sample_id"]) for row in rows]
    expected = list(range(args.num_sequences))
    if sorted(ids) != expected:
        raise ValueError("Generation shards do not contain every sample ID exactly once")
    if len({row["model"] for row in rows}) != 1 or len({row["paradigm"] for row in rows}) != 1:
        raise ValueError("Model or paradigm mismatch across generation shards")
    rows.sort(key=lambda row: int(row["sample_id"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields or [], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)

    invariant_keys = (
        "paradigm", "model", "base_model", "adapter", "adapter_weights",
        "num_requests", "base_seed", "decoding", "superfamily",
    )
    canonical = metadata[0]
    for record in metadata[1:]:
        for key in invariant_keys:
            if record.get(key) != canonical.get(key):
                raise ValueError(f"Metadata mismatch for {key}")
    canonical["sample_id_start"] = 0
    canonical["sample_id_stop"] = args.num_sequences
    canonical["generation_shards"] = args.num_shards
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(canonical, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
