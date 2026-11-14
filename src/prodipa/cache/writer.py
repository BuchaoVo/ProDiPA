"""Immutable cache writer used by synthetic and interface tests.

Production Parquet shards remain gated on the pinned GPU/data environment; the
same CacheRecord validation must run before their writer is introduced.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schema import CacheRecord


def write_jsonl_shard(path, records):
    path = Path(path)
    if path.suffix != ".jsonl":
        raise ValueError("portable fixture writer only accepts .jsonl shards")
    if path.exists():
        raise FileExistsError(f"cache shards are immutable: {path}")
    records = list(records)
    for record in records:
        if not isinstance(record, CacheRecord):
            raise TypeError("cache writer accepts CacheRecord values only")
        record.validate_for_training()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True, separators=(",", ":")) + "\n")
    return path
