#!/usr/bin/env python3
"""Write a live aggregate progress log for parallel generation shards."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import time
from pathlib import Path


def row_count(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle, delimiter="\t"))


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.num_sequences % args.num_shards:
        raise ValueError("num-sequences must be divisible by num-shards")
    width = args.num_sequences // args.num_shards
    previous: tuple[int, ...] | None = None

    while True:
        counts = tuple(
            row_count(args.input_dir / f"shard_{shard}.tsv")
            for shard in range(args.num_shards)
        )
        if any(count > width for count in counts):
            raise ValueError(f"Shard count exceeds budget: {counts}")
        if counts != previous:
            total = sum(counts)
            timestamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
            detail = " ".join(
                f"gpu{shard}={count}/{width}"
                for shard, count in enumerate(counts)
            )
            print(
                f"[{timestamp}] total={total}/{args.num_sequences} "
                f"progress={100.0 * total / args.num_sequences:.2f}% {detail}",
                flush=True,
            )
            previous = counts
        if sum(counts) == args.num_sequences:
            print("generation shards complete", flush=True)
            return
        if not process_alive(args.watch_pid):
            raise SystemExit("generation pipeline stopped before all shards completed")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
