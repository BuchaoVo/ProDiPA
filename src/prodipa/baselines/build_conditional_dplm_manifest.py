from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

try:
    from .common import atomic_write_json, atomic_write_jsonl, sha256_file, sha256_text
except ImportError:  # direct script execution
    from common import atomic_write_json, atomic_write_jsonl, sha256_file, sha256_text


OUTPUT_PATTERN = re.compile(r"^Seq=<([ACDEFGHIKLMNPQRSTVWY]+)>$")
DEFAULT_INPUTS = (
    "data/conditional/fold/train_plddt_300.json",
    "data/conditional/sol/train_sol_300.json",
    "data/conditional/thermo/train_thermo_300.json",
)


def load_pool(project_root: Path, inputs: list[str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_sequence: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    for value in inputs:
        path = (project_root / value).resolve()
        source_hashes[value] = sha256_file(path)
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"expected a JSON list: {path}")
        property_name = path.parent.name
        for index, record in enumerate(records):
            match = OUTPUT_PATTERN.fullmatch(str(record.get("output", "")))
            if match is None:
                raise ValueError(f"invalid Conditional sequence wrapper: {path}:{index}")
            sequence = match.group(1)
            item = by_sequence.setdefault(sequence, {
                "sequence": sequence,
                "sequence_sha256": sha256_text(sequence),
                "memberships": [],
            })
            item["memberships"].append({
                "property": property_name,
                "source": value,
                "source_index": index,
            })
    return sorted(by_sequence.values(), key=lambda row: row["sequence_sha256"]), source_hashes


def build_manifest(
    pool: list[dict[str, Any]], *, requests: int, seed: int,
    motif_length: int, minimum_flank_length: int,
) -> list[dict[str, Any]]:
    if len(pool) < requests:
        raise ValueError(f"unique TPR pool has {len(pool)} sequences, requires {requests}")
    selected = list(pool)
    random.Random(seed).shuffle(selected)
    selected = selected[:requests]
    manifest: list[dict[str, Any]] = []
    for index, record in enumerate(selected):
        sequence = record["sequence"]
        if len(sequence) < motif_length + 2 * minimum_flank_length:
            raise ValueError(
                f"sequence length {len(sequence)} cannot support motif={motif_length} "
                f"and flanks={minimum_flank_length}: {record['sequence_sha256']}"
            )
        start_rng = random.Random(seed * 1_000_003 + index)
        start = start_rng.randint(minimum_flank_length, len(sequence) - motif_length - minimum_flank_length)
        end = start + motif_length
        manifest.append({
            "request_id": f"dplm_650m_tpr_infilling_{index:06d}",
            "sample_id": index,
            "seed": seed,
            "family_id": "G3DSA:1.25.40.10",
            "family_name": "Tetratricopeptide-like helical domain superfamily",
            "target_length": len(sequence),
            "motif_start_0based": start,
            "motif_end_exclusive": end,
            "motif_sequence": sequence[start:end],
            "source_sequence_sha256": record["sequence_sha256"],
            "source_memberships": record["memberships"],
        })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Conditional DPLM TPR motif requests")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--input", action="append", dest="inputs")
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/evaluation/conditional/dplm_650m_tpr_infilling_manifest.jsonl"),
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--motif-length", type=int, default=34)
    parser.add_argument("--minimum-flank-length", type=int, default=8)
    args = parser.parse_args()

    root = args.project_root.resolve()
    inputs = args.inputs or list(DEFAULT_INPUTS)
    output = args.output if args.output.is_absolute() else root / args.output
    report_path = args.report or output.with_suffix(".report.json")
    if not report_path.is_absolute():
        report_path = root / report_path
    pool, source_hashes = load_pool(root, inputs)
    manifest = build_manifest(
        pool, requests=args.requests, seed=args.seed,
        motif_length=args.motif_length, minimum_flank_length=args.minimum_flank_length,
    )
    atomic_write_jsonl(output, manifest)
    atomic_write_json(report_path, {
        "status": "PASS",
        "selection": "unique sequences sorted by SHA256, deterministic shuffle without replacement",
        "seed": args.seed,
        "unique_pool_size": len(pool),
        "requests": len(manifest),
        "motif_length": args.motif_length,
        "minimum_flank_length": args.minimum_flank_length,
        "source_sha256": source_hashes,
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
    })
    print(f"wrote {len(manifest)} Conditional DPLM requests to {output}")


if __name__ == "__main__":
    main()
