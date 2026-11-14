#!/usr/bin/env python3
"""Compute frozen TPR text-to-sequence cosine similarity scores."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


DEFAULT_TEXT = "Tetratricopeptide-like helical domain superfamily"


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", newline="", encoding="utf-8", dir=path.parent,
        delete=False, prefix=f".{path.name}.",
    ) as handle:
        writer = csv.DictWriter(
            handle, ["sequence_id", "protrek_percent", "status"],
            delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--condition-text", default=DEFAULT_TEXT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle, delimiter="\t"))
    if not records:
        raise ValueError("empty input sequence table")
    checkpoint = args.model / "ProTrek_35M.pt"
    protein_config = args.model / "esm2_t12_35M_UR50D"
    text_config = args.model / "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    for path in (checkpoint, protein_config, text_config):
        if not path.exists():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(args.repo.resolve()))
    from model.ProTrek.protein_encoder import ProteinEncoder
    from model.ProTrek.text_encoder import TextEncoder

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    protein = ProteinEncoder(str(protein_config), 1024, load_pretrained=False)
    text = TextEncoder(str(text_config), 1024, load_pretrained=False)
    protein_state = {key[2:]: value for key, value in state.items() if key.startswith("1.")}
    text_state = {key[2:]: value for key, value in state.items() if key.startswith("2.")}
    # Recent Transformers releases no longer persist this deterministic buffer.
    # Remove only that compatibility-only entry; all learned weights stay strict.
    protein_state.pop("model.esm.embeddings.position_ids", None)
    text_state.pop("model.embeddings.position_ids", None)
    protein.load_state_dict(protein_state, strict=True)
    text.load_state_dict(text_state, strict=True)
    device = torch.device(args.device)
    protein.to(device=device, dtype=torch.bfloat16).eval()
    text.to(device=device, dtype=torch.bfloat16).eval()
    for module in (protein, text):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        text_vector = text.get_repr([args.condition_text], batch_size=1)[0].float()
    output_rows: list[dict[str, object]] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            protein_vectors = protein.get_repr(
                [record["sequence"] for record in batch], batch_size=args.batch_size
            ).float()
            similarities = 100.0 * (protein_vectors @ text_vector)
        for record, score in zip(batch, similarities.tolist()):
            output_rows.append({
                "sequence_id": record["sequence_id"],
                "protrek_percent": f"{score:.10g}",
                "status": "ok",
            })
        print(f"processed={len(output_rows)}/{len(records)}", flush=True)

    write_tsv(args.output, output_rows)
    revision = subprocess.check_output(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
    ).strip()
    model_manifest = json.loads((args.model / "DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8"))
    manifest = {
        "definition": "100 * cosine(sequence embedding, frozen condition-text embedding)",
        "condition_text": args.condition_text,
        "queries": len(records),
        "repository": str(args.repo.resolve()),
        "repository_commit": revision,
        "model": str(args.model.resolve()),
        "model_revision": model_manifest["revision"],
        "checkpoint": str(checkpoint.resolve()),
        "batch_size": args.batch_size,
        "precision": "bfloat16 encoders; float32 cosine",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
