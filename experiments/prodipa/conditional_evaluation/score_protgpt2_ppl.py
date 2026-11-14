#!/usr/bin/env python3
"""Score valid sequences with the frozen original ProtGPT2 reference model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


FIELDS = (
    "sequence_id",
    "bpe_tokens",
    "input_token_count",
    "scored_token_count",
    "nll_per_token",
    "nll",
    "ppl",
    "status",
    "error_reason",
)
PROTOCOL = "protgpt2_model_card_eos_newline_wrap60_newline_eos"
PRECISION = "float32"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_row(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS, delimiter="\t", lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {
        name: sha256_file(root / name)
        for name in names
        if (root / name).is_file()
    }


def combined_hash(hashes: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def format_model_card_text(sequence: str, eos_token: str) -> str:
    wrapped = "\n".join(
        sequence[index:index + 60]
        for index in range(0, len(sequence), 60)
    )
    return f"{eos_token}\n{wrapped}\n{eos_token}"


def shifted_mean_nll(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1].float()
    shifted_labels = input_ids[:, 1:]
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        reduction="mean",
    )


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, prefix=f".{path.name}."
    ) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def build_manifest(input_path: Path, model_path: Path) -> dict[str, object]:
    model_files = artifact_hashes(
        model_path,
        (
            "pytorch_model.bin",
            "model.safetensors",
            "config.json",
        ),
    )
    tokenizer_files = artifact_hashes(
        model_path,
        (
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer_config.json",
        ),
    )
    if not model_files or not tokenizer_files:
        raise ValueError(f"incomplete ProtGPT2 model/tokenizer artifacts: {model_path}")
    reproducibility = {
        "protocol": PROTOCOL,
        "precision": PRECISION,
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "model_path": str(model_path.resolve()),
        "model_sha256": combined_hash(model_files),
        "tokenizer_sha256": combined_hash(tokenizer_files),
        "scorer_sha256": sha256_file(Path(__file__).resolve()),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    return {
        "schema_version": 1,
        **reproducibility,
        "model_files": model_files,
        "tokenizer_files": tokenizer_files,
        "fingerprint": combined_hash({key: str(value) for key, value in reproducibility.items()}),
        "created_at": datetime.now().astimezone().isoformat(),
    }


def load_manifest(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def archive_stale_output(output: Path, manifest_path: Path) -> None:
    if not output.exists() and not manifest_path.exists():
        return
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
    history = output.parent / "history"
    history.mkdir(parents=True, exist_ok=True)
    for path in (output, manifest_path):
        if path.exists():
            os.replace(path, history / f"{stamp}_{path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows = read_tsv(args.input)
    sequence_ids = [row["sequence_id"] for row in rows]
    if len(sequence_ids) != len(set(sequence_ids)):
        raise ValueError("input sequence_id values must be unique")

    manifest_path = args.out.with_suffix(".manifest.json")
    expected_manifest = build_manifest(args.input, args.model)
    existing_manifest = load_manifest(manifest_path)
    cache_matches = (
        existing_manifest is not None
        and existing_manifest.get("fingerprint") == expected_manifest["fingerprint"]
    )
    if (args.out.exists() or manifest_path.exists()) and not cache_matches:
        archive_stale_output(args.out, manifest_path)
    atomic_json(manifest_path, expected_manifest)

    completed = read_tsv(args.out) if args.out.exists() else []
    done = {row["sequence_id"] for row in completed if row.get("status") == "ok"}
    if done == set(sequence_ids):
        print(f"reusing complete fingerprint-matched PPL table: {args.out}", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
        raise ValueError("ProtGPT2 tokenizer must define EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(args.device)
    model.eval()
    max_positions = int(
        getattr(model.config, "n_positions", 0)
        or getattr(model.config, "max_position_embeddings", 0)
    )

    for index, row in enumerate(rows, 1):
        if row["sequence_id"] in done:
            continue
        text = format_model_card_text(row["sequence"], tokenizer.eos_token)
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if token_ids[0] != tokenizer.eos_token_id or token_ids[-1] != tokenizer.eos_token_id:
            raise ValueError("ProtGPT2 model-card input did not tokenize to boundary EOS tokens")
        result = {
            "sequence_id": row["sequence_id"],
            "bpe_tokens": len(token_ids),
            "input_token_count": len(token_ids),
            "scored_token_count": max(len(token_ids) - 1, 0),
            "nll_per_token": "",
            "nll": "",
            "ppl": "",
            "status": "ok",
            "error_reason": "",
        }
        if max_positions and len(token_ids) > max_positions:
            result.update(
                status="failed",
                error_reason=f"bpe_tokens_exceed_context:{len(token_ids)}>{max_positions}",
            )
        else:
            ids = torch.tensor([token_ids], dtype=torch.long, device=args.device)
            try:
                with torch.inference_mode():
                    logits = model(input_ids=ids, use_cache=False).logits
                    nll = float(shifted_mean_nll(logits, ids))
                result["nll_per_token"] = f"{nll:.10g}"
                result["nll"] = f"{nll:.10g}"
                result["ppl"] = f"{math.exp(min(nll, 80.0)):.10g}"
            except Exception as error:
                result.update(status="failed", error_reason=repr(error))
                if "out of memory" in str(error).lower():
                    torch.cuda.empty_cache()
        append_row(args.out, result)
        print(f"PPL {index}/{len(rows)} status={result['status']}", flush=True)


if __name__ == "__main__":
    main()
