#!/usr/bin/env python3
"""Run audited ESM2, ESM3, or ASPO iterative masked-token generation."""

from __future__ import annotations

import argparse
import csv
import sys
import types
from pathlib import Path

import numpy as np
import torch

try:
    from .common import atomic_write_jsonl, build_generation_record, seed_everything
except ImportError:
    from common import atomic_write_jsonl, build_generation_record, seed_everything


def references(path: Path, expected: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != expected:
        raise ValueError(f"reference manifest has {len(rows)} rows, expected {expected}")
    if len({row["homology_group"] for row in rows}) != expected:
        raise ValueError("reference manifest repeats homology groups")
    return rows


def load_vector(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"steering-vector file must contain (positive, negative): {path}")
    return value[0].float() - value[1].float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("esm2", "esm3", "aspo"), required=True)
    parser.add_argument("--steering-repo", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--mask-ratio", type=float, default=0.1)
    parser.add_argument("--esm2-checkpoint", type=Path)
    parser.add_argument("--fold-vector", type=Path)
    parser.add_argument("--sol-vector", type=Path)
    parser.add_argument("--thermo-vector", type=Path)
    args = parser.parse_args()

    seed_everything(args.seed)
    rows = references(args.reference_manifest, args.requests)
    sys.path.insert(0, str(args.steering_repo.resolve()))
    steering = None
    if args.backend in {"esm2", "aspo"}:
        if not args.esm2_checkpoint or not args.esm2_checkpoint.is_file():
            raise FileNotFoundError("--esm2-checkpoint is required")
        from utils.esm2_utils import load_esm2_model, generate_sequences

        model, alphabet = load_esm2_model(
            "650M", device=args.device, ckpt_path=str(args.esm2_checkpoint)
        )
        batch_converter = alphabet.get_batch_converter()
        if args.backend == "aspo":
            from module.steerable_esm2 import steering_forward

            model.steering_forward = types.MethodType(steering_forward, model)
            vector_paths = (args.fold_vector, args.sol_vector, args.thermo_vector)
            if any(path is None or not path.is_file() for path in vector_paths):
                raise FileNotFoundError("ASPO requires Fold, Sol, and Thermo steering vectors")
            steering = (
                0.5 * load_vector(args.fold_vector)
                + 0.5 * load_vector(args.sol_vector)
                + 1.0 * load_vector(args.thermo_vector)
            ).to(args.device)

        def generate(sequence: str) -> str:
            _, _, tokens = batch_converter([("protein", sequence)])
            return generate_sequences(
                tokens.to(args.device), model, steering, args.mask_ratio, alphabet,
                temperature=args.temperature, top_p=args.top_p,
            )

    else:
        from utils.esm3_utils import load_esm3_model, generate_sequences

        model, tokenizer = load_esm3_model(device=args.device)

        def generate(sequence: str) -> str:
            tokens = torch.tensor(tokenizer.encode(sequence), dtype=torch.int64, device=args.device)
            return generate_sequences(
                tokens, model, None, args.mask_ratio, tokenizer,
                temperature=args.temperature, top_p=args.top_p,
            )

    model_id = {
        "esm2": "esm2_650m",
        "esm3": "esm3_open_1.4b",
        "aspo": "aspo_esm2_650m",
    }[args.backend]
    config_id = {
        "esm2": "esm2_650m_iterative_masked_infilling",
        "esm3": "esm3_open_iterative_masked_infilling",
        "aspo": "aspo_esm2_650m_fold_sol_thermo_combined",
    }[args.backend]
    generated = []
    for index, reference in enumerate(rows):
        torch.manual_seed(args.seed + index)
        np.random.seed(args.seed + index)
        sequence = generate(reference["sequence"])
        generated.append(build_generation_record(
            request_id=f"{model_id}_{index:04d}", benchmark="unconditional",
            model=model_id, seed=args.seed + index, condition="",
            raw_output=sequence, sequence=sequence, terminated=True,
            termination_reason="iterative_masked_infilling_complete",
            generation_method=("ASPO weighted activation steering" if args.backend == "aspo"
                               else "iterative masked-token infilling"),
            generation_config_id=config_id,
            method_metadata={
                "reference_sequence_id": reference["sequence_id"],
                "reference_homology_group": reference["homology_group"],
                "reference_length_aa": int(reference["length_aa"]),
                "mask_ratio": args.mask_ratio,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": "",
                "repetition_penalty": "",
                "max_new_tokens": "",
            },
        ))
    atomic_write_jsonl(args.output, generated)
    print(f"wrote {len(generated)} {model_id} requests to {args.output}")


if __name__ == "__main__":
    main()
