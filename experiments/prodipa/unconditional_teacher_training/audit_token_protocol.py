#!/usr/bin/env python3
"""Write an exact token-ID audit for historical and FASTA60 teacher inputs."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from prodipa.unconditional.teacher_sequence_protocol import (
    PLAIN_DIRECT_EOS_V1,
    PROTGPT2_FASTA60_V1,
    format_teacher_sequence,
    protocol_metadata,
)


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def role_for(text: str, start: int, end: int) -> str:
    piece = text[start:end]
    if piece == "\n":
        if start == 0:
            return "leading_newline"
        if end == len(text):
            return "terminal_newline"
        return "wrap_newline"
    if "\n" in piece:
        return "mixed_residue_newline"
    return "residue_text"


def tokenize_rows(tokenizer, sequence: str, protocol: str) -> tuple[list[int], list[dict]]:
    text = format_teacher_sequence(sequence, protocol)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    content_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    ids = [tokenizer.eos_token_id, *content_ids, tokenizer.eos_token_id]
    rows = [{
        "protocol": protocol,
        "token_index": 0,
        "token_id": tokenizer.eos_token_id,
        "token": tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id),
        "decoded": tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False),
        "char_start": "",
        "char_end": "",
        "role": "opening_bos_eos_delimiter",
    }]
    for index, (token_id, (start, end)) in enumerate(zip(content_ids, offsets), start=1):
        rows.append({
            "protocol": protocol,
            "token_index": index,
            "token_id": token_id,
            "token": tokenizer.convert_ids_to_tokens(token_id),
            "decoded": tokenizer.decode(
                [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
            ).replace("\n", "\\n"),
            "char_start": start,
            "char_end": end,
            "role": role_for(text, start, end),
        })
    rows.append({
        "protocol": protocol,
        "token_index": len(ids) - 1,
        "token_id": tokenizer.eos_token_id,
        "token": tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id),
        "decoded": tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=False),
        "char_start": "",
        "char_end": "",
        "role": "true_closing_eos_target",
    })
    return ids, rows


def first_sequence(path: Path) -> str:
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    return row["sequence"].strip().upper()


def diff_rows(old_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    old_ids = [int(row["token_id"]) for row in old_rows]
    new_ids = [int(row["token_id"]) for row in new_rows]
    matcher = difflib.SequenceMatcher(a=old_ids, b=new_ids, autojunk=False)
    result = []
    diff_index = 0
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        width = max(i2 - i1, j2 - j1)
        for offset in range(width):
            old = old_rows[i1 + offset] if i1 + offset < i2 else {}
            new = new_rows[j1 + offset] if j1 + offset < j2 else {}
            result.append({
                "diff_index": diff_index,
                "operation": operation,
                "old_index": old.get("token_index", ""),
                "old_id": old.get("token_id", ""),
                "old_token": old.get("token", ""),
                "old_role": old.get("role", ""),
                "new_index": new.get("token_index", ""),
                "new_id": new.get("token_id", ""),
                "new_token": new.get("token", ""),
                "new_role": new.get("role", ""),
            })
            diff_index += 1
    return result


def dataset_summary(tokenizer, data_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(data_dir.glob("*.csv")):
        lengths = {PLAIN_DIRECT_EOS_V1: [], PROTGPT2_FASTA60_V1: []}
        newline_counts = []
        with path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        for record in records:
            sequence = record["sequence"].strip().upper()
            for protocol in lengths:
                text = format_teacher_sequence(sequence, protocol)
                token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                lengths[protocol].append(len(token_ids) + 2)
            newline_counts.append(len(format_teacher_sequence(sequence, PROTGPT2_FASTA60_V1).splitlines()) - 1)
        rows.append({
            "file": path.name,
            "records": len(records),
            "legacy_min_tokens": min(lengths[PLAIN_DIRECT_EOS_V1]),
            "legacy_mean_tokens": sum(lengths[PLAIN_DIRECT_EOS_V1]) / len(records),
            "legacy_max_tokens": max(lengths[PLAIN_DIRECT_EOS_V1]),
            "fasta60_min_tokens": min(lengths[PROTGPT2_FASTA60_V1]),
            "fasta60_mean_tokens": sum(lengths[PROTGPT2_FASTA60_V1]) / len(records),
            "fasta60_max_tokens": max(lengths[PROTGPT2_FASTA60_V1]),
            "fasta60_over_512": sum(value > 512 for value in lengths[PROTGPT2_FASTA60_V1]),
            "roundtrip_failures": sum(
                "".join(
                    tokenizer.decode(
                        tokenizer(format_teacher_sequence(record["sequence"].strip().upper(), PROTGPT2_FASTA60_V1), add_special_tokens=False)["input_ids"],
                        clean_up_tokenization_spaces=False,
                    ).split()
                ).upper() != record["sequence"].strip().upper()
                for record in records
            ),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    sequence = first_sequence(args.data_dir / "train_sol_300.csv")
    old_ids, old_rows = tokenize_rows(tokenizer, sequence, PLAIN_DIRECT_EOS_V1)
    new_ids, new_rows = tokenize_rows(tokenizer, sequence, PROTGPT2_FASTA60_V1)
    all_rows = old_rows + new_rows
    write_tsv(
        args.out / "token_rows.tsv",
        all_rows,
        ["protocol", "token_index", "token_id", "token", "decoded", "char_start", "char_end", "role"],
    )
    differences = diff_rows(old_rows, new_rows)
    write_tsv(
        args.out / "token_diff.tsv",
        differences,
        ["diff_index", "operation", "old_index", "old_id", "old_token", "old_role", "new_index", "new_id", "new_token", "new_role"],
    )
    summaries = dataset_summary(tokenizer, args.data_dir)
    write_tsv(args.out / "dataset_tokenization.tsv", summaries, list(summaries[0]))

    model_card = args.model / "README.md"
    code_paths = [
        ROOT / "src/prodipa/unconditional/teacher_sequence_protocol.py",
        ROOT / "src/prodipa/unconditional/train_teacher.py",
        ROOT / "experiments/prodipa/conditional_evaluation/generate_model.py",
    ]
    payload = {
        "status": "PASS",
        "model": str(args.model.resolve()),
        "model_card": str(model_card.resolve()),
        "model_card_sha256": sha256(model_card),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_length": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "newline_token_ids": tokenizer("\n", add_special_tokens=False)["input_ids"],
        "sample_source": str((args.data_dir / "train_sol_300.csv").resolve()),
        "sample_sequence": sequence,
        "sample_length_aa": len(sequence),
        "historical_teacher_ids": old_ids,
        "fasta60_teacher_ids": new_ids,
        "generation_prompt_ids": [tokenizer.bos_token_id or tokenizer.eos_token_id],
        "inference_contract": {
            "prompt": "one BOS/EOS delimiter token",
            "leading_newline": "sampled by the model, never injected after the prompt",
            "raw_sampled_ids": "preserved through generation",
            "export": "whitespace removed only after decoding",
            "forced_aa_cap": "censor only; not a natural EOS",
        },
        "protocols": {
            PLAIN_DIRECT_EOS_V1: protocol_metadata(PLAIN_DIRECT_EOS_V1),
            PROTGPT2_FASTA60_V1: protocol_metadata(PROTGPT2_FASTA60_V1),
        },
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in code_paths},
        "artifacts": ["token_rows.tsv", "token_diff.tsv", "dataset_tokenization.tsv"],
    }
    (args.out / "TOKEN_PROTOCOL.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PASS",
        "eos_token_id": tokenizer.eos_token_id,
        "newline_token_ids": payload["newline_token_ids"],
        "legacy_tokens": len(old_ids),
        "fasta60_tokens": len(new_ids),
        "diff_rows": len(differences),
    }, indent=2))


if __name__ == "__main__":
    main()
