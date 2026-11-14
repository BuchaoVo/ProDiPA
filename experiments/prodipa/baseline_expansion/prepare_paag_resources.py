#!/usr/bin/env python3
"""Prepare the serialized model objects required by the PAAG runtime."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


PROTBERT_REPO = "Rostlab/prot_bert_bfd"
PROTBERT_REVISION = "6c5c8a55a52ff08a664dfd584aa1773f125a0487"
SCIBERT_REPO = "allenai/scibert_scivocab_uncased"
SCIBERT_REVISION = "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def download_snapshot(repo_id: str, revision: str, target: Path, endpoint: str) -> None:
    while True:
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=target,
                endpoint=endpoint,
                max_workers=1,
                resume_download=True,
            )
            return
        except Exception as exc:
            print(f"retrying {repo_id} after {type(exc).__name__}: {exc}", flush=True)
            time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()

    endpoint = "https://huggingface.co"
    os.environ["HF_ENDPOINT"] = endpoint
    sources = args.output / "sources"
    protbert_path = sources / "prot_bert_bfd"
    scibert_path = sources / "scibert_scivocab_uncased"
    state: dict[str, object] = {
        "state": "RUNNING",
        "stage": "download_sources",
        "endpoint": endpoint,
        "started_at": datetime.now().astimezone().isoformat(),
        "sources": {
            "protbert": {"repo_id": PROTBERT_REPO, "revision": PROTBERT_REVISION},
            "scibert": {"repo_id": SCIBERT_REPO, "revision": SCIBERT_REVISION},
        },
    }
    atomic_json(args.status, state)

    download_snapshot(PROTBERT_REPO, PROTBERT_REVISION, protbert_path, endpoint)
    download_snapshot(SCIBERT_REPO, SCIBERT_REVISION, scibert_path, endpoint)

    sys.path.insert(0, str(args.repo))
    from models.med import BertLMHeadModel, BertModel
    from transformers import BertConfig, BertTokenizer

    state["stage"] = "serialize_protbert"
    atomic_json(args.status, state)
    tokenizer = BertTokenizer.from_pretrained(
        protbert_path, do_lower_case=False, local_files_only=True
    )
    atomic_torch_save(tokenizer, args.output / "ProtBert_tokenizer.pt")
    del tokenizer

    encoder_config = BertConfig.from_json_file(args.repo / "configs" / "config_seq.json")
    encoder_config.encoder_width = 768
    encoder = BertModel.from_pretrained(
        protbert_path,
        config=encoder_config,
        add_pooling_layer=False,
        local_files_only=True,
    )
    atomic_torch_save(encoder, args.output / "ProtBert.pt")
    del encoder
    gc.collect()

    decoder = BertLMHeadModel.from_pretrained(
        protbert_path,
        config=str(args.repo / "configs" / "config_light_seq.json"),
        is_decoder=True,
        local_files_only=True,
    )
    atomic_torch_save(decoder, args.output / "ProtDecoder.pt")
    del decoder
    gc.collect()

    state["stage"] = "serialize_scibert"
    atomic_json(args.status, state)
    tokenizer = BertTokenizer.from_pretrained(
        scibert_path, do_lower_case=False, local_files_only=True
    )
    atomic_torch_save(tokenizer, args.output / "scibert_tokenizer.pt")
    del tokenizer
    encoder = BertModel.from_pretrained(
        scibert_path, add_pooling_layer=False, local_files_only=True
    )
    atomic_torch_save(encoder, args.output / "scibert.pt")
    del encoder
    gc.collect()

    state["state"] = "COMPLETE"
    state["stage"] = "complete"
    state["completed_at"] = datetime.now().astimezone().isoformat()
    state["outputs"] = [
        str(args.output / name)
        for name in (
            "ProtBert_tokenizer.pt",
            "ProtBert.pt",
            "ProtDecoder.pt",
            "scibert_tokenizer.pt",
            "scibert.pt",
        )
    ]
    atomic_json(args.status, state)
    print("PAAG runtime resources complete", flush=True)


if __name__ == "__main__":
    main()
