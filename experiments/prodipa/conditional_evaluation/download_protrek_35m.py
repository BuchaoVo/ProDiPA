#!/usr/bin/env python3
"""Download the pinned compact sequence/text alignment checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import requests


REVISION = "fa3ae6d84de7c204cabda5fc6ad225057466bb05"
FILES = (
    "ProTrek_35M.pt",
    "config.json",
    "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/config.json",
    "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/tokenizer_config.json",
    "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/vocab.txt",
    "esm2_t12_35M_UR50D/config.json",
    "esm2_t12_35M_UR50D/special_tokens_map.json",
    "esm2_t12_35M_UR50D/tokenizer_config.json",
    "esm2_t12_35M_UR50D/vocab.txt",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    records = []
    for name in FILES:
        target = args.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/westlake-repl/ProTrek_35M/resolve/{REVISION}/{name}"
        temporary = target.with_suffix(target.suffix + ".part")
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with session.get(url, headers=headers, stream=True, timeout=120) as response:
            response.raise_for_status()
            append = offset > 0 and response.status_code == 206
            if offset and not append:
                offset = 0
            mode = "ab" if append else "wb"
            expected = response.headers.get("x-linked-size") or response.headers.get("content-length")
            print(f"downloading {name} offset={offset} expected={expected}", flush=True)
            with temporary.open(mode) as handle:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        os.replace(temporary, target)
        records.append({
            "file": name,
            "bytes": target.stat().st_size,
            "etag": response.headers.get("x-linked-etag") or response.headers.get("etag"),
        })
    manifest = {
        "repository": "westlake-repl/ProTrek_35M",
        "revision": REVISION,
        "files": records,
    }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=args.output, delete=False, prefix=".DOWNLOAD_MANIFEST."
    ) as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
        temporary_manifest = handle.name
    os.replace(temporary_manifest, args.output / "DOWNLOAD_MANIFEST.json")


if __name__ == "__main__":
    main()
