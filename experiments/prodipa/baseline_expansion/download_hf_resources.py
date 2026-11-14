#!/usr/bin/env python3
"""Download pinned baseline resources with resumable, fail-closed retries."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


RESOURCES = (
    ("protdat_esm1b", "facebook/esm1b_t33_650M_UR50S", "ProtDAT/esm1b"),
    ("protdat_pubmedbert", "NeuML/pubmedbert-base-embeddings", "ProtDAT/pubmedbert"),
    ("paag", "ychaohao/PAAG", "PAAG"),
    ("progen3_762m", "Profluent-Bio/progen3-762m", "ProGen3-762M"),
    ("dplm2_650m", "airkingbd/dplm2_650m", "DPLM2-650M"),
    ("dplm2_struct_tokenizer", "airkingbd/struct_tokenizer", "DPLM2-struct-tokenizer"),
)

ALLOW_PATTERNS = {
    "paag": (
        ".gitattributes",
        "README.md",
        "paag_protbert_protannotation.ckpt",
    ),
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("__PRODIPA_MODELS__/external_weights"))
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--only",
        action="append",
        choices=[resource[0] for resource in RESOURCES],
        help="Download only the named resource; repeat to select multiple resources.",
    )
    args = parser.parse_args()
    endpoint = "https://huggingface.co"
    state: dict[str, object] = {
        "state": "RUNNING",
        "started_at": datetime.now().astimezone().isoformat(),
        "endpoint": endpoint,
        "resources": {},
    }
    atomic_json(args.status, state)
    print(f"START endpoint={endpoint}", flush=True)

    selected = set(args.only or ())
    pending = [resource for resource in RESOURCES if not selected or resource[0] in selected]
    while pending:
        next_pending = []
        for name, repo_id, relative in pending:
            target = args.root / relative
            error = None
            try:
                target.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=repo_id,
                    revision="main",
                    local_dir=target,
                    endpoint=endpoint,
                    max_workers=1,
                    resume_download=True,
                    allow_patterns=ALLOW_PATTERNS.get(name),
                )
                info = HfApi(endpoint=endpoint).model_info(repo_id, revision="main", timeout=60)
                state["resources"][name] = {
                    "state": "COMPLETE",
                    "repo_id": repo_id,
                    "revision": info.sha,
                    "path": str(target),
                    "endpoint": endpoint,
                    "completed_at": datetime.now().astimezone().isoformat(),
                }
                atomic_json(args.status, state)
                print(f"COMPLETE {name} revision={info.sha} path={target}", flush=True)
            except Exception as exc:  # retry the official endpoint after the polling delay
                error = f"{type(exc).__name__}: {exc}"
                print(f"RETRY {name} endpoint={endpoint} error={error}", flush=True)
                state["resources"][name] = {
                    "state": "WAITING_RETRY",
                    "repo_id": repo_id,
                    "path": str(target),
                    "endpoint": endpoint,
                    "last_error": error,
                    "updated_at": datetime.now().astimezone().isoformat(),
                }
                atomic_json(args.status, state)
                next_pending.append((name, repo_id, relative))
        pending = next_pending
        if pending:
            time.sleep(args.poll_seconds)

    state["state"] = "COMPLETE"
    state["completed_at"] = datetime.now().astimezone().isoformat()
    atomic_json(args.status, state)


if __name__ == "__main__":
    main()
