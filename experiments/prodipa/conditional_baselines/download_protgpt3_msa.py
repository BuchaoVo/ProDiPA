#!/usr/bin/env python3
"""Download and freeze the released homolog-context model snapshot."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download


REPOSITORY = "protgpt3/ProtGPT3-MSA"
REVISION = "fa69ff482788c79dd44079151346ab544c3aa456"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("__PRODIPA_MODELS__/ProtGPT3-MSA"))
    parser.add_argument("--retry-seconds", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=0, help="0 means retry until successful")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        attempt += 1
        try:
            resolved = snapshot_download(
                repo_id=REPOSITORY,
                revision=REVISION,
                local_dir=output,
                local_dir_use_symlinks=False,
            )
            break
        except Exception as error:
            print(f"download attempt {attempt} failed: {error!r}", flush=True)
            if args.max_attempts and attempt >= args.max_attempts:
                raise
            time.sleep(args.retry_seconds)
    manifest = {
        "status": "COMPLETE",
        "repository": REPOSITORY,
        "revision": REVISION,
        "resolved_path": str(Path(resolved).resolve()),
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    atomic_json(output / "DOWNLOAD_MANIFEST.json", manifest)
    (output / ".complete").write_text(REVISION + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
