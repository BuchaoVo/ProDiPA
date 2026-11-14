from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
REQUIRED_GENERATION_FIELDS = (
    "request_id",
    "benchmark",
    "model",
    "seed",
    "condition",
    "raw_output",
    "sequence",
    "length_aa",
    "valid",
    "terminated",
    "termination_reason",
    "generation_method",
    "generation_config_id",
    "sequence_sha256",
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON or YAML without making PyYAML a core ProDiPA dependency."""
    config_path = Path(path).expanduser().resolve()
    text = config_path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError(
                f"{config_path} is not JSON-compatible YAML and PyYAML is unavailable"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    value["_config_path"] = str(config_path)
    return value


def expand_path(value: str, *, base: str | Path | None = None) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path.resolve()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sequence(value: Any) -> str:
    if value is None:
        return ""
    return "".join(str(value).split()).upper()


def sequence_is_valid(sequence: str, min_length: int = 50, max_length: int = 512) -> bool:
    return min_length <= len(sequence) <= max_length and set(sequence) <= AA_ALPHABET


def build_generation_record(
    *,
    request_id: str,
    benchmark: str,
    model: str,
    seed: int,
    condition: str,
    raw_output: str,
    sequence: str,
    terminated: bool,
    termination_reason: str,
    generation_method: str,
    generation_config_id: str,
    method_metadata: Mapping[str, Any] | None = None,
    min_length: int = 50,
    max_length: int = 512,
) -> dict[str, Any]:
    normalized = normalize_sequence(sequence)
    return {
        "request_id": str(request_id),
        "benchmark": str(benchmark),
        "model": str(model),
        "seed": int(seed),
        "condition": str(condition),
        "raw_output": str(raw_output),
        "sequence": normalized,
        "length_aa": len(normalized),
        "valid": sequence_is_valid(normalized, min_length=min_length, max_length=max_length),
        "terminated": bool(terminated),
        "termination_reason": str(termination_reason),
        "generation_method": str(generation_method),
        "generation_config_id": str(generation_config_id),
        "sequence_sha256": sha256_text(normalized),
        "method_metadata": dict(method_metadata or {}),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False, prefix=f".{target.name}."
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False, prefix=f".{target.name}."
    ) as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
