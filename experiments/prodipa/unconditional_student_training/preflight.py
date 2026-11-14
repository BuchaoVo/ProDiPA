#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from peft import PeftConfig


EXPECTED_TEACHERS = ("fold", "sol", "thermo")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the frozen inputs for unconditional multi-teacher student training.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    config_path = args.teacher_config.resolve()
    output_dir = args.output_dir.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    backbone = Path(payload["teacher_backbone_path"]).resolve()
    if not backbone.is_dir():
        raise FileNotFoundError(backbone)
    teachers = payload.get("teachers")
    if not isinstance(teachers, list) or tuple(item.get("name") for item in teachers) != EXPECTED_TEACHERS:
        raise ValueError(f"Teacher order must be exactly {EXPECTED_TEACHERS}.")
    if abs(sum(float(item["weight"]) for item in teachers) - 1.0) > 1e-8:
        raise ValueError("Teacher weights must sum to one.")

    adapter_audit = []
    for item in teachers:
        adapter = Path(item["adapter_path"]).resolve()
        config_file = adapter / "adapter_config.json"
        weights_file = adapter / "adapter_model.safetensors"
        if not config_file.is_file() or not weights_file.is_file():
            raise FileNotFoundError(f"Incomplete teacher adapter: {adapter}")
        peft_config = PeftConfig.from_pretrained(str(adapter))
        if str(peft_config.peft_type).split(".")[-1] != "PREFIX_TUNING":
            raise ValueError(f"Teacher {item['name']} is not a Prefix adapter: {peft_config.peft_type}")
        if int(peft_config.num_virtual_tokens) != 100:
            raise ValueError(f"Teacher {item['name']} must use 100 virtual tokens.")
        if Path(peft_config.base_model_name_or_path).resolve() != backbone:
            raise ValueError(f"Teacher {item['name']} backbone mismatch.")
        adapter_audit.append(
            {
                "name": item["name"],
                "path": str(adapter),
                "weight": float(item["weight"]),
                "temperature": float(item["temperature"]),
                "peft_type": "PREFIX_TUNING",
                "num_virtual_tokens": 100,
                "adapter_config_sha256": sha256_file(config_file),
                "adapter_weights_sha256": sha256_file(weights_file),
            }
        )

    source_files = [
        root / "src/prodipa/unconditional/train.py",
        root / "src/prodipa/unconditional/trainer.py",
        root / "src/prodipa/unconditional/data.py",
    ]
    report = {
        "status": "PASS",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "teacher_config": str(config_path),
        "teacher_config_sha256": sha256_file(config_path),
        "teacher_backbone": str(backbone),
        "output_dir": str(output_dir),
        "teachers": adapter_audit,
        "source_sha256": {str(path.relative_to(root)): sha256_file(path) for path in source_files},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
