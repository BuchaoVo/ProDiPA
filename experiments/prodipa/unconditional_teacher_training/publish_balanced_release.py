#!/usr/bin/env python3
"""Replace the canonical Unconditional data and Prefix teachers with FASTA60 v1."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[3]
SOURCE_DATA = ROOT / "data/rebuild/unconditional_teacher_balanced"
SOURCE_RUN = ROOT / "outputs/experiments/unconditional_teacher_balanced_fasta60_v1"
TARGET_DATA = ROOT / "data/unconditional"
TARGET_TEACHERS = ROOT / "outputs/unconditional/protgpt2"
ATTRIBUTES = {
    "fold": ("train_plddt_300.csv", "test_plddt_100.csv"),
    "sol": ("train_sol_300.csv", "test_sol_100.csv"),
    "thermo": ("train_thermo_300.csv", "test_thermo_100.csv"),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_dir(staged: Path, target: Path) -> None:
    old = target.with_name(target.name + f".replaced.{os.getpid()}")
    if old.exists():
        shutil.rmtree(old)
    if target.exists():
        os.replace(target, old)
    try:
        os.replace(staged, target)
    except BaseException:
        if old.exists() and not target.exists():
            os.replace(old, target)
        raise
    if old.exists():
        shutil.rmtree(old)


def main() -> None:
    audit = json.loads((SOURCE_DATA / "FINAL_AUDIT.json").read_text())
    if not str(audit.get("status", "")).startswith("PASS"):
        raise RuntimeError("balanced teacher data audit is not PASS")
    comparison = json.loads(
        (SOURCE_RUN / "evaluation/results/COMPARISON_RESULTS.json").read_text()
    )
    if comparison.get("status") != "COMPLETE":
        raise RuntimeError("formal teacher comparison is incomplete")
    for attribute, names in ATTRIBUTES.items():
        for name in names:
            if not (SOURCE_DATA / name).is_file():
                raise FileNotFoundError(SOURCE_DATA / name)
        adapter = SOURCE_RUN / "teachers" / attribute / "best_adapter"
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(adapter)

    data_stage = TARGET_DATA.with_name(TARGET_DATA.name + f".next.{os.getpid()}")
    teacher_stage = TARGET_TEACHERS.with_name(TARGET_TEACHERS.name + f".next.{os.getpid()}")
    for path in (data_stage, teacher_stage):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

    for attribute, (train_name, test_name) in ATTRIBUTES.items():
        directory = data_stage / attribute
        directory.mkdir()
        shutil.copy2(SOURCE_DATA / train_name, directory / train_name)
        shutil.copy2(SOURCE_DATA / test_name, directory / test_name)
    for name in (
        "FINAL_AUDIT.json", "TOKENIZATION_AUDIT.json", "build_report.json",
        "selected_sequence_manifest.tsv",
    ):
        shutil.copy2(SOURCE_DATA / name, data_stage / name)
    readme = (SOURCE_DATA / "README.md").read_text()
    readme = readme.replace(
        "Status: **PASS, NOT PUBLISHED**. The canonical files under\n"
        "`data/unconditional/` are unchanged.",
        "Status: **PASS, PUBLISHED**. This is the canonical Unconditional teacher dataset.",
    )
    (data_stage / "README.md").write_text(readme)

    protocol = {
        "version": 1,
        "name": "protgpt2_fasta60_v1",
        "leading_newline": True,
        "line_width_aa": 60,
        "trailing_newline_before_eos": True,
        "generation_sequence_format": "legacy",
        "generation_max_aa": 512,
        "forced_terminal_is_complete": False,
    }
    teacher_hashes = {}
    for attribute in ATTRIBUTES:
        source = SOURCE_RUN / "teachers" / attribute
        target = teacher_stage / attribute
        target.mkdir()
        shutil.copytree(source / "best_adapter", target / "best_adapter")
        for name in (
            "DONE.json", "preflight.json", "run_metadata.json", "train_result.json",
            "training_history.tsv",
        ):
            if (source / name).is_file():
                shutil.copy2(source / name, target / name)
        (target / "best_adapter/TEACHER_SEQUENCE_PROTOCOL.json").write_text(
            json.dumps(protocol, indent=2) + "\n"
        )
        config = (SOURCE_RUN / "teachers/configs" / f"{attribute}.yaml").read_text()
        train_name, test_name = ATTRIBUTES[attribute]
        config = config.replace(str(SOURCE_DATA / train_name), str(TARGET_DATA / attribute / train_name))
        config = config.replace(str(SOURCE_DATA / test_name), str(TARGET_DATA / attribute / test_name))
        config = config.replace(str(source), str(target))
        (target / "training_config.yaml").write_text(config)
        teacher_hashes[attribute] = sha(target / "best_adapter/adapter_model.safetensors")

    release = {
        "status": "PUBLISHED",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "data_protocol": "base_length_matched_3to1_complexity_audited_v1",
        "teacher_sequence_protocol": protocol,
        "teacher_adapter_sha256": teacher_hashes,
        "data_sha256": {
            f"{attribute}/{name}": sha(data_stage / attribute / name)
            for attribute, names in ATTRIBUTES.items() for name in names
        },
        "known_formal_evaluation": comparison,
        "replacement_policy": "old canonical data and Prefix teacher directories deleted after atomic replacement",
    }
    (teacher_stage / "CANONICAL_RELEASE.json").write_text(
        json.dumps(release, indent=2) + "\n"
    )
    replace_dir(data_stage, TARGET_DATA)
    replace_dir(teacher_stage, TARGET_TEACHERS)
    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
