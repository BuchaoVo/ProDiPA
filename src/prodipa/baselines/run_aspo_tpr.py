from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any

try:
    from .audit_resources import audit_config
    from .common import atomic_write_json, load_config, normalize_sequence, sha256_file, sha256_text
except ImportError:  # direct script execution
    from audit_resources import audit_config
    from common import atomic_write_json, load_config, normalize_sequence, sha256_file, sha256_text


PROPERTY_SPECS = {
    "fold": ("plddt", 0.85),
    "sol": ("sol", 0.70),
    "thermo": ("thermo", 0.60),
}


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON array of objects: {path}")
    return value


def _sequence(row: dict[str, Any]) -> str:
    value = row.get("sequence", row.get("output", ""))
    if isinstance(value, str) and "Seq=<" in value:
        value = value.split("Seq=<", 1)[1].split(">", 1)[0]
    return normalize_sequence(value)


def audit_aspo_data(config: dict[str, Any], resource_report: dict[str, Any]) -> dict[str, Any]:
    paths = {
        item["name"]: Path(item["resolved_path"])
        for item in resource_report["resources"]
        if item["status"] == "ready" and "resolved_path" in item
    }
    errors: list[str] = []
    properties: dict[str, Any] = {}
    all_test_hashes: set[str] = set()
    for prop in PROPERTY_SPECS:
        test_path = paths.get(f"test_{prop}")
        if test_path:
            all_test_hashes.update(sha256_text(_sequence(row)) for row in _load_json_rows(test_path))

    for prop, (score_key, threshold) in PROPERTY_SPECS.items():
        item: dict[str, Any] = {}
        positive_path = paths.get(f"positive_{prop}")
        negative_path = paths.get(f"negative_{prop}")
        if not positive_path:
            errors.append(f"missing_positive_{prop}")
            properties[prop] = item
            continue
        positives = _load_json_rows(positive_path)
        positive_sequences = [_sequence(row) for row in positives]
        item.update(
            positive_file=str(positive_path),
            positive_sha256=sha256_file(positive_path),
            positive_count=len(positives),
            positive_unique_sequences=len(set(positive_sequences)),
            positive_median_length=median(map(len, positive_sequences)),
        )
        if len(positives) != 300:
            errors.append(f"{prop}_positive_count={len(positives)}")
        below = [row for row in positives if float(row.get(score_key, float("-inf"))) < threshold]
        if below:
            errors.append(f"{prop}_positives_below_threshold={len(below)}")
        if set(map(sha256_text, positive_sequences)) & all_test_hashes:
            errors.append(f"{prop}_positive_exact_overlap_with_test")

        if not negative_path:
            errors.append(f"missing_negative_{prop}")
            properties[prop] = item
            continue
        negatives = _load_json_rows(negative_path)
        negative_sequences = [_sequence(row) for row in negatives]
        negative_hashes = set(map(sha256_text, negative_sequences))
        item.update(
            negative_file=str(negative_path),
            negative_sha256=sha256_file(negative_path),
            negative_count=len(negatives),
            negative_unique_sequences=len(negative_hashes),
            negative_median_length=median(map(len, negative_sequences)) if negatives else None,
            negative_exact_test_overlap=len(negative_hashes & all_test_hashes),
        )
        if not negatives:
            errors.append(f"empty_negative_{prop}")
        if len(negative_hashes) != len(negative_sequences):
            errors.append(f"duplicate_negative_sequences_{prop}")
        if negative_hashes & all_test_hashes:
            errors.append(f"{prop}_negative_exact_overlap_with_test")
        negative_threshold = config.get("negative_thresholds", {}).get(prop)
        if negative_threshold is None:
            errors.append(f"unfrozen_negative_threshold_{prop}")
        else:
            above = [row for row in negatives if float(row.get(score_key, float("inf"))) > negative_threshold]
            if above:
                errors.append(f"{prop}_negatives_above_threshold={len(above)}")
        properties[prop] = item

    if "homology_manifest" not in paths:
        errors.append("test_homology_exclusion_unverifiable")
    return {"status": "ready" if not errors else "blocked", "errors": errors, "properties": properties}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit or run TPR multi-attribute ASPO steering")
    parser.add_argument("action", choices=("preflight", "run"))
    parser.add_argument("--config", type=Path, default=Path("configs/baselines/aspo_tpr.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = load_config(args.config)
    resources = audit_config(args.config, args.project_root)
    data = audit_aspo_data(config, resources)
    report = {
        "baseline": "aspo_tpr_multi_attribute",
        "status": "ready" if resources["status"] == "ready" and data["status"] == "ready" else "blocked",
        "resource_audit": resources,
        "data_audit": data,
        "steering_weights": config.get("steering_weights"),
        "condition": config.get("condition"),
    }
    atomic_write_json(args.report, report)
    print(f"ASPO preflight: {report['status']} report={args.report}")
    if report["status"] != "ready":
        raise SystemExit(2)
    if args.action == "run":
        entrypoint = next(
            Path(item["resolved_path"])
            for item in resources["resources"]
            if item["name"] == "multi_vector_entrypoint" and item["status"] == "ready"
        )
        output = Path(args.project_root) / str(config["protocol"]["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        # This is an explicit interface contract for the audited ProDiPA multi-vector patch.
        # It must write the common generations.jsonl schema; the caller audits all 512 rows.
        subprocess.run(
            [
                sys.executable,
                str(entrypoint),
                "--config",
                str(args.config.resolve()),
                "--project-root",
                str(args.project_root.resolve()),
                "--output",
                str(output.resolve()),
                "--device",
                args.device,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
