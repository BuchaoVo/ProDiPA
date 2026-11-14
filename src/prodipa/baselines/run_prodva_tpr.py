from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from .audit_resources import audit_config
    from .common import atomic_write_json, atomic_write_jsonl, build_generation_record, expand_path, load_config
except ImportError:  # direct script execution
    from audit_resources import audit_config
    from common import atomic_write_json, atomic_write_jsonl, build_generation_record, expand_path, load_config


def _resolved(audit: dict[str, Any], name: str) -> str:
    for item in audit["resources"]:
        if item["name"] == name and item["status"] == "ready":
            return str(item["resolved_path"])
    raise RuntimeError(f"resource is not ready: {name}")


def prepare_text_input(config: dict[str, Any], project_root: Path) -> Path:
    protocol = config["protocol"]
    output_dir = expand_path(protocol["output_dir"], base=project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "text_only_requests.json"
    requests = [
        {
            "request_id": f"prodva_tpr_{index:06d}",
            "instruction": config["prompt"],
            "sequence": "",
        }
        for index in range(1, int(protocol["requests"]) + 1)
    ]
    atomic_write_json(input_path, requests)
    return input_path


def prepare_eval_config(config: dict[str, Any], audit: dict[str, Any], input_path: Path) -> Path:
    protocol = config["protocol"]
    output_dir = input_path.parent
    result_path = output_dir / "official_eval_results.json"
    eval_config_path = output_dir / "official_eval_config.yaml"
    eval_config = {
        "model": {
            "model_name_or_path": _resolved(audit, "model_weights"),
            "text_encoder_path": _resolved(audit, "text_tokenizer"),
            "language_model_path": _resolved(audit, "language_model_tokenizer"),
            "phrase_encoder_path": _resolved(audit, "phrase_tokenizer"),
            "phrase_sampler_type": "PROTEIN_FRAGMENT",
            "phrase_encoder_batch_size": 100000,
            "protein_fragment_mapping_file": _resolved(audit, "fragment_mapping"),
        },
        "infer": {
            "doc_top_k": int(protocol["doc_top_k"]),
            "embedding_model_path": _resolved(audit, "embedding_model"),
            "data_file": _resolved(audit, "retriever_data"),
            "vector_store_path": _resolved(audit, "vector_store"),
            "do_sample": True,
            "max_new_tokens": int(protocol["max_new_tokens"]),
            "temperature": float(protocol["temperature"]),
            "top_k": int(protocol["top_k"]),
            "top_p": float(protocol["top_p"]),
            "protein_sequence_mapping_file": _resolved(audit, "protein_sequence_mapping"),
        },
        "eval": {
            "eval_seed": int(protocol["seed"]),
            "test_data_file": str(input_path),
            "batch_size": int(protocol["batch_size"]),
            "task_type": "PROTEIN_DESIGN",
            "save_results_path": str(result_path),
        },
    }
    atomic_write_json(eval_config_path, eval_config)
    return eval_config_path


def normalize(config: dict[str, Any], result_path: Path, output_path: Path) -> None:
    values = json.loads(result_path.read_text(encoding="utf-8"))
    expected = int(config["protocol"]["requests"])
    if not isinstance(values, list) or len(values) != expected:
        raise ValueError(f"ProDVa result count is not {expected}: {result_path}")
    records = []
    for index, row in enumerate(values, start=1):
        raw_output = str(row.get("prediction", ""))
        sequence = raw_output.replace("<|endoftext|>", "")
        sequence = "".join(sequence.split())
        records.append(
            build_generation_record(
                request_id=f"prodva_tpr_{index:06d}",
                benchmark="conditional_tpr",
                model="prodva_molinst_swissprotclap",
                seed=int(config["protocol"]["seed"]),
                condition=config["condition"],
                raw_output=raw_output,
                sequence=sequence,
                terminated=bool(raw_output),
                termination_reason="official_eval_complete" if raw_output else "empty_output",
                generation_method="retrieval_augmented_dynamic_vocabulary",
                generation_config_id=config["generation_config_id"],
                method_metadata={
                    "prompt": row.get("instruction", config["prompt"]),
                    "raw_token_ids": row.get("ids"),
                    "reference_was_empty": row.get("reference", None) == "",
                    "model_weights": config.get("model_id"),
                    "retrieval_index_version": config.get("retrieval_index_version"),
                    "doc_top_k": int(config["protocol"]["doc_top_k"]),
                },
            )
        )
    atomic_write_jsonl(output_path, records)


def main() -> None:
    parser = argparse.ArgumentParser(description="ProDVa text-only TPR adapter")
    parser.add_argument("action", choices=("preflight", "prepare", "run", "normalize"))
    parser.add_argument("--config", type=Path, default=Path("configs/baselines/prodva_tpr.yaml"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--report", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    audit = audit_config(args.config, args.project_root)
    if args.report:
        atomic_write_json(args.report, audit)
    print(f"ProDVa preflight: {audit['status']} blockers={audit['blockers']}")

    if args.action == "normalize":
        if not args.results or not args.output:
            raise ValueError("--results and --output are required for normalize")
        normalize(config, args.results, args.output)
        return
    if args.action == "prepare":
        input_path = prepare_text_input(config, args.project_root)
        print(f"text-only requests: {input_path}; all 512 reference sequence fields are empty")
        if audit["status"] != "ready":
            print("resource audit remains blocked; official eval config and generation were not started")
            return
    if audit["status"] != "ready":
        raise SystemExit(2)
    input_path = prepare_text_input(config, args.project_root)
    eval_config_path = prepare_eval_config(config, audit, input_path)
    print(f"text-only requests: {input_path}; reference sequence fields are empty")
    print(f"official eval config: {eval_config_path}")
    if args.action == "run":
        executable = _resolved(audit, "dvagen_executable")
        subprocess.run([executable, "eval", "--config_path", str(eval_config_path)], check=True)


if __name__ == "__main__":
    main()
