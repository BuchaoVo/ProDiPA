#!/usr/bin/env python3
"""Extract and apply TPR activation-steering vectors with standard HF LLaMA.

The released steering code targets a custom ``llama`` package exposing
``self_attn.head_out``.  ProLLaMA in this repository is a standard Hugging Face
LLaMA checkpoint, so this compatibility implementation maps that location to
``self_attn.o_proj``.  MLP steering, mean positive-minus-negative vectors, and
per-token norm preservation follow the released method.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, set_seed

try:
    from .common import (
        atomic_write_json,
        atomic_write_jsonl,
        build_generation_record,
        expand_path,
        load_config,
        read_jsonl,
        sha256_file,
        sha256_text,
    )
except ImportError:  # direct script execution
    from common import (  # type: ignore
        atomic_write_json,
        atomic_write_jsonl,
        build_generation_record,
        expand_path,
        load_config,
        read_jsonl,
        sha256_file,
        sha256_text,
    )


PROPERTIES = ("fold", "sol", "thermo")
AA_PATTERN = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def _resource_path(config: dict[str, Any], name: str, root: Path) -> Path:
    spec = config["resources"][name]
    value = spec["path"]
    return expand_path(value, base=root)


def _sequence(row: dict[str, Any]) -> str:
    value = str(row.get("sequence", row.get("output", "")))
    if "Seq=<" in value:
        value = value.split("Seq=<", 1)[1].split(">", 1)[0]
    sequence = "".join(value.split()).upper()
    if not AA_PATTERN.fullmatch(sequence):
        raise ValueError("noncanonical or empty protein sequence in steering dataset")
    return sequence


def _load_sequences(path: Path, count: int) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected JSON array: {path}")
    sequences = [_sequence(row) for row in value]
    if len(sequences) < count:
        raise ValueError(f"{path} contains {len(sequences)} rows; need {count}")
    selected = sequences[:count]
    if len(set(selected)) != len(selected):
        raise ValueError(f"duplicate sequences among first {count} rows: {path}")
    return selected


def _load_model(model_path: Path, device: str) -> tuple[Any, Any, torch.dtype]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval().to(device)
    return model, tokenizer, dtype


class ActivationCollector:
    def __init__(self, model: Any) -> None:
        self.head: list[torch.Tensor | None] = [None] * len(model.model.layers)
        self.mlp: list[torch.Tensor | None] = [None] * len(model.model.layers)
        self.handles = []
        for index, layer in enumerate(model.model.layers):
            self.handles.append(layer.self_attn.o_proj.register_forward_hook(self._hook(self.head, index)))
            self.handles.append(layer.mlp.register_forward_hook(self._hook(self.mlp, index)))

    @staticmethod
    def _hook(destination: list[torch.Tensor | None], index: int):
        def capture(_module: Any, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
            destination[index] = output[:, -1, :].detach().float().cpu()
        return capture

    def pop(self) -> tuple[torch.Tensor, torch.Tensor]:
        if any(item is None for item in self.head + self.mlp):
            raise RuntimeError(
                "missing activation layers: "
                f"attention={[i for i, value in enumerate(self.head) if value is None]} "
                f"mlp={[i for i, value in enumerate(self.mlp) if value is None]}"
            )
        head = torch.stack([item.squeeze(0) for item in self.head if item is not None])
        mlp = torch.stack([item.squeeze(0) for item in self.mlp if item is not None])
        # Hooks close over these list objects.  Clear them in place so later
        # examples keep writing into the active collector buffers.
        for index in range(len(self.head)):
            self.head[index] = None
            self.mlp[index] = None
        return head, mlp

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _mean_activations(
    model: Any,
    tokenizer: Any,
    sequences: list[str],
    prompt: str,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    collector = ActivationCollector(model)
    head_sum: torch.Tensor | None = None
    mlp_sum: torch.Tensor | None = None
    try:
        for index, sequence in enumerate(sequences, start=1):
            encoded = tokenizer(prompt + sequence, return_tensors="pt", add_special_tokens=True)
            max_positions = int(getattr(model.config, "max_position_embeddings", 0) or 0)
            if max_positions and encoded.input_ids.shape[1] > max_positions:
                raise ValueError(
                    f"steering example has {encoded.input_ids.shape[1]} tokens, above model limit {max_positions}"
                )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                model(**encoded, use_cache=False)
            head, mlp = collector.pop()
            head_sum = head.double() if head_sum is None else head_sum + head.double()
            mlp_sum = mlp.double() if mlp_sum is None else mlp_sum + mlp.double()
            print(f"activation {index}/{len(sequences)} tokens={encoded['input_ids'].shape[1]}", flush=True)
    finally:
        collector.close()
    assert head_sum is not None and mlp_sum is not None
    return (head_sum / len(sequences)).float(), (mlp_sum / len(sequences)).float()


def extract_vectors(config_path: Path, root: Path, device: str, properties: list[str]) -> None:
    config = load_config(config_path)
    method = config["method"]
    count = int(method.get("vector_samples_per_class", 100))
    family = config["condition"]["family_name"]
    prompt = f"[Generate by superfamily]Superfamily=<{family}>Seq=<"
    model_path = _resource_path(config, "prollama_base", root)
    model, tokenizer, dtype = _load_model(model_path, device)
    for prop in properties:
        positive_path = _resource_path(config, f"{prop}_positive_pool", root)
        negative_path = _resource_path(config, f"{prop}_negative_pool", root)
        output_path = _resource_path(config, f"{prop}_vectors", root)
        positive = _load_sequences(positive_path, count)
        negative = _load_sequences(negative_path, count)
        pos_head, pos_mlp = _mean_activations(model, tokenizer, positive, prompt, device)
        neg_head, neg_mlp = _mean_activations(model, tokenizer, negative, prompt, device)
        artifact = {
            "head": pos_head - neg_head,
            "mlp": pos_mlp - neg_mlp,
            "metadata": {
                "property": prop,
                "samples_per_class": count,
                "positive_source": str(positive_path),
                "positive_source_sha256": sha256_file(positive_path),
                "negative_source": str(negative_path),
                "negative_source_sha256": sha256_file(negative_path),
                "positive_sequence_set_sha256": sha256_text("\n".join(sorted(positive))),
                "negative_sequence_set_sha256": sha256_text("\n".join(sorted(negative))),
                "model": str(model_path),
                "prompt": prompt,
                "dtype": str(dtype),
                "compatibility_mapping": {
                    "released_self_attn_location": "self_attn.head_out",
                    "hf_location": "self_attn.o_proj",
                    "mlp_location": "mlp",
                },
            },
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(artifact, temporary)
        temporary.replace(output_path)
        atomic_write_json(output_path.with_suffix(".manifest.json"), artifact["metadata"])
        print(f"saved {prop} vectors: {output_path}", flush=True)


def _steer(output: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    original_dtype = output.dtype
    values = output.float()
    original_norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    steered = values + vector.to(device=values.device, dtype=values.dtype).view(1, 1, -1)
    new_norm = torch.linalg.vector_norm(steered, dim=-1, keepdim=True).clamp_min(1e-5)
    return (steered * (original_norm / new_norm)).to(original_dtype)


def _install_steering(model: Any, head: torch.Tensor, mlp: torch.Tensor) -> list[Any]:
    if head.shape != mlp.shape or head.shape[0] != len(model.model.layers):
        raise ValueError(f"steering vector shape mismatch: head={tuple(head.shape)} mlp={tuple(mlp.shape)}")
    handles = []
    for index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.o_proj.register_forward_hook(
            lambda _module, _inputs, output, vector=head[index]: _steer(output, vector)
        ))
        handles.append(layer.mlp.register_forward_hook(
            lambda _module, _inputs, output, vector=mlp[index]: _steer(output, vector)
        ))
    return handles


class StopAfterWrapperClose(StoppingCriteria):
    def __init__(self, tokenizer: Any, prompt_length: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor, **_kwargs: Any) -> bool:
        completions = self.tokenizer.batch_decode(
            input_ids[:, self.prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return bool(completions) and all(">" in value for value in completions)


def _load_combined_vectors(config: dict[str, Any], root: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    weights = config["method"]["steering_weights"]
    combined_head: torch.Tensor | None = None
    combined_mlp: torch.Tensor | None = None
    manifests: dict[str, Any] = {}
    for prop in PROPERTIES:
        path = _resource_path(config, f"{prop}_vectors", root)
        value = torch.load(path, map_location="cpu", weights_only=False)
        weight = float(weights[prop])
        head = value["head"].float() * weight
        mlp = value["mlp"].float() * weight
        combined_head = head if combined_head is None else combined_head + head
        combined_mlp = mlp if combined_mlp is None else combined_mlp + mlp
        manifests[prop] = {"path": str(path), "sha256": sha256_file(path), "weight": weight}
    assert combined_head is not None and combined_mlp is not None
    return combined_head, combined_mlp, manifests


def generate(config_path: Path, root: Path, output: Path, device: str) -> None:
    config = load_config(config_path)
    protocol = config["protocol"]
    family = config["condition"]["family_name"]
    prompt = f"[Generate by superfamily]Superfamily=<{family}>Seq=<"
    model_path = _resource_path(config, "prollama_base", root)
    model, tokenizer, dtype = _load_model(model_path, device)
    head, mlp, vector_metadata = _load_combined_vectors(config, root)
    handles = _install_steering(model, head, mlp)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    prompt_length = encoded.input_ids.shape[1]
    encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
    requests = int(protocol["requests"])
    base_seed = int(protocol["seed"])
    existing = read_jsonl(output) if output.exists() else []
    if len(existing) > requests:
        raise ValueError(f"existing output has {len(existing)} rows, expected at most {requests}")
    records = list(existing)
    try:
        for request_index in range(len(records), requests):
            seed = base_seed + request_index * 1009
            set_seed(seed)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=float(protocol["temperature"]),
                    top_p=float(protocol["top_p"]),
                    top_k=int(protocol["top_k"]),
                    repetition_penalty=float(protocol.get("repetition_penalty", 1.2)),
                    max_new_tokens=int(protocol["max_new_tokens"]),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([StopAfterWrapperClose(tokenizer, prompt_length)]),
                    use_cache=True,
                )[0, prompt_length:]
            completion = tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            wrapper_closed = ">" in completion
            payload = completion.split(">", 1)[0]
            sequence = "".join(payload.split()).upper()
            eos_reached = tokenizer.eos_token_id is not None and tokenizer.eos_token_id in generated.tolist()
            if wrapper_closed:
                reason = "wrapper_close"
            elif eos_reached:
                reason = "eos_before_wrapper_close"
            else:
                reason = "max_new_tokens"
            records.append(build_generation_record(
                request_id=f"aspo_tpr_hf_compat_{request_index:04d}",
                benchmark="conditional_tpr",
                model="aspo_tpr_hf_compat",
                seed=seed,
                condition=family,
                raw_output=prompt + completion,
                sequence=sequence,
                terminated=wrapper_closed,
                termination_reason=reason,
                generation_method="activation_steering_hf_compat",
                generation_config_id="aspo_tpr_multi_hf_compat",
                method_metadata={
                    "temperature": float(protocol["temperature"]),
                    "top_p": float(protocol["top_p"]),
                    "top_k": int(protocol["top_k"]),
                    "repetition_penalty": float(protocol.get("repetition_penalty", 1.2)),
                    "max_new_tokens": int(protocol["max_new_tokens"]),
                    "wrapper_closed": wrapper_closed,
                },
            ))
            atomic_write_jsonl(output, records)
            print(
                f"aspo_tpr_hf_compat {len(records)}/{requests} length={len(sequence)} "
                f"closed={wrapper_closed} eos={eos_reached}",
                flush=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    atomic_write_json(output.with_suffix(".metadata.json"), {
        "status": "COMPLETE",
        "method": "aspo_tpr_hf_compat",
        "requests": requests,
        "model": str(model_path),
        "model_dtype": str(dtype),
        "prompt": prompt,
        "steering_vectors": vector_metadata,
        "compatibility_mapping": "released self_attn.head_out -> Hugging Face self_attn.o_proj",
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("extract", "generate"), default="generate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--properties", default="fold,sol,thermo")
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config.resolve()
    if args.action == "extract":
        properties = [item.strip() for item in args.properties.split(",") if item.strip()]
        if not properties or any(item not in PROPERTIES for item in properties):
            raise ValueError(f"properties must be drawn from {PROPERTIES}")
        extract_vectors(config_path, root, args.device, properties)
    else:
        if args.output is None:
            config = load_config(config_path)
            output = expand_path(config["protocol"]["output"], base=root)
        else:
            output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        generate(config_path, root, output, args.device)


if __name__ == "__main__":
    main()
