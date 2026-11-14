#!/usr/bin/env python3

from __future__ import annotations

import argparse
import enum
import importlib.util
import logging
import os
import random
import subprocess
import sys
import tempfile
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Any as TypingAny

try:
    from .common import atomic_write_jsonl, build_generation_record, read_jsonl, seed_everything
except ImportError:
    from common import atomic_write_jsonl, build_generation_record, read_jsonl, seed_everything


DEFAULT_PROMPT = "Design novel protein sequences with high foldability, high solubility, and high thermostability."


def protein_dt(args: argparse.Namespace) -> list[str]:
    script = args.repo / "examples/downstream_Text2Protein/step_02_inference_ProteinDT.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    with tempfile.TemporaryDirectory(prefix="prodipa_proteindt_") as temporary:
        work = Path(temporary)
        prompts = work / "prompts.txt"
        raw = work / "raw.txt"
        prompts.write_text((args.prompt + "\n") * args.requests, encoding="utf-8")
        command = [
            sys.executable, str(script), "--device", "0", "--seed", str(args.seed),
            "--batch_size", str(args.batch_size), "--num_workers", "0",
            "--dataset_size", str(args.requests), "--input_text_file_path", str(prompts),
            "--pretrained_folder", str(args.pretrained_folder),
            "--step_04_folder", str(args.decoder_folder),
            "--output_text_file_path", str(raw), "--num_repeat", str(args.num_repeat),
            "--decoder_distribution", "T5Decoder", "--score_network_type", "T5Base",
            "--protein_max_sequence_len", "512", "--use_facilitator", "--AR_generation_mode", "01",
        ]
        subprocess.run(command, cwd=args.repo / "examples/downstream_Text2Protein", check=True)
        lines = [line.strip() for line in raw.read_text(encoding="utf-8").splitlines()]
    if len(lines) != 2 * args.requests:
        raise RuntimeError(f"ProteinDT returned {len(lines)} text lines, expected {2 * args.requests}")
    if any(lines[index] != args.prompt for index in range(0, len(lines), 2)):
        raise RuntimeError("ProteinDT output prompt does not match the frozen prompt")
    return lines[1::2]


def extract_pinal_sequence(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("sequence", "seq", "protein_sequence"):
            if isinstance(value.get(key), str):
                return value[key]
    if isinstance(value, (tuple, list)):
        # The official Pinal return tuple is
        # (structure_tokens, structure_logp, sequence, sequence_logp).
        if len(value) >= 3 and isinstance(value[2], str):
            sequence = "".join(value[2].split()).upper()
            if sequence and set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY"):
                return sequence
        for item in value:
            if isinstance(item, str) and set("".join(item.split()).upper()) <= set("ACDEFGHIKLMNPQRSTVWY"):
                return item
    raise TypeError(f"unsupported Pinal result type: {type(value).__name__}")


def pinal_batches(args: argparse.Namespace, start_index: int = 0):
    import torch
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode

    sys.path.insert(0, str(args.repo.resolve()))
    # Denovo-Pinal reads PINAL_MODEL_ROOT at import time.
    os.environ["PINAL_MODEL_ROOT"] = str(args.weights.resolve())
    os.chdir(args.repo)
    # The pinned official SaProt-T checkpoint stores training-only OmegaConf,
    # ZeRO and loss-scaler metadata beside the state dict. Inference does not
    # need DeepSpeed, so provide inert types for those exact pickle globals and
    # keep PyTorch's restricted weights-only unpickler enabled.
    module_names = (
        "deepspeed",
        "deepspeed.runtime",
        "deepspeed.runtime.fp16",
        "deepspeed.runtime.fp16.loss_scaler",
        "deepspeed.runtime.zero",
        "deepspeed.runtime.zero.config",
        "deepspeed.utils",
        "deepspeed.utils.tensor_fragment",
    )
    for module_name in module_names:
        module = types.ModuleType(module_name)
        module.__spec__ = importlib.util.spec_from_loader(module_name, loader=None)
        module.__path__ = []
        sys.modules[module_name] = module
        if "." in module_name:
            parent_name, child_name = module_name.rsplit(".", 1)
            setattr(sys.modules[parent_name], child_name, module)

    class LossScaler:
        def __init__(self, *values, **options):
            self.cur_scale = values[0] if values else options.get("scale", 1)

    LossScaler.__module__ = "deepspeed.runtime.fp16.loss_scaler"
    LossScaler.__qualname__ = "LossScaler"

    class ZeroStageEnum(enum.IntEnum):
        disabled = 0
        optimizer_states = 1
        gradients = 2
        weights = 3
        max_stage = 3

    ZeroStageEnum.__module__ = "deepspeed.runtime.zero.config"
    ZeroStageEnum.__qualname__ = "ZeroStageEnum"

    class fragment_address:
        def __init__(self, start: int = 0, numel: int = 0):
            self.start = start
            self.numel = numel

    fragment_address.__module__ = "deepspeed.utils.tensor_fragment"
    fragment_address.__qualname__ = "fragment_address"
    sys.modules[LossScaler.__module__].LossScaler = LossScaler
    sys.modules[ZeroStageEnum.__module__].ZeroStageEnum = ZeroStageEnum
    sys.modules[fragment_address.__module__].fragment_address = fragment_address
    torch.serialization.add_safe_globals([
        dict,
        list,
        int,
        TypingAny,
        defaultdict,
        DictConfig,
        ListConfig,
        Metadata,
        ContainerMetadata,
        AnyNode,
        LossScaler,
        ZeroStageEnum,
        fragment_address,
    ])

    upstream_torch_load = torch.load

    def trusted_official_checkpoint_load(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", True)
        return upstream_torch_load(*load_args, **load_kwargs)

    torch.load = trusted_official_checkpoint_load
    # The upstream generator only uses RichHandler for console formatting.
    # Keep that optional so inference does not require an unrelated UI package.
    try:
        import rich.logging  # noqa: F401
    except ImportError:
        # pip vendors a complete Rich copy even when Rich is not installed as
        # a standalone distribution. Reuse that implementation so both the
        # upstream logger and dependency probes see modules with valid specs.
        # A hand-made module without ``__spec__`` breaks Transformers import.
        try:
            import pip._vendor.rich as rich_module
            import pip._vendor.rich.logging as rich_logging_module
        except ImportError:
            rich_module = types.ModuleType("rich")
            rich_module.__spec__ = __import__("importlib.util").util.spec_from_loader("rich", loader=None)
            rich_logging_module = types.ModuleType("rich.logging")
            rich_logging_module.__spec__ = __import__("importlib.util").util.spec_from_loader(
                "rich.logging", loader=None
            )
            rich_logging_module.RichHandler = logging.StreamHandler
            rich_module.logging = rich_logging_module
        sys.modules["rich"] = rich_module
        sys.modules["rich.logging"] = rich_logging_module

    import utils.design_utils as design_utils
    from transformers import GenerationConfig

    def frozen_generation_config(temperature: float) -> GenerationConfig:
        return GenerationConfig(
            temperature=temperature,
            top_k=40,
            top_p=1.0,
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.0,
            max_length=args.generation_max_length,
            min_length=args.generation_min_length,
        )

    design_utils.T2StructDefaultGenerationConfig = frozen_generation_config
    PinalDesign = design_utils.PinalDesign
    load_pinal = design_utils.load_pinal

    components = load_pinal()
    if len(components) != 6:
        raise RuntimeError("unexpected load_pinal() return signature")
    names = (
        "t2struct", "text_tokenizer", "structure_tokenizer", "saprot",
        "saprot_text_tokenizer", "saprot_tokenizer",
    )
    models = dict(zip(names, components, strict=True))
    if start_index % args.batch_size:
        raise ValueError("Pinal resumable output must end on a complete generation batch")
    for start in range(start_index, args.requests, args.batch_size):
        seed_everything(args.seed + start)
        count = min(args.batch_size, args.requests - start)
        result = PinalDesign(
            desc=args.prompt, num=count, **models, temperature=0.7,
            saprot_sample_type="multinomial", multinomial_temperature=0.3,
        )
        sequences = [extract_pinal_sequence(item) for item in result]
        if len(sequences) != count:
            raise RuntimeError(f"Pinal returned {len(sequences)} sequences, expected {count}")
        yield start, sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("proteindt", "pinal"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-repeat", type=int, default=16)
    parser.add_argument("--pretrained-folder", type=Path)
    parser.add_argument("--decoder-folder", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--benchmark", default="unconditional")
    parser.add_argument("--model-id")
    parser.add_argument("--config-id")
    parser.add_argument("--generation-min-length", type=int, default=64)
    parser.add_argument("--generation-max-length", type=int, default=1024)
    args = parser.parse_args()
    seed_everything(args.seed)
    if args.backend == "proteindt":
        if not args.pretrained_folder or not args.decoder_folder:
            raise ValueError("ProteinDT requires --pretrained-folder and --decoder-folder")
        sequences = protein_dt(args)
        model_id = args.model_id or "proteindt_t5"
        config_id = args.config_id or "proteindt_t5_multi_property_prompt"
        method = "ProteinDT text-to-protein T5 decoder"
        metadata = {"temperature": 1.0, "top_p": 0.9, "top_k": 40,
                    "repetition_penalty": 1.0, "max_new_tokens": 512,
                    "num_repeat": args.num_repeat}
    else:
        if not args.weights:
            raise ValueError("Pinal requires --weights")
        model_id = args.model_id or "pinal_1.2b"
        config_id = args.config_id or "pinal_t2struc_1.2b_multi_property_prompt"
        method = "Pinal T2struc-1.2B plus SaProt-T"
        metadata = {"temperature": 0.7, "top_p": "", "top_k": "",
                    "repetition_penalty": "", "max_new_tokens": args.generation_max_length,
                    "min_new_tokens": args.generation_min_length,
                    "saprot_multinomial_temperature": 0.3}
        rows = read_jsonl(args.output) if args.output.exists() else []
        if len(rows) > args.requests:
            raise RuntimeError(f"existing output has {len(rows)} rows, expected at most {args.requests}")
        expected_ids = [f"{model_id}_{index:04d}" for index in range(len(rows))]
        if [row.get("request_id") for row in rows] != expected_ids:
            raise RuntimeError("existing Pinal output is not a contiguous request prefix")
        if len(rows) == args.requests:
            print(f"already complete: {args.output}")
            return
        for start, sequences in pinal_batches(args, len(rows)):
            for offset, sequence in enumerate(sequences):
                index = start + offset
                rows.append(build_generation_record(
                    request_id=f"{model_id}_{index:04d}", benchmark=args.benchmark,
                    model=model_id, seed=args.seed + index, condition=args.prompt,
                    raw_output=sequence, sequence=sequence, terminated=True,
                    termination_reason="model_generation_complete",
                    generation_method=method, generation_config_id=config_id,
                    method_metadata={**metadata, "prompt": args.prompt},
                ))
            atomic_write_jsonl(args.output, rows)
            print(f"Pinal requests complete: {len(rows)}/{args.requests}", flush=True)
        return
    rows = [build_generation_record(
        request_id=f"{model_id}_{index:04d}", benchmark=args.benchmark, model=model_id,
        seed=args.seed + index, condition=args.prompt, raw_output=sequence, sequence=sequence,
        terminated=True, termination_reason="model_generation_complete",
        generation_method=method, generation_config_id=config_id,
        method_metadata={**metadata, "prompt": args.prompt},
    ) for index, sequence in enumerate(sequences)]
    atomic_write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} {model_id} requests to {args.output}")


if __name__ == "__main__":
    main()
