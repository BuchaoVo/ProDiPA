"""Train one unconditional ProtGPT2 property teacher with Prefix-Tuning.

This maintained trainer keeps the frozen causal-LM objective and
Prefix-Tuning parameterization while adding
epoch-boundary resume state and explicit checkpoint roles.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import random
import shutil
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import peft
import torch
import torch.nn.functional as F
import transformers
import yaml
from peft import PeftModel, PrefixTuningConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_polynomial_decay_schedule_with_warmup,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeacherConfig:
    property_name: str
    model_name_or_path: str
    tokenizer_name_or_path: str
    train_path: str
    validation_path: str
    output_dir: str
    text_column: str = "sequence"
    id_column: str = "sequence_id"
    num_virtual_tokens: int = 100
    max_length: int = 512
    learning_rate: float = 5e-3
    lr_end: float = 1e-7
    lr_power: float = 3.0
    max_epochs: int = 50
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    seed: int = 42
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    dtype: str = "bfloat16"
    num_workers: int = 0
    log_every_optimizer_steps: int = 10
    resume: bool = True
    base_model_sha256: str = "UNKNOWN"
    source_code_revision: str = "UNKNOWN"
    generation_samples_per_epoch: int = 0
    generation_batch_size: int = 8
    generation_max_new_tokens: int = 512
    generation_temperature: float = 0.7
    generation_top_p: float = 0.9
    generation_top_k: int = 200
    generation_repetition_penalty: float = 1.2
    generation_seed: int = 4242
    generation_min_length_aa: int = 50
    generation_max_length_aa: int = 512
    generation_enforce_aa_limit: bool = False
    close_loss_weight: float = 1.0
    sequence_protocol: str = "plain_direct_eos_v1"

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def validate(self) -> None:
        from prodipa.unconditional.teacher_sequence_protocol import SUPPORTED_PROTOCOLS

        if self.sequence_protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"sequence_protocol must be one of {SUPPORTED_PROTOCOLS}; "
                f"got {self.sequence_protocol!r}"
            )
        if not math.isfinite(self.close_loss_weight) or self.close_loss_weight < 1.0:
            raise ValueError("close_loss_weight must be finite and at least 1.0")
        if self.property_name not in {"fold", "sol", "thermo"}:
            raise ValueError("property_name must be fold, sol, or thermo")
        for value in (
            self.model_name_or_path,
            self.tokenizer_name_or_path,
            self.train_path,
            self.validation_path,
        ):
            if not Path(value).exists():
                raise FileNotFoundError(value)
        if self.num_virtual_tokens <= 0 or self.max_length <= 0:
            raise ValueError("num_virtual_tokens and max_length must be positive")
        if self.learning_rate <= 0 or self.lr_end < 0:
            raise ValueError("invalid learning-rate settings")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.micro_batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch settings must be positive")
        if self.effective_batch_size != 8:
            raise ValueError(
                "This protocol fixes effective batch size at 8; got "
                f"{self.effective_batch_size}."
            )
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.dtype not in {"bfloat16", "float32"}:
            raise ValueError("dtype must be bfloat16 or float32")
        if self.generation_samples_per_epoch < 0:
            raise ValueError("generation_samples_per_epoch must be non-negative")
        if self.generation_samples_per_epoch and self.generation_batch_size <= 0:
            raise ValueError("generation_batch_size must be positive")
        if self.generation_max_new_tokens <= 0:
            raise ValueError("generation_max_new_tokens must be positive")
        if self.generation_temperature <= 0 or not 0 < self.generation_top_p <= 1:
            raise ValueError("invalid generation temperature/top-p")


def load_config(path: str | Path) -> TeacherConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Teacher config must be a YAML mapping")
    config = TeacherConfig(**payload)
    config.validate()
    return config


def _open_text(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


class ProteinCsvDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer,
        text_column: str,
        id_column: str,
        max_length: int,
        sequence_protocol: str = "plain_direct_eos_v1",
    ) -> None:
        self.records: list[dict[str, Any]] = []
        with _open_text(path) as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if text_column not in fields:
                raise ValueError(f"{path} is missing sequence column: {text_column}")
            has_explicit_id = id_column in fields
            seen_ids: set[str] = set()
            seen_sequences: set[str] = set()
            for row_index, row in enumerate(reader, start=2):
                sequence = row[text_column].strip().upper()
                if not sequence or not set(sequence).issubset(set("ACDEFGHIKLMNPQRSTVWY")):
                    raise ValueError(f"{path}:{row_index} contains an invalid protein sequence")
                sequence_id = row[id_column].strip() if has_explicit_id else ""
                if not sequence_id:
                    sequence_id = "sha256:" + hashlib.sha256(sequence.encode("ascii")).hexdigest()
                if sequence_id in seen_ids:
                    raise ValueError(f"{path}:{row_index} repeats sequence_id={sequence_id}")
                if sequence in seen_sequences:
                    raise ValueError(f"{path}:{row_index} repeats an exact sequence")
                seen_ids.add(sequence_id)
                seen_sequences.add(sequence)
                token_ids = encode_sequence(
                    tokenizer,
                    sequence,
                    max_length,
                    sequence_id=sequence_id,
                    sequence_protocol=sequence_protocol,
                )
                self.records.append(
                    {
                        "sequence_id": sequence_id,
                        "sequence_id_source": "csv" if has_explicit_id else "sequence_sha256",
                        "sequence": sequence,
                        "sequence_length_aa": len(sequence),
                        "input_ids": token_ids,
                    }
                )
        if not self.records:
            raise ValueError(f"No records found in {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]

    @property
    def token_lengths(self) -> list[int]:
        return [len(item["input_ids"]) for item in self.records]


def encode_sequence(
    tokenizer,
    sequence: str,
    max_length: int,
    sequence_id: str = "UNKNOWN",
    sequence_protocol: str = "plain_direct_eos_v1",
) -> list[int]:
    """Encode without truncation under an explicit ProtGPT2 text protocol."""
    from prodipa.unconditional.teacher_sequence_protocol import encode_teacher_sequence

    encoded = encode_teacher_sequence(tokenizer, sequence, sequence_protocol)
    if len(encoded) > max_length:
        raise ValueError(
            f"Sequence {sequence_id} of {len(sequence)} aa tokenizes to "
            f"{len(encoded)} tokens, "
            f"exceeding max_length={max_length}; truncation is forbidden."
        )
    return encoded


class CausalProteinCollator:
    """Right-pad to the longest sequence in a batch; padding is not supervised."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        width = max(len(record["input_ids"]) for record in records)
        input_ids, attention_mask, labels, close_token_mask = [], [], [], []
        for record in records:
            ids = record["input_ids"]
            if len(ids) < 2 or ids[0] != ids[-1]:
                raise ValueError("Expected a complete BOS/sequence/EOS record")
            pad = width - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            labels.append(ids + [-100] * pad)
            close_token_mask.append([False] * (len(ids) - 1) + [True] + [False] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "close_token_mask": torch.tensor(close_token_mask, dtype=torch.bool),
        }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_fingerprint(config: TeacherConfig) -> str:
    critical = {
        key: value
        for key, value in asdict(config).items()
        if key not in {"resume", "log_every_optimizer_steps"}
    }
    if not config.generation_enforce_aa_limit:
        # Keep historical epoch-boundary checkpoints resumable when opted out.
        critical.pop("generation_enforce_aa_limit", None)
    if config.close_loss_weight == 1.0:
        critical.pop("close_loss_weight", None)
    return hashlib.sha256(
        json.dumps(critical, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _length_summary(dataset: ProteinCsvDataset) -> dict[str, Any]:
    values = np.asarray(dataset.token_lengths)
    longest = max(dataset.records, key=lambda row: len(row["input_ids"]))
    return {
        "count": int(values.size),
        "min_tokens_with_eos": int(values.min()),
        "median_tokens_with_eos": float(np.median(values)),
        "p95_tokens_with_eos": float(np.quantile(values, 0.95)),
        "max_tokens_with_eos": int(values.max()),
        "max_tokens_sequence_id": longest["sequence_id"],
        "max_tokens_sequence_length_aa": longest["sequence_length_aa"],
        "sequence_id_source": dataset.records[0]["sequence_id_source"],
    }


def _configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "training.log", encoding="utf-8"),
        ],
        force=True,
    )


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _save_adapter(model, tokenizer, path: Path, metadata: dict[str, Any]) -> None:
    staging = path.with_name(path.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(staging, safe_serialization=True)
    tokenizer.save_pretrained(staging)
    _save_json(staging / "checkpoint_metadata.json", metadata)
    if path.exists():
        shutil.rmtree(path)
    staging.rename(path)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _save_resume_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    output_dir: Path,
    state: dict[str, Any],
) -> None:
    target = output_dir / "latest_resume"
    temporary = output_dir / "latest_resume.tmp"
    previous = output_dir / "latest_resume.previous"
    if temporary.exists():
        shutil.rmtree(temporary)
    _save_adapter(model, tokenizer, temporary, state["metadata"])
    torch.save(
        {
            **state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": _rng_state(),
        },
        temporary / "training_state.pt",
    )
    if previous.exists():
        shutil.rmtree(previous)
    if target.exists():
        target.rename(previous)
    temporary.rename(target)
    if previous.exists():
        shutil.rmtree(previous)


def _move_optimizer_state(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _optimizer_state_audit(optimizer) -> dict[str, Any]:
    floating = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value) and value.is_floating_point() and value.numel() > 1
    ]
    if not floating:
        raise RuntimeError("AdamW did not create floating-point optimizer state")
    invalid = [str(value.dtype) for value in floating if value.dtype != torch.float32]
    if invalid:
        raise RuntimeError(f"AdamW optimizer state is not FP32: {invalid[:10]}")
    if not all(torch.isfinite(value).all() for value in floating):
        raise FloatingPointError("AdamW optimizer state contains NaN/Inf")
    return {"floating_state_tensors": len(floating), "dtype": "float32", "finite": True}


def _append_history(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _make_loader(
    dataset: Dataset,
    collator: CausalProteinCollator,
    config: TeacherConfig,
    epoch: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed + epoch)
    return DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=True,
    )


def _autocast_context(config: TeacherConfig):
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=config.dtype == "bfloat16",
    )


def _causal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    close_token_mask: torch.Tensor | None = None,
    close_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """Weighted causal CE; only the genuine final EOS receives extra weight."""
    if not math.isfinite(close_loss_weight) or close_loss_weight < 1.0:
        raise ValueError("close_loss_weight must be finite and at least 1.0")
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    active = shift_labels.ne(-100)
    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    weights = active.to(per_token.dtype)
    if close_loss_weight != 1.0:
        if close_token_mask is None or close_token_mask.shape != labels.shape:
            raise ValueError("Weighted EOS CE requires an aligned close_token_mask")
        if close_token_mask[:, 0].any() or (close_token_mask & labels.eq(-100)).any():
            raise ValueError("BOS/padding must not receive terminal EOS supervision")
        closing = close_token_mask[:, 1:] & active
        if not torch.all(closing.sum(dim=1) == 1):
            raise ValueError("Expected exactly one genuine terminal EOS per example")
        positions = torch.arange(active.shape[1], device=labels.device).expand_as(active)
        last = positions.masked_fill(~active, -1).max(dim=1).values
        if not torch.all(closing.gather(1, last.clamp_min(0).unsqueeze(1))):
            raise ValueError("Closing marker must be the final supervised target")
        weights = torch.where(closing, close_loss_weight, weights)
    normalizer = weights.sum()
    if normalizer.item() <= 0:
        raise ValueError("batch has no supervised causal targets")
    return (per_token * weights).sum() / normalizer, float(normalizer.detach().cpu())


def _model_loss(
    model,
    batch: dict[str, torch.Tensor],
    close_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, float]:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    return _causal_cross_entropy(outputs.logits, batch["labels"],
                                 batch.get("close_token_mask"), close_loss_weight)


@torch.no_grad()
def _evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    config: TeacherConfig,
) -> float:
    model.eval()
    loss_sum = 0.0
    normalization_sum = 0.0
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with _autocast_context(config):
            loss, normalizer = _model_loss(model, batch, config.close_loss_weight)
        loss_sum += float(loss.detach().cpu()) * normalizer
        normalization_sum += normalizer
    return loss_sum / normalization_sum


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared = torch.zeros((), dtype=torch.float32, device=parameters[0].device)
    for parameter in parameters:
        if parameter.grad is not None:
            squared += parameter.grad.detach().float().square().sum()
    return float(squared.sqrt().cpu())


def _kmer_diversity(sequences: list[str], k: int = 3) -> float:
    if len(sequences) < 2:
        return 0.0
    kmers = [
        {sequence[index : index + k] for index in range(max(0, len(sequence) - k + 1))}
        for sequence in sequences
    ]
    distances = []
    for left in range(len(kmers)):
        for right in range(left + 1, len(kmers)):
            union = kmers[left] | kmers[right]
            distances.append(1.0 - len(kmers[left] & kmers[right]) / max(1, len(union)))
    return float(np.mean(distances))


@torch.no_grad()
def _generation_eval(
    model,
    tokenizer,
    device: torch.device,
    config: TeacherConfig,
    epoch: int,
    training_sequences: set[str],
    output_dir: Path,
) -> dict[str, Any]:
    count = config.generation_samples_per_epoch
    if count == 0:
        return {}
    saved_rng = _rng_state()
    _set_seed(config.generation_seed)
    was_training = model.training
    original_use_cache = model.config.use_cache
    model.eval()
    model.config.use_cache = True
    records = []
    canonical = set("ACDEFGHIKLMNPQRSTVWY")
    try:
        while len(records) < count:
            batch_size = min(config.generation_batch_size, count - len(records))
            prompt = torch.full(
                (batch_size, 1), tokenizer.eos_token_id, dtype=torch.long, device=device
            )
            length_processor = None
            extra_kwargs = {}
            if config.generation_enforce_aa_limit:
                from .length_control import ProteinLengthLogitsProcessor
                length_processor = ProteinLengthLogitsProcessor(tokenizer, config.generation_max_length_aa)
                extra_kwargs["logits_processor"] = [length_processor]
            generated = model.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                max_new_tokens=config.generation_max_new_tokens,
                do_sample=True,
                temperature=config.generation_temperature,
                top_p=config.generation_top_p,
                top_k=config.generation_top_k,
                repetition_penalty=config.generation_repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                **extra_kwargs,
            ).detach().cpu().tolist()
            forced_flags = length_processor.forced_flags() if length_processor is not None else [False] * len(generated)
            for row, forced in zip(generated, forced_flags):
                new_tokens = row[1:]
                eos_reached = tokenizer.eos_token_id in new_tokens
                if eos_reached:
                    new_tokens = new_tokens[: new_tokens.index(tokenizer.eos_token_id)]
                sequence = "".join(
                    tokenizer.decode(
                        new_tokens,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ).split()
                ).upper()
                valid = (
                    bool(sequence)
                    and set(sequence).issubset(canonical)
                    and eos_reached
                    and not forced
                    and config.generation_min_length_aa
                    <= len(sequence)
                    <= config.generation_max_length_aa
                )
                records.append(
                    {
                        "sample_id": len(records),
                        "sequence": sequence,
                        "length_aa": len(sequence),
                        "eos_reached": eos_reached,
                        "length_forced_eos": forced,
                        "natural_eos_reached": eos_reached and not forced,
                        "canonical_only": bool(sequence) and set(sequence).issubset(canonical),
                        "valid": valid,
                        "exact_training_duplicate": sequence in training_sequences,
                    }
                )
    finally:
        model.config.use_cache = original_use_cache
        if was_training:
            model.train()
        _restore_rng_state(saved_rng)

    sequences = [record["sequence"] for record in records]
    valid_sequences = [record["sequence"] for record in records if record["valid"]]
    lengths = np.asarray([record["length_aa"] for record in records], dtype=float)
    metrics = {
        "epoch": epoch,
        "samples": count,
        "valid_sequence_rate": float(np.mean([record["valid"] for record in records])),
        "unique_sequence_rate": len(set(sequences)) / count,
        "eos_termination_rate": float(np.mean([record["eos_reached"] for record in records])),
        "length_min": int(lengths.min()),
        "length_median": float(np.median(lengths)),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std()),
        "length_max": int(lengths.max()),
        "mean_pairwise_3mer_jaccard_distance_valid": _kmer_diversity(valid_sequences),
        "exact_training_duplicate_count": sum(
            record["exact_training_duplicate"] for record in records
        ),
        "exact_training_duplicate_rate": float(
            np.mean([record["exact_training_duplicate"] for record in records])
        ),
        "sampling_seed": config.generation_seed,
        "temperature": config.generation_temperature,
        "top_p": config.generation_top_p,
        "top_k": config.generation_top_k,
        "repetition_penalty": config.generation_repetition_penalty,
        "max_new_tokens": config.generation_max_new_tokens,
    }
    generation_dir = output_dir / "epoch_generation"
    generation_dir.mkdir(parents=True, exist_ok=True)
    with (generation_dir / f"epoch_{epoch}.json").open("w", encoding="utf-8") as handle:
        json.dump({"metrics": metrics, "records": records}, handle, indent=2)
    return metrics


def _build_model(config: TeacherConfig, resume_dir: Path | None):
    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        dtype=dtype,
        local_files_only=True,
    )
    base.config.use_cache = False
    if resume_dir is not None:
        model = PeftModel.from_pretrained(base, resume_dir, is_trainable=True)
    else:
        model = get_peft_model(
            base,
            PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=config.num_virtual_tokens,
            ),
        )
    # PEFT normally creates Prefix parameters in FP32. Keep this explicit so a
    # future PEFT default change cannot silently move AdamW state to BF16.
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    return model


def _validate_prefix_parameterization(model) -> dict[str, Any]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if not trainable or not frozen:
        raise RuntimeError("Prefix-Tuning must have both trainable Prefix and frozen backbone parameters")
    invalid_names = [
        name for name, _ in trainable
        if "prompt_encoder" not in name and "prefix" not in name
    ]
    if invalid_names:
        raise RuntimeError(f"Non-Prefix parameters are trainable: {invalid_names[:10]}")
    invalid_dtypes = [name for name, parameter in trainable if parameter.dtype != torch.float32]
    if invalid_dtypes:
        raise RuntimeError(f"Trainable Prefix parameters are not FP32: {invalid_dtypes[:10]}")
    return {
        "trainable_parameter_names": [name for name, _ in trainable],
        "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
        "frozen_parameters": sum(parameter.numel() for _, parameter in frozen),
        "trainable_dtype": "float32",
        "frozen_dtypes": sorted({str(parameter.dtype).removeprefix("torch.") for _, parameter in frozen}),
    }


def _validate_saved_adapter(config: TeacherConfig, adapter_path: Path) -> dict[str, Any]:
    if not (adapter_path / "adapter_config.json").is_file():
        raise RuntimeError(f"Saved adapter is missing adapter_config.json: {adapter_path}")
    base = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        dtype=torch.bfloat16 if config.dtype == "bfloat16" else torch.float32,
        local_files_only=True,
    )
    loaded = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    adapter_config = loaded.peft_config["default"]
    actual_virtual_tokens = int(adapter_config.num_virtual_tokens)
    if actual_virtual_tokens != config.num_virtual_tokens:
        raise RuntimeError(
            f"Reloaded adapter virtual-token mismatch: {actual_virtual_tokens} != "
            f"{config.num_virtual_tokens}"
        )
    adapter_tensors = [
        parameter.detach() for name, parameter in loaded.named_parameters()
        if "prompt_encoder" in name or "prefix" in name
    ]
    if not adapter_tensors or not all(torch.isfinite(parameter).all() for parameter in adapter_tensors):
        raise RuntimeError("Reloaded Prefix adapter contains no finite Prefix tensors")
    result = {
        "adapter_path": str(adapter_path.resolve()),
        "peft_type": str(adapter_config.peft_type),
        "num_virtual_tokens": actual_virtual_tokens,
        "finite_prefix_tensors": len(adapter_tensors),
        "load_api": "PeftModel.from_pretrained",
    }
    del loaded, base
    return result


def _metadata(
    config: TeacherConfig,
    train_path_sha256: str,
    validation_path_sha256: str,
    epoch: int,
    global_step: int,
    train_loss: float,
    validation_loss: float,
) -> dict[str, Any]:
    return {
        "property": config.property_name,
        "base_model": config.model_name_or_path,
        "base_model_sha256": config.base_model_sha256,
        "source_code_revision": config.source_code_revision,
        "train_path": config.train_path,
        "train_path_sha256": train_path_sha256,
        "validation_path": config.validation_path,
        "validation_path_sha256": validation_path_sha256,
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "effective_batch_size": config.effective_batch_size,
        "seed": config.seed,
    }


def train(config: TeacherConfig, smoke_optimizer_steps: int | None = None) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("ProtGPT2 teacher training requires CUDA")
    if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured bfloat16 but the selected GPU does not support it")

    output_dir = Path(config.output_dir)
    _configure_logging(output_dir)
    _set_seed(config.seed)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name_or_path,
        local_files_only=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("ProtGPT2 tokenizer must define EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = ProteinCsvDataset(
        config.train_path,
        tokenizer,
        config.text_column,
        config.id_column,
        config.max_length,
        config.sequence_protocol,
    )
    validation_dataset = ProteinCsvDataset(
        config.validation_path,
        tokenizer,
        config.text_column,
        config.id_column,
        config.max_length,
        config.sequence_protocol,
    )
    preflight = {
        "config": asdict(config),
        "config_fingerprint": _config_fingerprint(config),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "train": _length_summary(train_dataset),
        "validation": _length_summary(validation_dataset),
        "no_truncation": True,
        "sequence_protocol": config.sequence_protocol,
    }
    _save_json(output_dir / "preflight.json", preflight)

    train_sha = _sha256(config.train_path)
    validation_sha = _sha256(config.validation_path)
    resume_dir = output_dir / "latest_resume"
    state_path = resume_dir / "training_state.pt"
    do_resume = config.resume and state_path.exists()
    model = _build_model(config, resume_dir if do_resume else None).to(device)
    parameterization = _validate_prefix_parameterization(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable = sum(parameter.numel() for parameter in trainable_parameters)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable == 0 or trainable == total:
        raise RuntimeError(f"Invalid Prefix-Tuning trainable parameters: {trainable}/{total}")
    LOGGER.info("Trainable parameters: %d / %d (%.4f%%)", trainable, total, 100 * trainable / total)

    collator = CausalProteinCollator(tokenizer.pad_token_id)
    microbatches_per_epoch = math.ceil(len(train_dataset) / config.micro_batch_size)
    optimizer_steps_per_epoch = math.ceil(
        microbatches_per_epoch / config.gradient_accumulation_steps
    )
    total_optimizer_steps = optimizer_steps_per_epoch * config.max_epochs
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
    )
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_optimizer_steps,
        lr_end=config.lr_end,
        power=config.lr_power,
    )

    start_epoch = 1
    global_step = 0
    best_validation_loss = float("inf")
    best_epoch = 0
    no_improve_epochs = 0
    elapsed_seconds_before_resume = 0.0
    if do_resume:
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved["config_fingerprint"] != _config_fingerprint(config):
            raise RuntimeError("Resume config fingerprint does not match current config")
        if saved["train_path_sha256"] != train_sha or saved["validation_path_sha256"] != validation_sha:
            raise RuntimeError("Resume data fingerprint does not match current data")
        optimizer.load_state_dict(saved["optimizer"])
        _move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(saved["scheduler"])
        _restore_rng_state(saved["rng"])
        start_epoch = int(saved["epoch"]) + 1
        global_step = int(saved["global_step"])
        best_validation_loss = float(saved["best_validation_loss"])
        best_epoch = int(saved["best_epoch"])
        no_improve_epochs = int(saved["no_improve_epochs"])
        elapsed_seconds_before_resume = float(saved.get("elapsed_seconds", 0.0))
        LOGGER.info("Resumed after epoch=%d global_step=%d", start_epoch - 1, global_step)

    if start_epoch > config.max_epochs:
        result_path = output_dir / "train_result.json"
        if result_path.exists():
            with result_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        raise RuntimeError("Run is already complete but train_result.json is missing")

    _save_json(
        output_dir / "run_metadata.json",
        {
            **preflight,
            "train_path_sha256": train_sha,
            "validation_path_sha256": validation_sha,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "parameterization_audit": parameterization,
            "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
            "max_optimizer_steps": total_optimizer_steps,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "peft_version": peft.__version__,
            "cuda_version": torch.version.cuda,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        },
    )

    history_path = output_dir / "training_history.tsv"
    run_start = time.time()
    stop_after_smoke = False
    for epoch in range(start_epoch, config.max_epochs + 1):
        train_loader = _make_loader(train_dataset, collator, config, epoch, shuffle=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch_losses: list[float] = []
        batch_loss_normalizers: list[float] = []
        gradient_norms: list[float] = []
        supervised_tokens = 0
        epoch_start_step = global_step
        epoch_start = time.time()
        group_start = 0
        for batch_index, batch in enumerate(train_loader):
            remaining = len(train_loader) - group_start
            group_size = min(config.gradient_accumulation_steps, remaining)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            supervised_tokens += int((batch["labels"] != -100).sum().item())
            with _autocast_context(config):
                loss, loss_normalizer = _model_loss(model, batch, config.close_loss_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch} batch={batch_index}")
            batch_losses.append(float(loss.detach().cpu()))
            batch_loss_normalizers.append(loss_normalizer)
            (loss / group_size).backward()
            end_of_group = (batch_index - group_start + 1) == group_size
            if not end_of_group:
                continue
            gradient_norm = _gradient_norm(trainable_parameters)
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch={epoch} batch={batch_index}"
                )
            gradient_norms.append(gradient_norm)
            optimizer.step()
            if global_step == 0:
                optimizer_state_audit = _optimizer_state_audit(optimizer)
                LOGGER.info("AdamW state audit: %s", optimizer_state_audit)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            group_start = batch_index + 1
            if global_step % config.log_every_optimizer_steps == 0:
                LOGGER.info(
                    "epoch=%d step=%d loss=%.6f lr=%.8g grad_norm=%.6f",
                    epoch,
                    global_step,
                    float(loss.detach().cpu()),
                    scheduler.get_last_lr()[0],
                    gradient_norm,
                )
            if smoke_optimizer_steps is not None and global_step >= smoke_optimizer_steps:
                stop_after_smoke = True
                break

        train_loss = float(
            np.average(batch_losses, weights=np.asarray(batch_loss_normalizers, dtype=float))
        )
        validation_loader = _make_loader(
            validation_dataset, collator, config, epoch, shuffle=False
        )
        validation_loss = _evaluate(model, validation_loader, device, config)
        improved = validation_loss < best_validation_loss - config.early_stopping_min_delta
        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        epoch_seconds = time.time() - epoch_start
        metadata = _metadata(
            config,
            train_sha,
            validation_sha,
            epoch,
            global_step,
            train_loss,
            validation_loss,
        )
        _save_adapter(model, tokenizer, output_dir / "epochs" / f"epoch_{epoch}", metadata)
        if epoch == 1:
            _save_adapter(model, tokenizer, output_dir / "epoch_1", metadata)
        if improved:
            _save_adapter(model, tokenizer, output_dir / "best_adapter", metadata)
        _save_resume_checkpoint(
            model,
            tokenizer,
            optimizer,
            scheduler,
            output_dir,
            {
                "epoch": epoch,
                "global_step": global_step,
                "best_validation_loss": best_validation_loss,
                "best_epoch": best_epoch,
                "no_improve_epochs": no_improve_epochs,
                "config_fingerprint": _config_fingerprint(config),
                "train_path_sha256": train_sha,
                "validation_path_sha256": validation_sha,
                "elapsed_seconds": elapsed_seconds_before_resume + (time.time() - run_start),
                "metadata": metadata,
            },
        )
        generation_metrics = _generation_eval(
            model,
            tokenizer,
            device,
            config,
            epoch,
            {record["sequence"] for record in train_dataset.records},
            output_dir,
        )
        _append_history(
            history_path,
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_ppl": math.exp(min(train_loss, 80.0)),
                "validation_ppl": math.exp(min(validation_loss, 80.0)),
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_norm_mean": float(np.mean(gradient_norms)),
                "gradient_norm_max": float(np.max(gradient_norms)),
                "epoch_seconds": epoch_seconds,
                "optimizer_steps": global_step - epoch_start_step,
                "supervised_tokens": supervised_tokens,
                "supervised_targets": float(sum(batch_loss_normalizers)),
                "supervised_tokens_per_second": supervised_tokens / max(epoch_seconds, 1e-9),
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "no_improve_epochs": no_improve_epochs,
            },
        )
        if generation_metrics:
            _append_history(output_dir / "epoch_generation_metrics.tsv", generation_metrics)
        LOGGER.info(
            "epoch=%d train_loss=%.6f validation_loss=%.6f best_epoch=%d seconds=%.1f",
            epoch,
            train_loss,
            validation_loss,
            best_epoch,
            epoch_seconds,
        )
        if stop_after_smoke:
            LOGGER.info("Smoke run stopped after %d optimizer steps", global_step)
            break
        if no_improve_epochs >= config.early_stopping_patience:
            LOGGER.info(
                "Early stopping at epoch=%d; best epoch=%d validation_loss=%.6f",
                epoch,
                best_epoch,
                best_validation_loss,
            )
            break

    final_metadata = {
        **metadata,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "runtime_seconds": elapsed_seconds_before_resume + (time.time() - run_start),
        "gpu_hours": (elapsed_seconds_before_resume + (time.time() - run_start)) / 3600.0,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "smoke_run": smoke_optimizer_steps is not None,
        "optimizer_steps": global_step,
        "best_adapter": str((output_dir / "best_adapter").resolve()),
    }
    _save_adapter(model, tokenizer, output_dir / "final", final_metadata)
    final_metadata["adapter_reload_validation"] = _validate_saved_adapter(
        config, output_dir / "best_adapter"
    )
    _save_json(output_dir / "train_result.json", final_metadata)
    _save_json(
        output_dir / ("SMOKE_OK.json" if smoke_optimizer_steps is not None else "DONE.json"),
        final_metadata,
    )
    return final_metadata


def preflight(config: TeacherConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    _configure_logging(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name_or_path, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data = ProteinCsvDataset(
        config.train_path,
        tokenizer,
        config.text_column,
        config.id_column,
        config.max_length,
        config.sequence_protocol,
    )
    validation_data = ProteinCsvDataset(
        config.validation_path,
        tokenizer,
        config.text_column,
        config.id_column,
        config.max_length,
        config.sequence_protocol,
    )
    result = {
        "property": config.property_name,
        "effective_batch_size": config.effective_batch_size,
        "train": _length_summary(train_data),
        "validation": _length_summary(validation_data),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "max_length": config.max_length,
        "no_truncation": True,
        "sequence_protocol": config.sequence_protocol,
    }
    _save_json(output_dir / "preflight.json", result)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-optimizer-steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        if args.preflight_only:
            preflight(config)
        else:
            train(config, smoke_optimizer_steps=args.smoke_optimizer_steps)
    except Exception as error:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_json(
            output_dir / "FAILED.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "time_unix": time.time(),
            },
        )
        raise


if __name__ == "__main__":
    main()
