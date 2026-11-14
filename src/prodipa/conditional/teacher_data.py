"""Instruction-data preparation for conditional LoRA teachers.

The prompt tokens are context only.  Loss is computed on ``output`` plus EOS.
Token-budget violations fail closed; proteins are never silently truncated.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import datasets
import torch
import transformers
from datasets import concatenate_datasets, load_dataset

IGNORE_INDEX = -100
LOGGER = logging.getLogger(__name__)


def tokenize_instruction_batch(examples, tokenizer, max_seq_length: int) -> dict:
    """Tokenize a batch with response-only labels and a hard token limit."""

    prompts: list[str] = []
    targets: list[str] = []
    for instruction, input_text, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        if not isinstance(instruction, str) or not isinstance(output, str):
            raise ValueError("instruction and output must be strings")
        prompt = instruction
        if input_text is not None and str(input_text) != "":
            prompt += " " + str(input_text)
        prompts.append(prompt)
        targets.append(output)

    prompt_ids = tokenizer(prompts, return_attention_mask=False)["input_ids"]
    target_ids = tokenizer(
        targets, return_attention_mask=False, add_special_tokens=False
    )["input_ids"]
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    all_input_ids, all_labels = [], []
    for prompt, encoded_target in zip(prompt_ids, target_ids):
        # Do not append tokenizer.eos_token as text. Under datasets.map worker
        # serialization, the literal string can be split into ordinary tokens
        # instead of the configured special EOS token.
        target = list(encoded_target) + [int(tokenizer.eos_token_id)]
        total = len(prompt) + len(target)
        if total > max_seq_length:
            raise ValueError(
                f"Instruction example tokenizes to {total} tokens, exceeding "
                f"max_seq_length={max_seq_length}; truncation is forbidden."
            )
        all_input_ids.append(prompt + target)
        all_labels.append([IGNORE_INDEX] * len(prompt) + target)
    return {"input_ids": all_input_ids, "labels": all_labels}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_path(
    source: Path,
    cache_root: Path,
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_length: int,
) -> Path:
    identity = {
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_path": str(getattr(tokenizer, "name_or_path", "unknown")),
        "tokenizer_size": len(tokenizer),
        "max_seq_length": max_seq_length,
        "format": "instruction-output-explicit-eos-loss",
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    return cache_root / f"{source.stem}_{key}"


def tokenize_instruction_dataset(
    data_path: str | Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_length: int,
    data_cache_dir: str | None = None,
    preprocessing_num_workers: int | None = None,
) -> datasets.Dataset:
    """Load one or more instruction JSON files and tokenize without truncation."""

    sources = [data_path] if isinstance(data_path, str) else list(data_path)
    if not sources:
        raise ValueError("At least one instruction dataset is required")
    if preprocessing_num_workers not in {None, 0, 1}:
        raise ValueError(
            "Conditional teacher tokenization is intentionally single-process "
            "to preserve tokenizer special-token state"
        )
    cache_root = Path(data_cache_dir).resolve() if data_cache_dir else None
    prepared: list[datasets.Dataset] = []

    for value in sources:
        source = Path(value).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        root = cache_root or source.parent / ".tokenized_cache"
        root.mkdir(parents=True, exist_ok=True)
        cached = _cache_path(source, root, tokenizer, max_seq_length)
        if cached.is_dir():
            dataset = datasets.load_from_disk(str(cached))
            LOGGER.info("Loaded tokenized dataset cache: %s", cached)
        else:
            raw = load_dataset("json", data_files=str(source), cache_dir=str(root))["train"]
            required = {"instruction", "input", "output"}
            missing = required - set(raw.column_names)
            if missing:
                raise ValueError(f"{source} is missing fields: {sorted(missing)}")
            dataset = raw.map(
                lambda batch: tokenize_instruction_batch(
                    batch, tokenizer, max_seq_length
                ),
                batched=True,
                num_proc=None,
                remove_columns=raw.column_names,
                keep_in_memory=False,
                desc=f"Tokenizing {source.name}",
            )
            dataset.save_to_disk(str(cached))
        dataset.set_format("torch")
        prepared.append(dataset)
    return concatenate_datasets(prepared)


@dataclass
class DataCollatorForSupervisedDataset:
    """Right-pad inputs and mask padding from the supervised objective."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]
        lengths = torch.tensor([value.numel() for value in input_ids], dtype=torch.long)
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        # PAD may intentionally share EOS's token ID.  A token-ID comparison
        # would therefore mask the real terminal EOS as padding.
        attention_mask = (
            torch.arange(padded_inputs.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
        )
        close_token_mask = torch.zeros_like(padded_inputs, dtype=torch.bool)
        for row, label in enumerate(labels):
            supervised = torch.nonzero(label.ne(IGNORE_INDEX), as_tuple=False).flatten()
            if supervised.numel() < 2:
                raise ValueError("A conditional response must contain close and EOS tokens")
            close_position = int(supervised[-2])
            close_token_id = int(input_ids[row][close_position])
            if self.tokenizer.decode([close_token_id], skip_special_tokens=False) != ">":
                raise ValueError(
                    "The penultimate supervised token is not the response closing marker"
                )
            close_token_mask[row, close_position] = True
        return {
            "input_ids": padded_inputs,
            "labels": padded_labels,
            "attention_mask": attention_mask,
            "close_token_mask": close_token_mask,
        }
