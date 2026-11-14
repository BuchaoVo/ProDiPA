from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch

from .config import PromptConfig


IGNORE_INDEX = -100


class ConditionalProDiPADataCollator:
    """Build the fixed ProLLaMA prompt for every synthetic rollout step."""

    def __init__(self, tokenizer, prompt_config: PromptConfig) -> None:
        self.tokenizer = tokenizer
        self.prompt_config = prompt_config
        tokenized_prompt = tokenizer(
            prompt_config.prompt_text,
            return_attention_mask=False,
        )
        self.prompt_ids = list(tokenized_prompt["input_ids"])
        if len(self.prompt_ids) == 0:
            raise ValueError("Resolved ProLLaMA prompt token sequence is empty.")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor | int]:
        if len(features) == 0:
            raise ValueError("Received an empty batch in ProDiPA conditional distillation data collator.")
        batch_size = len(features)
        prompt_length = len(self.prompt_ids)
        prompt_ids = torch.tensor([self.prompt_ids for _ in range(batch_size)], dtype=torch.long)
        prompt_attention_mask = torch.ones((batch_size, prompt_length), dtype=torch.long)
        return {
            "prompt_ids": prompt_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "prompt_length": prompt_length,
            "prompt_lengths_per_example": torch.full((batch_size,), prompt_length, dtype=torch.long),
        }


class FixedRolloutDataset(torch.utils.data.Dataset):
    """Frozen complete sequences used by online and cached supervision."""

    def __init__(self, path: str | Path) -> None:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            self.rows = [row for row in csv.DictReader(handle, delimiter="\t") if row.get("valid_complete", "true") == "true"]
        if not self.rows:
            raise ValueError(f"No valid fixed rollouts found in {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, str | int]:
        row = self.rows[idx]
        return {"sample_id": idx, "sequence_id": row.get("sequence_id", str(idx)), "sequence": row["sequence"]}


class FixedRolloutCollator(ConditionalProDiPADataCollator):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor | int | list[str]]:
        if not features:
            raise ValueError("Received an empty fixed-rollout batch")
        target_ids = [
            self.tokenizer(
                f"Seq=<{feature['sequence']}>{self.tokenizer.eos_token}",
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            for feature in features
        ]
        rows = [torch.tensor(self.prompt_ids + target, dtype=torch.long) for target in target_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            rows, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        for index, row in enumerate(rows):
            attention_mask[index, :row.numel()] = 1
            labels[index, len(self.prompt_ids):row.numel()] = row[len(self.prompt_ids):]
        return {
            "fixed_input_ids": input_ids,
            "fixed_attention_mask": attention_mask,
            "fixed_labels": labels,
            "prompt_length": len(self.prompt_ids),
            "sequence_ids": [str(feature["sequence_id"]) for feature in features],
        }


class StaticCacheDataset(torch.utils.data.Dataset):
    def __init__(self, cache_path: str | Path) -> None:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.records = payload["records"]
        self.metadata = payload["metadata"]
        if not self.records:
            raise ValueError(f"Static cache is empty: {cache_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


class StaticCacheCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError("Exact full-vocabulary static cache currently requires batch size 1")
        record = features[0]
        return {
            "fixed_input_ids": record["input_ids"].unsqueeze(0),
            "fixed_attention_mask": record["attention_mask"].unsqueeze(0),
            "fixed_labels": record["labels"].unsqueeze(0),
            "cached_target_log_probs": record["target_log_probs"].unsqueeze(0),
            "prompt_length": int(record["prompt_length"]),
            "sequence_ids": [record["sequence_id"]],
        }
