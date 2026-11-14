"""Fixed-rollout full-PoE cache support for the paired Conditional OPD study."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import TrainerCallback

from prollama_opd_trainer import (
    BaseProLLaMAOpdTrainer,
    ProLLaMAProteinOPDTrainer,
)


class CacheSegmentBoundary(TrainerCallback):
    """Save and pause at a cache boundary without shortening the LR schedule."""

    def __init__(self, stop_step: int):
        self.stop_step = int(stop_step)

    def on_step_end(self, args, state, control, **kwargs):
        control.should_save = state.global_step >= self.stop_step
        if state.global_step >= self.stop_step:
            control.should_training_stop = True
        return control


def cache_record_path(cache_dir: Path, sample_id: int) -> Path:
    return cache_dir / "records" / f"{sample_id:06d}.pt"


def load_cache_record(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"cache record is not a mapping: {path}")
    required = {"sample_id", "input_ids", "prompt_length", "target_log_probs"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"cache record lacks {sorted(missing)}: {path}")
    return value


class FullPoeCacheDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: Path, sample_count: int) -> None:
        self.cache_dir = Path(cache_dir)
        self.sample_count = int(sample_count)
        manifest = json.loads((self.cache_dir / "CACHE_MANIFEST.json").read_text())
        if manifest.get("status") != "COMPLETE":
            raise ValueError("offline cache is not complete")
        if int(manifest.get("sample_count", -1)) != self.sample_count:
            raise ValueError("offline cache sample count does not match the experiment")
        for sample_id in range(self.sample_count):
            if not cache_record_path(self.cache_dir, sample_id).is_file():
                raise FileNotFoundError(cache_record_path(self.cache_dir, sample_id))

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, int]:
        return {"sample_id": int(index)}


class FullPoeCacheCollator:
    def __init__(self, cache_dir: Path, tokenizer) -> None:
        self.cache_dir = Path(cache_dir)
        self.pad_token_id = int(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def __call__(self, features: list[dict[str, int]]) -> dict[str, torch.Tensor | int]:
        records = [
            load_cache_record(cache_record_path(self.cache_dir, int(item["sample_id"])))
            for item in features
        ]
        prompt_lengths = {int(item["prompt_length"]) for item in records}
        vocab_sizes = {int(item["target_log_probs"].shape[-1]) for item in records}
        if len(prompt_lengths) != 1 or len(vocab_sizes) != 1:
            raise ValueError("cache batch mixes prompt lengths or vocabularies")
        prompt_length = prompt_lengths.pop()
        vocab_size = vocab_sizes.pop()
        total_lengths = [int(item["input_ids"].numel()) for item in records]
        completion_lengths = [length - prompt_length for length in total_lengths]
        if any(length <= 0 for length in completion_lengths):
            raise ValueError("cache record has an empty completion")
        max_total = max(total_lengths)
        max_completion = max(completion_lengths)
        batch = len(records)
        input_ids = torch.full((batch, max_total), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, max_total), dtype=torch.long)
        labels = torch.full((batch, max_completion), -100, dtype=torch.long)
        uniform_logp = -math.log(vocab_size)
        target = torch.full(
            (batch, max_completion, vocab_size),
            uniform_logp,
            dtype=torch.bfloat16,
        )
        sample_ids = torch.empty(batch, dtype=torch.long)
        for row, (record, total_length, completion_length) in enumerate(
            zip(records, total_lengths, completion_lengths)
        ):
            ids = record["input_ids"].long()
            cached_target = record["target_log_probs"]
            if tuple(cached_target.shape) != (completion_length, vocab_size):
                raise ValueError(f"misaligned cache target for sample {record['sample_id']}")
            input_ids[row, :total_length] = ids
            attention_mask[row, :total_length] = 1
            labels[row, :completion_length] = ids[prompt_length:]
            if not bool(record.get("supervise_terminal", True)):
                labels[row, completion_length - 1] = -100
            target[row, :completion_length] = cached_target.to(torch.bfloat16)
            sample_ids[row] = int(record["sample_id"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "shifted_labels": labels,
            "target_log_probs": target,
            "prompt_length": prompt_length,
            "sample_ids": sample_ids,
        }


class _MeasuredMixin:
    def _init_stage_timings(self) -> None:
        self.stage_seconds = {"total_compute": 0.0, "rollout": 0.0, "teacher": 0.0}

    @staticmethod
    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def timing_payload(self) -> dict[str, float]:
        total = float(self.stage_seconds["total_compute"])
        rollout = float(self.stage_seconds["rollout"])
        teacher = float(self.stage_seconds["teacher"])
        return {
            "compute_seconds": total,
            "rollout_seconds": rollout,
            "teacher_seconds": teacher,
            "student_and_loss_seconds": max(0.0, total - rollout - teacher),
        }


class MeasuredOnlineTrainer(_MeasuredMixin, ProLLaMAProteinOPDTrainer):
    """Existing online trainer with device-synchronized stage accounting only."""

    def __init__(self, *args, **kwargs) -> None:
        self._init_stage_timings()
        super().__init__(*args, **kwargs)

    def _generate_on_policy(self, *args, **kwargs):
        self._sync()
        started = time.perf_counter()
        result = super()._generate_on_policy(*args, **kwargs)
        self._sync()
        self.stage_seconds["rollout"] += time.perf_counter() - started
        return result

    def _forward_teacher_logits(self, *args, **kwargs):
        self._sync()
        started = time.perf_counter()
        result = super()._forward_teacher_logits(*args, **kwargs)
        self._sync()
        self.stage_seconds["teacher"] += time.perf_counter() - started
        return result

    def compute_loss(self, *args, **kwargs):
        self._sync()
        started = time.perf_counter()
        result = super().compute_loss(*args, **kwargs)
        self._sync()
        self.stage_seconds["total_compute"] += time.perf_counter() - started
        return result


class OfflineFullPoeTrainer(_MeasuredMixin, BaseProLLaMAOpdTrainer):
    """Recompute only current-student predictions against frozen full-PoE targets."""

    def __init__(self, *args, **kwargs) -> None:
        self._init_stage_timings()
        super().__init__(*args, **kwargs)

    def _prepare_teacher_model(self) -> None:
        # The offline phase deliberately has no teacher model on GPU.
        return None

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor | int],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        del num_items_in_batch
        self._sync()
        started = time.perf_counter()
        prompt_length = int(inputs["prompt_length"])
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = output.logits[:, prompt_length - 1 : -1, :]
        student_log_probs = F.log_softmax(logits / float(self.distill_temperature), dim=-1)
        target_log_probs = inputs["target_log_probs"].to(
            device=student_log_probs.device,
            dtype=student_log_probs.dtype,
        )
        if student_log_probs.shape != target_log_probs.shape:
            raise ValueError(
                f"student/cache shape mismatch: {student_log_probs.shape} != {target_log_probs.shape}"
            )
        loss = self._generalized_jsd_loss_from_log_probs(
            student_log_probs,
            target_log_probs,
            labels=inputs["shifted_labels"],
            beta=self.distill_beta,
        )
        self._buffer_train_metrics({
            "train/jsd_loss": loss.detach(),
            "train/regularized_loss": loss.detach(),
            "train/student_temperature": self.distill_temperature,
        })
        self._sync()
        self.stage_seconds["total_compute"] += time.perf_counter() - started
        if return_outputs:
            return loss, {"loss": loss.detach()}
        return loss


def write_rank_timing(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)
