"""Fixed-rollout cache trainers for unconditional multi-teacher OPD."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from prodipa.conditional.offline_opd import (
    FullPoeCacheCollator,
    FullPoeCacheDataset,
    cache_record_path,
    load_cache_record,
    write_rank_timing,
)

from .trainer import UnconditionalProDiPATrainer


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


class MeasuredOnlineUnconditionalTrainer(_MeasuredMixin, UnconditionalProDiPATrainer):
    def __init__(self, *args, **kwargs) -> None:
        self._init_stage_timings()
        super().__init__(*args, **kwargs)

    def _generate_on_policy(self, *args, **kwargs):
        self._sync()
        started = time.perf_counter()
        value = super()._generate_on_policy(*args, **kwargs)
        self._sync()
        self.stage_seconds["rollout"] += time.perf_counter() - started
        return value

    def _forward_teacher_logits(self, *args, **kwargs):
        self._sync()
        started = time.perf_counter()
        value = super()._forward_teacher_logits(*args, **kwargs)
        self._sync()
        self.stage_seconds["teacher"] += time.perf_counter() - started
        return value

    def compute_loss(self, *args, **kwargs):
        self._sync()
        started = time.perf_counter()
        value = super().compute_loss(*args, **kwargs)
        self._sync()
        self.stage_seconds["total_compute"] += time.perf_counter() - started
        return value


class OfflineUnconditionalFullPoeTrainer(_MeasuredMixin, UnconditionalProDiPATrainer):
    def __init__(self, *args, **kwargs) -> None:
        self._init_stage_timings()
        super().__init__(*args, **kwargs)

    def _prepare_teacher_model(self) -> None:
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
        student_log_probs = F.log_softmax(
            logits / float(self.distill_temperature), dim=-1
        )
        target_log_probs = inputs["target_log_probs"].to(
            device=student_log_probs.device,
            dtype=student_log_probs.dtype,
        )
        if student_log_probs.shape != target_log_probs.shape:
            raise ValueError(
                f"student/cache shape mismatch: {student_log_probs.shape} != "
                f"{target_log_probs.shape}"
            )
        loss = self._generalized_jsd_loss_from_log_probs(
            student_log_probs,
            target_log_probs,
            labels=inputs["shifted_labels"],
            beta=self.distill_beta,
        )
        self._buffer_train_metrics(
            {
                "train/jsd_loss": loss.detach(),
                "train/regularized_loss": loss.detach(),
                "train/student_temperature": self.distill_temperature,
            }
        )
        self._sync()
        self.stage_seconds["total_compute"] += time.perf_counter() - started
        if return_outputs:
            return loss, {"loss": loss.detach()}
        return loss


__all__ = [
    "FullPoeCacheCollator",
    "FullPoeCacheDataset",
    "MeasuredOnlineUnconditionalTrainer",
    "OfflineUnconditionalFullPoeTrainer",
    "cache_record_path",
    "load_cache_record",
    "write_rank_timing",
]
