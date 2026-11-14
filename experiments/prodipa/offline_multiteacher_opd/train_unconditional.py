#!/usr/bin/env python3
"""Run paired unconditional online or fixed-cache OPD continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import TrainerCallback, set_seed

from prodipa.conditional.offline_opd import CacheSegmentBoundary, FullPoeCacheCollator, FullPoeCacheDataset
from prodipa.unconditional.data import UnconditionalProDiPADataCollator
from prodipa.unconditional.offline_opd import (
    MeasuredOnlineUnconditionalTrainer,
    OfflineUnconditionalFullPoeTrainer,
    write_rank_timing,
)
from prodipa.unconditional.train import (
    UnconditionalPromptDataset,
    UnconditionalScriptArguments,
    UnconditionalTrainingArguments,
    _load_student_model,
    _load_teacher_config,
    _load_teacher_model_and_specs,
    _load_tokenizer,
)
from prodipa.unconditional.trainer import TeacherAdapterSpec


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_weight(path: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def latest_recovery_checkpoint(
    output_dir: Path, initial_step: int, target_step: int
) -> Path | None:
    """Return the newest valid continuation checkpoint, if one exists."""
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if initial_step < step < target_step and (path / "trainer_state.json").is_file():
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


class RecoveryCheckpointCallback(TrainerCallback):
    """Force recovery saves even when a resumed state has an older interval."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % 10 == 0:
            control.should_save = True
        return control


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("online", "offline"), required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--theta-zero", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--online-sample-count", type=int, default=8192)
    parser.add_argument("--offline-sample-count", type=int, default=512)
    parser.add_argument("--online-epochs", type=float, default=1.0)
    parser.add_argument("--offline-epochs", type=float, default=16.0)
    parser.add_argument("--initial-global-step", type=int, default=200)
    parser.add_argument("--target-global-step", type=int, default=256)
    parser.add_argument("--stop-global-step", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-format", choices=("legacy", "unwrapped"), default="legacy")
    parser.add_argument("--max-length-aa", type=int, default=512)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise ValueError("the paired protocol requires exactly two training GPUs")
    if args.mode == "offline" and args.cache_dir is None:
        raise ValueError("offline mode requires --cache-dir")
    if args.target_global_step <= args.initial_global_step:
        raise ValueError("invalid continuation step range")

    process_started_unix = time.time()
    process_started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recovery_checkpoint = latest_recovery_checkpoint(
        args.output_dir, args.initial_global_step, args.target_global_step
    )
    resume_checkpoint = args.resume_checkpoint or recovery_checkpoint or args.theta_zero
    set_seed(args.seed)

    sample_count = (
        args.online_sample_count if args.mode == "online" else args.offline_sample_count
    )
    epochs = args.online_epochs if args.mode == "online" else args.offline_epochs
    script_args = UnconditionalScriptArguments(
        num_train_samples=sample_count,
        student_model_name_or_path=str(args.base_model),
        teacher_config_path=str(args.teacher_config),
        prompt_mode="unconditional",
        prompt_prefix_length=32,
        max_new_tokens=512,
        max_length_aa=args.max_length_aa,
        sequence_format=args.sequence_format,
        temperature=1.0,
        repetition_penalty=1.2,
        top_p=0.95,
        top_k=500,
        beta=0.5,
        top_k_loss=0,
        useloss="jsd",
        use_entropy_rule=True,
        entropy_alpha=0.0,
        save_generations=False,
        student_tune_mode="lora",
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        lora_target_modules=["c_attn", "c_proj", "c_fc"],
    )
    training_args = UnconditionalTrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=False,
        eval_strategy="no",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.0,
        num_train_epochs=epochs,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        logging_strategy="steps",
        logging_steps=1,
        report_to=[],
        save_strategy="steps",
        # Keep recovery points inside the short 56-update continuation.  A
        # 100-step interval produces no checkpoint between theta-zero and the
        # formal endpoint, so an external interruption loses the whole run.
        save_steps=10,
        save_total_limit=2,
        save_safetensors=True,
        seed=args.seed,
        bf16=True,
        remove_unused_columns=False,
        optim="adamw_torch_fused",
    )

    tokenizer = _load_tokenizer(str(args.base_model))
    student, prefix_enabled = _load_student_model(script_args, training_args)
    if prefix_enabled:
        raise ValueError("the formal unconditional continuation must use the LoRA student")
    if student.get_input_embeddings().num_embeddings != len(tokenizer):
        student.resize_token_embeddings(len(tokenizer))

    teacher_config = _load_teacher_config(str(args.teacher_config), 1.0)
    common = dict(
        model=student,
        args=training_args,
        tokenizer=tokenizer,
        beta=script_args.beta,
        temperature=script_args.temperature,
        use_curriculum_tem=False,
        curriculum_tem_start=0.7,
        top_k_loss=0,
        max_new_tokens=512,
        max_length_aa=script_args.max_length_aa,
        sequence_format=script_args.sequence_format,
        repetition_penalty=1.2,
        top_p=0.95,
        top_k=500,
        useloss="jsd",
        useZ_t=False,
        use_entropy_rule=True,
        entropy_alpha=0.0,
        use_wi_entropy=False,
        use_wi_distance=False,
        wi_entropy_tau=1.0,
        wi_distance_beta=0.9,
        wi_distance_tau=1.0,
        debug_teacher_diff=False,
        debug_teacher_diff_steps=0,
        save_generations=False,
        generation_save_steps=5,
        student_prefix_init_enabled=False,
        log_teacher_entropy=teacher_config.log_teacher_entropy,
    )
    if args.mode == "online":
        dataset = UnconditionalPromptDataset(sample_count)
        collator = UnconditionalProDiPADataCollator(tokenizer=tokenizer)
        teacher, specs = _load_teacher_model_and_specs(teacher_config, training_args)
        if teacher.get_input_embeddings().num_embeddings != len(tokenizer):
            teacher.resize_token_embeddings(len(tokenizer))
        trainer = MeasuredOnlineUnconditionalTrainer(
            train_dataset=dataset,
            data_collator=collator,
            teacher_model=teacher,
            teacher_specs=specs,
            **common,
        )
    else:
        dataset = FullPoeCacheDataset(args.cache_dir, sample_count)
        collator = FullPoeCacheCollator(args.cache_dir, tokenizer)
        specs = [
            TeacherAdapterSpec(
                name=item.name,
                adapter_name=item.name,
                weight=item.weight,
                temperature=item.temperature,
            )
            for item in teacher_config.teachers
        ]
        trainer = OfflineUnconditionalFullPoeTrainer(
            train_dataset=dataset,
            data_collator=collator,
            teacher_model=torch.nn.Identity(),
            teacher_specs=specs,
            **common,
        )
    trainer.add_callback(RecoveryCheckpointCallback())
    if args.stop_global_step is not None:
        if not args.initial_global_step < args.stop_global_step <= args.target_global_step:
            raise ValueError("invalid cache segment boundary")
        trainer.args.save_total_limit = None
        trainer.add_callback(CacheSegmentBoundary(args.stop_global_step))

    if rank == 0:
        continuation_updates = args.target_global_step - args.initial_global_step
        effective_trajectories = (
            continuation_updates * 8 * 2 * world_size
            if args.mode == "online"
            else sample_count
        )
        protocol = {
            "status": "RUNNING",
            "mode": args.mode,
            "theta_zero": str(args.theta_zero.resolve()),
            "resume_checkpoint": str(resume_checkpoint.resolve()),
            "theta_zero_weights_sha256": sha256(adapter_weight(args.theta_zero)),
            "base_model": str(args.base_model.resolve()),
            "teacher_config": str(args.teacher_config.resolve()),
            "teacher_config_sha256": sha256(args.teacher_config),
            "initial_global_step": args.initial_global_step,
            "target_global_step": args.target_global_step,
            "configured_dataset_samples": sample_count,
            "unique_continuation_trajectories": effective_trajectories,
            "reuse_epochs": float(epochs),
            "world_size": world_size,
            "per_device_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "optimizer": "AdamW fused",
            "learning_rate": 2e-5,
            "scheduler": "linear",
            "warmup_ratio": 0.1,
            "training_arguments": training_args.to_dict(),
            "script_arguments": asdict(script_args),
        }
        (args.output_dir / "TRAINING_PROTOCOL.json").write_text(
            json.dumps(protocol, indent=2, default=str) + "\n"
        )

    train_result = trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    if int(trainer.state.global_step) != (args.stop_global_step or args.target_global_step):
        raise RuntimeError(
            f"training ended at step {trainer.state.global_step}, expected "
            f"{args.target_global_step}"
        )
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    trainer.save_state()
    timing = {
        "status": "COMPLETE",
        "mode": args.mode,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "started_at_unix": process_started_unix,
        "completed_at_unix": time.time(),
        "allocated_gpu_seconds": time.perf_counter() - process_started,
        "trainer_runtime_seconds": float(train_result.metrics.get("train_runtime", 0.0)),
        "initial_global_step": args.initial_global_step,
        "final_global_step": int(trainer.state.global_step),
        "student_updates": int(trainer.state.global_step) - args.initial_global_step,
        **trainer.timing_payload(),
    }
    write_rank_timing(args.output_dir / "timings" / f"rank_{rank}.json", timing)
    if rank == 0:
        protocol = json.loads((args.output_dir / "TRAINING_PROTOCOL.json").read_text())
        protocol.update(
            {
                "status": "COMPLETE" if timing["final_global_step"] == args.target_global_step else "SEGMENT_COMPLETE",
                "completed_at_unix": timing["completed_at_unix"],
                "final_global_step": timing["final_global_step"],
                "student_updates": timing["student_updates"],
            }
        )
        (args.output_dir / "TRAINING_PROTOCOL.json").write_text(
            json.dumps(protocol, indent=2, default=str) + "\n"
        )


if __name__ == "__main__":
    main()
