#!/usr/bin/env python3
"""Run the paired online or fixed-cache Conditional multi-teacher continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import set_seed

from prodipa.conditional.offline_opd import (
    CacheSegmentBoundary,
    FullPoeCacheCollator,
    FullPoeCacheDataset,
    MeasuredOnlineTrainer,
    OfflineFullPoeTrainer,
    write_rank_timing,
)
from prollama_opd_config import load_opd_config
from prollama_opd_data import ProLLaMAOpdDataCollator
from prollama_opd_train import (
    FixedPromptDataset,
    OpdTrainingArguments,
    _enable_gradient_checkpointing_for_peft,
    _load_model,
    _load_teacher_model_and_specs,
    _load_tokenizer,
)
from prollama_opd_trainer import TeacherAdapterSpec


def file_sha256(path: Path) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("online", "offline"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--theta-zero", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--online-sample-count", type=int, default=2048)
    parser.add_argument("--offline-sample-count", type=int, default=512)
    parser.add_argument("--online-epochs", type=float, default=1.0)
    parser.add_argument("--offline-epochs", type=float, default=4.0)
    parser.add_argument("--resume-optimizer-state", action="store_true")
    parser.add_argument("--initial-global-step", type=int, default=0)
    parser.add_argument("--target-global-step", type=int, default=64)
    parser.add_argument("--stop-global-step", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    process_started_unix = time.time()
    process_started = time.perf_counter()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise ValueError("the frozen core protocol requires exactly two training GPUs")
    if args.mode == "offline" and args.cache_dir is None:
        raise ValueError("offline mode requires --cache-dir")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    config = load_opd_config(str(args.config))
    sample_count = (
        args.online_sample_count if args.mode == "online" else args.offline_sample_count
    )
    epochs = args.online_epochs if args.mode == "online" else args.offline_epochs
    if sample_count <= 0 or epochs <= 0:
        raise ValueError("sample counts and epochs must be positive")
    if args.initial_global_step < 0 or args.target_global_step <= args.initial_global_step:
        raise ValueError("invalid continuation step range")
    training_args = OpdTrainingArguments(
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
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_strategy="steps",
        logging_steps=5,
        report_to=[],
        save_strategy="steps",
        save_steps=64,
        save_total_limit=1,
        save_safetensors=True,
        seed=args.seed,
        data_seed=args.seed,
        bf16=True,
        load_in_kbits=16,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
    )
    tokenizer = _load_tokenizer(config.student.backbone_path)
    student_base = _load_model(config.student.backbone_path, training_args)
    student = PeftModel.from_pretrained(
        student_base, str(args.theta_zero), is_trainable=True
    )
    student.config.use_cache = False
    _enable_gradient_checkpointing_for_peft(student)
    if hasattr(student, "print_trainable_parameters") and rank == 0:
        student.print_trainable_parameters()
    if student.get_input_embeddings().num_embeddings != len(tokenizer):
        student.resize_token_embeddings(len(tokenizer))

    common = dict(
        model=student,
        args=training_args,
        tokenizer=tokenizer,
        sequence_regex=config.prompt.sequence_regex,
        temperature=config.generation.student_temperature,
        use_curriculum_tem=config.generation.use_curriculum_tem,
        curriculum_tem_start=config.generation.curriculum_tem_start,
        max_new_tokens=config.generation.max_new_tokens,
        repetition_penalty=config.generation.repetition_penalty,
        top_p=config.generation.top_p,
        top_k=config.generation.top_k,
        save_generations=False,
        generation_save_steps=config.generation.generation_save_steps,
        sample=False,
        sample_steps=config.generation.sample_steps,
        sample_num_sequences=config.generation.sample_num_sequences,
        sample_temperature=config.generation.sample_temperature,
        sample_batch_size=config.generation.sample_batch_size,
        beta=config.distill.beta,
        top_k_loss=config.distill.top_k_loss,
        useloss=config.distill.loss_type,
        useZ_t=config.distill.use_z_t,
        use_entropy_rule=config.distill.use_entropy_rule,
        entropy_alpha=config.distill.entropy_alpha,
        use_wi_entropy=config.distill.use_wi_entropy,
        use_wi_distance=config.distill.use_wi_distance,
        wi_entropy_tau=config.distill.wi_entropy_tau,
        wi_distance_beta=config.distill.wi_distance_beta,
        wi_distance_tau=config.distill.wi_distance_tau,
        debug_teacher_diff=False,
        debug_teacher_diff_steps=0,
        log_teacher_entropy=False,
    )
    if args.mode == "online":
        dataset = FixedPromptDataset(sample_count)
        collator = ProLLaMAOpdDataCollator(tokenizer, config.prompt)
        teacher, specs = _load_teacher_model_and_specs(config, training_args)
        if teacher.get_input_embeddings().num_embeddings != len(tokenizer):
            teacher.resize_token_embeddings(len(tokenizer))
        trainer = MeasuredOnlineTrainer(
            train_dataset=dataset,
            data_collator=collator,
            teacher_model=teacher,
            teacher_specs=specs,
            sample_prompt_ids=collator.prompt_ids,
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
            for item in config.teachers.adapters
        ]
        trainer = OfflineFullPoeTrainer(
            train_dataset=dataset,
            data_collator=collator,
            teacher_model=torch.nn.Identity(),
            teacher_specs=specs,
            sample_prompt_ids=[],
            **common,
        )

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
            "theta_zero_weights_sha256": file_sha256(adapter_weight(args.theta_zero)),
            "opd_config": str(args.config.resolve()),
            "opd_config_sha256": file_sha256(args.config),
            "initial_global_step": args.initial_global_step,
            "target_global_step": args.target_global_step,
            "configured_dataset_samples": sample_count,
            "unique_continuation_trajectories": effective_trajectories,
            "reuse_epochs": float(epochs),
            "resume_optimizer_and_scheduler": bool(args.resume_optimizer_state),
            "world_size": world_size,
            "per_device_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "optimizer": "AdamW fused",
            "learning_rate": 2e-5,
            "scheduler": "cosine",
            "warmup_ratio": 0.1,
            "training_arguments": training_args.to_dict(),
            "opd": asdict(config),
        }
        (args.output_dir / "TRAINING_PROTOCOL.json").write_text(
            json.dumps(protocol, indent=2, default=str) + "\n"
        )

    initial_global_step = args.initial_global_step
    train_result = trainer.train(
        resume_from_checkpoint=str(args.resume_checkpoint or args.theta_zero) if args.resume_optimizer_state else None
    )
    trainer.save_model(str(args.output_dir))
    if int(trainer.state.global_step) != (args.stop_global_step or args.target_global_step):
        raise RuntimeError(
            f"training ended at step {trainer.state.global_step}, expected {args.target_global_step}"
        )
    tokenizer.save_pretrained(str(args.output_dir))
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
        "initial_global_step": initial_global_step,
        "final_global_step": int(trainer.state.global_step),
        "student_updates": int(trainer.state.global_step) - initial_global_step,
        **trainer.timing_payload(),
    }
    write_rank_timing(args.output_dir / "timings" / f"rank_{rank}.json", timing)
    if rank == 0:
        protocol = json.loads((args.output_dir / "TRAINING_PROTOCOL.json").read_text())
        protocol["status"] = "COMPLETE" if timing["final_global_step"] == args.target_global_step else "SEGMENT_COMPLETE"
        protocol["completed_at_unix"] = timing["completed_at_unix"]
        protocol["final_global_step"] = timing["final_global_step"]
        protocol["student_updates"] = timing["student_updates"]
        (args.output_dir / "TRAINING_PROTOCOL.json").write_text(
            json.dumps(protocol, indent=2, default=str) + "\n"
        )


if __name__ == "__main__":
    main()
