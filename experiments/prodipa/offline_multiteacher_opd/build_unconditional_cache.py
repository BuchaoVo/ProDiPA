#!/usr/bin/env python3
"""Build a fixed unconditional rollout cache with exact full-vocabulary PoE targets."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM
from prodipa.unconditional.length_control import ProteinLengthLogitsProcessor

from prodipa.unconditional.offline_opd import cache_record_path, load_cache_record
from prodipa.unconditional.train import (
    UnconditionalTrainingArguments,
    _load_teacher_config,
    _load_teacher_model_and_specs,
    _load_tokenizer,
)
from prodipa.unconditional.trainer import UnconditionalProDiPATrainer


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def canonicalize(
    generated_ids: torch.Tensor,
    prompt_length: int,
    eos_token_id: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    rows: list[torch.Tensor] = []
    lengths: list[int] = []
    for row in generated_ids:
        completion = row[prompt_length:].tolist()
        if pad_token_id != eos_token_id:
            while completion and completion[-1] == pad_token_id:
                completion.pop()
        while completion and completion[-1] == eos_token_id:
            completion.pop()
        tensor = torch.tensor(
            [eos_token_id] + completion + [eos_token_id],
            dtype=torch.long,
            device=generated_ids.device,
        )
        rows.append(tensor)
        lengths.append(tensor.numel())
    maximum = max(lengths)
    batch = torch.full(
        (len(rows), maximum), pad_token_id, dtype=torch.long, device=generated_ids.device
    )
    attention = torch.zeros_like(batch)
    for index, row in enumerate(rows):
        batch[index, : row.numel()] = row
        attention[index, : row.numel()] = 1
    return batch, attention, lengths


def generalized_jsd(
    student_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    return UnconditionalProDiPATrainer._generalized_jsd_loss_from_log_probs(
        student_log_probs, target_log_probs, labels=None, beta=beta
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--teacher-config", type=Path, required=True)
    parser.add_argument("--theta-zero", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--sample-start", type=int, required=True)
    parser.add_argument("--sample-stop", type=int, required=True)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length-aa", type=int, default=512)
    args = parser.parse_args()
    if not (0 <= args.sample_start < args.sample_stop <= args.sample_count):
        raise ValueError("invalid cache shard range")

    started_unix = time.time()
    started = time.perf_counter()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.cache_dir / "records").mkdir(parents=True, exist_ok=True)
    timing = {"rollout_seconds": 0.0, "teacher_seconds": 0.0, "cache_io_seconds": 0.0}

    training_args = UnconditionalTrainingArguments(
        output_dir=str(args.cache_dir / "loader_state"),
        do_train=True,
        bf16=True,
        report_to=[],
    )
    tokenizer = _load_tokenizer(str(args.base_model))
    eos = int(tokenizer.eos_token_id)
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos)
    prompt = torch.tensor([[eos]], dtype=torch.long, device=args.device)
    prompt_length = 1

    student_base = AutoModelForCausalLM.from_pretrained(
        str(args.base_model), torch_dtype=torch.bfloat16
    )
    student = PeftModel.from_pretrained(
        student_base, str(args.theta_zero), is_trainable=False
    ).eval().to(args.device)
    teacher_config = _load_teacher_config(str(args.teacher_config), 1.0)
    teacher, teacher_specs = _load_teacher_model_and_specs(teacher_config, training_args)
    teacher.eval().to(args.device)
    if student.get_input_embeddings().num_embeddings != len(tokenizer):
        student.resize_token_embeddings(len(tokenizer))
    if teacher.get_input_embeddings().num_embeddings != len(tokenizer):
        teacher.resize_token_embeddings(len(tokenizer))

    first_audit: dict[str, object] | None = None
    produced = 0
    for batch_start in range(args.sample_start, args.sample_stop, args.batch_size):
        sample_ids = list(range(batch_start, min(batch_start + args.batch_size, args.sample_stop)))
        batch_size = len(sample_ids)
        seed_all(args.seed + batch_start)
        prompts = prompt.repeat(batch_size, 1)
        prompt_attention = torch.ones_like(prompts)
        torch.cuda.synchronize()
        stage_started = time.perf_counter()
        with torch.inference_mode():
            length_processor = ProteinLengthLogitsProcessor(
                tokenizer, args.max_length_aa, prompt_length
            )
            raw = student.generate(
                input_ids=prompts,
                attention_mask=prompt_attention,
                max_new_tokens=512,
                do_sample=True,
                temperature=1.0,
                repetition_penalty=1.2,
                top_p=0.95,
                top_k=500,
                pad_token_id=pad,
                eos_token_id=eos,
                use_cache=True,
                logits_processor=[length_processor],
            )
        torch.cuda.synchronize()
        timing["rollout_seconds"] += time.perf_counter() - stage_started
        input_ids, attention_mask, total_lengths = canonicalize(
            raw, prompt_length, eos, pad
        )
        forced_flags = length_processor.forced_flags()
        natural_flags = []
        for generated_row, forced in zip(raw, forced_flags):
            completion = generated_row[prompt_length:].tolist()
            natural_flags.append((eos in completion) and not forced)

        torch.cuda.synchronize()
        stage_started = time.perf_counter()
        teacher_log_probs: list[torch.Tensor] = []
        with torch.inference_mode():
            for spec in teacher_specs:
                teacher.set_adapter(spec.adapter_name)
                logits = teacher(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits[:, prompt_length - 1 : -1, :]
                teacher_log_probs.append(
                    F.log_softmax(logits / float(spec.temperature), dim=-1)
                )
            weights = torch.tensor(
                [spec.weight for spec in teacher_specs],
                dtype=teacher_log_probs[0].dtype,
                device=teacher_log_probs[0].device,
            )
            aggregated = (
                weights.view(1, 1, -1, 1)
                * torch.stack(teacher_log_probs, dim=2)
            ).sum(dim=2)
            live_target = F.log_softmax(aggregated, dim=-1)
        torch.cuda.synchronize()
        timing["teacher_seconds"] += time.perf_counter() - stage_started

        for row, (sample_id, total_length, natural_terminal) in enumerate(
            zip(sample_ids, total_lengths, natural_flags)
        ):
            completion_length = total_length - prompt_length
            record = {
                "schema_version": 1,
                "sample_id": sample_id,
                "input_ids": input_ids[row, :total_length].detach().cpu(),
                "prompt_length": prompt_length,
                "target_log_probs": live_target[row, :completion_length]
                .detach()
                .to(torch.bfloat16)
                .cpu(),
                "supervise_terminal": bool(natural_terminal),
            }
            target = cache_record_path(args.cache_dir, sample_id)
            temporary = target.with_suffix(f".pt.{os.getpid()}.tmp")
            io_started = time.perf_counter()
            torch.save(record, temporary)
            os.replace(temporary, target)
            timing["cache_io_seconds"] += time.perf_counter() - io_started
            produced += 1

        if first_audit is None:
            with torch.inference_mode():
                logits = student(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits[:, prompt_length - 1 : -1, :]
                student_log_probs = F.log_softmax(logits, dim=-1)
            live_losses: list[torch.Tensor] = []
            cached_losses: list[torch.Tensor] = []
            max_target_diff = 0.0
            mean_target_diffs: list[float] = []
            for row, (sample_id, total_length) in enumerate(zip(sample_ids, total_lengths)):
                length = total_length - prompt_length
                live = live_target[row, :length]
                cached = load_cache_record(cache_record_path(args.cache_dir, sample_id))[
                    "target_log_probs"
                ].to(device=live.device, dtype=live.dtype)
                difference = (live - cached).abs().float()
                max_target_diff = max(max_target_diff, float(difference.max().item()))
                mean_target_diffs.append(float(difference.mean().item()))
                live_losses.append(generalized_jsd(student_log_probs[row, :length], live, 0.5))
                cached_losses.append(generalized_jsd(student_log_probs[row, :length], cached, 0.5))
            live_loss = torch.stack(live_losses).mean()
            cached_loss = torch.stack(cached_losses).mean()
            loss_diff = abs(float(live_loss.item()) - float(cached_loss.item()))
            first_audit = {
                "status": "PASS" if max_target_diff == 0.0 and loss_diff == 0.0 else "FAIL",
                "sample_ids": sample_ids,
                "max_abs_target_logprob_diff": max_target_diff,
                "mean_abs_target_logprob_diff": float(np.mean(mean_target_diffs)),
                "online_live_loss": float(live_loss.item()),
                "offline_cached_loss": float(cached_loss.item()),
                "absolute_loss_diff": loss_diff,
            }

    report = {
        "status": "COMPLETE",
        "rank": args.rank,
        "sample_start": args.sample_start,
        "sample_stop": args.sample_stop,
        "produced": produced,
        "started_at_unix": started_unix,
        "completed_at_unix": time.time(),
        "allocated_gpu_seconds": time.perf_counter() - started,
        **timing,
        "consistency_audit": first_audit,
    }
    (args.cache_dir / f"shard_{args.rank}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
