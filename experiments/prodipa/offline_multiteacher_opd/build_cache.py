#!/usr/bin/env python3
"""Generate fixed theta-zero rollouts and cache exact full-vocabulary PoE targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel

from prodipa.conditional.offline_opd import cache_record_path, load_cache_record
from prollama_opd_config import load_opd_config
from prollama_opd_data import ProLLaMAOpdDataCollator
from prollama_opd_train import (
    OpdTrainingArguments,
    _load_model,
    _load_teacher_model_and_specs,
    _load_tokenizer,
)
from prollama_opd_trainer import BaseProLLaMAOpdTrainer


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        value = row[:prompt_length].tolist() + completion + [eos_token_id]
        tensor = torch.tensor(value, dtype=torch.long, device=generated_ids.device)
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
    return BaseProLLaMAOpdTrainer._generalized_jsd_loss_from_log_probs(
        student_log_probs, target_log_probs, labels=None, beta=beta
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--theta-zero", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--sample-start", type=int, required=True)
    parser.add_argument("--sample-stop", type=int, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not (0 <= args.sample_start < args.sample_stop <= args.sample_count):
        raise ValueError("invalid cache shard range")

    job_started = time.time()
    monotonic_started = time.perf_counter()
    config = load_opd_config(str(args.config))
    if config.distill.loss_type != "jsd" or config.distill.top_k_loss != 0:
        raise ValueError("the core offline study requires the existing full-vocabulary JSD")
    if config.distill.use_wi_entropy or config.distill.use_wi_distance:
        raise ValueError("dynamic teacher weights cannot be frozen by this minimal cache")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.cache_dir / "records").mkdir(parents=True, exist_ok=True)
    timing = {"rollout_seconds": 0.0, "teacher_seconds": 0.0, "cache_io_seconds": 0.0}

    training_args = OpdTrainingArguments(
        output_dir=str(args.cache_dir / "loader_state"),
        do_train=True,
        bf16=True,
        load_in_kbits=16,
        report_to=[],
        gradient_checkpointing=False,
    )
    tokenizer = _load_tokenizer(config.student.backbone_path)
    prompt = ProLLaMAOpdDataCollator(tokenizer, config.prompt)
    prompt_ids_single = torch.tensor(prompt.prompt_ids, dtype=torch.long, device=args.device)
    prompt_length = int(prompt_ids_single.numel())
    eos = int(tokenizer.eos_token_id)
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos)

    student_base = _load_model(config.student.backbone_path, training_args)
    student = PeftModel.from_pretrained(
        student_base, str(args.theta_zero), is_trainable=False
    ).eval().to(args.device)
    teacher, teacher_specs = _load_teacher_model_and_specs(config, training_args)
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
        prompts = prompt_ids_single.unsqueeze(0).repeat(batch_size, 1)
        prompt_attention = torch.ones_like(prompts)
        generation_kwargs = {
            "max_new_tokens": config.generation.max_new_tokens,
            "do_sample": True,
            "temperature": config.generation.student_temperature,
            "repetition_penalty": config.generation.repetition_penalty,
            "top_p": config.generation.top_p,
            "top_k": config.generation.top_k,
            "pad_token_id": pad,
            "eos_token_id": eos,
            "use_cache": True,
        }
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            raw = student.generate(
                input_ids=prompts,
                attention_mask=prompt_attention,
                **generation_kwargs,
            )
        torch.cuda.synchronize()
        timing["rollout_seconds"] += time.perf_counter() - started
        input_ids, attention_mask, total_lengths = canonicalize(
            raw, prompt_length, eos, pad
        )

        torch.cuda.synchronize()
        started = time.perf_counter()
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
        timing["teacher_seconds"] += time.perf_counter() - started

        for row, (sample_id, total_length) in enumerate(zip(sample_ids, total_lengths)):
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
            }
            target = cache_record_path(args.cache_dir, sample_id)
            temporary = target.with_suffix(f".pt.{os.getpid()}.tmp")
            io_started = time.perf_counter()
            torch.save(record, temporary)
            os.replace(temporary, target)
            timing["cache_io_seconds"] += time.perf_counter() - io_started
            produced += 1

        if first_audit is None:
            # Same student logits and same rollout: compare live online target/loss
            # with the serialized offline target/loss.
            with torch.inference_mode():
                student_logits = student(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits[:, prompt_length - 1 : -1, :]
                student_log_probs = F.log_softmax(
                    student_logits / float(config.generation.student_temperature), dim=-1
                )
            cached_targets = []
            for sample_id, total_length in zip(sample_ids, total_lengths):
                completion_length = total_length - prompt_length
                cached_targets.append(
                    load_cache_record(cache_record_path(args.cache_dir, sample_id))[
                        "target_log_probs"
                    ][:completion_length]
                )
            live_losses = []
            cached_losses = []
            max_target_diff = 0.0
            mean_target_diff_values = []
            for row, cached in enumerate(cached_targets):
                length = cached.shape[0]
                live = live_target[row, :length]
                cached = cached.to(device=live.device, dtype=live.dtype)
                difference = (live - cached).abs().float()
                max_target_diff = max(max_target_diff, float(difference.max().item()))
                mean_target_diff_values.append(float(difference.mean().item()))
                live_losses.append(
                    generalized_jsd(student_log_probs[row, :length], live, config.distill.beta)
                )
                cached_losses.append(
                    generalized_jsd(student_log_probs[row, :length], cached, config.distill.beta)
                )
            live_loss = torch.stack(live_losses).mean()
            cached_loss = torch.stack(cached_losses).mean()
            first_audit = {
                "status": "PASS" if max_target_diff == 0.0 and torch.equal(live_loss, cached_loss) else "FAIL",
                "sample_ids": sample_ids,
                "max_abs_target_logprob_diff": max_target_diff,
                "mean_abs_target_logprob_diff": float(np.mean(mean_target_diff_values)),
                "online_live_loss": float(live_loss.item()),
                "offline_cached_loss": float(cached_loss.item()),
                "absolute_loss_diff": abs(float(live_loss.item()) - float(cached_loss.item())),
            }
        print(f"rank={args.rank} cached={produced}/{args.sample_stop-args.sample_start}", flush=True)

    completed = time.time()
    report = {
        "status": "COMPLETE",
        "rank": args.rank,
        "sample_start": args.sample_start,
        "sample_stop": args.sample_stop,
        "sample_count": produced,
        "started_at_unix": job_started,
        "completed_at_unix": completed,
        "allocated_gpu_seconds": time.perf_counter() - monotonic_started,
        **timing,
        "consistency_audit": first_audit,
        "theta_zero_adapter": str(args.theta_zero.resolve()),
        "theta_zero_adapter_sha256": sha256(args.theta_zero / "adapter_model.bin"),
        "config_sha256": sha256(args.config),
    }
    (args.cache_dir / f"shard_{args.rank}.json").write_text(json.dumps(report, indent=2) + "\n")
    if first_audit is None or first_audit["status"] != "PASS":
        raise RuntimeError("cache/live consistency audit failed")


if __name__ == "__main__":
    main()
