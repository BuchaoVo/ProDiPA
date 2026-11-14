from __future__ import annotations

import logging
import os
import sys
import time
import csv
import gc
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file as load_safetensors
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from .config import ConditionalProDiPAConfig, load_config
from .data import (
    ConditionalProDiPADataCollator,
    FixedRolloutCollator,
    FixedRolloutDataset,
    StaticCacheCollator,
    StaticCacheDataset,
)
from .trainer import ConditionalProDiPATrainer, StaticCachedConditionalTrainer, TeacherAdapterSpec


logger = logging.getLogger(__name__)


def _load_causal_lm_with_attention_backend(
    model_path: str,
    use_flash_attention_2: bool = False,
    **model_kwargs,
) -> AutoModelForCausalLM:
    if not use_flash_attention_2:
        return AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            **model_kwargs,
        )
    except TypeError as error:
        if "attn_implementation" not in str(error):
            raise
        logger.warning("`attn_implementation` is unsupported; retrying with legacy `use_flash_attention_2=True`.")
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            use_flash_attention_2=True,
            **model_kwargs,
        )


def prepare_model_for_kbit_training(model, use_gradient_checkpointing: bool = True):
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.parameters():
        if ((param.dtype == torch.float16) or (param.dtype == torch.bfloat16)) and loaded_in_kbit:
            param.data = param.data.to(torch.float32)
    for name, module in model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)
    if loaded_in_kbit and use_gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(_module, _input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        model.gradient_checkpointing_enable()
    return model


@dataclass
class ConditionalScriptArguments:
    config_path: str = field(default="")
    fusion_rule: str | None = field(default=None)
    teacher_names: str | None = field(default=None)
    teacher_weights: str | None = field(default=None)
    teacher_adapter: str | None = field(default=None)
    teacher_temperature: float | None = field(default=None)
    loss_type: str | None = field(default=None)
    num_train_samples: int | None = field(default=None)
    max_new_tokens: int | None = field(default=None)
    fixed_rollouts: str | None = field(default=None)
    build_cache: str | None = field(default=None)
    cache_path: str | None = field(default=None)
    share_teacher_backbone: bool = field(default=False)
    check_shared_teacher_parity: str | None = field(default=None)


@dataclass
class ConditionalTrainingArguments(TrainingArguments):
    output_dir: str = field(default="")
    load_in_kbits: int = field(default=16)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    use_flash_attention_2: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="none")


class FixedPromptDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int):
        self.num_samples = int(num_samples)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, int]:
        return {"sample_id": int(idx)}


def _apply_script_overrides(
    config: ConditionalProDiPAConfig,
    script_args: ConditionalScriptArguments,
) -> ConditionalProDiPAConfig:
    adapters = list(config.teachers.adapters)
    if script_args.teacher_names:
        requested = [name.strip() for name in script_args.teacher_names.split(",") if name.strip()]
        adapters = [adapter for adapter in adapters if adapter.name in requested]
        missing = set(requested) - {adapter.name for adapter in adapters}
        if missing:
            raise ValueError(f"Unknown --teacher_names: {sorted(missing)}")
    if not adapters:
        raise ValueError("At least one teacher must remain after --teacher_names filtering.")
    if script_args.teacher_adapter is not None:
        if len(adapters) != 1:
            raise ValueError("--teacher_adapter requires exactly one selected teacher")
        adapter_path = str(Path(script_args.teacher_adapter).expanduser().resolve())
        if not (Path(adapter_path) / "adapter_config.json").is_file():
            raise ValueError(f"Invalid --teacher_adapter path: {adapter_path}")
        adapters = [replace(adapters[0], adapter_path=adapter_path)]

    weights = {adapter.name: adapter.weight for adapter in adapters}
    if script_args.teacher_weights:
        parsed: dict[str, float] = {}
        for item in script_args.teacher_weights.split(","):
            name, separator, raw_weight = item.partition("=")
            if not separator:
                raise ValueError("--teacher_weights must use name=value comma-separated entries.")
            parsed[name.strip()] = float(raw_weight)
        missing = {adapter.name for adapter in adapters} - parsed.keys()
        extra = parsed.keys() - {adapter.name for adapter in adapters}
        if missing or extra:
            raise ValueError(f"--teacher_weights mismatch: missing={sorted(missing)} extra={sorted(extra)}")
        weights = parsed
    total_weight = sum(weights.values())
    if total_weight <= 0 or any(weight < 0 for weight in weights.values()):
        raise ValueError("Selected teacher weights must be non-negative with a positive sum.")
    adapters = [
        replace(adapter, weight=weights[adapter.name] / total_weight)
        for adapter in adapters
    ]
    if script_args.teacher_temperature is not None:
        if script_args.teacher_temperature <= 0:
            raise ValueError("--teacher_temperature must be positive.")
        adapters = [
            replace(adapter, temperature=float(script_args.teacher_temperature))
            for adapter in adapters
        ]

    distill = config.distill
    if script_args.loss_type is not None:
        loss_type = script_args.loss_type.strip().lower()
        if loss_type not in {"hard_ce", "jsd", "pg", "sft"}:
            raise ValueError("--loss_type must be one of: hard_ce, jsd, pg, sft")
        distill = replace(distill, loss_type=loss_type)
    if script_args.fusion_rule is not None:
        fusion_rule = script_args.fusion_rule.strip().lower()
        if fusion_rule not in {"poe", "probability_average"}:
            raise ValueError("--fusion_rule must be one of: poe, probability_average")
        distill = replace(distill, fusion_rule=fusion_rule)

    generation = config.generation
    if script_args.num_train_samples is not None:
        if script_args.num_train_samples <= 0:
            raise ValueError("--num_train_samples must be positive.")
        generation = replace(generation, num_train_samples=script_args.num_train_samples)
    if script_args.max_new_tokens is not None:
        if script_args.max_new_tokens <= 0:
            raise ValueError("--max_new_tokens must be positive.")
        generation = replace(generation, max_new_tokens=script_args.max_new_tokens)

    return replace(
        config,
        teachers=replace(config.teachers, adapters=adapters),
        distill=distill,
        generation=generation,
    )


def _resolve_torch_dtype(training_args: ConditionalTrainingArguments) -> torch.dtype | None:
    if training_args.bf16:
        return torch.bfloat16
    if training_args.fp16:
        return torch.float16
    return None


def _build_quantization_config(training_args: ConditionalTrainingArguments) -> tuple[BitsAndBytesConfig | None, dict[str, Any]]:
    if training_args.load_in_kbits not in {4, 8, 16}:
        raise ValueError("`load_in_kbits` must be one of: 4, 8, 16")
    if training_args.load_in_kbits == 16:
        return None, {}
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=training_args.load_in_kbits == 4,
        load_in_8bit=training_args.load_in_kbits == 8,
        llm_int8_threshold=6.0,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=training_args.double_quant,
        bnb_4bit_quant_type=training_args.quant_type,
    )
    return quantization_config, {"": int(os.environ.get("LOCAL_RANK") or 0)}


def _load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token and no eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _load_model(model_path: str, training_args: ConditionalTrainingArguments) -> AutoModelForCausalLM:
    model_dtype = _resolve_torch_dtype(training_args)
    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if model_dtype is not None:
        model_kwargs["torch_dtype"] = model_dtype
    quantization_config, device_map = _build_quantization_config(training_args)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = device_map
        model_kwargs["load_in_4bit"] = training_args.load_in_kbits == 4
        model_kwargs["load_in_8bit"] = training_args.load_in_kbits == 8
    elif os.environ.get("LOCAL_RANK") is not None:
        model_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    model = _load_causal_lm_with_attention_backend(
        model_path,
        use_flash_attention_2=training_args.use_flash_attention_2,
        **model_kwargs,
    )
    if training_args.load_in_kbits in {4, 8}:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )
    return model


def _enable_gradient_checkpointing_for_peft(model) -> None:
    model_modules_to_save = getattr(model, "modules_to_save", None)
    if model_modules_to_save and "embed_tokens" in model_modules_to_save:
        return
    hook_target = getattr(model, "base_model", model)
    if hasattr(hook_target, "enable_input_require_grads"):
        hook_target.enable_input_require_grads()
    elif hasattr(hook_target, "get_input_embeddings"):
        def make_inputs_require_grad(_module, _input, output):
            output.requires_grad_(True)

        hook_target.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


def _load_student_model(
    config: ConditionalProDiPAConfig,
    training_args: ConditionalTrainingArguments,
) -> AutoModelForCausalLM:
    student_model = _load_model(config.student.backbone_path, training_args)
    if training_args.seed is not None:
        set_seed(training_args.seed)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.student.lora.target_modules,
        inference_mode=False,
        r=config.student.lora.r,
        lora_alpha=config.student.lora.alpha,
        lora_dropout=config.student.lora.dropout,
        bias="none",
    )
    student_model = get_peft_model(student_model, lora_config)
    student_model.config.use_cache = False
    if training_args.gradient_checkpointing:
        _enable_gradient_checkpointing_for_peft(student_model)
    if hasattr(student_model, "print_trainable_parameters"):
        student_model.print_trainable_parameters()
    return student_model


def _load_teacher_model_and_specs(
    config: ConditionalProDiPAConfig,
    training_args: ConditionalTrainingArguments,
) -> tuple[PeftModel, list[TeacherAdapterSpec]]:
    teacher_base_model = _load_model(config.teachers.backbone_path, training_args)
    first_teacher = config.teachers.adapters[0]
    teacher_model = PeftModel.from_pretrained(
        teacher_base_model,
        first_teacher.adapter_path,
        is_trainable=False,
    )
    specs: list[TeacherAdapterSpec] = [
        TeacherAdapterSpec(
            name=first_teacher.name,
            adapter_name="default",
            weight=first_teacher.weight,
            temperature=first_teacher.temperature,
        )
    ]
    for idx, teacher in enumerate(config.teachers.adapters[1:], start=1):
        adapter_name = teacher.name if teacher.name != "default" else f"teacher_{idx}"
        try:
            teacher_model.load_adapter(teacher.adapter_path, adapter_name=adapter_name, is_trainable=False)
        except TypeError:
            teacher_model.load_adapter(teacher.adapter_path, adapter_name=adapter_name)
        _restore_adapter_weights_exact(teacher_model, teacher.adapter_path, adapter_name)
        specs.append(
            TeacherAdapterSpec(
                name=teacher.name,
                adapter_name=adapter_name,
                weight=teacher.weight,
                temperature=teacher.temperature,
            )
        )
    teacher_model.requires_grad_(False)
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    return teacher_model, specs


def _restore_adapter_weights_exact(model: PeftModel, adapter_path: str, adapter_name: str) -> None:
    """Undo PEFT's BF16-then-FP32 rounding when appending adapters to a BF16 base."""
    checkpoint = load_safetensors(str(Path(adapter_path) / "adapter_model.safetensors"))
    parameters = dict(model.named_parameters())
    missing: list[str] = []
    with torch.no_grad():
        for checkpoint_name, value in checkpoint.items():
            if not checkpoint_name.endswith(".weight"):
                missing.append(checkpoint_name)
                continue
            parameter_name = checkpoint_name[:-7] + f".{adapter_name}.weight"
            parameter = parameters.get(parameter_name)
            if parameter is None:
                missing.append(parameter_name)
                continue
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    if missing:
        raise KeyError(f"Adapter `{adapter_name}` has unmapped checkpoint tensors: {missing[:3]}")


def _load_shared_student_and_teachers(
    config: ConditionalProDiPAConfig,
    training_args: ConditionalTrainingArguments,
) -> tuple[PeftModel, list[TeacherAdapterSpec]]:
    first = config.teachers.adapters[0]
    first_name = f"teacher_0_{first.name}"
    student_model = PeftModel.from_pretrained(
        _load_model(config.teachers.backbone_path, training_args),
        first.adapter_path,
        adapter_name=first_name,
        is_trainable=False,
    )
    specs = [TeacherAdapterSpec(
        name=first.name,
        adapter_name=first_name,
        weight=first.weight,
        temperature=first.temperature,
    )]
    for index, teacher in enumerate(config.teachers.adapters[1:], start=1):
        adapter_name = f"teacher_{index}_{teacher.name}"
        student_model.load_adapter(teacher.adapter_path, adapter_name=adapter_name, is_trainable=False)
        _restore_adapter_weights_exact(student_model, teacher.adapter_path, adapter_name)
        specs.append(TeacherAdapterSpec(
            name=teacher.name,
            adapter_name=adapter_name,
            weight=teacher.weight,
            temperature=teacher.temperature,
        ))
    if training_args.seed is not None:
        set_seed(training_args.seed)
    student_model.add_adapter("default", LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.student.lora.target_modules,
        inference_mode=False,
        r=config.student.lora.r,
        lora_alpha=config.student.lora.alpha,
        lora_dropout=config.student.lora.dropout,
        bias="none",
    ))
    student_model.set_adapter("default")
    student_model.config.use_cache = False
    if training_args.gradient_checkpointing:
        _enable_gradient_checkpointing_for_peft(student_model)
    if hasattr(student_model, "print_trainable_parameters"):
        student_model.print_trainable_parameters()
    return student_model, specs


def _validate_training_setup(
    config: ConditionalProDiPAConfig,
    script_args: ConditionalScriptArguments,
    training_args: ConditionalTrainingArguments,
) -> None:
    if not config.teachers.adapters:
        raise ValueError("ProDiPA distillation training requires configured teachers.")
    diagnostic_only = bool(script_args.check_shared_teacher_parity)
    if not training_args.output_dir and not script_args.build_cache and not diagnostic_only:
        raise ValueError("`output_dir` is required.")
    if float(training_args.num_train_epochs) != 1.0:
        raise ValueError("`num_train_epochs` must be 1.0; use `generation.num_train_samples` to control training size.")
    if not training_args.do_train and not script_args.build_cache and not diagnostic_only:
        raise ValueError("ProLLaMA ProDiPA distillation training requires `--do_train`.")
    if diagnostic_only and not script_args.fixed_rollouts:
        raise ValueError("`--check_shared_teacher_parity` requires `--fixed_rollouts`.")
    if script_args.build_cache and not script_args.fixed_rollouts:
        raise ValueError("`--build_cache` requires `--fixed_rollouts`.")
    if script_args.build_cache and script_args.cache_path:
        raise ValueError("`--build_cache` and `--cache_path` are mutually exclusive.")
    if script_args.cache_path and script_args.fixed_rollouts:
        raise ValueError("Cached training reads its rollouts from the cache; omit `--fixed_rollouts`.")
    if script_args.cache_path and training_args.per_device_train_batch_size != 1:
        raise ValueError("Exact full-vocabulary cached training currently requires batch size 1.")


def _build_exact_static_cache(
    cache_path: str,
    dataset: FixedRolloutDataset,
    collator: FixedRolloutCollator,
    teacher_model: PeftModel,
    teacher_specs: list[TeacherAdapterSpec],
    config: ConditionalProDiPAConfig,
) -> None:
    """Materialize the same exact full-vocabulary target used by online JSD."""
    if config.distill.fusion_rule != "poe":
        raise ValueError("Static cached-distillation comparison currently supports exact PoE only.")
    if torch.cuda.is_available():
        teacher_model.to(torch.device("cuda", int(os.environ.get("LOCAL_RANK") or 0)))
    device = next(teacher_model.parameters()).device
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)
    weights = torch.tensor([spec.weight for spec in teacher_specs], dtype=torch.float32, device=device)
    records: list[dict[str, Any]] = []
    teacher_forward_tokens = 0
    started = time.perf_counter()
    teacher_model.eval()
    for batch_index, batch in enumerate(loader):
        input_ids = batch["fixed_input_ids"].to(device)
        attention_mask = batch["fixed_attention_mask"].to(device)
        labels = batch["fixed_labels"]
        prompt_length = int(batch["prompt_length"])
        teacher_log_probs: list[torch.Tensor] = []
        with torch.no_grad():
            for spec in teacher_specs:
                teacher_model.set_adapter(spec.adapter_name)
                logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits
                logits = logits[:, prompt_length - 1:-1].float() / float(spec.temperature)
                teacher_log_probs.append(F.log_softmax(logits, dim=-1))
        stacked = torch.stack(teacher_log_probs, dim=2)
        poe_scores = (stacked * weights.view(1, 1, -1, 1)).sum(dim=2)
        target_log_probs = F.log_softmax(poe_scores, dim=-1)
        valid_tokens = int((labels[:, prompt_length:] != -100).sum().item())
        teacher_forward_tokens += valid_tokens * len(teacher_specs)
        records.append(
            {
                "sequence_id": batch["sequence_ids"][0],
                "input_ids": input_ids[0].cpu(),
                "attention_mask": attention_mask[0].cpu(),
                "labels": labels[0].cpu(),
                "prompt_length": prompt_length,
                "target_log_probs": target_log_probs[0].to(torch.float16).cpu(),
            }
        )
        if (batch_index + 1) % 16 == 0:
            logger.info("Static cache: %d/%d sequences", batch_index + 1, len(dataset))

    elapsed = time.perf_counter() - started
    payload = {
        "metadata": {
            "format": "prodipa_exact_full_vocab_poe_v1",
            "fusion_rule": "poe",
            "teacher_weights": {spec.name: float(spec.weight) for spec in teacher_specs},
            "teacher_adapters": {spec.name: spec.adapter_name for spec in teacher_specs},
            "teacher_forward_tokens": teacher_forward_tokens,
            "records": len(records),
            "target_dtype": "float16",
            "build_seconds": elapsed,
        },
        "records": records,
    }
    destination = Path(cache_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    logger.info(
        "Saved exact PoE cache: path=%s records=%d teacher_forward_tokens=%d seconds=%.1f size_gb=%.3f",
        destination,
        len(records),
        teacher_forward_tokens,
        elapsed,
        destination.stat().st_size / 1e9,
    )


def _check_shared_teacher_parity(
    report_path: str,
    rollout_path: str,
    tokenizer,
    config: ConditionalProDiPAConfig,
    training_args: ConditionalTrainingArguments,
) -> None:
    """Compare independent teacher loads with shared-base adapter switching."""
    dataset = FixedRolloutDataset(rollout_path)
    collator = FixedRolloutCollator(tokenizer=tokenizer, prompt_config=config.prompt)
    batch = collator([dataset[0]])
    prompt_length = int(batch["prompt_length"])
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK") or 0)) if torch.cuda.is_available() else torch.device("cpu")
    input_ids = batch["fixed_input_ids"].to(device)
    attention_mask = batch["fixed_attention_mask"].to(device)
    independent: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    for teacher in config.teachers.adapters:
        model = PeftModel.from_pretrained(
            _load_model(config.teachers.backbone_path, training_args),
            teacher.adapter_path,
            is_trainable=False,
        ).to(device).eval()
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, prompt_length - 1:-1].float()
            log_probs = F.log_softmax(logits / float(teacher.temperature), dim=-1)
        independent[teacher.name] = (logits.cpu(), log_probs.cpu())
        del logits, log_probs, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    shared, specs = _load_shared_student_and_teachers(config, training_args)
    shared.to(device).eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for spec in specs:
            shared.set_adapter(spec.adapter_name)
            logits = shared(input_ids=input_ids, attention_mask=attention_mask).logits[:, prompt_length - 1:-1].float().cpu()
            log_probs = F.log_softmax(logits / float(spec.temperature), dim=-1)
            independent_logits, independent_log_probs = independent[spec.name]
            logit_diff = (logits - independent_logits).abs()
            logprob_diff = (log_probs - independent_log_probs).abs()
            rows.append({
                "teacher": spec.name,
                "tokens_compared": int(logits.shape[1]),
                "vocab_size": int(logits.shape[2]),
                "max_abs_logit_diff": float(logit_diff.max().item()),
                "mean_abs_logit_diff": float(logit_diff.mean().item()),
                "max_abs_logprob_diff": float(logprob_diff.max().item()),
            })
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    max_diff = max(row["max_abs_logprob_diff"] for row in rows)
    if max_diff > 1e-4:
        raise RuntimeError(f"Shared teacher adapter parity failed: max_abs_logprob_diff={max_diff:.6g}")
    logger.info("Shared teacher adapter parity passed: report=%s max_abs_logprob_diff=%.6g", destination, max_diff)


def main() -> None:
    parser = HfArgumentParser((ConditionalScriptArguments, ConditionalTrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    config = _apply_script_overrides(load_config(script_args.config_path), script_args)
    _validate_training_setup(config, script_args, training_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(training_args.get_process_log_level())
    logger.info(
        "Resolved training setup: fusion_rule=%s teachers=%s num_train_samples=%d max_new_tokens=%d",
        config.distill.fusion_rule,
        {adapter.name: adapter.weight for adapter in config.teachers.adapters},
        config.generation.num_train_samples,
        config.generation.max_new_tokens,
    )
    if training_args.seed is not None:
        set_seed(training_args.seed)
    if training_args.output_dir:
        Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(config.student.backbone_path)
    if script_args.check_shared_teacher_parity:
        _check_shared_teacher_parity(
            script_args.check_shared_teacher_parity,
            script_args.fixed_rollouts,
            tokenizer,
            config,
            training_args,
        )
        return
    if script_args.build_cache:
        teacher_model, teacher_specs = _load_teacher_model_and_specs(config, training_args)
        if teacher_model.get_input_embeddings().num_embeddings != len(tokenizer):
            teacher_model.resize_token_embeddings(len(tokenizer))
        fixed_dataset = FixedRolloutDataset(script_args.fixed_rollouts)
        fixed_collator = FixedRolloutCollator(tokenizer=tokenizer, prompt_config=config.prompt)
        _build_exact_static_cache(
            script_args.build_cache,
            fixed_dataset,
            fixed_collator,
            teacher_model,
            teacher_specs,
            config,
        )
        return

    student_model = None
    teacher_model = None
    teacher_specs: list[TeacherAdapterSpec] = []
    if script_args.cache_path:
        student_model = _load_student_model(config, training_args)
        train_dataset = StaticCacheDataset(script_args.cache_path)
        data_collator = StaticCacheCollator(tokenizer.pad_token_id)
    elif script_args.fixed_rollouts:
        train_dataset = FixedRolloutDataset(script_args.fixed_rollouts)
        data_collator = FixedRolloutCollator(tokenizer=tokenizer, prompt_config=config.prompt)
        if script_args.share_teacher_backbone:
            student_model, teacher_specs = _load_shared_student_and_teachers(config, training_args)
            teacher_model = student_model
        else:
            student_model = _load_student_model(config, training_args)
            teacher_model, teacher_specs = _load_teacher_model_and_specs(config, training_args)
    else:
        student_model = _load_student_model(config, training_args)
        train_dataset = FixedPromptDataset(config.generation.num_train_samples)
        data_collator = ConditionalProDiPADataCollator(tokenizer=tokenizer, prompt_config=config.prompt)
        teacher_model, teacher_specs = _load_teacher_model_and_specs(config, training_args)

    if student_model is None:
        raise RuntimeError("Student model was not initialized.")
    if student_model.get_input_embeddings().num_embeddings != len(tokenizer):
        student_model.resize_token_embeddings(len(tokenizer))
    if teacher_model is not None and teacher_model.get_input_embeddings().num_embeddings != len(tokenizer):
        teacher_model.resize_token_embeddings(len(tokenizer))

    trainer_class = StaticCachedConditionalTrainer if script_args.cache_path else ConditionalProDiPATrainer
    trainer_kwargs: dict[str, Any] = dict(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        temperature=config.generation.student_temperature,
        beta=config.distill.beta,
    )
    if not script_args.cache_path:
        trainer_kwargs.update(
            teacher_model=teacher_model,
            teacher_specs=teacher_specs,
            sequence_regex=config.prompt.sequence_regex,
            use_curriculum_tem=config.generation.use_curriculum_tem,
            curriculum_tem_start=config.generation.curriculum_tem_start,
            max_new_tokens=config.generation.max_new_tokens,
            repetition_penalty=config.generation.repetition_penalty,
            top_p=config.generation.top_p,
            top_k=config.generation.top_k,
            save_generations=config.generation.save_generations,
            generation_save_steps=config.generation.generation_save_steps,
            top_k_loss=config.distill.top_k_loss,
            useloss=config.distill.loss_type,
            fusion_rule=config.distill.fusion_rule,
            useZ_t=config.distill.use_z_t,
            use_entropy_rule=config.distill.use_entropy_rule,
            entropy_alpha=config.distill.entropy_alpha,
            use_wi_entropy=config.distill.use_wi_entropy,
            use_wi_distance=config.distill.use_wi_distance,
            wi_entropy_tau=config.distill.wi_entropy_tau,
            wi_distance_beta=config.distill.wi_distance_beta,
            wi_distance_tau=config.distill.wi_distance_tau,
            debug_teacher_diff=config.distill.prodipa.debug_teacher_diff,
            debug_teacher_diff_steps=config.distill.prodipa.debug_teacher_diff_steps,
            log_teacher_entropy=config.distill.log_teacher_entropy,
        )
    trainer = trainer_class(**trainer_kwargs)
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    metrics = dict(train_result.metrics)
    metrics["train_samples"] = len(train_dataset)
    if hasattr(trainer, "teacher_forward_tokens"):
        metrics["teacher_forward_tokens"] = int(trainer.teacher_forward_tokens)
        metrics["student_training_tokens"] = int(trainer.student_training_tokens)
        metrics["teacher_inference_seconds"] = float(trainer.teacher_inference_seconds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if config.generation.save_generations and hasattr(trainer, "_save_generation_outputs"):
        trainer._save_generation_outputs(int(trainer.state.global_step))
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
