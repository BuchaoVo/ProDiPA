#!/usr/bin/env python3
"""Train or evaluate one conditional ProLLaMA LoRA property teacher.

This is the maintained ProDiPA-local training entry point.  It freezes the
base model, supervises only response tokens, saves PEFT-compatible adapters,
and retains Hugging Face Trainer checkpoint/resume semantics.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import datasets
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

try:
    from .teacher_data import (
        DataCollatorForSupervisedDataset,
        tokenize_instruction_dataset,
    )
except ImportError:  # Direct execution by path.
    from teacher_data import (  # type: ignore
        DataCollatorForSupervisedDataset,
        tokenize_instruction_dataset,
    )

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Local original ProLLaMA directory"})
    tokenizer_name_or_path: Optional[str] = None
    config_name: Optional[str] = None
    cache_dir: Optional[str] = None
    model_revision: str = "main"
    use_auth_token: bool = False
    use_fast_tokenizer: bool = True
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={"choices": ["auto", "bfloat16", "float16", "float32"]},
    )
    merge_when_finished: bool = False


@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    dataset_dir: Optional[str] = None
    data_cache_dir: Optional[str] = None
    preprocessing_num_workers: Optional[int] = None
    validation_split_percentage: float = 0.05
    max_seq_length: int = 512


@dataclass
class TeacherTrainingArguments(TrainingArguments):
    trainable: str = "q_proj,v_proj"
    lora_rank: int = 8
    lora_dropout: float = 0.1
    lora_alpha: float = 32.0
    modules_to_save: Optional[str] = None
    peft_path: Optional[str] = None
    use_flash_attention_2: bool = False
    double_quant: bool = True
    quant_type: str = "nf4"
    load_in_kbits: int = 16
    close_loss_weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "Relative token weight for the true response-closing marker. "
                "1.0 exactly recovers ordinary token-averaged causal CE."
            )
        },
    )


class ConditionalTeacherTrainer(Trainer):
    """Trainer with an optional weighted CE on the true closing marker."""

    def __init__(self, *args, close_loss_weight: float = 1.0, **kwargs):
        if close_loss_weight < 1.0:
            raise ValueError("close_loss_weight must be at least 1.0")
        self.close_loss_weight = float(close_loss_weight)
        super().__init__(*args, **kwargs)
        # This custom loss is already normalized over active token weights and
        # does not consume Trainer's num_items_in_batch.  PEFT forward accepts
        # arbitrary kwargs, which can otherwise make Trainer skip the expected
        # gradient-accumulation normalization.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, **_kwargs):
        close_token_mask = inputs.pop("close_token_mask")
        labels = inputs["labels"]
        outputs = model(**inputs)
        if self.close_loss_weight == 1.0:
            loss = outputs.loss
        else:
            shifted_logits = outputs.logits[:, :-1, :].float()
            shifted_labels = labels[:, 1:]
            active = shifted_labels.ne(-100)
            token_loss = F.cross_entropy(
                shifted_logits.reshape(-1, shifted_logits.shape[-1]),
                shifted_labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(shifted_labels)
            weights = active.to(token_loss.dtype)
            shifted_close = close_token_mask[:, 1:] & active
            if not bool(torch.all(shifted_close.sum(dim=1) == 1)):
                raise ValueError("Expected exactly one supervised closing token per example")
            weights = torch.where(
                shifted_close,
                torch.full_like(weights, self.close_loss_weight),
                weights,
            )
            loss = (token_loss * weights).sum() / weights.sum()
        return (loss, outputs) if return_outputs else loss


def _require_path(value: Optional[str], name: str, directory: bool) -> Path:
    if not value:
        raise ValueError(f"--{name} is required")
    path = Path(value).expanduser().resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise ValueError(f"--{name} must be an existing {kind}: {path}")
    return path


def _dataset_files(path: str) -> list[str]:
    root = _require_path(path, "dataset_dir", directory=True)
    files = sorted({str(item.resolve()) for pattern in ("*.json", "*.jsonl") for item in root.glob(pattern)})
    if not files:
        raise ValueError(f"No JSON instruction datasets found in {root}")
    return files


def _dtype(name: Optional[str]):
    if name in {None, "auto"}:
        return name
    return getattr(torch, name)


def _prepare_quantization(args: TeacherTrainingArguments):
    if args.load_in_kbits not in {4, 8, 16}:
        raise ValueError("load_in_kbits must be 4, 8, or 16")
    if args.load_in_kbits == 16:
        return False, False, None
    compute_dtype = torch.float16 if args.fp16 else (
        torch.bfloat16 if args.bf16 else torch.float32
    )
    skip = None
    if args.modules_to_save:
        skip = [value.strip() for value in args.modules_to_save.split(",") if value.strip()]
    config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_kbits == 4,
        load_in_8bit=args.load_in_kbits == 8,
        llm_int8_threshold=6.0,
        load_in_8bit_skip_modules=skip,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.double_quant,
        bnb_4bit_quant_type=args.quant_type,
    )
    return args.load_in_kbits == 4, args.load_in_kbits == 8, config


def _load_base_model(
    model_args: ModelArguments,
    training_args: TeacherTrainingArguments,
    config,
):
    load_in_4bit, load_in_8bit, quantization = _prepare_quantization(training_args)
    device_index = int(os.environ.get("LOCAL_RANK") or 0)
    kwargs = {
        "config": config,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": True if model_args.use_auth_token else None,
        "torch_dtype": _dtype(model_args.torch_dtype),
        "low_cpu_mem_usage": True,
        "device_map": {"": device_index},
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
        "quantization_config": quantization,
    }
    if training_args.use_flash_attention_2:
        kwargs["attn_implementation"] = "flash_attention_2"
    try:
        return AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **kwargs)
    except TypeError as error:
        # Compatibility with the Transformers version used by the original run.
        if "token" in str(error):
            kwargs["use_auth_token"] = kwargs.pop("token")
            return AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **kwargs)
        if "attn_implementation" in str(error):
            kwargs.pop("attn_implementation")
            kwargs["use_flash_attention_2"] = True
            return AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **kwargs)
        raise


def _enable_checkpoint_input_grads(model) -> None:
    target = getattr(model, "base_model", model)
    if hasattr(target, "enable_input_require_grads"):
        target.enable_input_require_grads()
    else:
        target.get_input_embeddings().register_forward_hook(
            lambda _module, _inputs, output: output.requires_grad_(True)
        )


def _attach_adapter(model, args: TeacherTrainingArguments):
    if args.peft_path:
        adapter_path = _require_path(args.peft_path, "peft_path", directory=True)
        adapter_config = PeftConfig.from_pretrained(adapter_path)
        if str(adapter_config.peft_type).split(".")[-1].lower() != "lora":
            raise ValueError(f"Expected a LoRA adapter at {adapter_path}")
        try:
            return PeftModel.from_pretrained(
                model, adapter_path, is_trainable=args.do_train
            )
        except TypeError:
            return PeftModel.from_pretrained(model, adapter_path)

    targets = [value.strip() for value in args.trainable.split(",") if value.strip()]
    if not targets:
        raise ValueError("At least one LoRA target module is required")
    modules = None
    if args.modules_to_save:
        modules = [value.strip() for value in args.modules_to_save.split(",") if value.strip()]
    return get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=targets,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            modules_to_save=modules,
        ),
    )


def _parameter_audit(model, do_train: bool) -> dict:
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if do_train and not trainable:
        raise RuntimeError("No trainable LoRA parameters were found")
    invalid = [
        name for name, _ in trainable
        if "lora_" not in name and "modules_to_save" not in name
    ]
    if invalid:
        raise RuntimeError(f"Non-LoRA parameters are trainable: {invalid[:10]}")
    for _name, parameter in trainable:
        if parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
    return {
        "trainable_parameters": sum(value.numel() for _, value in trainable),
        "total_parameters": sum(value.numel() for value in model.parameters()),
        "trainable_names": [name for name, _ in trainable],
        "trainable_dtype": "float32" if trainable else None,
    }


def _load_split(sources, tokenizer, data_args):
    return tokenize_instruction_dataset(
        sources,
        tokenizer,
        data_args.max_seq_length,
        data_args.data_cache_dir,
        data_args.preprocessing_num_workers,
    )


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TeacherTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_path = _require_path(model_args.model_name_or_path, "model_name_or_path", True)
    tokenizer_path = _require_path(
        model_args.tokenizer_name_or_path or model_args.model_name_or_path,
        "tokenizer_name_or_path",
        True,
    )
    if training_args.do_train and not (data_args.train_file or data_args.dataset_dir):
        raise ValueError("Training requires --train_file or --dataset_dir")
    if training_args.do_eval and not (
        data_args.validation_file or data_args.train_file or data_args.dataset_dir
    ):
        raise ValueError("Evaluation requires an instruction dataset")
    if training_args.close_loss_weight < 1.0:
        raise ValueError("close_loss_weight must be at least 1.0")
    if data_args.train_file:
        _require_path(data_args.train_file, "train_file", False)
    if data_args.validation_file:
        _require_path(data_args.validation_file, "validation_file", False)
    Path(training_args.output_dir).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    level = training_args.get_process_log_level()
    LOGGER.setLevel(level)
    datasets.utils.logging.set_verbosity(level)
    transformers.utils.logging.set_verbosity(level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    output_dir = Path(training_args.output_dir)
    last_checkpoint = None
    if output_dir.is_dir() and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(str(output_dir))
        ignored = {"training.log", "dataset_cache"}
        unexpected = [item for item in output_dir.iterdir() if item.name not in ignored]
        if last_checkpoint is None and unexpected:
            raise ValueError(
                f"Output directory is nonempty and not resumable: {output_dir}"
            )
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name or str(model_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_fast=model_args.use_fast_tokenizer,
        local_files_only=True,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_sources = None
    if data_args.train_file:
        train_sources = [str(Path(data_args.train_file).resolve())]
    elif data_args.dataset_dir:
        train_sources = _dataset_files(data_args.dataset_dir)
    validation_sources = (
        [str(Path(data_args.validation_file).resolve())]
        if data_args.validation_file else None
    )
    train_dataset = None
    eval_dataset = None
    with training_args.main_process_first(desc="instruction tokenization"):
        full_train = _load_split(train_sources, tokenizer, data_args) if train_sources else None
        if validation_sources:
            eval_dataset = _load_split(validation_sources, tokenizer, data_args)
        if full_train is not None:
            if training_args.do_eval and eval_dataset is None:
                split = full_train.train_test_split(
                    test_size=data_args.validation_split_percentage,
                    seed=training_args.seed,
                )
                train_dataset, eval_dataset = split["train"], split["test"]
            else:
                train_dataset = full_train

    model = _load_base_model(model_args, training_args, config)
    model.config.use_cache = False
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model = _attach_adapter(model, training_args)
    if training_args.gradient_checkpointing:
        _enable_checkpoint_input_grads(model)
    audit = _parameter_audit(model, training_args.do_train)
    LOGGER.info(
        "LoRA trainable parameters: %d / %d",
        audit["trainable_parameters"], audit["total_parameters"],
    )

    trainer = ConditionalTeacherTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSupervisedDataset(tokenizer),
        close_loss_weight=training_args.close_loss_weight,
    )
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = result.metrics
        metrics["train_samples"] = len(train_dataset)
        metrics["trainable_parameters"] = audit["trainable_parameters"]
        metrics["total_parameters"] = audit["total_parameters"]
        (output_dir / "parameter_audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        if model_args.merge_when_finished:
            merged_path = output_dir.with_name(output_dir.name + "_merged")
            model.merge_and_unload().save_pretrained(merged_path)
            tokenizer.save_pretrained(merged_path)

    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)
        if "eval_loss" in metrics:
            try:
                metrics["perplexity"] = math.exp(metrics["eval_loss"])
            except OverflowError:
                metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
