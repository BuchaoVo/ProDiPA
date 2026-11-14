#!/usr/bin/env python3
"""Reload one TPR LoRA adapter and verify finite next-token logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.data.read_text())
    tokenizer = LlamaTokenizer.from_pretrained(args.base, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": 0}, local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    prompt = f"{rows[0]['instruction']} {rows[0]['input']}"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**inputs).logits[:, -1]
    payload = {
        "status": "PASS", "adapter": str(args.adapter.resolve()),
        "logits_shape": list(logits.shape), "finite_logits": bool(torch.isfinite(logits).all()),
        "logits_dtype": str(logits.dtype), "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    if not payload["finite_logits"] or logits.shape[-1] != len(tokenizer):
        raise RuntimeError(f"adapter verification failed: {payload}")
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
