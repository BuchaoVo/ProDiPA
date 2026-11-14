#!/usr/bin/env python3
"""Independent BF16 AMP ProtGPT2 Prefix teacher trainer.

Does NOT modify existing ProDiPA code. Uses EOS + raw-sequence + EOS,
fixed 512 BPE tokens, PEFT PrefixTuning, AdamW/poly schedule, and early stop.
Validation inputs are *not* an unseen blind test; checkpoint selection uses them.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, help='Existing local ProtGPT2 model directory')
    ap.add_argument('--tokenizer', default=None, help='Existing local tokenizer directory (default: --model)')
    ap.add_argument('--train', required=True)
    ap.add_argument('--validation', required=True)
    ap.add_argument('--output', required=True, help='New output directory; no old adapter is overwritten')
    ap.add_argument('--attribute', required=True, choices=('plddt', 'sol', 'thermo'))
    ap.add_argument('--max-length', type=int, default=512, help='BPE tokens including 2 EOS, NOT amino acids')
    ap.add_argument('--virtual-tokens', type=int, default=100)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--learning-rate', type=float, default=0.005)
    ap.add_argument('--lr-end', type=float, default=1e-7)
    ap.add_argument('--lr-power', type=float, default=3.0)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--min-delta', type=float, default=0.0001)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--smoke-steps', type=int, default=0, help='Safety test only, never writes DONE')
    ap.add_argument('--disable-tf32', action='store_true')
    return ap.parse_args()


def read_sequences(path: Path, score_column: str) -> list[str]:
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['sequence', score_column]:
            raise ValueError(f'Unexpected CSV headers in {path}: {reader.fieldnames}')
        seqs = [r['sequence'].strip() for r in reader]
    if not seqs:
        raise ValueError(f'Empty training/validation CSV: {path}')
    return seqs


def load_batches(seqs, tokenizer, max_length: int):
    """Model-side serialization exactly matches official trainer semantics.
    Fail-closed if token budget is exceeded; never truncate any protein.
    """
    import torch
    from torch.utils.data import Dataset

    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError('ProtGPT2 tokenizer has no EOS token')
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos
        tokenizer.pad_token_id = eos

    class Rows(Dataset):
        def __init__(self, s):
            self.rows = []
            self.longest = 0
            for i, seq in enumerate(s):
                toks = tokenizer(seq, add_special_tokens=False)['input_ids']
                ids = [eos] + toks + [eos]
                self.longest = max(self.longest, len(ids))
                if len(ids) > max_length:
                    raise ValueError(
                        f'Token budget exceeded: row={i} len_aa={len(seq)} '
                        f'tokens_with_eos={len(ids)} max_length={max_length}. '
                        f'No truncation permitted.'
                    )
                padding = max_length - len(ids)
                self.rows.append({
                    'input_ids': torch.tensor(ids + [pad] * padding, dtype=torch.long),
                    'attention_mask': torch.tensor([1] * len(ids) + [0] * padding, dtype=torch.long),
                    'labels': torch.tensor(ids + [-100] * padding, dtype=torch.long),
                })
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, idx):
            return self.rows[idx]
    return Rows(seqs)


def save_best(model, tokenizer, target: Path, meta: dict) -> None:
    tmp = target.with_name(target.name + '.staging')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tokenizer.save_pretrained(tmp)
    (tmp / 'best_epoch.json').write_text(json.dumps(meta, indent=2) + '\n')
    # Replace only this run's output. Previous versions live in different run paths.
    if target.exists():
        shutil.rmtree(target)
    os.replace(tmp, target)


def main():
    args = parse_args()
    if args.smoke_steps < 0:
        raise ValueError('--smoke-steps must be nonnegative')
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError('epochs/batch-size/patience must be positive')
    model_path = Path(args.model).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer or args.model).expanduser().resolve()
    for path in (model_path, tokenizer_path):
        if not path.is_dir():
            raise ValueError(f'Model/tokenizer directory does not exist: {path}')
    train_file = Path(args.train).expanduser().resolve()
    val_file = Path(args.validation).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'DONE.json').exists():
        raise ValueError(f'Already finished, refusing to overwrite: {output}')
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise ValueError('GPU queue required: CUDA_VISIBLE_DEVICES not set')

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from peft import PrefixTuningConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_polynomial_decay_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; cannot run teacher training on CPU')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Selected GPU does not support BF16')
    torch.backends.cuda.matmul.allow_tf32 = not args.disable_tf32
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda:0')

    score_col = {'plddt': 'pLDDT', 'sol': 'sol', 'thermo': 'thermo'}[args.attribute]
    train_seqs = read_sequences(train_file, score_col)
    val_seqs = read_sequences(val_file, score_col)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    train_ds = load_batches(train_seqs, tokenizer, args.max_length)
    val_ds = load_batches(val_seqs, tokenizer, args.max_length)
    print(f'Parsed train={len(train_ds)}, val={len(val_ds)}, max_BPE_tokens_train={train_ds.longest}, max_BPE_tokens_val={val_ds.longest}', flush=True)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    base = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float32, local_files_only=True)
    base.config.use_cache = False
    prefix_cfg = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=args.virtual_tokens)
    model = get_peft_model(base, prefix_cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or any(p.dtype != torch.float32 for p in params):
        raise RuntimeError('Expected trainable FP32 Prefix parameters')
    trainable = sum(p.numel() for p in params)
    total = sum(p.numel() for p in model.parameters())
    if trainable > total / 10:
        raise RuntimeError('Too many trainable parameters; backbone may not be frozen')
    print(f'Params trainable={trainable} total={total} trainable_ratio={trainable/total:.4%}', flush=True)
    opt = torch.optim.AdamW(params, lr=args.learning_rate)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        opt, num_warmup_steps=0, num_training_steps=len(train_loader) * args.epochs,
        lr_end=args.lr_end, power=args.lr_power)
    signature = {
        'model': str(model_path), 'tokenizer': str(tokenizer_path),
        'train_sha256': digest_file(train_file), 'val_sha256': digest_file(val_file),
        'config': vars(args), 'torch_version': torch.__version__,
        'cuda_device': torch.cuda.get_device_name(0), 'precision': args.precision,
        'trainable_params': trainable, 'torch_cuda': torch.version.cuda,
    }
    atomic_json(output / 'run_manifest.json', signature)
    hist_path = output / 'metrics.jsonl'
    best = float('inf'); best_epoch = 0; no_improve = 0; steps_done = 0
    use_amp = args.precision == 'bf16'
    started = time.monotonic()
    with hist_path.open('a', encoding='utf-8', buffering=1) as h:
        for epoch in range(1, args.epochs + 1):
            model.train(); sum_train = 0.0; train_batches = 0
            for batch in train_loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    loss = model(**batch).loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f'Nonfinite train loss, epoch={epoch}, step={steps_done}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip, error_if_nonfinite=True)
                opt.step(); scheduler.step()
                sum_train += float(loss.detach().item()); train_batches += 1
                steps_done += 1
                if args.smoke_steps and steps_done >= args.smoke_steps:
                    print(f'SMOKE_OK steps={steps_done} loss={float(loss.item()):.6f} peak_mem_mib={torch.cuda.max_memory_allocated()/2**20:.1f}', flush=True)
                    return
            model.eval(); eval_sum = 0.0; eval_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                        v = model(**batch).loss
                    if not bool(torch.isfinite(v)):
                        raise FloatingPointError(f'Nonfinite validation loss at epoch={epoch}')
                    eval_sum += float(v.detach().item()); eval_batches += 1
            tr = sum_train / max(train_batches, 1); ev = eval_sum / max(eval_batches, 1)
            record = {'epoch': epoch, 'train_loss': tr, 'val_loss': ev, 'step': steps_done,
                      'lr': opt.param_groups[0]['lr'], 'peak_cuda_mem_mib': round(torch.cuda.max_memory_allocated()/2**20, 1)}
            h.write(json.dumps(record) + '\n')
            print(f'Epoch {epoch:03d} loss={tr:.6f} val={ev:.6f} best={best:.6f} cuda_peak_MiB={record["peak_cuda_mem_mib"]}', flush=True)
            if ev < best - args.min_delta:
                best = ev; best_epoch = epoch; no_improve = 0
                save_best(model, tokenizer, output / 'best_adapter', {'best_epoch': best_epoch, 'best_val_loss': best})
            else:
                no_improve += 1
            if no_improve >= args.patience:
                print(f'Early stop at epoch {epoch}; best epoch {best_epoch}', flush=True)
                break
    if not (output / 'best_adapter' / 'adapter_config.json').is_file():
        raise RuntimeError('No PEFT adapter_config.json found after training')
    atomic_json(output / 'DONE.json', {'best_epoch': best_epoch, 'best_val_loss': best,
        'steps': steps_done, 'seconds': round(time.monotonic() - started, 1),
        'data_train_sha256': signature['train_sha256'], 'data_val_sha256': signature['val_sha256'],
        'precision': args.precision, 'output_adapter': str(output / 'best_adapter')})
    print(f'FINISHED attribute={args.attribute} best_epoch={best_epoch} val_loss={best:.6f}', flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as ex:
        print(f'TRAIN_FAILED {type(ex).__name__}: {ex}', file=sys.stderr, flush=True)
        raise
