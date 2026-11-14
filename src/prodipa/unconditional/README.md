# ProDiPA ProtGPT2 — 300/100 Prefix teachers + BF16 GPU queue

This formal pipeline trains three independent Prefix adapters from the frozen
ProDiPA 300-train/100-validation views. It does not overwrite earlier teachers,
students, or the six CSV files.

## Inputs

```text
data/unconditional/
├── fold/{train_plddt_300.csv,test_plddt_100.csv}
├── sol/{train_sol_300.csv,test_sol_100.csv}
└── thermo/{train_thermo_300.csv,test_thermo_100.csv}
```

The positive thresholds are pLDDT >= 0.85, Sol >= 0.7 and Thermo >= 0.6. The
`test_*` files are Validation1K-derived checkpoint-selection views, not a blind
test. CSVs without `sequence_id` receive an in-memory
`sha256:<sequence hash>` identifier; source CSVs are never rewritten.

## Preflight and queue

```bash
cd __PRODIPA_ROOT__
export PROTGPT2_MODEL=__PRODIPA_MODELS__/ProtGPT2
export PYTHON_BIN=__PRODIPA_ENVS__/ProDiPA/bin/python

$PYTHON_BIN src/prodipa/unconditional/prefix_data.py \
  --data-dir data/unconditional \
  --write-report /tmp/prodipa_unconditional_preflight.json

bash src/prodipa/unconditional/run_prefix_gpu_queue.sh --once
bash src/prodipa/unconditional/run_prefix_gpu_queue.sh --dry-run

nohup bash src/prodipa/unconditional/run_prefix_gpu_queue.sh \
  > protgpt2_prefix_queue.out 2>&1 < /dev/null &
echo $! > protgpt2_prefix_queue.pid
```

Defaults require an 80GB-class GPU, at least 51,200 MiB free, utilization <=15%,
and three consecutive checks 60 seconds apart. The selected device is passed by
GPU UUID, locked cooperatively with `flock`, and rechecked immediately before
launch. No CUDA model is loaded while waiting.

The queue runs Fold -> Sol -> Thermo. Before formal training it runs a two-step
BF16 engineering check that verifies forward/backward, FP32 Prefix and AdamW
state, finite loss/gradients, adapter save and `PeftModel.from_pretrained()`
reload. Temporary engineering-check checkpoints are deleted immediately; the
report and log remain.

## Formal configuration

| Field | Value |
|---|---|
| Backbone | original pretrained ProtGPT2, frozen BF16 weights |
| Adapter | PEFT Prefix-Tuning, 100 virtual tokens |
| Trainable precision | Prefix FP32; AdamW state FP32 |
| Forward | BF16 autocast; no GradScaler |
| LR | 0.005 |
| Scheduler | polynomial decay, power 3, final LR 1e-7, no warmup |
| Batch | 8; accumulation 1 |
| Epochs | at most 50 |
| Early stopping | patience 5; minimum improvement 1e-4 |
| Seed | 42 |
| Token budget | 512 BPE tokens including leading/trailing EOS; no truncation |

## Outputs

```text
outputs/unconditional/protgpt2/
├── data_preflight.json
├── engineering_check.json
├── gpu_queue.log
├── configs/{fold,sol,thermo}.yaml
├── fold/
│   ├── best_adapter/
│   ├── latest_resume/
│   ├── training_history.tsv
│   ├── training.log
│   ├── run_metadata.json
│   └── DONE.json
├── sol/
└── thermo/
```

An interrupted epoch can resume only from the last complete epoch. Resume checks
the configuration and both data SHA256 values. `DONE.json` is written only after
the best adapter successfully reloads through `PeftModel.from_pretrained()`.

## Difference from the local reference implementation

Scientific hyperparameters remain Prefix-100, AdamW 0.005, polynomial power 3,
batch 8, 50 epochs and patience 5. ProDiPA additionally uses BF16 autocast,
dynamic right-padding, token-weighted validation loss, finite-value audits,
atomic checkpoint roles, optimizer/scheduler/RNG resume, detailed manifests and
post-save adapter reload. Dynamic right-padding changes compute efficiency but
not supervised token content or causal positions.

The local reference script defaults to `max_length=256`; this formal
task explicitly fixes the limit at 512 BPE tokens. All six current files are at
most 182 tokens including both EOS markers, so neither setting would
truncate these data.

The student `train.py`, `trainer.py` and two-process `accelerate.yaml` are not
used for these three single-property teachers.

## Maintained implementation

The complete teacher runtime lives in this directory and has no sibling-tree
Python dependency:

- `train_teacher.py`: canonical Prefix trainer with BF16 forward, FP32 Prefix
  and optimizer state, atomic best/resume checkpoints and adapter reload audit.
- `prefix_data.py`: frozen-data and homology preflight.
- `gpu_prefix_queue.py`: GPU selection, engineering check and sequential
  Fold -> Sol -> Thermo execution.
- `run_prefix_gpu_queue.sh`: stable command-line entry point.

`prefix_trainer_bf16.py` is retained only for compatibility with earlier run
records; new formal runs use `train_teacher.py`.
