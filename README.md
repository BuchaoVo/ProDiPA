# ProDiPA

**ProDiPA** is a multi-objective protein design framework using multi-teacher knowledge distillation to balance **structural quality**, **solubility**, and **thermostability**. It provides implementations and configs for conditional/unconditional generation, online/offline distillation, and quality-aware cache strategies.

## 1. Installation

Requires **Linux**, **Python 3.10**, and a **CUDA-enabled GPU**. Install a PyTorch build matching your CUDA driver, then install the remaining dependencies:

```bash
python -m pip install -r requirements-reproduction.txt
python -m pip install -e '.[dev,evaluation]'
```

- `configs/reproduction/observed_environment.json` records exact package versions (including CUDA PyTorch) from the archived environment; `environment.yml` is a historical snapshot and may not resolve on all machines.
- Verify GPU availability after installation, as CPU dependencies can install without a working CUDA setup.

## 2. Local Path Configuration

Configure paths to pretrained models, databases, and Conda environments before running experiments:

```bash
python scripts/configure_release.py \
  --models /absolute/path/to/models \
  --databases /absolute/path/to/databases \
  --envs /absolute/path/to/conda/envs \
  --conda /absolute/path/to/conda

export PRODIPA_PYTHON="$(command -v python)"
export PYTHON_BIN="$PRODIPA_PYTHON"
export TORCHRUN_BIN="$(command -v torchrun)"
export PROJECT_ROOT="$PWD"
```

## 3. External Resources

| Resource | Expected Location / Notes |
|---|---|
| ProtGPT2 | `<models>/ProtGPT2` (upstream: `nferruz/ProtGPT2`) |
| ProLLaMA | `<models>/ProLLaMA` (verify against original fingerprint and config) |
| ESMFold HF weights | `<models>/esmfold_v1_hf` (used by structure evaluation scripts) |
| TemBERTureCLS adapter/head | `tools/TemBERTure/temBERTure/temBERTure_CLS/` (weights not included) |
| ProtBERT-BFD | Official `Rostlab/prot_bert_bfd` checkpoint (known revision in `configs/reproduction/temberture_base.json`) |
| ProTrek | Source in `third_party/ProTrek/`; weights in `<models>/ProTrek_35M` |
| MMseqs2 | `tools/mmseqs/bin/mmseqs`, or set `MMSEQS_BIN` |
| Novelty-U database | `<databases>/uniprotkb/2026_03/mmseqs/uniprotkb_2026_03` plus database-ready marker |
| External baselines | Configure under `configs/baselines/` (large weights not included) |

## 4. Dataset Verification

Use the provided splits directly without regenerating them.

| Dataset | Sequences |
|---|---:|
| Train | 47,342 |
| Validation | 5,918 |
| Test-H | 5,918 |
| Complete three-property training subset | 25,000 |
| Subset with `in_prior_15k=true` | 15,000 |
| Validation1K (stored separately) | 1,000 |

- **Conditional:** 300 training and 100 held-out sequences per property (fixed TPR superfamily condition).
- **Unconditional:** 300 training and 100 validation sequences per property (defined by the finalized Sol V3 audit and manifest).

Run lightweight data and source checks:

```bash
python scripts/configure_release.py --check-data
python -m pytest -q tests/test_pae_hv_report.py tests/cache/test_schema.py
```

These checks do not replace GPU model loading, tokenization, or training tests. Do not reuse `data/release/prodipa_dataset/normalization.json` if target property definitions change.

## 5. Teacher Model Training & Evaluation

### 5.1 Unconditional Teachers (ProtGPT2 Prefix Tuning)

Configs: `configs/reproduction/unconditional_{fold,sol,thermo}.yaml` (prefix length: `100`, max length: `FASTA60`, lr: `0.005`, effective batch size: `8`, max epochs: `50`, patience: `5`, seed: `42`).

```bash
# Preflight check
python src/prodipa/unconditional/train_teacher.py \
  --config configs/reproduction/unconditional_fold.yaml --preflight-only

# Train property teachers
CUDA_VISIBLE_DEVICES=0 python src/prodipa/unconditional/train_teacher.py \
  --config configs/reproduction/unconditional_fold.yaml

CUDA_VISIBLE_DEVICES=1 python src/prodipa/unconditional/train_teacher.py \
  --config configs/reproduction/unconditional_sol.yaml

CUDA_VISIBLE_DEVICES=2 python src/prodipa/unconditional/train_teacher.py \
  --config configs/reproduction/unconditional_thermo.yaml
```

- Checkpoints save to `outputs/unconditional/protgpt2/<property>/`. Resume from saved state instead of overwriting; do not use legacy scripts requiring `CANONICAL_RELEASE.json` in an empty output directory.

### 5.2 Conditional Teachers (ProLLaMA LoRA Fine-Tuning)

Uses three finalized JSON datasets (300 train / 100 held-out per property) with a fixed prompt and output format:

```text
# Prompt
[Generate by superfamily]
Superfamily=<Tetratricopeptide-like domain helical superfamily>

# Output
Seq=<...>
```

Train each property teacher (`8` epochs, LoRA rank `8`, alpha `16`, dropout `0.1`, lr `2e-5`, effective batch size `32`, closing-token loss weight `4`):

```bash
bash experiments/prodipa/conditional_teacher_training/run_one_teacher.sh fold 0
bash experiments/prodipa/conditional_teacher_training/run_one_teacher.sh sol 1
bash experiments/prodipa/conditional_teacher_training/run_one_teacher.sh thermo 2
```

- The 100 sequences in `test_*` are for post-training diagnostics only (no gradient updates) and are not an independent `Test-H` benchmark.
- Intermediate checkpoints are removed automatically upon completion; set an explicit retention policy beforehand if full trajectories are needed.

### 5.3 Teacher Evaluation

Benchmark each teacher against its Base model under identical prompts, sampling parameters, and budgets using `experiments/prodipa/conditional_evaluation/generate_model.py` (supports both conditional and unconditional generation).

Evaluate generated sequences across:
1. Sequence validity and amino acid composition
2. Natural EOS termination
3. Sequence length distribution
4. Repetition and redundancy
5. Structural quality (`pLDDT`, `pTM`, `pAE`)
6. Predicted solubility
7. Predicted thermostability
8. Target-property improvement over the Base model (stratified by length)

Only advance teachers to student training after verifying generation quality and target-property gains—lower training loss alone is insufficient.

## 6. Student Training & Distillation

### 6.1 Initial Multi-Teacher Student Models

```bash
# Conditional student
TRAIN_GPUS=0,1 EVAL_GPUS=0,1,2,3 \
  bash experiments/prodipa/conditional_multiteacher_student/run_all.sh

# Unconditional student
TRAINING_GPUS=0,1 \
  bash experiments/prodipa/unconditional_student_training/run_training.sh
```

### 6.2 Online Distillation & Fixed Offline Cache

Requires `outputs/conditional/prollama_multiteacher_student/checkpoint-192` to exist beforehand:

```bash
TRAIN_GPUS=0,1 EVAL_GPUS=0,1,2,3 \
  bash experiments/prodipa/offline_multiteacher_opd/run_core_experiment.sh
```

## 7. Quality-Aware Cache Experiments

```bash
ADAPTIVE_PARADIGMS=conditional ADAPTIVE_GPUS=0,1,2,3 \
  python experiments/prodipa/adaptive_cache/run.py
```

## 8. Unified Evaluation

Run evaluation for conditional or unconditional generations (pass `--help` to inspect CLI arguments):

```bash
# Conditional
bash experiments/prodipa/conditional_evaluation/run_method_evaluation.sh \
  --model-id METHOD_ID \
  --generation-tsv /absolute/path/to/generation.tsv \
  --gpus 0,1,2,3

# Unconditional
bash experiments/prodipa/unconditional_evaluation/run_method_evaluation.sh \
  --model-id METHOD_ID \
  --generation-tsv /absolute/path/to/generation.tsv \
  --gpus 0,1,2,3
```

Main property scoring scripts:

| Script | Evaluation |
|---|---|
| `scripts/run_esmfold_properties.py` | Structure prediction and structural quality |
| `scripts/run_protein_sol.py` | Predicted protein solubility |
| `scripts/run_temBERTure_cls.py` | TemBERTure-based thermostability classification |