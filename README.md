# ProDiPA

ProDiPA is a multi-objective protein design framework that leverages multi-teacher knowledge distillation to improve protein sequence generation across multiple functional properties.

By integrating property-specific teacher models into a unified student model, ProDiPA aims to balance structural quality, solubility, and thermostability in protein design.

This repository provides the implementation and experimental configurations for conditional and unconditional generation, online and offline distillation, and quality-aware cache strategies.

## 1. Environment Setup and External Resources

### 1.1 Software Environment

We recommend Linux, Python 3.10, and a CUDA-enabled GPU. Install a version of PyTorch compatible with your CUDA driver first, followed by the remaining dependencies:

```bash
python -m pip install -r requirements-reproduction.txt
python -m pip install -e '.[dev,evaluation]'
```

The file `configs/reproduction/observed_environment.json` records the package versions observed when the experimental environment was archived, including the original CUDA-enabled PyTorch installation.

The `environment.yml` file is retained as a historical environment record. It may not be directly reproducible on every machine. In particular, successfully installing the CPU dependencies does not imply that the GPU environment has been properly configured.

This repository does not currently include a valid `uv.lock`, so reproducibility through `uv sync --locked` is not assumed.

### 1.2 Local Path Configuration

Before running the experiments, configure the paths to the pretrained models, databases, and Conda environments on your machine:

```bash
python scripts/configure_release.py \
  --models /absolute/path/to/models \
  --databases /absolute/path/to/databases \
  --envs /absolute/path/to/conda/envs \
  --conda /absolute/path/to/conda

export PYTHONPATH="$PWD/src:$PWD/third_party/ProteinOPD/conditional/proteinopd"
export PRODIPA_PYTHON="$(command -v python)"
export PYTHON_BIN="$PRODIPA_PYTHON"
export TORCHRUN_BIN="$(command -v torchrun)"
export PROJECT_ROOT="$PWD"
```

The main Conda environments under `--envs` are named `ProDiPA` and `temberture`. Some legacy experiment scripts refer to these environment names directly. If different names are used, check the remaining resource paths reported by the configuration script.

The configuration script only updates local paths. It does not modify the loss functions, dataset membership, random seeds, or evaluation parameters used in the original experiments.

### 1.3 Pretrained Models and External Resources

The following resources are required for the corresponding experiments.

| Resource | Purpose / Expected Location |
|---|---|
| ProtGPT2 pretrained weights and tokenizer | `<models>/ProtGPT2`; upstream model: `nferruz/ProtGPT2` |
| ProLLaMA pretrained weights and tokenizer | `<models>/ProLLaMA`; verify against the original model fingerprint and configuration |
| ESMFold Hugging Face weights | `<models>/esmfold_v1_hf`; use the existing evaluation scripts rather than replacing the structure predictor |
| TemBERTureCLS adapter and classification head | `tools/TemBERTure/temBERTure/temBERTure_CLS/`; weights are not included in this package |
| ProtBERT-BFD | Official `Rostlab/prot_bert_bfd` checkpoint; the known revision is recorded in `configs/reproduction/temberture_base.json`, where available |
| ProTrek | Source code under `third_party/ProTrek/` and pretrained weights under `<models>/ProTrek_35M` |
| MMseqs2 | Install at `tools/mmseqs/bin/mmseqs` or specify its location through `MMSEQS_BIN` |
| Novelty-U database | `<databases>/uniprotkb/2026_03/mmseqs/uniprotkb_2026_03`, together with the corresponding database-ready marker |
| External baseline weights and indices | Configure individual resources under `configs/baselines/`; downloaded large model weights are not included in this package |

The source-code commits of third-party dependencies are recorded in `configs/reproduction/third_party_sources.json`.

For pretrained models whose original revisions cannot be verified from the available records, the revision should be treated as uncertain. Matching directory names alone are not sufficient to establish that a checkpoint is identical to the one used in the original experiments.

**TemBERTure environment.** TemBERTure requires a separate environment based on an older version of `adapter-transformers`. The main project's Transformers installation should not be used as a direct replacement.

Prepare the environment using `tools/TemBERTure/temBERTure/requirements.txt` and its accompanying README. Also verify the classification head, input preprocessing, and extraction of the thermophilic-class probability. The CLS-based classification model must not be replaced with a Tm regression model.

## 2. Dataset Verification

The experiments use the existing dataset splits and finalized data artifacts. The original splits should be used as provided rather than reconstructed.

### 2.1 Original Dataset Statistics

| Dataset | Number of sequences |
|---|---:|
| Train | 47,342 |
| Validation | 5,918 |
| Test-H | 5,918 |
| Complete three-property training subset | 25,000 |
| Subset with `in_prior_15k=true` | 15,000 |
| Validation1K | 1,000 |

The 25K subset contains sequences with all three historical property annotations. Validation1K is maintained as a separately stored dataset.

### 2.2 Current Experimental Datasets

The datasets used in the current experiments are organized as follows:

- **Conditional:** 300 training sequences and 100 held-out sequences per property. The TPR family condition is fixed across the experiments.
- **Unconditional:** 300 training sequences and 100 validation sequences per property. The finalized Sol V3 audit and its associated manifest define the dataset used in the current experiments.

The compressed TSV file containing the historical 25K dataset can be read directly using `gzip.open(..., 'rt')` or `pandas.read_csv(..., sep='\t')`.

Only the compressed finalized table is retained to avoid redundant JSONL and CSV copies. The original sequence scores, identifiers, and dataset membership have not been recalculated.

The following commands can be used to inspect the available data and run the corresponding lightweight tests:

```bash
python scripts/configure_release.py --check-data
python -m pytest -q tests/test_pae_hv_report.py tests/cache/test_schema.py
```

These checks verify the packaged data and source files. They do not replace GPU-based model loading, tokenization, or actual training.

The file `data/release/prodipa_dataset/normalization.json` belongs to the historical experiments. It should not be automatically reused in new experiments that introduce different target definitions.

## 3. Teacher Model Training

The teacher models are trained separately for the three target properties: folding-related structural quality, solubility, and thermostability.

### 3.1 Unconditional Teachers: ProtGPT2 Prefix Tuning

The configurations in:

```text
configs/reproduction/unconditional_{fold,sol,thermo}.yaml
```

are derived directly from the configurations used in the current teacher-training experiments. Only the output directories have been updated to point to the canonical teacher locations.

The original training settings are retained:

| Parameter | Setting |
|---|---|
| Prefix length | 100 |
| Maximum sequence length | FASTA60 |
| Learning rate | 0.005 |
| Effective batch size | 8 |
| Maximum epochs | 50 |
| Early stopping patience | 5 |
| Random seed | 42 |

These settings correspond to the current teacher experiments, not the earlier approach that trained unconditional teachers on the entire 25K dataset.

Run the following commands:

```bash
python src/prodipa/unconditional/train_teacher.py \
  --config configs/reproduction/unconditional_fold.yaml --preflight-only

CUDA_VISIBLE_DEVICES=0 python src/prodipa/unconditional/train_teacher.py \
  --config configs/reproduction/unconditional_fold.yaml

CUDA_VISIBLE_DEVICES=1 python src/prodipa/unconditional/train_teacher.py \
  --config configs/reproduction/unconditional_sol.yaml

CUDA_VISIBLE_DEVICES=2 python src/prodipa/unconditional/train_teacher.py \
  --config configs/reproduction/unconditional_thermo.yaml
```

The three training jobs can be executed sequentially or in parallel, depending on available GPU resources.

The resulting checkpoints are stored under:

```text
outputs/unconditional/protgpt2/<property>/
```

If training needs to be resumed, use the saved training state rather than overwriting an existing experiment.

The historical release scripts that require an existing `CANONICAL_RELEASE.json` should not be used to initialize training in an empty output directory.

### 3.2 Conditional Teachers: ProLLaMA LoRA Fine-Tuning

The conditional teachers are trained using the three finalized JSON datasets, each containing 300 training sequences and 100 held-out sequences.

The conditioning prompt is fixed as:

```text
[Generate by superfamily]
Superfamily=<Tetratricopeptide-like helical domain superfamily>
```

The generated output follows the format:

```text
Seq=<...>
```

Run the three property-specific training jobs:

```bash
bash experiments/prodipa/conditional_teacher_training/run_one_teacher.sh fold 0

bash experiments/prodipa/conditional_teacher_training/run_one_teacher.sh sol 1

bash experiments/prodipa/conditional_teacher_training/run_one_teacher.sh thermo 2
```

The original training configuration is retained:

- Training duration: 8 epochs.
- LoRA rank: 8.
- LoRA alpha: 16.
- LoRA dropout: 0.1.
- Learning rate: 2e-5.
- Effective batch size: 32.
- Closing-token loss weight: 4.

The 100 sequences in each `test_*` file are used for post-training diagnostics and do not contribute to gradient updates in this training procedure. They should not be described as an independent final Test-H evaluation set.

This historical training entry point removes its intermediate checkpoints after completion. If the complete training trajectory is required, an explicit checkpoint-retention strategy should be established before starting a new experiment.

### 3.3 Teacher Model Evaluation

After training, each teacher should be evaluated against its corresponding Base model under the same input format, sampling parameters, and generation budget.

Use the existing generation script:

```text
experiments/prodipa/conditional_evaluation/generate_model.py
```

This script supports both conditional and unconditional generation paradigms.

The generated sequences should then be evaluated using the existing property-scoring scripts.

The evaluation should cover:

1. Sequence validity and amino acid composition.
2. Natural end-of-sequence (EOS) termination.
3. Sequence length distributions.
4. Sequence repetition and redundancy.
5. Structural quality, including pLDDT, pTM, and pAE.
6. Predicted solubility (Sol).
7. Predicted thermostability (Thermo).
8. Improvements in the target property relative to the corresponding Base model, stratified by sequence length.

A decreasing training loss alone does not establish that a teacher model is effective.

Teacher models should only be considered suitable for subsequent student experiments after their generation quality and target-property improvements have been evaluated. A teacher that fails these checks should not be directly used in the formal student comparisons.

## 4. Student Training and Online/Offline Comparisons

### 4.1 Initial Multi-Teacher Student Models

The initial conditional and unconditional student experiments can be launched using the following commands:

**Conditional student:**

```bash
TRAIN_GPUS=0,1 EVAL_GPUS=0,1,2,3 \
  bash experiments/prodipa/conditional_multiteacher_student/run_all.sh
```

**Unconditional student:**

```bash
TRAINING_GPUS=0,1 \
  bash experiments/prodipa/unconditional_student_training/run_training.sh
```

Before running the conditional experiments, verify that the three adapter paths in the YAML configurations point to the intended teacher checkpoints.

The existing conditional shell scripts still use compatibility entry points from:

```text
third_party/ProteinOPD/conditional/proteinopd/
```

This directory should not be removed, since the current experiments depend on parts of its source code.

The unconditional experiments include several different training protocols, including:

- FASTA60 teacher-based training.
- The original student model.
- Unwrapped student variants.
- Length-controlled student variants.

These variants do not represent a single unified experimental comparison. Their differences in initialization and training protocols should be retained when interpreting the results.

The follow-up experiment scripts and related fixes are maintained under:

```text
experiments/prodipa/unconditional_student_training/
```

Historical student checkpoints that are no longer available cannot be used directly for continuation. For experiments that depend on historical pretrained student models, their initial states must first be reconstructed, and the additional training cost should be reported.

### 4.2 Online Distillation and Fixed Offline Cache

The main conditional online/offline comparison can be launched using:

```bash
TRAIN_GPUS=0,1 EVAL_GPUS=0,1,2,3 \
  bash experiments/prodipa/offline_multiteacher_opd/run_core_experiment.sh
```

This comparison requires the following conditional student checkpoint to exist beforehand:

```text
outputs/conditional/prollama_multiteacher_student/checkpoint-192
```

Before running the experiment, verify the consistency of:

- Student warm-start checkpoints.
- Optimizer states.
- Teacher model checkpoints.
- Rollout information.

The offline cache is generated using `build_cache.py` and is not included in the packaged repository.

Importantly, this experiment uses full-vocabulary Product-of-Experts (PoE) and Jensen-Shannon Divergence (JSD). It should therefore be described as full-vocabulary distillation rather than top-k distillation.

Other cross-paradigm experiment entry points are available in the same directory. Refer to the accompanying README and `run_full_training.sh` for the relevant commands, and check the required checkpoint dependencies before execution.

## 5. Quality-Aware Cache Experiments

The quality-aware cache experiments investigate how the selection and updating of cached training samples affect student learning and computational cost.

The current experiment can be launched using:

```bash
ADAPTIVE_PARADIGMS=conditional ADAPTIVE_GPUS=0,1,2,3 \
  python experiments/prodipa/adaptive_cache/run.py
```

This experiment should be performed only after the preceding warm-start, paired online training, and initial cache preparation have been completed.

The historical execution progress described in the directory README does not necessarily represent the current state of the packaged experiments.

For a meaningful comparison, fixed, periodic, and adaptive caching strategies should use the same continuation budget.

The following quantities should be recorded:

- Initial cache construction cost.
- Oracle-related cost.
- Teacher forward-pass cost.
- Student training cost.
- Costs associated with failed runs and restarts.

For the unconditional paradigm, the experimental starting point must be re-established using the corresponding current teacher and student protocols. The cache from a historically paused experiment should not be reused directly, since its training conditions may not match the current setup.

## 6. Unified Evaluation and Main Results

### 6.1 Evaluation Procedures

Conditional and unconditional models are evaluated using their respective existing evaluation scripts.

**Conditional evaluation:**

```bash
bash experiments/prodipa/conditional_evaluation/run_method_evaluation.sh \
  --model-id METHOD_ID \
  --generation-tsv /absolute/path/to/generation.tsv \
  --gpus 0,1,2,3
```

**Unconditional evaluation:**

```bash
bash experiments/prodipa/unconditional_evaluation/run_method_evaluation.sh \
  --model-id METHOD_ID \
  --generation-tsv /absolute/path/to/generation.tsv \
  --gpus 0,1,2,3
```

Check the current command-line arguments using `--help` before execution.

The main scoring scripts are:

| Script | Evaluation |
|---|---|
| `scripts/run_esmfold_properties.py` | Structure prediction and structural quality assessment |
| `scripts/run_protein_sol.py` | Predicted protein solubility |
| `scripts/run_temBERTure_cls.py` | TemBERTure-based thermostability classification |

The automatic 35 GiB memory-based switching option has not been approved for these experiments and should remain disabled.

For consistent comparisons, retain the original ESMFold recycling and numerical precision settings, and record the actual parameters used during evaluation.

Results obtained under different scoring profiles should not be silently combined into a single evaluation table.

### 6.2 Main Results and Hypervolume Analysis

The current hypervolume (HV) definitions are preserved as configuration files under:

```text
configs/reproduction/*_hv_definition.json
```

These files specify the evaluation objectives used in the original experiments. The scaling parameters from previous runs should not be reused as new training signals.

For the conditional experiments, the four objectives are:

| Objective | Optimization direction |
|---|---|
| pLDDT | Maximize |
| pAE | Minimize |
| Solubility (Sol) | Maximize |
| Thermostability (Thermo) | Maximize |

The corresponding four-dimensional objective space is therefore:

\[
\mathrm{HV} =
\mathrm{HV}(
\mathrm{pLDDT}\uparrow,
\mathrm{pAE}\downarrow,
\mathrm{Sol}\uparrow,
\mathrm{Thermo}\uparrow
)
\]

The main results table and computational cost curves should only be regenerated after per-sequence evaluation has been completed for all methods included in the comparison.

Hypervolume must be calculated from the complete set of evaluated objective vectors, using a consistent reference point and normalization scheme.

It should not be approximated by multiplying the mean property scores of a model. Such a calculation does not represent the hypervolume of the underlying multi-objective solution set.

The same evaluation definitions and reference settings should be maintained across methods to ensure that the resulting hypervolume values are directly comparable.

