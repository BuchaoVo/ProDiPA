<div align="center">

# ProDiPA

**Multi-Objective Protein Design via Multi-Teacher Knowledge Distillation**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)](#1-installation)
[![Objectives](https://img.shields.io/badge/Objectives-Foldability%20%C2%B7%20Solubility%20%C2%B7%20Thermostability-0969DA?style=flat-square)](#overview)

[Overview](#overview) •
[Installation](#1-installation) •
[Path Configuration](#2-local-path-configuration) •
[External Resources](#3-external-resources) •
[Dataset Verification](#4-dataset-verification) •
[Teacher Training](#5-teacher-model-training--evaluation) •
[Student Distillation](#6-student-training--distillation) •
[Adaptive Cache](#7-quality-aware-cache-experiments) •
[Unified Evaluation](#8-unified-evaluation)

</div>

---

## Overview

**ProDiPA** is a multi-objective protein design framework using multi-teacher knowledge distillation to jointly balance **structural quality**, **solubility**, and **thermostability**. It provides standardized implementations and reproduction configurations for:

* **Generation Paradigms** — Conditional (superfamily-guided) and unconditional protein sequence generation.
* **Distillation Strategies** — Multi-teacher student initialization, online distillation, and fixed offline cache distillation.
* **Cache Optimization** — Quality-aware adaptive cache refreshing across target properties.

---

## 1. Installation

### System Requirements

* **OS:** Linux
* **Runtime:** Python `3.10`
* **Hardware:** CUDA-enabled GPU

### Setup Steps

Install a **PyTorch** build matching your local CUDA driver first, then install the remaining dependencies:

```bash
python -m pip install -r requirements-reproduction.txt
python -m pip install -e '.[dev,evaluation]'
```

> [!NOTE]
> * `configs/reproduction/observed_environment.json` records exact package versions (including CUDA PyTorch) from the archived environment.
> * `environment.yml` is provided as a historical snapshot and may not resolve cleanly across all machines.
> * Verify GPU availability after installation, as CPU dependencies can install without a working CUDA setup.

---

## 2. Local Path Configuration

Configure paths to pretrained models, sequence databases, and Conda environments before running experiments:

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

---

## 3. External Resources

Ensure all required checkpoints, tools, and databases are placed at their expected locations:

| Resource | Expected Location | Notes / Upstream Reference |
| :--- | :--- | :--- |
| **ProtGPT2** | `<models>/ProtGPT2` | Upstream: `nferruz/ProtGPT2` |
| **ProLLaMA** | `<models>/ProLLaMA` | Verify against original fingerprint and config |
| **ESMFold HF weights** | `<models>/esmfold_v1_hf` | Used by structure evaluation scripts |
| **TemBERTureCLS adapter/head** | `tools/TemBERTure/temBERTure/temBERTure_CLS/` | Weights not included |
| **ProtBERT-BFD** | Official `Rostlab/prot_bert_bfd` checkpoint | Known revision recorded in `configs/reproduction/temberture_base.json` |
| **ProTrek** | Source: `third_party/ProTrek/`<br>Weights: `<models>/ProTrek_35M` | Sequence-structure-function scoring model |
| **MMseqs2** | `tools/mmseqs/bin/mmseqs` | Or specify executable path via `MMSEQS_BIN` |
| **Novelty-U database** | `<databases>/uniprotkb/2026_03/mmseqs/uniprotkb_2026_03` | Requires database-ready marker |
| **External baselines** | `configs/baselines/` | Large baseline weights not included |

---

## 4. Dataset Verification

Use the provided dataset splits directly without regenerating them.

### Primary Benchmark Splits

| Dataset Split | Sequences | Description |
| :--- | :---: | :--- |
| **Train** | `47,342` | Main training split |
| **Validation** | `5,918` | Main validation split |
| **Test-H** | `5,918` | Held-out test benchmark |
| **Complete three-property subset** | `25,000` | Training subset with complete annotations across all three properties |
| **Prior subset (`in_prior_15k=true`)** | `15,000` | Filtered training subset |
| **Validation1K** | `1,000` | Stored separately for validation |

### Paradigm-Specific Teacher Splits

* **Conditional:** `300` training and `100` held-out sequences per property (fixed TPR superfamily condition).
* **Unconditional:** `300` training and `100` validation sequences per property (defined by the finalized Sol V3 audit and manifest)
