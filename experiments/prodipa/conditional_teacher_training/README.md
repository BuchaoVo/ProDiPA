# Conditional teacher training

This is the only maintained Conditional teacher training workflow. It trains
three independent Fold, Sol, and Thermo LoRA teachers from the original
`__PRODIPA_MODELS__/ProLLaMA` base with closing-token loss weight 4.
Every teacher is initialized independently; no previous adapter is resumed.

## Frozen protocol

- Data: approved `data/conditional/{fold,sol,thermo}` Train300/Test100 files.
- Thresholds: Fold >= 85, Sol >= 0.70, Thermo >= 0.60.
- Training: 8 epochs for every teacher, seed 42, BF16, max 512 tokens without
  silent truncation.
- LoRA: rank 8, alpha 16, dropout 0.1, all seven projection modules.
- Optimization: AdamW, learning rate 2e-5, cosine schedule, warmup ratio 0.05,
  effective batch size 32.
- Loss: response-only causal CE with weight 4 on the true closing marker.
- Evaluation: 512 requests each for Base and three teachers, followed by the
  canonical Conditional scoring and pooled-cohort summary.

## Maintained files

- `src/prodipa/conditional/train_teacher.py`: LoRA training and evaluation.
- `src/prodipa/conditional/teacher_data.py`: tokenization, response labels,
  closing-marker mask and dynamic padding.
- `run_one_teacher.sh`: one property teacher, adapter verification and test
  diagnostics.
- `run_all.sh`: three-teacher training, generation, scoring and finalization.
- `verify_adapter.py`: adapter reload and finite-logit check.
- `quick_diagnosis.py`: teacher-forced body and termination diagnostics.

Run the complete workflow with:

```bash
cd __PRODIPA_ROOT__
nohup bash experiments/prodipa/conditional_teacher_training/run_all.sh \
  > /dev/null 2>&1 < /dev/null &
```

Formal training outputs remain under `outputs/conditional/prollama/`; formal
generation and evaluation outputs remain under `outputs/evaluation/conditional/`.
The pipeline validates the approved data audit before allocating a GPU and
refuses to overwrite an incomplete nonempty adapter directory.

The complete pipeline log is
`outputs/conditional/prollama/training_pipeline.log`; no duplicate launcher
log is required.
