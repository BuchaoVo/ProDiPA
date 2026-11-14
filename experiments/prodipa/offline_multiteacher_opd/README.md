# Offline Multi-Teacher OPD core validation

This experiment compares online and fixed-cache Conditional multi-teacher
distillation from the same effective student checkpoint.

## Frozen paired protocol

- theta-zero: `outputs/conditional/prollama_multiteacher_student/checkpoint-192`
- teachers: current Fold, Sol, and Thermo Conditional adapters
- aggregation/loss: equal-weight geometric PoE and full-vocabulary JSD
- Online: 2,048 changing-policy trajectories, one epoch, 64 student updates
- Offline: 512 theta-zero trajectories cached once and reused for four epochs,
  64 student updates
- optimizer: fused AdamW, learning rate `2e-5`, cosine schedule, 10% warmup
- training: two GPUs, per-device batch 8, gradient accumulation 2, BF16
- evaluation: 512 requests per model, seed 42, frozen Conditional protocol

The offline cache stores complete BF16 PoE log-probability distributions over
the 32k vocabulary. It does not use top-k approximation. Before training, a
fail-closed audit compares live teacher targets and JSD with the serialized
cache on the same theta-zero batch.

## Run

```bash
cd __PRODIPA_ROOT__ && setsid -f nohup env TRAIN_GPUS=0,3 EVAL_GPUS=0,1,2,3 bash experiments/prodipa/offline_multiteacher_opd/run_core_experiment.sh >> outputs/experiments/offline_multiteacher_opd/launcher.log 2>&1 < /dev/null
```

## Tracking and results

- pipeline: `outputs/experiments/offline_multiteacher_opd/pipeline.log`
- state: `outputs/experiments/offline_multiteacher_opd/STATUS.json`
- cache ranks: `outputs/experiments/offline_multiteacher_opd/logs/cache_rank_{0,1}.log`
- online training: `outputs/experiments/offline_multiteacher_opd/logs/online_training.log`
- offline training: `outputs/experiments/offline_multiteacher_opd/logs/offline_training.log`
- cache parity: `outputs/experiments/offline_multiteacher_opd/cache/CONSISTENCY_AUDIT.json`
- paired table: `outputs/experiments/offline_multiteacher_opd/results/core_metrics.tsv`
- final report: `outputs/experiments/offline_multiteacher_opd/results/FINAL_REPORT.md`

Distillation GPU-hours sum actual per-rank allocated wall seconds. Cache
construction is included in Offline cost. Evaluation is recorded separately
and excluded from both Online and Offline distillation GPU-hours.
