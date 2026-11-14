# External baseline generation adapters

This directory implements a fail-closed boundary between four official generation
systems and ProDiPA's frozen evaluation pipeline. It does not replace model code,
invent missing weights, or convert engineering checks into scientific results.

## Standard contract

Every completed method writes exactly 512 rows to `generations.jsonl`. Failed,
invalid, non-terminated, and duplicate attempts remain rows. Run
`audit_generation.py` before any scoring job. DPLM's lack of autoregressive EOS is
recorded as `diffusion_iterations_complete`, not treated as invalid.

## Current resource state (2026-11-05)

- ProGen2: official source and `progen2-base` weights are absent.
- DPLM: official source, `airkingbd/dplm_650m`, and the frozen Train25K length
  manifest are absent. The current repository no longer contains Train25K.
- ASPO: ProLLaMA Base and Train300/Test100 positives exist; official Steering-PLMs,
  real low-property TPR pools, frozen negative thresholds, and a cross-test homology
  exclusion manifest are absent.
- ProDVa: official source commit `ff2d41c...` exists, but all model/tokenizer,
  fragment-mapping, retriever, vector-index, and sequence-mapping resources are
  absent.

No formal generation is authorized while these checks are blocked.

## Commands

```bash
cd "$PRODIPA_ROOT"

python src/prodipa/baselines/audit_resources.py \
  --project-root . \
  --config configs/baselines/progen2.yaml \
  --config configs/baselines/dplm.yaml \
  --config configs/baselines/aspo_tpr.yaml \
  --config configs/baselines/prodva_tpr.yaml \
  --output outputs/baseline_resource_audit.json

python src/prodipa/baselines/run_progen2.py preflight \
  --report outputs/unconditional/external_baselines/progen2_base/preflight.json
python src/prodipa/baselines/run_dplm.py preflight \
  --report outputs/unconditional/external_baselines/dplm_650m/preflight.json
python src/prodipa/baselines/run_aspo_tpr.py preflight \
  --report outputs/conditional/tpr_external_baselines/aspo_multi/preflight.json
python src/prodipa/baselines/run_prodva_tpr.py preflight \
  --report outputs/conditional/tpr_external_baselines/prodva/preflight.json
```

After official sources and weights are installed and the preflight passes, use the
`run` action for ProGen2/DPLM. First run 8--16 attempts with a separate engineering
configuration and delete those temporary outputs after verification. Do not change
the formal 512-request configs based on engineering-test property scores.

ProDVa's official evaluator requires a `sequence` field. The adapter deliberately
sets it to the empty string for every request; no TPR Test100 sequence is supplied
to generation. ASPO remains blocked until the real negative cohorts and a pinned
multi-vector entrypoint pass audit.
