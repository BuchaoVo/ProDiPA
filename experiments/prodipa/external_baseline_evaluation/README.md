# External baseline post-evaluation queue

This queue starts only after the existing 4-model teacher evaluation required by the selected
scope reports `COMPLETE`. It handles four independent, fail-closed baselines:

- Unconditional: `progen2_base`, `dplm_650m`
- Conditional TPR: `aspo_tpr`, `prodva_tpr`

Each method must pass its resource audit before receiving a GPU. Formal output must contain
exactly 512 standardized requests and pass `audit_generation.py`. The two paradigms are scored
separately with the same frozen scorers and Novelty-T references used by the existing teacher
tables. Unconditional per-method scores then enter the maintained cohort in
`configs/evaluation/unconditional.json`; Novelty-U and HV are published only after the complete
cohort is ready. Conditional rows remain separate. The result table names are:

- `unconditional_prodipa_main.tsv`
- `conditional_prodipa_main.tsv`

The queue never substitutes unavailable weights, source code, negative pools, retrieval indexes,
or training references. Populate `configs/baselines/runtime_paths.env` from the provided example;
the running queue reloads it on every resource poll.

```bash
nohup env BASELINE_SCOPE=unconditional \
  bash experiments/prodipa/external_baseline_evaluation/run_external_baseline_pipeline_queue.sh \
  > outputs/evaluation/external_baselines/launcher.log 2>&1 < /dev/null &
```

`BASELINE_SCOPE` accepts `unconditional`, `conditional`, or `both` (default). This allows the
ready Unconditional baselines to complete without waiting for Conditional-only resources.

Monitor `outputs/evaluation/external_baselines/PIPELINE_STATUS.json` and
`pipeline_queue.log`. Generated and per-sequence scored outputs remain in the external-baseline
evaluation directory. The maintained Unconditional table is
`outputs/evaluation/unconditional/results/unconditional_prodipa_main.tsv`; see
`experiments/prodipa/unconditional_evaluation/README.md` before adding another method.
