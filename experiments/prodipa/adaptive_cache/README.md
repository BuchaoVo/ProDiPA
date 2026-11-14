# Quality-Aware Adaptive Cache

## Unconditional hold (2026-11-11)

The old Unconditional evaluation is paused by user request. The durable marker
`outputs/experiments/adaptive_cache/unconditional/PAUSED.json` excludes
that paradigm from orchestration and prevents new jobs/GPU allocations. Conditional
continues from saved jobs and generation shards; combined publication is withheld
while one paradigm is paused. Existing artifacts are retained, not deleted.

The candidate replacement is under
`outputs/experiments/unconditional_length_control_20261111/`. Its
`MODEL_PROMOTION_POLICY.json` records the requested conditional replacement: actual
comparable quality evidence must establish teacher/OPD success before official
models are switched. Successful training or bounded length alone is insufficient.
After promotion, rebuild Unconditional innovation-II caches and paired comparisons
in a new versioned run. Do not remove the pause marker merely to continue old
cached teacher targets with replacement weights.

Run or resume the complete experiment from the repository root:

```bash
ADAPTIVE_GPUS=0,1,2,3 __PRODIPA_ENVS__/ProDiPA/bin/python experiments/prodipa/adaptive_cache/run.py
```

The running background session is `prodipa_adaptive_cache_parallel`. Its main log is
`outputs/experiments/adaptive_cache/pipeline.log`; `STATUS.json` records
the active stage, and `logs/` contains individual training, rollout and scoring logs.
Conditional and Unconditional training run concurrently using the available GPUs
0–3 through shared allocation locks. Within each paradigm, Fixed, Periodic and
Adaptive still run in order. Each paradigm has its own `STATUS.json`; the root
status lists both throughout training and evaluation. Each paradigm starts its
independent evaluation immediately after its own training finishes; it does not
wait for the other paradigm. Both share the dynamically available GPUs 0–3.
Automatic joint reporting waits for both paradigms' evaluation to finish.
An explicitly selected, completed paradigm can be published independently:

```bash
__PRODIPA_ENVS__/ProDiPA/bin/python -u experiments/prodipa/adaptive_cache/report.py --paradigm conditional > outputs/experiments/adaptive_cache/conditional_report.log 2>&1
```

This only reads existing scores and updates the selected paper TSV and original
main TSV, plus scoped metrics, cost curves and the core report. It never launches
training, generation or scoring, and never changes the other paradigm's table or
pause marker. `REPORT_STATUS.json` records publication separately from the overall
`PARTIAL` experiment status. No `.tex` files are generated. Back up the selected
tables before publication if retaining their previous display values is required.
The orchestrator skips completed jobs on restart and resumes completed segment
checkpoints, including optimizer, scheduler and RNG state. It stops visibly on a
failed job rather than claiming an incomplete score table is final.

## Current recovery and remaining workflow

The 2026-11-10 device-selection recovery runs in `prodipa_adaptive_cache_recovery`,
with `recovery.log` and `RECOVERY_STATUS.json` under the experiment output root.
It retries only the eight failed ESMFold jobs using explicit local `cuda:0`, keeps
all previous attempts in GPU-hour accounting, and allows the existing CPU novelty
searches to finish before restarting `run.py`. Completed training and score jobs
are skipped. After handoff, the continued pipeline output is also in `recovery.log`.

The complete sequence is:

1. Resume both paradigms in parallel. Fixed and Periodic are already at their
   endpoint; finish Conditional Adaptive's third refresh and final training
   segment, and Unconditional Adaptive's initial scoring and three refreshes.
2. At each refresh, score the fixed 96-request candidate batch, apply frozen
   quality/Pareto/diversity gates, cache teacher targets only for accepted samples,
   and preserve all 128 anchor slots. A rejected batch does not trigger replacement.
3. As each paradigm finishes training, independently generate 512 requests at
   the shared reference and each method's four quarter checkpoints. Evaluate
   the reused paired Online final model under the same independent final seeds.
   This can overlap the other paradigm's remaining training and cache refreshes.
   Each paradigm records `EVALUATION_COMPLETE.json` only after all its nodes finish.
4. Score every valid independent output with the frozen predictors; retain invalid
   requests in denominators. Use separate training and evaluation score stores.
5. Produce frozen-scale final HV, applicable Retention, common-cost archive HV-AUC,
   HQ discovery, training/evaluation GPU-hours and refresh statistics. Update both
   paper tables and the original Conditional/Unconditional main tables, preserving
   newly completed baselines when compatible sequence-score evidence is available.

Intermediate training selections are not independent discoveries. Results are
not final until their requested publication scope is complete and the core report
is written. A single-paradigm report does not make the overall experiment complete.

`PROTOCOL.json` is frozen before model updates. Existing full training endpoints
are Conditional 192→256 and Unconditional 200→256. Each comparison uses the same
512 initial trajectories, initial optimizer state, OPD loss and teacher weights.
All methods run four segments with checkpoints at continuation quarters. Fixed is
replayed because its historical intermediate checkpoints were not retained.
Online uses existing paired final weights, with independent evaluation under the
same new seeds. No teacher or online retraining is performed.

Periodic and Adaptive use 128 fixed anchors and a 96-request candidate budget at
each of three refresh points. Periodic randomly accepts legal candidates. Adaptive
uses frozen quality gates, exact three-property Pareto contribution, initial-cache
quartile coverage and 3-mer Jaccard diversity. It reserves up to 24 admission slots
for diversity and never changes the distillation loss. Teacher targets are computed
only for accepted trajectories. Initial cache files are linked read-only; refresh
replaces links in experiment-owned directories. Historical caches remain intact.

The quality gates are pLDDT≥70, pAE≤10, Sol≥0.7, Thermo≥0.6, positive Novelty-U and
Novelty-T, maximum residue fraction 0.4, and maximum trimer fraction 0.2. Novelty-U
uses the project's frozen MMseqs2 search against UniProtKB 2026_03; Novelty-T uses
the teacher-training union. Missing scores cannot satisfy a gate. These are
computational prediction criteria, not claims of wet-lab viability.

Each independent evaluation node uses 512 requests, including invalid outputs.
Evaluation seeds are separate from cache seeds. The report preserves the canonical
five-objective HV definitions using copies of the existing frozen scales. Training
selection uses the explicitly separate three-property [pLDDT/100, Sol, Thermo]
scale. Novelty-U and Conditional ProTrek are never interchanged.

`jobs/*.json` records actual GPU allocation intervals, including failed attempts.
Training costs include initial cache construction, fresh rollout, quality oracles,
teacher inference, student training, model loading and GPU-held file I/O. CPU-only
work without a GPU reservation has zero GPU-hours. Evaluation jobs are separate.
Reported cost is continuation cost; the common warm-start is excluded.

After all scoring completes, `report.py` creates the selected paper `.tsv`
tables, `adaptive_cache_core_results.md`, and `adaptive_cache_hv_cost.tsv` under
`outputs/evaluation/paper/`, and merges results into the original Conditional and
Unconditional main tables. HV Retention is N/A for a nonpositive or negligible
Online gain. HV-AUC uses the shared supported cost interval, without extrapolation.
HQ discovery uses only deduplicated independent evaluation candidates passing
the frozen gate and final nondominance, with both full-run and common-cost rates.

Implementation checks (no model training):

```bash
__PRODIPA_ENVS__/ProDiPA/bin/python experiments/prodipa/adaptive_cache/test_metrics.py
```
