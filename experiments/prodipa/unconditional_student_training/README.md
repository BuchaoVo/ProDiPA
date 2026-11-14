# Unconditional multi-teacher student training

## Student EOS supervision: full-sequence repair

The failed EOS-only experiment (260/512 empty requests), its dedicated scripts,
weights, outputs and diagnostic files were permanently removed at the user's
request. The first full-sequence repair completed 256 updates in
`outputs/experiments/unconditional_student_sequence_ce/`; its independent
quality evaluation has not run. Those weights predate the format/mask changes below.

### Unwrapped rollout and terminal-mask protocol

New training defaults to `SEQUENCE_FORMAT=unwrapped` (`--sequence_format unwrapped`).
The existing teacher-training CSVs and CE anchors contain contiguous residues.
Rollout sampling now blocks every byte-level BPE token containing ASCII whitespace,
including mixed residue/whitespace tokens. EOS is not blocked or boosted. This is
a **sampling-policy change**, not an aa-length limit or postprocessing edit.
Sampled token IDs remain unchanged and are passed identically to student and
teachers. PoE/JS and teacher weights remain unchanged over the full vocabulary;
the loss distributions are not renormalized over allowed sampling tokens.

Naturally sampled EOS remains supervised, even when EOS and PAD have the same
ID. Artificial EOS after token-budget exhaustion or an optional aa-cap boundary
is always masked, independently of whether aa control is enabled. Padding and
BOS are excluded by position. No artificial endpoint is a CE anchor target.

`SEQUENCE_PROTOCOL.json` is saved with each new checkpoint. A new output directory
and weights-only continuation are required when moving from older checkpoints;
resuming optimizer state across this protocol change is rejected. `legacy` is an
explicit formatting option, but still uses corrected artificial-terminal masking.
The completed run's controller remains a no-op when DONE exists; it is not an
entry point for overwriting that experiment.

Both `prodipa.unconditional.generate` and the formal
`experiments/prodipa/conditional_evaluation/generate_model.py` generator default to
`sequence_format=auto`: new checkpoint metadata selects unwrapped sampling;
old checkpoints without metadata retain legacy formatting. Explicit overrides
may not contradict recorded checkpoint metadata. Formal output records include
the new format to prevent mixing resumed shards with historical output. This
does not make old/new sampling protocols interchangeable for scientific comparison.

Logs split `train/position_{body,whitespace,natural_eos}_*` into token counts,
JS (or PG) loss, student/PoE/individual-teacher EOS probabilities, and student/PoE
whitespace probability mass. Body means non-whitespace, non-EOS target tokens;
it is not a claim of biochemical validity. Probabilities are from unconstrained
model distributions, not masked sampling logits. Counts and sums are reduced
across GPUs before computing per-microbatch token means; logging averages these
microbatch statistics, not a pooled epoch statistic. Empty groups emit count 0
and omit means (not a fabricated zero loss). Under unwrapped sampling the observed
whitespace group should be empty; its residual model probability is still tracked
on body/end positions. `train/end_{natural,forced,token_budget}_count` distinguishes
true endings from censorship. No added loss term, retraining, evaluation or main-table
replacement is launched by this code update.

The optional auxiliary objective now supervises complete training sequences:
`BOS → first residue → subsequent residues → true EOS`. It uses the existing
teacher-training proteins, never validation/test proteins or
artificially truncated generated endpoints. BOS is an input, not a target;
padding is masked by sequence length even when PAD and EOS share a token ID.
Before a new run, rebuild the deduplicated anchor CSV from the original three
teacher-training CSVs under `data/unconditional/`; the old experiment's derived
880-protein anchor CSV was removed along with its outputs.
Residue targets have weight 1, and only the true EOS target has weight `w`:

`L = JS_PoE + lambda * (sum residue CE + w * sum terminal EOS CE) / (N_residue_tokens + w * N_sequences)`.

`--eos_anchor_weight` / `EOS_ANCHOR_WEIGHT` controls `w` (at least 1; the shell
entry defaults to 4). This is **not** the coefficient relative to JS.
`--eos_anchor_loss_weight` / `EOS_ANCHOR_LOSS_WEIGHT` controls `lambda` and must
be explicitly positive when anchors are enabled; there is no implicit legacy
coefficient. Full-sequence CE has a different scale from EOS-only CE, so this
coefficient must be chosen and recorded before a new experiment. Tests verify
alignment and gradient direction, not that any particular coefficient preserves
protein quality. Teacher weights and PoE/JS remain unchanged.

A new run must use a new output directory and the pre-EOS student, e.g.
`outputs/experiments/unconditional_length_control_20261111/student`, rather than
the collapsed student. `EOS_SUPERVISION.json` records objective version
`full_sequence_weighted_ce_v2`, anchor-data hash, `w` and `lambda`. Resuming an
old EOS-only checkpoint or silently changing these settings is rejected.

Training logs separately report residue/terminal CE, weighted auxiliary loss,
BOS/nonterminal/terminal EOS probabilities, empty rate, under-50-aa rate,
over-512-aa rate and valid-complete rate. A high EOS rate or shorter mean length
alone is not success. Independent generation must still validate these rates
under the evaluation sampling settings, which differ from rollout sampling.
No teacher retraining, model promotion, main-table edit, or restart of the paused
Unconditional Adaptive Cache is part of this repair. The subsequent unwrapped
protocol above restricts whitespace sampling only.

### Full-sequence CE training run

`run_sequence_ce.py` launches a separate 256-update continuation from the
pre-EOS student with fresh optimizer/scheduler state, unchanged three teachers,
terminal weight 4 and auxiliary coefficient 0.01. The coefficient is a fixed
initial choice, not a validated optimum. Rollouts retain a 512-token safety
budget without an aa hard cap; naturally ending within 512 aa is not guaranteed.
It rebuilds complete, deduplicated anchors from the three training CSVs and
records input/code hashes before training. GPUs 1 and 2 are checked for available
headroom and reserved using the shared ProDiPA GPU locks in that completed run.
The controller now supports `--output-dir` and `--gpus auto`; automatic selection
chooses two GPUs within 0–3 with at least 64,000 MiB free and at most 30% current
utilization, preferring the most free memory. A locked/busy device is not taken.

```bash
cd __PRODIPA_ROOT__ && __PRODIPA_ENVS__/ProDiPA/bin/python -u experiments/prodipa/unconditional_student_training/run_sequence_ce.py
```

Outputs: `outputs/experiments/unconditional_student_sequence_ce/`, with
`logs/student_training.log`, `STATUS.json`, `PROTOCOL.json` and `student/`.
No automatic attribute scoring or main-table promotion is attached to this run.

The unwrapped-protocol retraining uses the **same initial student** as the first
CE repair (not its final weights), the same frozen teachers, and fresh optimizer
state. This isolates the new formatting and terminal-mask protocol within the
same 256-update/8,192-request training budget. Run it in its separate directory:

```bash
cd __PRODIPA_ROOT__ && __PRODIPA_ENVS__/ProDiPA/bin/python -u experiments/prodipa/unconditional_student_training/run_sequence_ce.py --output-dir outputs/experiments/unconditional_student_unwrapped --gpus auto
```

Its `STATUS.json`, `PROTOCOL.json`, `pipeline.log` and
`logs/student_training.log` live under `outputs/experiments/unconditional_student_unwrapped/`.
The launch validation includes a CPU-only tiny-model integration test using real
LoRA/Prefix adapters: sampling, identical teacher/student contexts, combined
loss backward, frozen teachers, and saved generation-protocol metadata. It makes
no optimizer update and does not substitute for the full scientific experiment.

This experiment distils the frozen Fold, Sol and Thermo Prefix teachers into a
single ProtGPT2 LoRA student. The three teachers are loaded on one frozen
teacher backbone and switched by adapter name. The student samples its own
rollouts, and the training target is the equally weighted geometric aggregation
of the three teacher token distributions.

The formal run keeps the frozen reference settings: 8,192 online samples,
Jensen-Shannon loss, two GPUs, BF16, LoRA rank 64/alpha 128, batch 8 per GPU,
gradient accumulation 2, learning rate 2e-5 and 512 new-token budget. No
single-property teacher is updated.

```bash
cd __PRODIPA_ROOT__
TRAINING_GPUS=2,3 nohup bash \
  experiments/prodipa/unconditional_student_training/run_training.sh \
  > outputs/unconditional/protgpt2_multiteacher_student/launcher.log 2>&1 < /dev/null &
```

The entry point validates all three Prefix-100 adapters before allocating CUDA.
It resumes from the newest complete `checkpoint-*` directory and writes
`DONE.json` only after the final student adapter, trainer state and immutable
input hashes have been saved.

For a fresh end-to-end run that continues through four-GPU generation, formal
scoring, Novelty-U and pooled HV publication, use:

```bash
TRAINING_GPUS=0,1 EVALUATION_GPUS=0,1,2,3 \
  bash experiments/prodipa/unconditional_student_training/run_full_pipeline.sh
```

## Explicit 512-aa length control and fresh training

ProtGPT2 BPE tokens can encode several residues: `max_new_tokens=512` alone is
not a 512-aa constraint. The opt-in residue processor masks a proposed token
before it crosses 512 non-whitespace amino-acid characters. It never slices a
decoded protein. A boundary-forced EOS is recorded separately from natural EOS,
is not counted as a valid complete generation, and its terminal position is
excluded from the student's OPD loss. The original PoE/JS loss is unchanged.
The historical protocol remains the default (`MAX_LENGTH_AA=0`).

The fresh-run entry point below audits all teacher data without truncation,
trains the three Prefix teachers in separate output directories, then trains a
fresh Base-initialized LoRA student using those new teachers. Teacher settings
remain max 50 epochs with patience-5 early stopping; the existing full student
budget is 8192 prompts / 2 GPUs / batch 8 / accumulation 2 / 1 epoch = 256 updates.
It uses available GPUs 0–3 with shared allocation locks and does not stop other
jobs. No terminal-loss reweighting is introduced.

```bash
cd __PRODIPA_ROOT__ && PYTHONPATH=src __PRODIPA_ENVS__/ProDiPA/bin/python -u experiments/prodipa/unconditional_student_training/run_length_control.py
```

Outputs and logs: `outputs/experiments/unconditional_length_control_20261111/`.
After training, 512 independent generation requests for each of Base, three
teachers and student use the same residue cap, with natural/forced EOS and
valid-complete counts in `LENGTH_VALIDATION.json`. This validates length control;
it does not establish attribute-quality improvement. No historical main-table
rows are overwritten. Full attribute scoring is a separate subsequent step.

Standalone formal generation enables the cap with `--generation-max-aa 512`
in `conditional_evaluation/generate_model.py` (unconditional mode only).
The module `prodipa.unconditional.generate` exposes `--enforce_aa_limit` with
`--max_length_aa 512`; these flags do not change sampling temperature or top-p.

### Post-training scoring and conditional result replacement

Run the following alongside the training controller. It waits without reserving
a GPU until `LENGTH_VALIDATION.json` exists, then uses GPU0–3 allocation locks:

```bash
cd __PRODIPA_ROOT__ && PYTHONPATH=src __PRODIPA_ENVS__/ProDiPA/bin/python -u experiments/prodipa/unconditional_student_training/evaluate_length_control.py
```

`EVALUATION_PROTOCOL.json` freezes the current HV bounds, scorers, old weight hashes
and acceptance rules before scoring. The new Base/three teachers/student and four
old adapters are evaluated with the same 512-aa cap, 512 requests and seeds. Reuse
of identical sequence scores is restricted to this run and frozen protocol. Costs
are recorded separately in `evaluation/jobs/`.

Automatic promotion requires all three teachers to retain their target attribute
versus old teachers and exceed Base, with no loss of valid-complete requests. The
student must improve HV by more than max(1e-6, 1%) over both Base and old student,
and retain old pLDDT, Sol, Thermo and valid-complete count. All metrics and scores
must be complete and finite. These are conservative single-budget decision rules,
not a statistical-significance claim. Forced EOS never counts as valid complete.

When every gate passes, stage and audit the updated Unconditional canonical and
paper tables, back up existing publication files under `publication_backup/`, and
replace tables, sequence evidence and score registry together. Failed publication
rolls back. Frozen HV bounds are retained on subsequent cohort rebuilds. Other
baselines remain; old continuation rows are labeled historical and not presented
as controls for the new student. No `.tex` files are generated and no physical
checkpoint is overwritten. If any gate fails, retain official results unchanged.

Follow `post_training_evaluation.log`, `EVALUATION_STATUS.json`, `comparison.tsv`,
and `PROMOTION_DECISION.json`. A new Unconditional innovation-II run still needs
new caches and matched initialization; the old experiment remains paused.
