# TPR family-conditioned teacher datasets

**Status: COMPLETE AT THE 85/.70/.60 THRESHOLDS.** All three 300-sequence
training files and three 100-sequence evaluation files were atomically
published on 2026-11-06 after the joint six-dataset audit passed. The exact
selection is recorded in `data/provenance/conditional_teacher_manifest.tsv`;
`data/provenance/conditional_teacher_audit.json` has status `PASS`.
This package never fabricates sequences or property scores, duplicates samples to pad counts, cuts proteins to length quotas, or borrows training examples for a test set.

## Scientific specification

- Family: `G3DSA:1.25.40.10` (CATH-Gene3D); native ProLLaMA conditioning label: `Tetratricopeptide-like helical domain superfamily`.
- Prompt: `instruction = "[Generate by superfamily]"`, `input = "Superfamily=<Tetratricopeptide-like helical domain superfamily>"`.
- Current positive cutoffs based on *full-length* frozen ProDiPA scores: Fold pLDDT >=85 (0–100); Protein-Sol >=0.70; TemBERTureCLS >=0.60.
- Each JSON record contains `instruction`, `input`, `output = "Seq=<...>"`, and **one target score field** (`plddt`, `sol`, or `thermo`). The `plddt` JSON field uses the frozen 0–1 training scale and is obtained by dividing pLDDT by 100.
- Target six files: `train_plddt_300.json`, `train_sol_300.json`, `train_thermo_300.json`, `test_plddt_100.json`, `test_sol_100.json`, `test_thermo_100.json`.
- Length quotas are optimized separately for each attribute and split from the
  real independent-homology-group capacity. The primary target is the frozen
  512-request ProLLaMA TPR Base distribution (7/406/99/0 across the four bins),
  followed by one-dimensional distribution distance and Train/Test
  comparability. The resulting deterministic quotas are:

| Property / split | 50–149 | 150–249 | 250–349 | 350–512 | Total |
|---|---:|---:|---:|---:|---:|
| Fold Train | 0 | 192 | 108 | 0 | 300 |
| Fold Test | 0 | 52 | 48 | 0 | 100 |
| Sol Train | 27 | 189 | 84 | 0 | 300 |
| Sol Test | 25 | 51 | 24 | 0 | 100 |
| Thermo Train | 5 | 162 | 133 | 0 | 300 |
| Thermo Test | 2 | 43 | 55 | 0 | 100 |

Fold has no qualifying 50–149 aa group at pLDDT >=85, so its short-bin quota is
zero rather than being filled with an ineligible sequence. Sol Test also
requires a small adjustment from the marginal-capacity optimum because some
homology groups span multiple length bins. Every selected record remains a
genuine scored positive satisfying the frozen threshold.

## Split and homology safety

- An individual attribute set contains at most **one** sequence from a homology group. The same verified sequence may participate in multiple property sets **inside one split**.
- Train, Validation, and Test **must never share a homology group across any property**, enforced on the assembled TPR input table.
- Existing 176 Train25K members retain `split=train`; existing 2 Validation1K members retain `split=validation`. They are **not** relabeled as independent test records.
- Additional, newly scored candidates are assigned `split=train` or `split=test` *before* final selection and must use homology groups derived consistently with the full project corpus. Confirm against the **entire** Train25K/Validation1K and any other external/heldout data, not only the 178 included TPR sequences. Internal table split isolation cannot by itself prove external-corpus independence.
- All candidates require true target superfamily evidence. A native prompt label alone is not a biological family annotation.

## Current file availability

| Split | Fold | Sol | Thermo | Missing slots |
|---|---:|---:|---:|---:|
| Training | 300 | 300 | 300 | 0 |
| Test | 100 | 100 | 100 | 0 |

Every dataset contains the stated number of distinct sequences and distinct
homology groups. Across all three attributes together, Train and Test share
zero sequences and zero homology groups.

## Data files

- `fold/train_plddt_300.json`, `fold/test_plddt_100.json`.
- `sol/train_sol_300.json`, `sol/test_sol_100.json`.
- `thermo/train_thermo_300.json`, `thermo/test_thermo_100.json`.
- `data/provenance/conditional_teacher_manifest.tsv`: final IDs, sequences,
  splits, homology groups, attributes, proxy scores and family evidence.
- `data/provenance/conditional_teacher_audit.json`: joint audit with status
  `PASS`.
- `data/provenance/conditional_teacher_capacity_matrix.tsv`: qualifying
  sequence and independent-group capacity for all 24 split/property/bin cells.
- `data/provenance/conditional_teacher_quota_optimization.tsv`: old fixed
  quotas, Base targets, available capacity, optimized quotas and reasons.
- `data/provenance/conditional_teacher_selection_comparison.tsv`: before/after
  length distributions, target means and group counts.
- `data/provenance/conditional_teacher_build_report.json`: frozen inputs,
  hashes, seed, optimization definition and full build metrics.

## Rebuilding

The scored candidate table must include `sequence_id`, `sequence`,
`homology_group`, `split`, `cath_superfamily`, family evidence and all three
frozen proxy scores. Original validation and explicitly excluded candidates
remain outside Train/Test selection.

```bash
bash scripts/run_tpr_teacher_data_rebuild.sh
```

The build stage reuses complete score files and accepts the frozen ProLLaMA TPR
Base generation TSV as its length-distribution reference. It remains
fail-closed: all six files are first written to staging and jointly audited;
publication occurs only when every dataset reaches 300/100 and every constraint
passes. Existing published JSON files are preserved on any failure.

## Evaluation limitation

These are positive-only **teacher datasets**, not a natural-distribution test of a generated model. The requested 100-positive-per-property test sets can assess in-family teacher fit/recognition but **cannot** themselves estimate JSR, false-positive behavior, model generation quality or family-match yield. Continue using separately generated samples and full-request scoring for JSR and overall generation evaluation; ideally reserve a natural-distribution independent family holdout as well.
