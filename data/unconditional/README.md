# Canonical Unconditional teacher data

Status: **PASS, PUBLISHED**.

## Sol-Designable-V3-Strict70-15

The canonical Sol Train/Validation data were replaced by the P3 joint-homology
split on 2026-11-12.  The 300/100 rows use
length-bin quotas `[45, 132, 65, 58]` and `[15, 44, 22, 19]`; all satisfy
Sol>=0.70, pLDDT>=70 and mean pAE<=15.  Explicit Fragment records were excluded.

Sol, Fold and Thermo were jointly audited at 30% identity and 80% bidirectional
coverage.  There are zero direct Train–Validation homology hits across the three
attributes.  FASTA-60 remains the teacher input protocol.  The existing Sol
Teacher adapter predates this data revision and must be retrained before it can
be described as the Sol V3 Teacher.

Final Sol V3 provenance and audits are retained under `sol/SOL_V3_P3_*`.

## Previous balanced release notes

# Staged balanced Unconditional teacher data

Status: **PASS, PUBLISHED**. This is the canonical Unconditional teacher dataset.

This package was built from the frozen train/validation partitions with the
same Fold >= 85, Sol >= 0.70 and Thermo >= 0.60 thresholds. It adds:

- exact 300:100 train/validation ratios in every feasible length bin;
- quotas optimized against 335 valid-complete 50–512-aa ProtGPT2 Base samples;
- one sequence per homology group and zero cross-split homology overlap;
- explicit low-complexity gates;
- score-range coverage instead of top-score-only selection;
- structure-aware Sol selection with matched 72% train/validation qualification;
- full accession, structure, property and complexity provenance;
- `protgpt2_fasta60_v1` tokenization audit with no truncation.

The immutable Sol validation partition has only 48/18/5 eligible independent
groups in the 150–249/250–349/350–512 bins after complexity filtering. The
resulting Sol distribution therefore cannot exactly reproduce the Base length
distribution without changing the frozen split or lowering the property
threshold. The builder preserves the split and records this support limitation.

Rebuild only the staging package:

```bash
cd /mnt/data/users/zbc/opd/ProDiPA
PYTHONPATH=src /home/zbc/data/software/miniconda/envs/ProDiPA/bin/python \
  scripts/build_unconditional_teacher_datasets_balanced.py \
  > data/rebuild/unconditional_teacher_balanced/build.log 2>&1
```

Do not point a formal run at this directory until the staged report is accepted.
If accepted, use a fresh output root and explicitly request
`--sequence-protocol protgpt2_fasta60_v1`; historical adapters are not resumable
under the changed data and token protocol.
