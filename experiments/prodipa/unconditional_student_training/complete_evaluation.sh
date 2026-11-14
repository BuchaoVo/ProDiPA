#!/usr/bin/env bash
set -euo pipefail

ROOT="__PRODIPA_ROOT__"
GEN_ROOT="$ROOT/outputs/unconditional/protgpt2_multiteacher_student"
GEN_OUT="$GEN_ROOT/generation.tsv"
SHARD0="$GEN_ROOT/generation.shard_0_171.tsv"
SHARD1="$GEN_ROOT/generation.shard_171_342.tsv"
SHARD2="$GEN_ROOT/generation.shard_342_512.tsv"
LOCK="$GEN_ROOT/.evaluation_watcher.lock"
PYTHON="__PRODIPA_ENVS__/ProDiPA/bin/python"

exec 9>"$LOCK"
flock -n 9 || { echo "Another evaluation watcher is active"; exit 1; }

cd "$ROOT"
echo "[$(date --iso-8601=seconds)] waiting for all generation shards"
while :; do
  if [[ -f "${GEN_OUT%.tsv}.metadata.json" \
     && -f "${SHARD1%.tsv}.metadata.json" \
     && -f "${SHARD2%.tsv}.metadata.json" ]]; then
    break
  fi
  sleep 60
done

cp "$GEN_OUT" "$SHARD0"
GEN_OUT="$GEN_OUT" SHARD0="$SHARD0" SHARD1="$SHARD1" SHARD2="$SHARD2" \
  "$PYTHON" - <<'PY'
import csv
import os
from pathlib import Path

paths = [Path(os.environ[name]) for name in ("SHARD0", "SHARD1", "SHARD2")]
rows = []
fieldnames = None
for path in paths:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if fieldnames is None:
            fieldnames = reader.fieldnames
        elif reader.fieldnames != fieldnames:
            raise SystemExit(f"schema mismatch: {path}")
        rows.extend(reader)

sample_ids = [int(row["sample_id"]) for row in rows]
if sorted(sample_ids) != list(range(512)):
    raise SystemExit("generation shards do not contain exactly sample IDs 0..511")
if len({int(row["seed"]) for row in rows}) != 512:
    raise SystemExit("generation seeds are not unique")
if any(row["model"] != "protgpt2_multiteacher_student" for row in rows):
    raise SystemExit("unexpected model label in generation shards")

rows.sort(key=lambda row: int(row["sample_id"]))
out = Path(os.environ["GEN_OUT"])
temporary = out.with_suffix(".merged.tmp")
with temporary.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
temporary.replace(out)

valid = sum(row["valid_complete"].lower() == "true" for row in rows)
print(f"merged_requests={len(rows)} valid_complete={valid} valid_rate={valid / len(rows):.6f}")
PY

echo "[$(date --iso-8601=seconds)] generation merge complete; starting formal evaluation"
bash experiments/prodipa/unconditional_evaluation/run_method_evaluation.sh \
  --model-id protgpt2_multiteacher_student \
  --generation-tsv "$GEN_OUT" \
  --gpus 1,2,3 \
  --publish
echo "[$(date --iso-8601=seconds)] formal evaluation and publication complete"
