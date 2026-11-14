#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
REFERENCE_ROOT="${UNIPROTKB_REFERENCE_ROOT:-__PRODIPA_DATABASES__/uniprotkb/2026_03}"
MMSEQS="${MMSEQS_BIN:-$ROOT/tools/mmseqs/bin/mmseqs}"
THREADS="${UNIPROTKB_THREADS:-32}"
MIN_FREE_GIB="${UNIPROTKB_MIN_FREE_GIB:-180}"
SOURCE="$REFERENCE_ROOT/source"
DB_ROOT="$REFERENCE_ROOT/mmseqs"
DB="$DB_ROOT/uniprotkb_2026_03"
STATUS="$REFERENCE_ROOT/STATUS.json"
LOG="$REFERENCE_ROOT/prepare.log"

mkdir -p "$SOURCE" "$DB_ROOT"
exec 7>"$REFERENCE_ROOT/.prepare.lock"
flock -n 7 || { echo "another UniProtKB preparation is active" >&2; exit 1; }
exec > >(tee -a "$LOG") 2>&1

write_status() {
  local state="$1" stage="$2"
  python - "$STATUS" "$state" "$stage" "$DB" <<'PY'
import json, os, sys, tempfile
from datetime import datetime
from pathlib import Path
path, state, stage, database = map(str, sys.argv[1:5])
target = Path(path)
value = {
    "state": state,
    "stage": stage,
    "release": "UniProtKB 2026_03",
    "database_prefix": database,
    "updated_at": datetime.now().astimezone().isoformat(),
}
with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False, prefix=f".{target.name}.") as h:
    json.dump(value, h, indent=2); h.write("\n"); temporary = h.name
os.replace(temporary, target)
PY
}
stage=startup
on_error() { local rc=$?; write_status FAILED "$stage"; exit "$rc"; }
on_stop() { write_status STOPPED "$stage"; exit 143; }
trap on_error ERR
trap on_stop INT TERM

if [[ -s "$REFERENCE_ROOT/READY" && -s "$DB" && -s "$DB.dbtype" ]]; then
  write_status COMPLETE complete
  echo "UniProtKB MMseqs2 database is already ready: $DB"
  exit 0
fi

available_gib="$(df -Pk "$REFERENCE_ROOT" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
if (( available_gib < MIN_FREE_GIB )); then
  echo "insufficient free disk: ${available_gib} GiB available, ${MIN_FREE_GIB} GiB required" >&2
  exit 2
fi

stage=download
write_status RUNNING "$stage"
unset HTTPS_PROXY HTTP_PROXY ALL_PROXY https_proxy http_proxy all_proxy
wget -c --timeout=60 --tries=40 --progress=dot:giga \
  https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz \
  -O "$SOURCE/uniprot_sprot.fasta.gz"
wget -c --timeout=60 --tries=40 --progress=dot:giga \
  https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz \
  -O "$SOURCE/uniprot_trembl.fasta.gz"

stage=verify_download
write_status RUNNING "$stage"
printf '%s  %s\n' \
  bc9d398533e6df582b563c6c03093bd0 "$SOURCE/uniprot_sprot.fasta.gz" \
  80cf05f7c90e0a7ed33c9498d7be1df9 "$SOURCE/uniprot_trembl.fasta.gz" \
  | md5sum --check --status
[[ "$(stat -c %s "$SOURCE/uniprot_sprot.fasta.gz")" == 93801562 ]]
[[ "$(stat -c %s "$SOURCE/uniprot_trembl.fasta.gz")" == 40628134612 ]]

stage=create_database
write_status RUNNING "$stage"
"$MMSEQS" createdb \
  "$SOURCE/uniprot_sprot.fasta.gz" "$SOURCE/uniprot_trembl.fasta.gz" "$DB" \
  --threads "$THREADS" --compressed 1
[[ -s "$DB" && -s "$DB.dbtype" ]]

stage=manifest
write_status RUNNING "$stage"
python - "$REFERENCE_ROOT/manifest.json" "$DB" "$MMSEQS" <<'PY'
import json, os, subprocess, sys, tempfile
from datetime import datetime
from pathlib import Path
target, database, mmseqs = map(Path, sys.argv[1:4])
files = sorted(database.parent.glob(database.name + "*"))
value = {
    "reference": "UniProtKB",
    "release": "2026_03",
    "release_date": "2026-09-03",
    "source": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/",
    "source_files": {
        "uniprot_sprot.fasta.gz": {"bytes": 93801562, "md5": "bc9d398533e6df582b563c6c03093bd0"},
        "uniprot_trembl.fasta.gz": {"bytes": 40628134612, "md5": "80cf05f7c90e0a7ed33c9498d7be1df9"},
    },
    "mmseqs_version": subprocess.check_output([str(mmseqs), "version"], text=True).strip(),
    "database_prefix": str(database),
    "database_files": {str(p.name): p.stat().st_size for p in files if p.is_file()},
    "created_at": datetime.now().astimezone().isoformat(),
}
with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False, prefix=f".{target.name}.") as h:
    json.dump(value, h, indent=2); h.write("\n"); temporary = h.name
os.replace(temporary, target)
PY

# The verified compressed downloads are no longer needed once MMseqs2 has ingested them.
unlink "$SOURCE/uniprot_sprot.fasta.gz"
unlink "$SOURCE/uniprot_trembl.fasta.gz"
date --iso-8601=seconds > "$REFERENCE_ROOT/READY"
stage=complete
write_status COMPLETE "$stage"
echo "UniProtKB 2026_03 MMseqs2 reference ready: $DB"
