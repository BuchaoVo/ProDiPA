#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
DEST="__PRODIPA_MODELS__/external_weights/EvoDiff/oaar-640M.tar"
PART_ROOT="__PRODIPA_MODELS__/external_weights/EvoDiff/oaar-640M.parts"
URL="https://zenodo.org/api/records/8045076/files/oaar-640M.tar/content"
EXPECTED_BYTES=7716114912
EXPECTED_MD5=e6e0f2a05be4b284dd7879e30cc6c421
CHUNK_BYTES=$((256 * 1024 * 1024))
LOG_ROOT="$ROOT/outputs/evaluation/baseline_expansion/downloads"
POLL_SECONDS="${EVODIFF_RETRY_SECONDS:-10}"

mkdir -p "$PART_ROOT" "$LOG_ROOT"
exec 7>"$LOG_ROOT/.evodiff_checkpoint.lock"
flock -n 7 || { echo "EvoDiff checkpoint downloader is already active"; exit 0; }
exec > >(exec 7>&-; tee -a "$LOG_ROOT/evodiff_oaar_640m_chunked.log") 2>&1

write_status() {
  printf '{"state":"%s","stage":"%s","completed_parts":%s,"total_parts":%s,"current_part_bytes":%s,"current_part_expected_bytes":%s,"updated_at":"%s"}\n' \
    "$1" "$2" "$3" "$4" "${5:-0}" "${6:-0}" "$(date --iso-8601=seconds)" \
    > "$LOG_ROOT/EVODIFF_DOWNLOAD_STATUS.json"
}

total_parts=$(((EXPECTED_BYTES + CHUNK_BYTES - 1) / CHUNK_BYTES))
if [[ -f "$DEST" ]] && [[ "$(stat -c %s "$DEST")" -eq "$EXPECTED_BYTES" ]]; then
  observed_md5="$(md5sum "$DEST" | awk '{print $1}')"
  [[ "$observed_md5" == "$EXPECTED_MD5" ]] || {
    echo "existing EvoDiff checkpoint checksum mismatch: $observed_md5" >&2
    exit 2
  }
  write_status COMPLETE verified "$total_parts" "$total_parts" 0 0
  exit 0
fi

completed=0
for ((part=0; part<total_parts; part++)); do
  start=$((part * CHUNK_BYTES))
  end=$((start + CHUNK_BYTES - 1))
  (( end >= EXPECTED_BYTES )) && end=$((EXPECTED_BYTES - 1))
  expected=$((end - start + 1))
  final_part="$PART_ROOT/part_$(printf '%03d' "$part")"
  temporary_part="$final_part.incomplete"
  attempt_part="$temporary_part.attempt"
  if [[ -f "$final_part" ]] && [[ "$(stat -c %s "$final_part")" -eq "$expected" ]]; then
    completed=$((completed + 1))
    continue
  fi
  while true; do
    observed=0
    [[ -f "$temporary_part" ]] && observed="$(stat -c %s "$temporary_part")"
    if (( observed > expected )); then
      echo "part=$part partial file is larger than expected: $observed > $expected" >&2
      write_status FAILED "invalid_part_$part" "$completed" "$total_parts" "$observed" "$expected"
      exit 4
    fi
    if (( observed == expected )); then
      mv "$temporary_part" "$final_part"
      completed=$((completed + 1))
      break
    fi
    remaining_start=$((start + observed))
    write_status RUNNING "download_part_$part" "$completed" "$total_parts" "$observed" "$expected"
    echo "[$(date --iso-8601=seconds)] downloading part=$part range=$remaining_start-$end existing_bytes=$observed"
    : >> "$temporary_part"
    : > "$attempt_part"
    set +e
    http_code="$(curl -fL --connect-timeout 30 --max-time 1800 \
      --range "$remaining_start-$end" --output "$attempt_part" \
      --write-out '%{http_code}' "$URL")"
    rc=$?
    set -e
    attempt_bytes="$(stat -c %s "$attempt_part")"
    remaining=$((expected - observed))
    if [[ "$http_code" == "206" ]] && (( attempt_bytes <= remaining )); then
      if (( attempt_bytes > 0 )); then
        cat "$attempt_part" >> "$temporary_part"
      fi
    else
      echo "[$(date --iso-8601=seconds)] discarded part=$part attempt http=$http_code bytes=$attempt_bytes remaining=$remaining" >&2
    fi
    : > "$attempt_part"
    observed="$(stat -c %s "$temporary_part")"
    write_status RUNNING "download_part_$part" "$completed" "$total_parts" "$observed" "$expected"
    if (( rc == 0 && observed == expected )); then
      mv "$temporary_part" "$final_part"
      completed=$((completed + 1))
      break
    fi
    echo "[$(date --iso-8601=seconds)] part=$part attempt_finished rc=$rc http=$http_code bytes=$observed expected=$expected; continuing"
    sleep "$POLL_SECONDS"
  done
done

write_status RUNNING assembly "$completed" "$total_parts" 0 0
assembled="$DEST.assembled.incomplete"
truncate -s "$EXPECTED_BYTES" "$assembled"
for ((part=0; part<total_parts; part++)); do
  input="$PART_ROOT/part_$(printf '%03d' "$part")"
  dd if="$input" of="$assembled" bs=1M seek="$((part * 256))" conv=notrunc status=none
done
[[ "$(stat -c %s "$assembled")" -eq "$EXPECTED_BYTES" ]]
observed_md5="$(md5sum "$assembled" | awk '{print $1}')"
[[ "$observed_md5" == "$EXPECTED_MD5" ]] || {
  echo "assembled EvoDiff checkpoint checksum mismatch: $observed_md5" >&2
  write_status FAILED checksum "$completed" "$total_parts" 0 0
  exit 3
}
mv "$assembled" "$DEST"
write_status COMPLETE verified "$completed" "$total_parts" 0 0
echo "EvoDiff OA-DM-640M checkpoint ready: $DEST"
