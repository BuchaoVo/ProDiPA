#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
REPO="__PRODIPA_MODELS__/external_repos/progen3"
ENV_ROOT="__PRODIPA_MODELS__/external_envs/progen3"
LOG_ROOT="$ROOT/outputs/evaluation/baseline_expansion/runtime"
POLL_SECONDS="${RUNTIME_RETRY_SECONDS:-900}"
UV="${UV_BIN:-__PRODIPA_CONDA__/bin/uv}"
PIP_WHEEL="__PRODIPA_CONDA__/pkgs/python-3.10.20-h17756b0_1/lib/python3.10/ensurepip/_bundled/pip-23.0.1-py3-none-any.whl"
SETUPTOOLS_WHEEL="__PRODIPA_CONDA__/pkgs/python-3.10.20-h17756b0_1/lib/python3.10/ensurepip/_bundled/setuptools-79.0.1-py3-none-any.whl"
mkdir -p "$LOG_ROOT"
exec 7>"$LOG_ROOT/.progen3_runtime.lock"
flock -n 7 || { echo "ProGen3 runtime setup is already active"; exit 0; }
exec > >(exec 7>&-; tee -a "$LOG_ROOT/progen3_setup.log") 2>&1

write_status() {
  printf '{"state":"%s","stage":"%s","updated_at":"%s"}\n' \
    "$1" "$2" "$(date --iso-8601=seconds)" > "$LOG_ROOT/PROGEN3_RUNTIME_STATUS.json"
}

runtime_ready() {
  [[ -x "$ENV_ROOT/bin/python" ]] && \
    PYTHONPATH="$REPO/src" "$ENV_ROOT/bin/python" -c \
      'import flash_attn, megablocks, progen3, torch, transformers; assert transformers.__version__.startswith(("4.42", "4.43", "4.44", "4.45", "4.46", "4.47", "4.48")); print(torch.__version__, transformers.__version__)'
}

while ! runtime_ready; do
  write_status RUNNING environment_install
  set +e
  if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
    "$UV" venv --python /usr/bin/python3.10 --seed "$ENV_ROOT"
  fi
  if [[ ! -x "$ENV_ROOT/bin/pip" ]]; then
    "$ENV_ROOT/bin/python" "$PIP_WHEEL/pip" install --no-index \
      "$PIP_WHEEL" "$SETUPTOOLS_WHEEL"
  fi
  "$ENV_ROOT/bin/pip" install \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    -e "$REPO" > >(exec 7>&-; tee -a "$LOG_ROOT/progen3_dependencies.log") 2>&1
  dependency_rc=$?
  if (( dependency_rc == 0 )); then
    CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=4 "$ENV_ROOT/bin/pip" install \
      'megablocks[gg]==0.7.0' --no-build-isolation \
      > >(exec 7>&-; tee -a "$LOG_ROOT/progen3_megablocks.log") 2>&1
    megablocks_rc=$?
  else
    megablocks_rc=1
  fi
  if (( megablocks_rc == 0 )); then
    CUDA_HOME=/usr/local/cuda-12.8 MAX_JOBS=4 "$ENV_ROOT/bin/pip" install \
      'flash-attn==2.7.4.post1' --no-build-isolation \
      > >(exec 7>&-; tee -a "$LOG_ROOT/progen3_flash_attn.log") 2>&1
    flash_rc=$?
  else
    flash_rc=1
  fi
  set -e
  if (( dependency_rc == 0 && megablocks_rc == 0 && flash_rc == 0 )) && runtime_ready; then
    break
  fi
  write_status WAITING retry
  echo "[$(date --iso-8601=seconds)] runtime setup incomplete; retrying in ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
done

write_status COMPLETE complete
date --iso-8601=seconds > "$LOG_ROOT/PROGEN3_RUNTIME_READY"
echo "ProGen3 runtime is ready"
