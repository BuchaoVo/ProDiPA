#!/usr/bin/env bash
set -euo pipefail
ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
RUNTIME_ENV="${BASELINE_RUNTIME_ENV_FILE:-$ROOT/configs/baselines/runtime_paths.env}"
if [[ -f "$RUNTIME_ENV" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV"
  set +a
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" "$ROOT/experiments/prodipa/conditional_baselines/run_requested_queue.py" \
  --project-root "$ROOT" "$@"
