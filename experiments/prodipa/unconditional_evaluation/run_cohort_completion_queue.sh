#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
PY="${PRODIPA_PYTHON:-__PRODIPA_ENVS__/ProDiPA/bin/python}"
CONFIG="${UNCONDITIONAL_EVALUATION_CONFIG:-$ROOT/configs/evaluation/unconditional.json}"
exec "$PY" "$ROOT/experiments/prodipa/unconditional_evaluation/run_cohort_completion_queue.py" \
  --project-root "$ROOT" \
  --config "$CONFIG" \
  "$@"
