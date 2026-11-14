#!/usr/bin/env bash
set -euo pipefail

# Compatibility entrypoint. The formal implementation lives under the shared
# Unconditional evaluation standard so future methods use the same cohort logic.
ROOT="${PROJECT_ROOT:-__PRODIPA_ROOT__}"
exec bash "$ROOT/experiments/prodipa/unconditional_evaluation/run_cohort_completion_queue.sh" "$@"
