#!/usr/bin/env bash
# Prepare data through the actual pipeline; keep these check outputs under _validation/.
# Usage: bash _validation/run_overnight_no_training.sh pi05|xr1|both [--smoke-test]
set -euo pipefail
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OUTPUT_DIR="$PIPELINE_DIR/_validation/overnight_flow"
exec bash "$PIPELINE_DIR/run_overnight.sh" "$@" --prepare-only
