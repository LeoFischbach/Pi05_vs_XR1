#!/usr/bin/env bash
# Print the arguments assembled by the actual training script, without starting it.
set -euo pipefail
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$PIPELINE_DIR/pi05/train_pi05.sh" --print-args "$@"
