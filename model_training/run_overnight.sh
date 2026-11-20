#!/usr/bin/env bash
# Raw recordings -> datasets -> training, for pi0.5, XR-1 or both (one after the other).
#
#   ./run_overnight.sh pi05|xr1|both [--smoke-test] [--prepare-only]
#
# Steps (each logs to outputs/logs/<start time>_<step>.log):
#   1     select episodes             common/select_episodes.py      -> outputs/episodes/
#   pi0.5 2 LeRobot dataset           pi05/build_lerobot_dataset.py  -> outputs/pi05/dataset/
#         3 relative-action stats     pi05/compute_relative_action_stats.py
#         4 training                  pi05/train_pi05.sh             -> outputs/pi05/runs/<run>/
#   XR-1  2 XR-1 dataset              xr1/build_xr1_dataset.py       -> outputs/xr1/dataset/
#         3 normalisation stats       xr1/compute_xr1_stats.py       -> outputs/xr1/configs/data/
#         4 training                  xr1/train_xr1.sh               -> outputs/xr1/runs/<run>/
#
# Reuses existing dataset directories, episode files and statistics markers by existence only;
# it does not verify their contents or check for changed settings. Builders discard leftover
# .building directories when rebuilding. Training uses a timestamped run path and refuses an
# existing one. --prepare-only runs data preparation without GPU training. Stops at the first error.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL="${1:-}"
case "$MODEL" in pi05|xr1|both) ;; *) echo "usage: $0 pi05|xr1|both [--smoke-test] [--prepare-only]" >&2; exit 2 ;; esac
shift
PREPARE_ONLY=0
for option in "$@"; do
  case "$option" in
    --smoke-test) export SMOKE_TEST=1 ;;
    --prepare-only) PREPARE_ONLY=1 ;;
    *) echo "unknown option $option" >&2; exit 2 ;;
  esac
done
source ./config.sh

[ -x .venv/bin/python ] || { echo "no .venv: run 'uv sync' in $(pwd) first" >&2; exit 1; }
NAME="$(.venv/bin/python -c 'from common.dataset_settings import DATASET_NAME; print(DATASET_NAME)')"
START="$(date +%Y%m%d_%H%M%S)"
LOGS="$OUTPUT_DIR/logs"
mkdir -p "$LOGS"

log () { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOGS/${START}_overnight.log"; }
STEP="start"
trap 'log "FAILED in step: $STEP (see $LOGS/${START}_*.log)"' ERR
# run_step <name> <command...>: runs the command, its output into a log of its own
run_step () {
  STEP="$1"; shift
  log "$STEP ... (log: ${START}_${STEP}.log)"
  "$@" > "$LOGS/${START}_${STEP}.log" 2>&1
  log "$STEP done: $(tail -1 "$LOGS/${START}_${STEP}.log")"
}
limit=(); [ "$MAX_EPISODES" -gt 0 ] && limit=(--max-episodes "$MAX_EPISODES")

log "pipeline $MODEL for $NAME$([ "${SMOKE_TEST:-0}" = 1 ] && echo ' (SMOKE TEST)'), outputs in $OUTPUT_DIR"
if [ "$PREPARE_ONLY" = 0 ]; then
  gpus="$(nvidia-smi -L 2>/dev/null | wc -l)"
  [ "$gpus" -ge "$NUM_GPUS" ] || { log "config.sh asks for $NUM_GPUS GPUs, this machine has $gpus"; exit 1; }
fi

# --- 1: episodes (shared by both models) -------------------------------------------------------
EPISODES="$OUTPUT_DIR/episodes/${NAME}_episodes.json"
if [ -f "$EPISODES" ]; then
  log "1 episodes: reusing $EPISODES"
else
  reference=(); [ -n "$REFERENCE_EPISODES" ] && reference=(--reference "$REFERENCE_EPISODES")
  run_step 1_select_episodes .venv/bin/python -m common.select_episodes \
      --raw-project-dir "$RAW_PROJECT_DIR" --manual-cuts "$MANUAL_CUTS" --output "$EPISODES" "${reference[@]}"
fi

# --- pi0.5 -------------------------------------------------------------------------------------
if [ "$MODEL" = pi05 ] || [ "$MODEL" = both ]; then
  DATASET="$OUTPUT_DIR/pi05/dataset/$NAME"
  if [ -d "$DATASET" ]; then
    log "pi05 2 dataset: reusing $DATASET"
  else
    run_step pi05_2_lerobot_dataset .venv/bin/python -m pi05.build_lerobot_dataset \
        --episodes "$EPISODES" --output "$DATASET" "${limit[@]}"
  fi
  if [ -f "$DATASET/meta/relative_actions.json" ]; then
    log "pi05 3 relative-action stats: already in $DATASET"
  else
    run_step pi05_3_relative_action_stats .venv/bin/python -m pi05.compute_relative_action_stats --dataset "$DATASET"
  fi
  if [ "$PREPARE_ONLY" = 1 ]; then
    log "pi05 data prepared; training skipped (--prepare-only)"
  else
    RUN="$OUTPUT_DIR/pi05/runs/pi05_${NAME}_${START}"
    mkdir -p "$(dirname "$RUN")"
    run_step pi05_4_training bash pi05/train_pi05.sh "$DATASET" "$RUN"
    log "pi05 checkpoints: $RUN/checkpoints/"
  fi
fi

# --- XR-1 --------------------------------------------------------------------------------------
if [ "$MODEL" = xr1 ] || [ "$MODEL" = both ]; then
  XR1_DATASET="$OUTPUT_DIR/xr1/dataset/$NAME"
  XR1_CONFIGS="$OUTPUT_DIR/xr1/configs"
  if [ -d "$XR1_DATASET" ]; then
    log "xr1 2 dataset: reusing $XR1_DATASET"
  else
    run_step xr1_2_dataset .venv/bin/python -m xr1.build_xr1_dataset \
        --episodes "$EPISODES" --output "$XR1_DATASET" "${limit[@]}"
  fi
  if [ -f "$XR1_CONFIGS/data/$NAME.yaml" ]; then
    log "xr1 3 statistics: reusing $XR1_CONFIGS/data/$NAME.yaml"
  else
    run_step xr1_3_statistics env PYTHONPATH="$XR1_REPO/xr1" "$XR1_ENV/bin/python" xr1/compute_xr1_stats.py \
        --json-dir "$XR1_DATASET/json" --out "$XR1_CONFIGS/data/$NAME.yaml"
  fi
  if [ "$PREPARE_ONLY" = 1 ]; then
    log "xr1 data prepared; training skipped (--prepare-only)"
  else
    RUN="$OUTPUT_DIR/xr1/runs/xr1_${NAME}_${START}"
    mkdir -p "$(dirname "$RUN")"
    run_step xr1_4_training bash xr1/train_xr1.sh "$XR1_CONFIGS" "$NAME" "$RUN"
    log "xr1 checkpoints: $RUN/project_xiaomi-robotics-1/$(basename "$RUN")/"
  fi
fi

log "all done"
