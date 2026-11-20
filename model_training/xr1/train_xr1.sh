#!/usr/bin/env bash
# Step 4 (XR-1): start XR-1's trainer with parameters from config.sh.
#   bash xr1/train_xr1.sh <folder with data/<name>.yaml> <name> <run folder (must not exist)>
# Parameters from ../config.sh. Full fine-tune (model=posttrain: Qwen3-VL backbone and action
# expert) from Xiaomi's pretrained Xiaomi-Robotics-1-5B, DeepSpeed, bf16 mixed precision.
#
# Start from the run folder to keep generated config and Hydra logs there.
# Our data config is found through hydra.searchpath.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
CONFIG_DIR="$(realpath "$1")"; DATA_NAME="$2"; RUN_DIR="$3"
[ -f "$CONFIG_DIR/data/$DATA_NAME.yaml" ] || { echo "no $CONFIG_DIR/data/$DATA_NAME.yaml" >&2; exit 1; }
[ ! -e "$RUN_DIR" ] || { echo "$RUN_DIR exists; choose a new run folder" >&2; exit 1; }
mkdir -p "$RUN_DIR/assets"
RUN_DIR="$(realpath "$RUN_DIR")"

extra=()
[ "$XR1_SAVE_LAST_AS_LINK" = "1" ] && extra+=(+trainer.save_last=link)
LAUNCH=(torchrun --nnodes=1 --node_rank=0 --nproc_per_node="$NUM_GPUS" --master_addr=localhost --master_port=29500)
if [ "${XR1_DRY_RUN:-0}" = "1" ]; then     # check only: print the composed config and stop
  LAUNCH=(python); extra+=(--cfg job)
fi

cd "$RUN_DIR"
export PYTHONPATH="$XR1_REPO/xr1" CUDA_HOME="$XR1_ENV" PATH="$XR1_ENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE WANDB_DIR="$RUN_DIR"
# Token budget of ONE GPU's packed batch (not per sample): XR-1's collate silently drops samples
# that do not fit. One sample of this data is ~570 tokens, batch 16 needs ~9100. 20000 = XR-1's
# scripts/train.sh; unset, the collate falls back to 4096 and trains on about 7 of 16 samples.
export MAX_LENGTH=20000

exec "${LAUNCH[@]}" "$XR1_REPO/xr1/tools/train.py" \
  "hydra.searchpath=[file://$CONFIG_DIR]" \
  "hydra.run.dir=$RUN_DIR/hydra" \
  data="$DATA_NAME" \
  model=posttrain \
  model.params.pretrained="$XR1_PRETRAINED" \
  data.params.train_datasets.batch_size="$XR1_BATCH_PER_GPU" \
  trainer.max_steps="$XR1_STEPS" \
  trainer.save_interval="$XR1_SAVE_EVERY" \
  trainer.scheduler.params.num_warmup_steps="$XR1_WARMUP_STEPS" \
  trainer.scheduler.params.max_lr="$XR1_LR" \
  trainer.scheduler.params.min_lr="$XR1_FINAL_LR" \
  +trainer.log_every_n_steps=50 \
  trainer.project=xiaomi-robotics-1 \
  trainer.exp_name="$(basename "$RUN_DIR")" \
  trainer.default_root_dir="$RUN_DIR" \
  "${extra[@]}"
