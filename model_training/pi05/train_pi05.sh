#!/usr/bin/env bash
# Step 4 (pi0.5): start LeRobot training with parameters from config.sh.
#   bash pi05/train_pi05.sh [--print-args] <dataset folder> <new run folder>
# Parameters from ../config.sh. Full fine-tune of pi0.5 (vision encoder, language model and action
# expert), relative joint actions, trainable weights in float32 under bf16 autocast.
set -euo pipefail
PRINT_ARGS=0
if [ "${1:-}" = --print-args ]; then PRINT_ARGS=1; shift; fi
[ "$#" = 2 ] || { echo "usage: $0 [--print-args] <dataset folder> <new run folder>" >&2; exit 2; }
source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"
DATASET="$(realpath "$1")"; RUN_DIR="$2"
[ ! -e "$RUN_DIR" ] || { echo "$RUN_DIR exists; choose a new run folder" >&2; exit 1; }

# Colour and sharpness jitter on every camera (LeRobot's image_transforms, up to 3 per sample).
COLOUR_JITTER='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.8,1.2]}},'
COLOUR_JITTER+='"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.8,1.2]}},'
COLOUR_JITTER+='"saturation":{"weight":1.0,"type":"ColorJitter","kwargs":{"saturation":[0.8,1.2]}},'
COLOUR_JITTER+='"hue":{"weight":1.0,"type":"ColorJitter","kwargs":{"hue":[-0.02,0.02]}},'
COLOUR_JITTER+='"sharpness":{"weight":1.0,"type":"SharpnessJitter","kwargs":{"sharpness":[0.5,1.5]}}}'

if [ "$NUM_GPUS" -gt 1 ]; then
  LAUNCH=(accelerate launch --multi_gpu --num_machines=1 --num_processes="$NUM_GPUS"
          --mixed_precision=bf16 --dynamo_backend=no)
else
  LAUNCH=(python)
fi

cd "$PIPELINE_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# --tolerance_s 0.005: LeRobot packs many episodes into one mp4; deep into a long file float
# rounding puts frames ~0.1 ms off its default tolerance. Frames are 33 ms apart.
# PyAV provides video decoding independently of torchcodec's CUDA libraries.
TRAIN_ARGS=(
  --dataset.repo_id="dobot/$(basename "$DATASET")" \
  --dataset.root="$DATASET" \
  --dataset.eval_split=0.0 \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --dataset.image_transforms.random_order=true \
  --dataset.image_transforms.tfs="$COLOUR_JITTER" \
  --fixed_camera_jitter.degrees=1 \
  --fixed_camera_jitter.translate=0.01 \
  --episode_repeats="$DATASET/meta/episode_sources.json" \
  --policy.type=pi05 \
  --policy.pretrained_path="$PI05_BASE_MODEL" \
  --policy.dtype=bfloat16 \
  --policy.use_relative_actions=true \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=false \
  --policy.gradient_checkpointing=true \
  --policy.optimizer_lr="$PI05_LR" \
  --policy.scheduler_decay_lr="$PI05_FINAL_LR" \
  --policy.scheduler_warmup_steps="$PI05_WARMUP_STEPS" \
  --policy.scheduler_decay_steps="$PI05_STEPS" \
  --policy.optimizer_weight_decay="$PI05_WEIGHT_DECAY" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size="$PI05_BATCH_PER_GPU" \
  --steps="$PI05_STEPS" \
  --save_freq="$PI05_STEPS" \
  --save_steps="$PI05_SAVE_STEPS" \
  --num_workers="$PI05_DATALOADER_WORKERS" \
  --log_freq=25 \
  --eval_steps=0 \
  --env_eval_freq=0 \
  --tolerance_s=0.005 \
  --wandb.enable="$([ "$WANDB_MODE" = disabled ] && echo false || echo true)" \
  --wandb.mode="$WANDB_MODE" \
  --wandb.project=anton_x_bocconi \
  --wandb.disable_artifact=true \
  --output_dir="$RUN_DIR" \
  --job_name="$(basename "$RUN_DIR")"
)

if [ "$PRINT_ARGS" = 1 ]; then
  printf '%s\n' "${TRAIN_ARGS[@]}"
  exit 0
fi
exec "$PIPELINE_DIR/.venv/bin/${LAUNCH[0]}" "${LAUNCH[@]:1}" -m pi05.train_pi05 "${TRAIN_ARGS[@]}"
