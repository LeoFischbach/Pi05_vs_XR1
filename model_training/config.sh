# Paths, GPUs and training parameters for both models.
# Read by run_overnight.sh, pi05/train_pi05.sh and xr1/train_xr1.sh. The data rules (task text,
# camera crop, skipped recordings, pause cut) are in common/dataset_settings.py.

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Paths --------------------------------------------------------------------------------------
# The recording project: the folder that holds collect_data/ (raw recordings; only read).
RAW_PROJECT_DIR="/home/anton/dobot/dobot_xtrainer/datasets/ANTON_X_BOCCONI"
# Everything this pipeline writes (episode list, datasets, runs, logs).
OUTPUT_DIR="${OUTPUT_DIR:-$PIPELINE_DIR/outputs}"
# Manual frame cuts and the expected episode selection. Set REFERENCE_EPISODES="" to
# process other recordings without comparing the selection to the supplied list.
MANUAL_CUTS="$PIPELINE_DIR/common/reference/clean4_manual_cuts.json"
REFERENCE_EPISODES="$PIPELINE_DIR/common/reference/clean4_trained_episodes.json"

# XR-1: a clone of github.com/XiaomiRobotics/Xiaomi-Robotics-1 (only read), its conda env
# (xr1/setup_xr1_env.sh) and the pretrained checkpoint (Hugging Face XiaomiRobotics/Xiaomi-Robotics-1-5B).
XR1_REPO="$HOME/Xiaomi-Robotics-1"
XR1_ENV="${XR1_ENV:-/opt/miniforge3/envs/xr1}"
XR1_PRETRAINED="$XR1_REPO/xr1/pretrained_ckpt/model_states.pt"

# --- Hardware -----------------------------------------------------------------------------------
# Effective batch = NUM_GPUS x the model's BATCH_PER_GPU (no gradient accumulation).
# To preserve the approximate number of sampled frames when changing batch size:
# new_steps = old_steps x old_effective_batch / new_effective_batch, rounded up.
# Scale save intervals and warmup steps by the same ratio to preserve their sample positions.
# This preserves the sample budget, not identical optimisation behaviour.
NUM_GPUS="${NUM_GPUS:-4}"

# --- pi0.5 ------------
PI05_BASE_MODEL="lerobot/pi05_base"      # Hugging Face model id or a local pretrained_model folder
PI05_BATCH_PER_GPU=16
PI05_STEPS=17324                         # total optimisation steps
PI05_SAVE_STEPS="4331,8662,12993,17324"  # checkpoint steps
PI05_LR=2e-5                             # peak of the cosine schedule (as XR-1)
PI05_FINAL_LR=5e-6
PI05_WARMUP_STEPS=500
PI05_WEIGHT_DECAY=1e-10                  # openpi's value, i.e. off
PI05_DATALOADER_WORKERS="${PI05_DATALOADER_WORKERS:-11}"  # per GPU; about (CPU cores / GPUs) - 1

# --- XR-1 -------------------------
XR1_BATCH_PER_GPU=16
XR1_STEPS=17324                          # total optimisation steps
XR1_SAVE_EVERY=4331                      # checkpoint interval in steps
XR1_LR=2e-5                              # XR-1's defaults (configs/trainer/deepspeed.yaml)
XR1_FINAL_LR=5e-6
XR1_WARMUP_STEPS=500
# 1 only after applying xr1/patches/save_last_link.patch to the XR-1 clone (saves ~70 GB of disk:
# last.ckpt becomes a link instead of a second copy). The training itself is the same.
XR1_SAVE_LAST_AS_LINK="${XR1_SAVE_LAST_AS_LINK:-0}"

# --- Logging ------------------------------------------------------------------------------------
# Weights & Biases: "offline" writes the run into its run folder (sync later with `wandb sync`),
# "online" needs `wandb login`, "disabled" turns it off.
WANDB_MODE="offline"

# --- Short test run (./run_overnight.sh <model> --smoke-test) -----------------------------------
# Only checks that every step runs: datasets of 3 episodes, 4 training steps with batch 1, all in
# outputs/smoke_test/. Training still needs the real GPUs (a full fine-tune does not fit in 32 GB).
MAX_EPISODES=0                           # 0 = all episodes
if [ "${SMOKE_TEST:-0}" = "1" ]; then
  OUTPUT_DIR="$OUTPUT_DIR/smoke_test"
  MAX_EPISODES=3
  PI05_STEPS=4; PI05_SAVE_STEPS=4; PI05_BATCH_PER_GPU=1; PI05_WARMUP_STEPS=1
  XR1_STEPS=4; XR1_SAVE_EVERY=4; XR1_BATCH_PER_GPU=1; XR1_WARMUP_STEPS=1
fi
