#!/usr/bin/env bash
# Submit from model_training/: sbatch pi05/train_pi05.sbatch <dataset> <new run folder>
#SBATCH --job-name=pi05-train
#SBATCH --partition=stud
#SBATCH --gres=gpu:7g.80gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --export=ALL
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from model_training/}"
export NUM_GPUS=1
exec bash pi05/train_pi05.sh "$@"
