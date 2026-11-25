# Pi05 vs XR-1

This repository compares two robot models, **$\pi_{0.5}$ (Pi05)** and **Xiaomi-Robotics-1 (XR-1)**, on a dual-arm task: grasp a pin, hand it to the other arm, and place it in a tray. It contains the scientific report, training code, small sample datasets, and analyses of the real-world rollouts.

## Repository structure

| Path | Contents |
| --- | --- |
| `README.md` | This overview and the HPC instructions. |
| `model_training/` | Prepares recordings and fine-tunes both models. |
| `model_training/common/` | Shared rules for episode selection, camera images, and robot states. `dataset_settings.py` defines the task text and image crop, among other settings; `select_episodes.py` selects usable recordings. |
| `model_training/common/reference/` | Manual frame cuts and the expected episode list for the dataset used in the report. |
| `model_training/pi05/` | Builds the LeRobot dataset, computes statistics for relative actions, and starts Pi05 training. |
| `model_training/xr1/` | Builds XR-1 JSON files and videos, computes normalization statistics, and starts XR-1 training. `env/` records exact package versions; `patches/` contains an optional patch that saves checkpoint disk space. |
| `model_training/_validation/` | Three prepared episodes for each model and `smoke_test.py` to check the sample data. XR-1 videos are in `xr1_dataset_3ep/videos/`, with their descriptions in `json/`. For Pi05, values are in `data/`, metadata in `meta/`, and camera videos in `videos/`; `pi05_dataset_3ep_rel/` contains the relative actions needed for training. |
| `model_training/config.sh` | Paths, GPU count, and training settings. `run_overnight.sh` runs the full pipeline from raw recordings through training. `pyproject.toml` and `uv.lock` define the Python environment for Pi05 and shared data preparation. |
| `model_rollout/raw_data/` | CSV files with rollout results and training losses. |
| `model_rollout/*.ipynb` | Notebooks for evaluation by training epoch and for robustness tests. |
| `Results/epoch_evaluation/` and `Results/robustness_evaluation/` | Figures produced from those evaluations. |
| `report/` | LaTeX report (`main.tex`), references (`references.bib`), CVPR style files, and `images/robot_setup.png`. `report/README.md` explains how to build the PDF. |
| `Papers/` | PDFs of the two model papers. On some local file systems, this folder appears as `papers/`. |

The full **raw recordings and model weights are not in the Git repository**. The included three-episode sample datasets are enough for the short training runs below, but not for reproducing the report results. `model_training/outputs/` is created when the pipeline runs; it holds generated datasets, logs, and checkpoints.

## Pi05 and XR-1 sample training on the Bocconi HPC

The commands in this section use the **included three episodes** for four-step smoke tests; they do not reproduce the models in the report. The Pi05 smoke test completed on Bocconi. The XR-1 smoke test has not completed there: the latest one-GPU attempt failed at the first optimizer step. Unless stated otherwise, run commands from `model_training/`. Replace `BOCCONI_ID` in the first command with your HPC username. Downloads and training require access to Hugging Face.

### 1. Log in and open a GPU session

```bash
ssh BOCCONI_ID@slogin.hpc.unibocconi.it
git clone https://github.com/LeoFischbach/Pi05_vs_XR1.git "$HOME/Pi05_vs_XR1"
cd "$HOME/Pi05_vs_XR1/model_training"
srun --partition=stud --gres=gpu:7g.80gb:1 --cpus-per-task=6 --mem=96G --time=04:00:00 --pty bash
cd "$HOME/Pi05_vs_XR1/model_training"
nvidia-smi
```

`srun` requests a compute node with one 80 GB MIG GPU for the Pi05 smoke test. The XR-1 full fine-tune also ran out of memory on one 80 GB MIG GPU with batch size 1 and no optimizer offload (Bocconi job 787898, 23 November 2026). Its first Adam step tried to allocate another 17.60 GiB with only 6.74 GiB free. The successful GCP run used four A100 80 GB GPUs with DeepSpeed ZeRO-2, which shards optimizer state and gradients across GPUs. Four GPUs are the validated configuration for that run; the minimum GPU count on Bocconi has not been tested. The GPU allocation comes from Slurm's `--gres` request, not `NUM_GPUS`. Run training on a compute node, not the login node. Skip `git clone` if you already have the repository. The Slurm request must fit the available partition and your time limit.

### 2. Install required programs and packages

The compute node needs `git`, `ffmpeg`/`ffprobe`, `uv`, `conda`, a C++ compiler, and CUDA-capable NVIDIA GPUs. The **Pi05 environment** uses Python 3.12 and is installed from `pyproject.toml` and `uv.lock`. Its main packages include `lerobot[pi,training]`, PyTorch, `torchvision`, `torchcodec`, `transformers`, `accelerate`, OpenCV, NumPy, Pandas, and `wandb`.

```bash
python3 -m pip install --user uv
"$HOME/.local/bin/uv" sync --frozen
command -v ffmpeg
command -v ffprobe
command -v conda
```

The **XR-1 environment** is separate. `xr1/setup_xr1_env.sh` installs the packages pinned in `xr1/env/`, including Python 3.10, PyTorch 2.8, CUDA 12.8, `transformers` 4.57.1, DeepSpeed, Lightning, Decord, and FlashAttention. The script also downloads the required Qwen processor files. Installation needs network access and disk space.

```bash
git clone https://github.com/XiaomiRobotics/Xiaomi-Robotics-1.git "$HOME/Xiaomi-Robotics-1"
export XR1_ENV="$HOME/xr1_env"
bash xr1/setup_xr1_env.sh "$XR1_ENV"
mkdir -p "$HOME/Xiaomi-Robotics-1/xr1/pretrained_ckpt"
"$XR1_ENV/bin/hf" download XiaomiRobotics/Xiaomi-Robotics-1-5B model_states.pt --local-dir "$HOME/Xiaomi-Robotics-1/xr1/pretrained_ckpt"
```

The last command downloads the **XR-1 base checkpoint** to the location expected by `config.sh`. Pi05 automatically downloads `lerobot/pi05_base` from Hugging Face when training first starts. If `uv`, `conda`, or `ffmpeg` are already available as cluster modules, you can use those installations. Do not create a second clone or environment if one already exists.

You can optionally check the included data before training, without using a GPU:

```bash
python3 _validation/smoke_test.py
```

### 3. Train Pi05 on the three sample episodes

```bash
mkdir -p outputs
SMOKE_TEST=1 NUM_GPUS=1 PI05_DATALOADER_WORKERS=1 \
  bash pi05/train_pi05.sh \
  "$PWD/_validation/pi05_dataset_3ep_rel" \
  "$PWD/outputs/sample_pi05_$(date +%Y%m%d_%H%M%S)"
```

`SMOKE_TEST=1` sets four training steps and a batch size of 1 per GPU. It writes logs under `outputs/` but skips the full checkpoint to keep the smoke test within HPC storage quotas. Each run needs a new name; the timestamp provides one here.

### 4. Train XR-1 on the three sample episodes

The included XR-1 JSON files contain video paths from the machine where they were created. The commands therefore **copy** the files and update the paths in the copy to point to the HPC location. The original repository files stay unchanged.

```bash
export XR1_ENV="$HOME/xr1_env"
export SAMPLE_DIR="$PWD/outputs/xr1_sample_$(date +%Y%m%d_%H%M%S)"
cp -a _validation/xr1_dataset_3ep "$SAMPLE_DIR"
python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ['SAMPLE_DIR'])
for file in (root / 'json').glob('*.json'):
    episode = json.loads(file.read_text())
    for views in episode['observations'].values():
        for view in views:
            view['path'] = str(root / 'videos' / Path(view['path']).name)
    file.write_text(json.dumps(episode))
PY
mkdir -p outputs/xr1/configs/data
PYTHONPATH="$HOME/Xiaomi-Robotics-1/xr1" \
  "$XR1_ENV/bin/python" xr1/compute_xr1_stats.py \
  --json-dir "$SAMPLE_DIR/json" \
  --out "$PWD/outputs/xr1/configs/data/sample_xr1.yaml"
```

The statistics file `sample_xr1.yaml` describes this specific copy of the dataset. If you repeat these steps, give the statistics file a new name or reuse the existing file with the same dataset copy. To run XR-1 without optimizer offload, first obtain a compute-node allocation with four 80 GB GPUs, then run the following from `model_training/`. This matches the GPU count of the successful GCP run but has not yet been validated on Bocconi:

```bash
SMOKE_TEST=1 NUM_GPUS=4 XR1_ENV="$XR1_ENV" \
  bash xr1/train_xr1.sh \
  "$PWD/outputs/xr1/configs" sample_xr1 \
  "$PWD/outputs/sample_xr1_$(date +%Y%m%d_%H%M%S)"
```

An alternative for a one-GPU smoke test is DeepSpeed optimizer offload to CPU RAM (`+trainer.strategy.params.offload_optimizer=true`), but that path has not been tested on Bocconi.

### Submit the Pi05 smoke test to the Slurm queue

After installing the Pi05 environment, you can submit its smoke test from the **login node** instead of running it in an interactive `srun` session. From `model_training/`:

```bash
SMOKE_TEST=1 PI05_DATALOADER_WORKERS=1 \
  sbatch pi05/train_pi05.sbatch \
  "$PWD/_validation/pi05_dataset_3ep_rel" \
  "$PWD/outputs/sample_pi05_$(date +%Y%m%d_%H%M%S)"
```

The Pi05 `.sbatch` script requests one 80 GB MIG GPU and sets `NUM_GPUS=1`. Its Slurm log is written as `slurm-<job ID>.out` in `model_training/`. The included `xr1/train_xr1.sbatch` also requests only one GPU and does not enable optimizer offload, so it is not suitable for the full XR-1 fine-tune described above.

### Full pipeline with raw recordings

To train on all recordings, `RAW_PROJECT_DIR` in `model_training/config.sh` must point to the folder containing `collect_data/`. The raw recordings are not in the repository. After `uv sync` and XR-1 setup, start the pipeline with:

```bash
cd "$HOME/Pi05_vs_XR1/model_training"
bash run_overnight.sh pi05
bash run_overnight.sh xr1
```

`run_overnight.sh` selects episodes, builds the dataset for each model, computes statistics, and trains. `bash run_overnight.sh both` runs both models one after the other; `--prepare-only` stops before training. Logs go to `outputs/logs/`. The pipeline reuses existing datasets and statistics files based on their paths. After changing data rules or raw recordings, choose a new `OUTPUT_DIR`.

For XR-1, allocate multiple GPUs before starting the full pipeline. The GCP run used four A100 80 GB GPUs without optimizer offload; the one-GPU Bocconi smoke test did not fit in GPU memory.
