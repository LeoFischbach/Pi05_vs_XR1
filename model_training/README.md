# Training pipeline: π0.5 and XR-1

Two ways to try this project: check the supplied samples on a Mac, or prepare raw data
and train the full models on a Linux GPU server. Run all commands from `model_training/`.

## 1. Smoke test on a Mac — no GPU or model installation

Requirements: **Python 3**, **FFmpeg** (including `ffprobe`), and the downloaded
`_validation/` folder with its sample datasets.

```bash
python3 _validation/smoke_test.py
```

The test checks source syntax, sample episode metadata, XR-1 annotations, and decoding
of all sample videos. Successful checks print `PASS` and the command exits with code 0.
It reads the supplied samples; it does not rebuild raw data, load a model or train.

**Do not run `uv sync` for this Mac check.** The training dependencies below include
NVIDIA CUDA packages and are intended for the Linux GPU server.

## 2. Training on a Linux server with large NVIDIA GPUs

The default configuration uses **4 GPUs** and full-model fine-tuning. Plan for GPUs
in the **80 GB VRAM class**, such as NVIDIA A100 80 GB, and substantial disk space
for datasets and checkpoints (hundreds of GB for full training). Memory requirements
depend on model and batch size; even a short training test loads the full model.

You also need Linux, a CUDA-12.8-compatible NVIDIA driver, FFmpeg, `uv`, the raw
recordings, and access to the pretrained model weights. XR-1 additionally needs
Conda and a C++ compiler.

### Install the training environment

On the GPU server:

```bash
uv sync --frozen
.venv/bin/hf auth login
```

For π0.5, your Hugging Face account needs access to the PaliGemma tokenizer.
For **XR-1**, also install its separate repository and environment:

```bash
git clone https://github.com/XiaomiRobotics/Xiaomi-Robotics-1.git ~/Xiaomi-Robotics-1
git -C ~/Xiaomi-Robotics-1 checkout cfcab04
bash xr1/setup_xr1_env.sh ./envs/xr1
.venv/bin/hf download XiaomiRobotics/Xiaomi-Robotics-1-5B model_states.pt \
  --revision ee21d524b5c52ac961d941e1bc7d6d92836c3d5e \
  --local-dir ~/Xiaomi-Robotics-1/xr1/pretrained_ckpt
```

### Set the paths and GPU count

Edit `config.sh`:

- `RAW_PROJECT_DIR`: the directory containing `collect_data/`.
- `NUM_GPUS`: the number of GPUs to use; the default is 4.
- For XR-1: `XR1_REPO`, `XR1_ENV` and `XR1_PRETRAINED`. Set `XR1_ENV` to the
  absolute path of the environment created above, such as `<this folder>/envs/xr1`.

Raw recordings are not included in the repository. Their expected structure is:

```text
collect_data/<recording>/
  meta.json
  observation/<frame>.pkl
  topImg/<frame>.jpg
  leftImg/<frame>.jpg
  rightImg/<frame>.jpg
```

Both models share the episode selection, then build their own datasets and statistics:

`Raw recordings → episode selection → model-specific dataset → normalisation → training`

Data rules are in `common/dataset_settings.py`. For other recordings, set
`REFERENCE_EPISODES=""` and provide appropriate manual cuts or set `MANUAL_CUTS=""`.
Changing GPU count changes effective batch size; `config.sh` explains how to adjust
step counts for approximately the same sampled-frame budget.

### First run a short GPU training test

```bash
./run_overnight.sh pi05 --smoke-test
# or
./run_overnight.sh xr1 --smoke-test
```

This selects episodes, builds a dataset from the first **3 episodes**, computes
normalisation statistics, and performs **4 training steps with batch size 1 per GPU**.
Outputs and logs go to `outputs/smoke_test/`. A successful run ends with `all done`.
Unlike the Mac check, this executes real model training.

### Then run full training

```bash
./run_overnight.sh pi05
./run_overnight.sh xr1
# Alternatively, run both models in sequence:
./run_overnight.sh both
```

These commands use the full dataset and training parameters from `config.sh`.
Datasets, checkpoints and logs are written under `outputs/`, separated by model.
For data preparation without training, add `--prepare-only`.

Existing dataset directories, episode files and statistics markers are reused by existence,
without checking completeness or changed settings. Use a fresh `OUTPUT_DIR` after changing
data settings or after an interrupted statistics step. Builders remove leftover `.building`
directories when rebuilding. Training refuses an existing run path.

Code attribution is in `NOTICE.md`; additional checks are described in `_validation/README.md`.
