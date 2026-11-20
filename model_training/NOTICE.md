# Origin and licences

## Code in this folder

Adapted and reduced from the dobot_xtrainer repository (Dobot X-Trainer rig code and training
adaptations). Source mapping within that repository:

- `common/dataset_settings.py`: `ModelTrain/constants.py` and `scripts/pauses.py`.
- `common/raw_recordings.py`: `script_collect2train.py`, `timing.py`, `image_crop.py`
  and `recording_ids.py` under `scripts/`.
- `common/nova2_kinematics.py`: `experiments/run_control.py` and
  `scripts/convert_dobot_to_xr1.py` (Nova 2 DH parameters).
- `common/select_episodes.py`: `scripts/pauses.py`, `scripts/convert_dobot_to_lerobot.py`
  and the episode-weighting preparation scripts.
- `pi05/build_lerobot_dataset.py`: `scripts/convert_dobot_to_lerobot.py` and
  `scripts/global_quantiles.py`.
- `pi05/compute_relative_action_stats.py`: `scripts/make_relative_dataset.py`.
- `pi05/train_pi05.py`, `pi05/train_pi05.sh` and `run_overnight.sh`:
  `scripts/train_lerobot.py` and `scripts/train_overnight.sh`.
- `xr1/build_xr1_dataset.py`, `xr1/compute_xr1_stats.py`: the corresponding
  `scripts/convert_dobot_to_xr1.py` and `scripts/compute_xr1_stats.py`.
- `xr1/train_xr1.sh`, `xr1/setup_xr1_env.sh` and `xr1/env/`: the XR-1 launch/setup
  scripts and exported environment locks; the launcher also follows upstream XR-1's `scripts/train.sh`.

The earlier data preparation used an HDF5 intermediate and a separate relative-action dataset
copy. This version reads recordings directly and updates the dedicated pi0.5 dataset's statistics
in place, keeping a backup of the absolute-action statistics.

The vendor code is distributed with this MIT notice:

```
MIT License

Copyright (c) 2024 Shenzhen Yuejiang Technology Co., Ltd. （Dobot）

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The Nova 2 DH parameters (`common/nova2_kinematics.py`) come from Dobot's vendor code in that
repository (`experiments/run_control.py`).

## Libraries used (not copied)

- **LeRobot** 0.6.1, Apache License 2.0, © The HuggingFace Inc. team: dataset format, pi0.5
  implementation and trainer. `pi05/train_pi05.py` replaces five of its functions at run time; the
  sampler subclass builds on the internals of `lerobot.datasets.sampler.EpisodeAwareSampler`.
- **Xiaomi-Robotics-1 (XR-1)**, Apache License 2.0, © 2026 Xiaomi Corporation: model, data format,
  trainer. `xr1/build_xr1_dataset.py` writes its documented data format (`xr1/docs/data_format.md`),
  `xr1/compute_xr1_stats.py` calls its `mibot.utils.io` functions and mirrors the action
  computation of its `mibot/data/datasets/json_dataset.py`. `xr1/patches/save_last_link.patch` is a
  change to its `tools/train.py` (Apache-2.0 applies to it).
- **flash-attention**, DeepSpeed, PyTorch Lightning, Hydra and the other packages in the lock files
  under their own licences.

## Model weights

`lerobot/pi05_base` (with the PaliGemma tokenizer `google/paligemma-3b-pt-224`),
`XiaomiRobotics/Xiaomi-Robotics-1-5B` and `Qwen/Qwen3-VL-4B-Instruct` come with their own licences
and terms of use. Check each model page on Hugging Face before using or publishing results.

## Data

The recordings (`ANTON_X_BOCCONI`) are not part of this folder. `common/reference/*.json` list only
recording names and frame ranges of the reference build.
