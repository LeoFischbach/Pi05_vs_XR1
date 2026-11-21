
"""Run LeRobot training with five targeted adaptations.

Keep trainable weights in float32 under mixed precision, repeat selected episodes,
save at requested steps, jitter fixed-camera images, and reduce DDP gradient memory.

Accepts LeRobot arguments plus --episode_repeats=<json>, --save_steps=<comma list>,
--fixed_camera_jitter.degrees=<degrees> and --fixed_camera_jitter.translate=<fraction>.
See train_pi05.sh for the training command and NOTICE.md for code attribution.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch


def take_argument(name: str) -> str | None:
    """Remove --<name>=<value> from sys.argv (LeRobot's parser rejects unknown arguments)."""
    value = None
    for argument in list(sys.argv[1:]):
        if argument.startswith(f"--{name}="):
            value = argument.split("=", 1)[1]
            sys.argv.remove(argument)
    return value


# 1 ------------------------------------------------------------------------------------------
def keep_trainable_weights_in_float32(lerobot_train) -> None:
    original = lerobot_train.make_optimizer_and_scheduler

    def make_optimizer_and_scheduler(cfg, policy):
        half = [p for p in policy.parameters() if p.requires_grad and p.dtype in (torch.bfloat16, torch.float16)]
        for parameter in half:
            parameter.data = parameter.data.float()
        logging.info(f"trainable weights in float32: {sum(p.numel() for p in half) / 1e9:.2f} B parameters "
                     f"cast from bfloat16")
        return original(cfg, policy)

    lerobot_train.make_optimizer_and_scheduler = make_optimizer_and_scheduler


# 2 ------------------------------------------------------------------------------------------
def repeat_episodes_in_sampler(lerobot_train, episodes_file: str) -> None:
    # the dataset's own episode list (written by build_lerobot_dataset): entry i = dataset episode i
    episodes = json.loads(Path(episodes_file).read_text())
    repeats = np.array([e["repeats"] for e in episodes], dtype=np.int64)
    BaseSampler = lerobot_train.EpisodeAwareSampler

    class EpisodeSamplerWithRepeats(BaseSampler):
        def __init__(self, dataset_from_indices, dataset_to_indices, *args, **kwargs):
            super().__init__(dataset_from_indices, dataset_to_indices, *args, **kwargs)
            if kwargs.get("episode_indices_to_use") is not None or kwargs.get("drop_n_first_frames") \
                    or kwargs.get("drop_n_last_frames") or len(repeats) != len(dataset_from_indices):
                raise SystemExit("episode repeats expect every episode of the dataset, no frames dropped")
            starts = np.asarray(dataset_from_indices, dtype=np.int64)
            lengths = np.asarray(dataset_to_indices, dtype=np.int64) - starts
            before = self._num_frames
            # the base class's frame list (episode starts + cumulative lengths), each episode repeated
            self._starts = np.repeat(starts, repeats)
            self._cum_lengths = np.cumsum(np.repeat(lengths, repeats))
            self._num_frames = int(self._cum_lengths[-1])
            logging.info(f"episode repeats from {episodes_file}: {before} -> {self._num_frames} samples per epoch")

    lerobot_train.EpisodeAwareSampler = EpisodeSamplerWithRepeats


# 3 ------------------------------------------------------------------------------------------
def save_at_steps(lerobot_train, steps: set[int]) -> None:
    lerobot_train.should_save_checkpoint = lambda step, save_freq, total_steps: step in steps or step == total_steps


# 4 ------------------------------------------------------------------------------------------
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402  (after the helpers above)


class LeRobotDatasetWithFixedCameraJitter(LeRobotDataset):
    """LeRobotDataset whose training samples get a random shift/rotation on the fixed cameras.

    Defined at module level so the data-loader workers (spawned processes) can unpickle it."""

    def __getitem__(self, index):
        item = super().__getitem__(index)         # includes LeRobot's colour jitter
        for key in self.fixed_camera_keys:
            item[key] = self.fixed_camera_jitter(item[key])
        return item


def jitter_fixed_cameras(lerobot_train, degrees: float, translate: float) -> None:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import RandomAffine

    original = lerobot_train.make_train_eval_datasets

    def make_train_eval_datasets(cfg):
        dataset, eval_dataset = original(cfg)
        if type(dataset) is not LeRobotDataset:
            raise SystemExit(f"fixed-camera jitter expects a LeRobotDataset, got {type(dataset).__name__}")
        dataset.__class__ = LeRobotDatasetWithFixedCameraJitter
        dataset.fixed_camera_keys = [key for key in dataset.meta.camera_keys if "wrist" not in key]
        # bilinear: nearest neighbour turns a 1 degree rotation into jagged edges
        dataset.fixed_camera_jitter = RandomAffine(degrees=degrees, translate=(translate, translate),
                                                   interpolation=InterpolationMode.BILINEAR)
        logging.info(f"fixed-camera jitter +-{degrees} deg, +-{translate:.0%} shift on {dataset.fixed_camera_keys}")
        return dataset, eval_dataset

    lerobot_train.make_train_eval_datasets = make_train_eval_datasets


# 5 ------------------------------------------------------------------------------------------
def ddp_gradients_as_bucket_view() -> None:
    import accelerate.utils

    original = accelerate.utils.DistributedDataParallelKwargs

    def ddp_kwargs(**kwargs):
        kwargs.setdefault("gradient_as_bucket_view", True)
        return original(**kwargs)

    accelerate.utils.DistributedDataParallelKwargs = ddp_kwargs   # LeRobot imports it inside train()


def main() -> None:
    episodes_file = take_argument("episode_repeats")
    save_steps = take_argument("save_steps")
    degrees = float(take_argument("fixed_camera_jitter.degrees") or 0)
    translate = float(take_argument("fixed_camera_jitter.translate") or 0)

    from lerobot.scripts import lerobot_train

    keep_trainable_weights_in_float32(lerobot_train)
    if episodes_file:
        repeat_episodes_in_sampler(lerobot_train, episodes_file)
    if os.environ.get("SMOKE_TEST") == "1":
        # A full checkpoint is much larger than the sample data and can exceed HPC home quotas.
        lerobot_train.should_save_checkpoint = lambda step, save_freq, total_steps: False
    elif save_steps:
        save_at_steps(lerobot_train, {int(s) for s in save_steps.split(",")})
    if degrees > 0 or translate > 0:
        jitter_fixed_cameras(lerobot_train, degrees, translate)
    ddp_gradients_as_bucket_view()
    lerobot_train.main()


if __name__ == "__main__":
    main()
