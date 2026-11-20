"""Validation only (CPU, no model): LeRobot's config from the arguments of pi05/train_pi05.sh,
then the patched dataset / sampler / data loader exactly as lerobot_train.train() builds them."""
import json, sys
from pathlib import Path
import torch
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts import lerobot_train
import pi05.train_pi05 as T

def main():
    DATASET = Path("_validation/pi05_dataset_3ep_rel").resolve()
    args = [f"--dataset.repo_id=dobot/x", f"--dataset.root={DATASET}", "--dataset.eval_split=0.0",
            "--dataset.image_transforms.enable=false", "--policy.type=pi05", "--policy.pretrained_path=lerobot/pi05_base",
            "--policy.dtype=bfloat16", "--policy.use_relative_actions=true", "--policy.device=cpu",
            "--policy.push_to_hub=false", "--batch_size=4", "--steps=4", "--tolerance_s=0.005",
            "--wandb.enable=false", "--output_dir=/nonexistent/check", "--num_workers=2"]
    cfg = parser.parse_and_load(TrainPipelineConfig, args) if hasattr(parser, "parse_and_load") else None
    if cfg is None:
        import draccus
        cfg = draccus.parse(TrainPipelineConfig, args=args)
    cfg.validate()
    T.repeat_episodes_in_sampler(lerobot_train, str(DATASET / "meta/episode_sources.json"))
    T.jitter_fixed_cameras(lerobot_train, 1.0, 0.01)
    dataset, _ = lerobot_train.make_train_eval_datasets(cfg)
    print("dataset class:", type(dataset).__name__, "| jittered cameras:", dataset.fixed_camera_keys)
    sampler = lerobot_train.EpisodeAwareSampler(dataset.meta.episodes["dataset_from_index"], dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes, drop_n_last_frames=0, shuffle=True, seed=1000,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx)
    episodes = json.loads((DATASET / "meta/episode_sources.json").read_text())
    expected = sum(e["kept_frames"] * e["repeats"] for e in episodes)
    print(f"samples per epoch: {len(sampler)} (expected {expected} = kept frames x repeats)")
    assert len(sampler) == expected
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=2,
                                         multiprocessing_context="spawn", collate_fn=lerobot_train.lerobot_collate_fn)
    batch = next(iter(loader))
    print("batch:", {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, "shape")})
    # wrist images untouched, top image shifted/rotated (no colour jitter in this check)
    index = 10
    plain = type(dataset).__mro__[1].__getitem__(dataset, index)
    jittered = dataset[index]
    for k in ("observation.images.top", "observation.images.left_wrist", "observation.images.right_wrist"):
        print(f"  {k}: changed by the jitter: {not torch.equal(plain[k], jittered[k])}")


if __name__ == '__main__':
    main()
