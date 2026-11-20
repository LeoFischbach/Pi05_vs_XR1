"""Build a LeRobot dataset from an episode list and raw recordings.

Each frame stores a 16-D state (joints, grippers, loads), a 14-D action and three
480x480 RGB views. Global quantiles provide normalisation across all selected frames.
The CLI refuses an existing output directory. A leftover <output>.building directory
is removed before rebuilding; the completed build is then renamed to <output>.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from common import dataset_settings as settings
from common.raw_recordings import load_camera_view, load_recording

JOINT_NAMES = ([f"left_joint_{i}" for i in range(1, 7)] + ["left_gripper"]
               + [f"right_joint_{i}" for i in range(1, 7)] + ["right_gripper"])
STATE_NAMES = JOINT_NAMES + ["left_gripper_load", "right_gripper_load"]
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def dataset_features() -> dict:
    width, height = settings.IMAGE_SIZE
    features = {
        "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": JOINT_NAMES},
    }
    for camera in settings.CAMERA_FOLDERS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channel"]}
    return features


def rewrite_quantiles_globally(dataset_root: Path, feature_names: list[str]) -> None:
    """Replace the q-keys of these features in meta/stats.json by quantiles over every frame.

    Averaging per-episode quantiles can narrow the range when episodes cover different poses.
    Compute quantiles across all frames; retain LeRobot's mean/std/min/max.
    """
    stats_path = Path(dataset_root) / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    parquet_files = sorted((Path(dataset_root) / "data").glob("*/*.parquet"))
    for feature in feature_names:
        values = np.concatenate([np.stack(pd.read_parquet(f, columns=[feature])[feature].to_numpy())
                                 for f in parquet_files]).astype(np.float64)
        quantiles = np.quantile(values, list(QUANTILES.values()), axis=0)
        for row, key in enumerate(QUANTILES):
            stats[feature][key] = quantiles[row].tolist()
        outside = ((values < quantiles[0]) | (values > quantiles[-1])).mean()
        print(f"  {feature}: global quantiles, {outside:.1%} of the values outside q01..q99")
    stats_path.write_text(json.dumps(stats, indent=4))


def build_dataset(episodes_file: Path, output: Path, max_episodes: int = 0) -> None:
    selection = json.loads(Path(episodes_file).read_text())
    episodes = selection["episodes"][:max_episodes or None]
    collect_dir = Path(selection["raw_project_dir"]) / "collect_data"
    building = output.with_name(output.name + ".building")
    if building.exists():
        print(f"removing {building}, left from a build that did not finish")
        shutil.rmtree(building)

    dataset = LeRobotDataset.create(
        repo_id=f"dobot/{output.name}", fps=selection["fps"], features=dataset_features(),
        root=building, robot_type="dobot_xtrainer", use_videos=True,
        # frames go straight to the video encoders (no image files in between): ~12 GB RAM
        streaming_encoding=True,
    )
    for number, episode in enumerate(episodes, 1):
        recording = load_recording(collect_dir / episode["recording"])
        state = np.concatenate([recording["joint_state"], recording["gripper_load"]], axis=1)
        for start, end in episode["kept_ranges"]:
            for t in range(start, end):
                frame = {"observation.state": state[t], "action": recording["joint_action"][t],
                         "task": selection["task"]}
                for camera in settings.CAMERA_FOLDERS:
                    view = load_camera_view(recording["jpeg_paths"][camera][t], camera)
                    frame[f"observation.images.{camera}"] = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
                dataset.add_frame(frame)
        dataset.save_episode()
        print(f"[{number}/{len(episodes)}] {episode['recording']}: {episode['kept_frames']} frames")
    dataset.finalize()

    rewrite_quantiles_globally(building, ["observation.state", "action"])
    meta = building / "meta"
    (meta / "image_crop.json").write_text(json.dumps({k: list(v) for k, v in settings.CROP_BOXES.items()}) + "\n")
    (meta / "image_size.json").write_text(json.dumps({"size": list(settings.IMAGE_SIZE)}) + "\n")
    (meta / "episode_sources.json").write_text(json.dumps(episodes, indent=1) + "\n")

    info = json.loads((meta / "info.json").read_text())
    expected = sum(e["kept_frames"] for e in episodes)
    if (info["total_episodes"], info["total_frames"]) != (len(episodes), expected):
        raise SystemExit(f"built {info['total_episodes']} episodes / {info['total_frames']} frames, "
                         f"expected {len(episodes)} / {expected}; left in {building}")
    building.rename(output)
    print(f"wrote {output}: {len(episodes)} episodes, {expected} frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", required=True, type=Path, help="episodes.json from step 1")
    parser.add_argument("--output", required=True, type=Path, help="dataset folder to create (must not exist)")
    parser.add_argument("--max-episodes", type=int, default=0, help="only the first N episodes (tests)")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; delete it to build again")
    build_dataset(args.episodes, args.output.resolve(), args.max_episodes)


if __name__ == "__main__":
    main()
