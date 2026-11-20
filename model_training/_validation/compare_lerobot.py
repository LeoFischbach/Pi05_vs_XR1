"""Validation only: compare the 3-episode build with the first 3 episodes of the historical
ANTON_X_BOCCONI_clean4 dataset (read-only): numbers exactly, images decoded from both videos."""
import json, sys
import numpy as np, pandas as pd, torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ours_root = Path("_validation/pi05_dataset_3ep")
theirs_root = Path("/home/anton/dobot/dobot_xtrainer/datasets/lerobot/ANTON_X_BOCCONI_clean4")
ours = pd.concat(pd.read_parquet(f) for f in sorted((ours_root / "data").glob("*/*.parquet")))
theirs = pd.concat(pd.read_parquet(f) for f in sorted((theirs_root / "data").glob("*/*.parquet")))
theirs = theirs[theirs["episode_index"] < 3]
for key in ("observation.state", "action", "timestamp", "frame_index", "episode_index"):
    a, b = np.stack(ours[key].to_numpy()), np.stack(theirs[key].to_numpy())
    print(f"{key:20s} shape {a.shape} identical: {a.shape == b.shape and np.array_equal(a, b)}")
info_a = json.loads((ours_root / "meta/info.json").read_text()); info_b = json.loads((theirs_root / "meta/info.json").read_text())
print("features identical:", info_a["features"] == info_b["features"], "| fps", info_a["fps"], info_b["fps"])
for f in ("image_crop.json", "image_size.json"):
    print(f, json.loads((ours_root / "meta" / f).read_text()) == json.loads((theirs_root / "meta" / f).read_text()))
print("task identical:", pd.read_parquet(ours_root / "meta/tasks.parquet").index.tolist() == pd.read_parquet(theirs_root / "meta/tasks.parquet").index.tolist())
# decoded images: same frames of both datasets
a = LeRobotDataset("dobot/a", root=ours_root, video_backend="pyav")
b = LeRobotDataset("dobot/b", root=theirs_root, episodes=[0, 1, 2], video_backend="pyav")
worst = 0.0
for i in (0, 100, 200, 300, 475):
    x, y = a[i], b[i]
    for cam in ("top", "left_wrist", "right_wrist"):
        k = f"observation.images.{cam}"
        d = (x[k] - y[k]).abs().max().item(); worst = max(worst, d)
print(f"decoded image frames: largest pixel difference {worst * 255:.1f} / 255")
