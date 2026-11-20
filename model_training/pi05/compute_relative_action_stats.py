"""Compute pi0.5 normalisation statistics for 50-step relative-action chunks.

Reads a prepared LeRobot dataset and updates meta/stats.json in place. Joint targets
are offsets from the chunk's initial state; grippers remain absolute. Absolute-action
statistics are backed up in meta/stats_absolute_actions.json. A relative_actions.json
marker records completion; state quantiles are recomputed globally.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")      # everything is local here

from lerobot.datasets.dataset_tools import recompute_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from pi05.build_lerobot_dataset import rewrite_quantiles_globally

CHUNK_SIZE = 50                  # = pi0.5's chunk_size
ABSOLUTE_ACTION_NAMES = ["gripper"]   # action names containing this stay absolute


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path, help="LeRobot dataset from step 2")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    meta = args.dataset / "meta"
    marker = meta / "relative_actions.json"
    if marker.exists():
        raise SystemExit(f"{args.dataset} has relative-action statistics already ({marker.name})")

    if not (meta / "stats_absolute_actions.json").exists():     # not again after a failed attempt
        shutil.copy(meta / "stats.json", meta / "stats_absolute_actions.json")
    dataset = LeRobotDataset(f"dobot/{args.dataset.name}", root=args.dataset)
    recompute_stats(dataset, relative_action=True, relative_exclude_joints=ABSOLUTE_ACTION_NAMES,
                    chunk_size=CHUNK_SIZE, num_workers=args.workers)
    rewrite_quantiles_globally(args.dataset, ["observation.state"])
    marker.write_text(json.dumps({"chunk_size": CHUNK_SIZE, "exclude": ABSOLUTE_ACTION_NAMES}, indent=2) + "\n")

    stats = json.loads((meta / "stats.json").read_text())["action"]
    names = dataset.meta.features["action"]["names"]
    for i, name in enumerate(names):
        print(f"  {name:15s} relative q01 {stats['q01'][i]:+.3f} .. q99 {stats['q99'][i]:+.3f}")
    print(f"wrote relative-action statistics into {meta / 'stats.json'}")


if __name__ == "__main__":
    main()
