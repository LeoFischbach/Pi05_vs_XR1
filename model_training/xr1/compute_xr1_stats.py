"""Compute XR-1 normalisation statistics and write its YAML data configuration.

Run in the XR-1 environment with mibot on PYTHONPATH. Reads all episode JSON files,
including repeats. Uses XR-1's packing functions for 30x60 relative-action chunks
(mean/std) and the 60-D state (q01/q99). Only complete action windows enter the stats.
The output YAML is selected by train_xr1.sh through data=<dataset name>.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from mibot.utils.io import ACTION_PARTS, compose_action, compose_state, get_value, rotm2aa_batch

ACTION_LENGTH = 30       # XR-1's action chunk (data.params.train_datasets.action_length)
MIN_STD = 1e-3           # floor for the std of moving dimensions (no blow-up of near-constant ones)
DECIMALS = 6


def array(trajectory: dict, key: str, rows) -> np.ndarray:
    return np.asarray(get_value(trajectory, key)[rows], dtype=np.float32)


def pad_to_length(values: np.ndarray, steps: int) -> np.ndarray:
    """Repeat the last row up to ACTION_LENGTH rows (as the loader pads an episode's end)."""
    if steps == ACTION_LENGTH:
        return values
    return np.concatenate([values, np.repeat(values[-1:], ACTION_LENGTH - steps, axis=0)], axis=0)


def relative_arm_action(trajectory: dict, arm: str, frame: int, steps: int):
    """Target position and rotation in the current gripper frame, and the gripper change."""
    rotation = array(trajectory, f"proprios.{arm}_ee_rotm", frame).reshape(3, 3)
    position = array(trajectory, f"proprios.{arm}_ee_pos", frame)
    window = slice(frame, frame + steps)
    target_position = array(trajectory, f"actions.{arm}_ee_pos", window)
    target_rotation = array(trajectory, f"actions.{arm}_ee_rotm", window).reshape(-1, 3, 3)
    gripper_now = array(trajectory, f"proprios.{arm}_gripper_pos", frame)
    gripper_target = array(trajectory, f"actions.{arm}_gripper_pos", window)
    return (pad_to_length((rotation.T @ (target_position - position).T).T, steps),
            pad_to_length(rotm2aa_batch(rotation.T @ target_rotation), steps),
            pad_to_length(gripper_target - gripper_now, steps))


def packed_action(trajectory: dict, frame: int) -> np.ndarray:
    """The (30, 60) unnormalised action XR-1's loader builds for this frame."""
    steps = min(ACTION_LENGTH, int(trajectory["num_frames"]) - frame)
    window = slice(frame, frame + steps)
    waist = pad_to_length(array(trajectory, "actions.waist_pos", window)
                          - array(trajectory, "proprios.waist_pos", frame), steps)
    base = pad_to_length(array(trajectory, "actions.base_vel", window), steps)
    return compose_action(*relative_arm_action(trajectory, "left", frame, steps),
                          *relative_arm_action(trajectory, "right", frame, steps),
                          waist, base, action_length=ACTION_LENGTH)


def packed_state(trajectory: dict, frame: int) -> np.ndarray:
    return compose_state(left_gripper=array(trajectory, "proprios.left_gripper_pos", frame),
                         left_joint=array(trajectory, "proprios.left_arm_joint", frame),
                         right_gripper=array(trajectory, "proprios.right_gripper_pos", frame),
                         right_joint=array(trajectory, "proprios.right_arm_joint", frame))[0]


def yaml_rows(rows: np.ndarray) -> str:
    return "\n".join("      - [" + ", ".join(f"{v:g}" for v in row) + "]" for row in np.round(rows, DECIMALS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="data config to write (must not exist)")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; delete it to compute again")
    files = sorted(args.json_dir.glob("**/*.json"))
    if not files:
        raise SystemExit(f"no JSON files in {args.json_dir}")

    action_sum = np.zeros((ACTION_LENGTH, 60))
    action_square_sum = np.zeros((ACTION_LENGTH, 60))
    windows, frames, states = 0, 0, []
    for number, path in enumerate(files, 1):
        trajectory = json.loads(path.read_text())
        count = int(trajectory["num_frames"])
        states += [packed_state(trajectory, t) for t in range(count)]
        for t in range(max(count - ACTION_LENGTH + 1, 0)):        # complete windows only
            action = packed_action(trajectory, t).astype(np.float64)
            action_sum += action
            action_square_sum += action * action
            windows += 1
        frames += count
        if number % 100 == 0 or number == len(files):
            print(f"[{number}/{len(files)}] files read", flush=True)

    mean = action_sum / windows
    std = np.sqrt(np.maximum(action_square_sum / windows - mean * mean, 0.0))
    used = np.zeros(60, dtype=bool)                  # the dimensions XR-1 actually fills
    for _, part in ACTION_PARTS:
        used[part] = True
    moving = used & (std > 0).any(axis=0)
    std[:, moving] = np.maximum(std[:, moving], MIN_STD)
    mean[:, ~used] = 0.0
    std[:, ~used] = 0.0
    states = np.stack(states)
    q01 = np.percentile(states, 1, axis=0)[None, :]
    q99 = np.percentile(states, 99, axis=0)[None, :]
    constant = (q99 <= q01) & ~((q01 == 0.0) & (q99 == 0.0))   # XR-1 rejects q01 == q99 != 0
    q01[constant] -= 1e-3
    q99[constant] += 1e-3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"""# @package _global_
# Generated by compute_xr1_stats.py from {len(files)} files / {frames} frames ({windows} action windows).
# Statistics describe THIS dataset; do not reuse them for other recordings.

data:
  type: BaseDataModule
  params:
    type: json
    max_steps: ${{trainer.max_steps}}
    train_datasets:
      batch_size: 48
      action_length: {ACTION_LENGTH}
      paths:
      - {args.json_dir.resolve()}
      mean:
{yaml_rows(mean)}
      std:
{yaml_rows(std)}
      q01:
{yaml_rows(q01)}
      q99:
{yaml_rows(q99)}
""")
    print(f"wrote {args.out} ({len(files)} files, {frames} frames, {windows} action windows)")


if __name__ == "__main__":
    main()
