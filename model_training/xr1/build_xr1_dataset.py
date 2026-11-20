"""Build XR-1 annotations and three H.264 videos per selected episode.

Reads the shared episode list and raw recordings. Forward kinematics converts joint
states and commands into gripper poses in each arm's base frame. JSON duplicates
implement episode weighting while sharing videos. Waist/base values are zero.
Builds in <output>.building, removing a leftover temporary build before retrying.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from common import dataset_settings as settings
from common.nova2_kinematics import gripper_centre_poses
from common.raw_recordings import load_camera_view, load_recording

# Our camera -> the view name XR-1 expects; the prompt's <image> placeholders follow this order.
XR1_VIEW_OF_CAMERA = {"top": "ego", "left_wrist": "wrist_left", "right_wrist": "wrist_right"}
VIDEO_QUALITY_CRF = 18           # libx264, visually lossless
VIDEO_KEYFRAME_INTERVAL = 30     # frames; training reads random frames, each decoded from a keyframe


def write_video(frames_bgr: list[np.ndarray], path: Path, fps: int) -> None:
    """Pipe BGR frames into ffmpeg -> H.264 mp4 (XR-1 reads it with decord)."""
    height, width = frames_bgr[0].shape[:2]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
               "-c:v", "libx264", "-preset", "medium", "-crf", str(VIDEO_QUALITY_CRF),
               "-g", str(VIDEO_KEYFRAME_INTERVAL), "-pix_fmt", "yuv420p",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for frame in frames_bgr:
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def xr1_prompt(task: str) -> dict:
    """The conversation template of XR-1's data format, one <image> per view."""
    text = ("The following observations are captured from multiple views.\n"
            "# Ego View\n<image>\n# Left-Wrist View\n<image>\n# Right-Wrist View\n<image>\n"
            f"Generate robot actions for the task:\n{task}")
    return {"images": [f"observations.{view}" for view in ("ego", "wrist_left", "wrist_right")],
            "conversations": [{"from": "human", "value": text}, {"from": "gpt", "value": ""}]}


def arm_poses_and_grippers(joints14: np.ndarray, arm: str) -> dict:
    """End-effector pose, joints and gripper of one arm from (T, 14) joint values."""
    first = 0 if arm == "left" else 7
    joints = joints14[:, first:first + 6].astype(np.float64)
    gripper = joints14[:, first + 6].astype(np.float64)
    positions, rotations = gripper_centre_poses(joints)
    return {"ee_pos": positions.astype(np.float32).tolist(),
            "ee_rotm": rotations.reshape(-1, 9).astype(np.float32).tolist(),
            "arm_joint": joints.astype(np.float32).tolist(),
            "gripper_pos": [[float(v)] for v in gripper]}


def convert_episode(recording_dir: Path, episode: dict, stem: str, building: Path, output: Path,
                    task: str, fps: int) -> dict:
    """Write the three videos of one episode into `building` and return its XR-1 JSON content,
    whose video paths point to `output` (the folder `building` is renamed to when complete)."""
    recording = load_recording(recording_dir)
    frames = np.concatenate([np.arange(start, end) for start, end in episode["kept_ranges"]])
    observations = {}
    for camera, view in XR1_VIEW_OF_CAMERA.items():
        video_name = f"{stem}_{view}.mp4"
        images = [load_camera_view(recording["jpeg_paths"][camera][t], camera) for t in frames]
        write_video(images, building / "videos" / video_name, fps)
        # absolute paths: XR-1 reads the video paths from the JSON
        observations[view] = [{"path": str(output / "videos" / video_name), "start": 0, "crop_bbox": None}]

    state, action = recording["joint_state"][frames], recording["joint_action"][frames]
    proprios, actions = {}, {}
    for arm in ("left", "right"):
        measured, commanded = arm_poses_and_grippers(state, arm), arm_poses_and_grippers(action, arm)
        for key in ("ee_pos", "ee_rotm", "arm_joint", "gripper_pos"):
            proprios[f"{arm}_{key}"] = measured[key]
        for key in ("ee_pos", "ee_rotm", "gripper_pos"):
            actions[f"{arm}_{key}"] = commanded[key]
    count = len(frames)
    proprios["waist_pos"] = [[0.0]] * count
    actions["waist_pos"] = [[0.0]] * count
    actions["base_vel"] = [[0.0, 0.0, 0.0]] * count
    return {"trajectory_type": "success", "time": stem, "num_frames": count,
            "instruction": {"general": [xr1_prompt(task)]},
            "observations": observations, "proprios": proprios, "actions": actions}


def build_dataset(episodes_file: Path, output: Path, max_episodes: int = 0) -> None:
    """Build in <output>.building, renamed to <output> once complete."""
    selection = json.loads(Path(episodes_file).read_text())
    episodes = selection["episodes"][:max_episodes or None]
    collect_dir = Path(selection["raw_project_dir"]) / "collect_data"
    building = output.with_name(output.name + ".building")
    if building.exists():
        print(f"removing {building}, left from a build that did not finish")
        shutil.rmtree(building)
    (building / "json").mkdir(parents=True)
    (building / "videos").mkdir()
    total = repeated = 0
    for number, episode in enumerate(episodes, 1):
        # Use the stable recording number in filenames: XR-1 sorts its files by
        # path, so the same names give the same sample order.
        stem = f"episode_init_{episode['recording_number']}"
        content = json.dumps(convert_episode(collect_dir / episode["recording"], episode, stem, building, output,
                                             selection["task"], selection["fps"]))
        for copy in range(episode["repeats"]):
            name = f"{stem}.json" if copy == 0 else f"{stem}__dup{copy}.json"
            (building / "json" / name).write_text(content)
        total += episode["kept_frames"]
        repeated += episode["kept_frames"] * episode["repeats"]
        print(f"[{number}/{len(episodes)}] {episode['recording']} -> {stem}: {episode['kept_frames']} frames"
              + (f" x{episode['repeats']}" if episode["repeats"] > 1 else ""))
    (building / "meta.json").write_text(json.dumps({
        "dataset": selection["dataset"], "task": selection["task"], "fps": selection["fps"],
        "crop_boxes": settings.CROP_BOXES, "image_size": settings.IMAGE_SIZE,
        "episodes": len(episodes), "frames": total, "frames_with_repeats": repeated,
    }, indent=2) + "\n")
    building.rename(output)
    print(f"wrote {output}: {len(episodes)} episodes, {total} frames ({repeated} with the repeated files)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", required=True, type=Path, help="episodes.json from step 1")
    parser.add_argument("--output", required=True, type=Path, help="folder to create (must not exist)")
    parser.add_argument("--max-episodes", type=int, default=0, help="only the first N episodes (tests)")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; delete it to build again")
    build_dataset(args.episodes, args.output.resolve(), args.max_episodes)


if __name__ == "__main__":
    main()
