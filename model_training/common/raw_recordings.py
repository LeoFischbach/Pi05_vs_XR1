"""Read raw robot recordings without modifying them.

Each collect_data/<recording>/ folder contains meta.json, observation/<frame>.pkl
and matching JPEGs in topImg/, leftImg/ and rightImg/. Helpers load robot values,
check timing, and crop/resize images according to dataset_settings.py.
"""

import json
import pickle
import re
from pathlib import Path

import cv2
import numpy as np

from common import dataset_settings as settings

_LABELLED = re.compile(r"^(slow|correction)_(\d+)$")
_INTERVENTION = re.compile(r"^intervention_(\d+)(?:_(\d+))?$")


def recording_sort_key(name: str):
    """Time order of a recording folder (None = not a recording folder)."""
    if name.isdigit():
        return (int(name), 0)
    match = _LABELLED.match(name)
    if match:
        return (int(match.group(2)), 0)
    match = _INTERVENTION.match(name)
    if match:
        return (int(match.group(1)), int(match.group(2) or 1))
    return None


def list_recordings(collect_dir: Path) -> list[str]:
    """All recording folder names in time order."""
    names = [p.name for p in Path(collect_dir).iterdir() if p.is_dir()]
    unknown = [n for n in names if recording_sort_key(n) is None]
    if unknown:
        print(f"  note: {len(unknown)} folders are no recordings and are ignored: {unknown[:5]}")
    return sorted((n for n in names if recording_sort_key(n) is not None), key=recording_sort_key)


def read_meta(recording_dir: Path) -> dict:
    path = Path(recording_dir) / "meta.json"
    return json.loads(path.read_text()) if path.exists() else {}


def usable_reason(recording_dir: Path) -> str:
    """'' if the recording may be used, else why not.

    An intervention is used only if the recorder finished it ("complete") and marked it valid
    (not aborted, long enough, no gap at the takeover). A recording marked with a moved camera
    (camera_alignment status "off") is never used.
    """
    meta = read_meta(recording_dir)
    if (meta.get("camera_alignment") or {}).get("status") == "off":
        return "a camera was off its reference position"
    if Path(recording_dir).name.startswith("intervention_"):
        if meta.get("status") != "complete":
            return f"intervention {meta.get('status') or 'without status'}"
        if meta.get("valid") is not True:
            return f"intervention not valid: {meta.get('invalid_reason') or '?'}"
    return ""


def load_recording(recording_dir: Path) -> dict:
    """The arrays of one recording, one row per frame.

    joint_state   (T, 14) float32  measured [L j1..j6, L gripper, R j1..j6, R gripper],
                                   joints in rad, gripper opening 0 = closed .. 1 = open
    joint_action  (T, 14) float32  the commanded values (from the master hands)
    gripper_load  (T, 2)  float32  measured gripper load, left / right
    timestamps    (T,)    float64  capture time in seconds from the first frame
    camera_frame_numbers (T, 3)    which camera frame each step used (top, left, right)
    jpeg_paths    {camera: [T paths]}

    Only frames that have a pickle and an image from every camera are kept, in index order. A
    pickle cut off mid-write ends the recording at the last readable frame.
    """
    recording_dir = Path(recording_dir)
    index_of = lambda path: int(path.stem)
    pickles = {index_of(p): p for p in (recording_dir / "observation").glob("*.pkl")}
    images = {camera: {index_of(p): p for p in (recording_dir / folder).glob("*.jpg")}
              for camera, folder in settings.CAMERA_FOLDERS.items()}
    common_indices = set(pickles)
    for camera_images in images.values():
        common_indices &= set(camera_images)
    indices = sorted(common_indices)

    rows = []
    for i in indices:
        try:
            with open(pickles[i], "rb") as file:
                rows.append(pickle.load(file))
        except (EOFError, pickle.UnpicklingError):
            print(f"  warning: {recording_dir.name}: unreadable frame {i}, keeping the {len(rows)} before it")
            break
    indices = indices[:len(rows)]
    if not rows:
        raise ValueError(f"{recording_dir}: no complete frame")

    timestamps = np.array([row["timestamp"] for row in rows], dtype=np.float64)
    return {
        # Use float32 for the dataset state and action values.
        "joint_state": np.array([row["joint_positions"] for row in rows]).astype(np.float32),
        "joint_action": np.array([row["control"] for row in rows]).astype(np.float32),
        "gripper_load": np.array([row["gripper_load"] for row in rows]).astype(np.float32),
        "timestamps": timestamps - timestamps[0],
        "camera_frame_numbers": np.array([row["cam_frame_number"] for row in rows], dtype=np.int64),
        "jpeg_paths": {camera: [images[camera][i] for i in indices] for camera in settings.CAMERA_FOLDERS},
    }


def timing_problems(timestamps: np.ndarray, camera_frame_numbers: np.ndarray, fps: float) -> list[str]:
    """Quality check of a recording's timing; [] = fine.

    A dataset claims frame i is at i / fps. A recording that ran too slow, had a long gap or
    stored the same camera image again and again is not the data that label promises.
    """
    period = 1.0 / fps
    intervals = np.diff(timestamps)
    if len(intervals) == 0:
        return ["fewer than two frames"]
    problems = []
    duration = timestamps[-1] - timestamps[0]
    mean_rate = len(intervals) / duration if duration > 0 else float("inf")
    if (intervals <= 0).any():
        problems.append(f"{int((intervals <= 0).sum())} non-increasing timestamps")
    if abs(mean_rate / fps - 1.0) > settings.FPS_TOLERANCE:
        problems.append(f"mean rate {mean_rate:.2f} Hz")
    if intervals.max() > settings.MAX_GAP_PERIODS * period:
        problems.append(f"gap of {intervals.max() * 1e3:.0f} ms at frame {int(intervals.argmax()) + 1}")
    late_fraction = float(np.mean(intervals > 1.5 * period))
    if late_fraction > settings.MAX_LATE_FRACTION:
        problems.append(f"{late_fraction:.1%} of frames late")
    steps = np.diff(camera_frame_numbers, axis=0)
    for column, camera in enumerate(("top", "left", "right")):
        repeated = float(np.mean(steps[:, column] == 0))
        if repeated > settings.MAX_CAMERA_REPEAT_FRACTION:
            problems.append(f"camera {camera} repeats {repeated:.1%} of frames")
        if (steps[:, column] < 0).any():
            problems.append(f"camera {camera} frame counter went backwards")
    return problems


def load_camera_view(jpeg_path: Path, camera: str) -> np.ndarray:
    """One camera image as the policies see it: decoded (BGR, as OpenCV gives it), cropped
    (top camera) and resized to settings.IMAGE_SIZE with INTER_AREA."""
    image = cv2.imdecode(np.fromfile(jpeg_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode {jpeg_path}")
    if camera in settings.CROP_BOXES:
        x0, y0, x1, y1 = settings.CROP_BOXES[camera]
        image = image[y0:y1, x0:x1]
    width, height = settings.IMAGE_SIZE
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
