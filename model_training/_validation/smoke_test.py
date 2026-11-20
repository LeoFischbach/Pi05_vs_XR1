"""Check downloaded sample datasets without models or a GPU.

Run: python3 _validation/smoke_test.py
Requires Python 3, ffmpeg and ffprobe. Reads files only; does not train or rebuild raw data.
"""

import ast
import filecmp
import json
import math
import shutil
import subprocess
from pathlib import Path

SAMPLES = Path(__file__).resolve().parent
PIPELINE = SAMPLES.parent


def read_json(path):
    return json.loads(path.read_text())


def video_command_output(command, path):
    result = subprocess.run(
        command, capture_output=True, text=True,
    )
    if result.returncode:
        tool = shutil.which(command[0]) or command[0]
        raise SystemExit(
            f"Video check failed: {path}\nTool: {tool}\n"
            f"{result.stderr.strip() or 'The video could not be read.'}\n"
            "Check that the file is fully downloaded and FFmpeg supports its codec "
            "(AV1 for pi0.5, H.264 for XR-1)."
        )
    return result.stdout


def check_video(path, expected_frames, fps):
    metadata = video_command_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", str(path)], path,
    )
    streams = json.loads(metadata).get("streams", [])
    if not streams:
        raise SystemExit(f"No readable video stream: {path}")
    stream = streams[0]
    numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
    assert (stream["width"], stream["height"]) == (480, 480), path
    assert numerator / denominator == fps, path

    # Count decoded frames directly: ffprobe may omit nb_read_frames on some installations.
    # Passthrough preserves each decoded frame without adding or dropping frames.
    progress = video_command_output(
        ["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(path),
         "-map", "0:v:0", "-an", "-fps_mode", "passthrough", "-progress", "pipe:1", "-nostats", "-f", "null", "-"], path,
    )
    counts = [int(line.split("=", 1)[1]) for line in progress.splitlines() if line.startswith("frame=")]
    if not counts:
        raise SystemExit(f"FFmpeg returned no decoded frame count: {path}")
    if counts[-1] != expected_frames:
        raise SystemExit(f"Frame count mismatch: {path}\nExpected {expected_frames}, decoded {counts[-1]}.")


def main():
    for tool in ("bash", "ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"Missing requirement: {tool}")

    source_files = []
    for folder in (PIPELINE / "common", PIPELINE / "pi05", PIPELINE / "xr1"):
        source_files.extend(folder.glob("*.py"))
    for path in source_files:
        ast.parse(path.read_text(), filename=str(path))
    shell_files = [PIPELINE / "config.sh", PIPELINE / "run_overnight.sh"]
    shell_files += list((PIPELINE / "pi05").glob("*.sh")) + list((PIPELINE / "xr1").glob("*.sh"))
    for path in shell_files:
        subprocess.run(["bash", "-n", str(path)], check=True)
    print(f"PASS: syntax of {len(source_files)} Python files and {len(shell_files)} shell files", flush=True)

    absolute = SAMPLES / "pi05_dataset_3ep"
    relative = SAMPLES / "pi05_dataset_3ep_rel"
    episodes = read_json(absolute / "meta/episode_sources.json")
    selected = read_json(SAMPLES / "episodes.json")["episodes"]
    reference = read_json(PIPELINE / "common/reference/clean4_trained_episodes.json")["episodes"]
    assert len(selected) == len(reference)
    for actual, expected in zip(selected, reference):
        for key in ("recording", "kept_ranges", "repeats"):
            assert actual[key] == expected[key], (actual["recording"], key)
    for episode, expected in zip(episodes, selected):
        for key in ("recording", "kept_ranges", "repeats"):
            assert episode[key] == expected[key], key
        assert episode["kept_frames"] == sum(end - start for start, end in episode["kept_ranges"])
    assert episodes == read_json(relative / "meta/episode_sources.json")
    frames = sum(e["kept_frames"] for e in episodes)
    weighted_frames = sum(e["kept_frames"] * e["repeats"] for e in episodes)
    for dataset in (absolute, relative):
        info = read_json(dataset / "meta/info.json")
        assert info["total_episodes"] == len(episodes) == 3
        assert info["total_frames"] == frames
        assert info["features"]["observation.state"]["shape"] == [16]
        assert info["features"]["action"]["shape"] == [14]
    assert read_json(relative / "meta/relative_actions.json") == {"chunk_size": 50, "exclude": ["gripper"]}
    parquet_files = list((absolute / "data").rglob("*.parquet"))
    assert parquet_files, "No pi0.5 sample data files"
    for path in parquet_files:
        assert filecmp.cmp(path, relative / path.relative_to(absolute), shallow=False)
    assert read_json(absolute / "meta/stats.json") == read_json(relative / "meta/stats_absolute_actions.json")
    videos = [(path, frames) for path in (absolute / "videos").rglob("*.mp4")]
    assert len(videos) == 3
    for path, _ in videos:
        assert filecmp.cmp(path, relative / path.relative_to(absolute), shallow=False)
    print(f"PASS: episode selection and pi0.5 samples ({len(episodes)} episodes, {frames} frames)", flush=True)

    xr1 = SAMPLES / "xr1_dataset_3ep"
    meta = read_json(xr1 / "meta.json")
    assert meta["episodes"] == len(episodes) and meta["frames"] == frames
    assert meta["frames_with_repeats"] == weighted_frames
    annotations = sorted((xr1 / "json").glob("*.json"))
    assert len(annotations) == sum(e["repeats"] for e in episodes)
    for episode in episodes:
        stem = f"episode_init_{episode['recording_number']}"
        annotation = read_json(xr1 / "json" / f"{stem}.json")
        count = episode["kept_frames"]
        assert annotation["num_frames"] == count
        for part in ("proprios", "actions"):
            for values in annotation[part].values():
                assert len(values) == count
                assert all(math.isfinite(value) for row in values for value in row)
        for duplicate in range(1, episode["repeats"]):
            assert read_json(xr1 / "json" / f"{stem}__dup{duplicate}.json") == annotation
        for view in ("ego", "wrist_left", "wrist_right"):
            # Workstation paths are resolved against the downloaded video folder for this check.
            # Annotations stay untouched; this is not a model-loader path test.
            filename = Path(annotation["observations"][view][0]["path"]).name
            videos.append((xr1 / "videos" / filename, count))
    print(f"PASS: XR-1 annotations ({len(annotations)} files, {weighted_frames} weighted frames)", flush=True)

    for path, count in videos:
        check_video(path, count, meta["fps"])
    print(f"PASS: all {len(videos)} unique sample videos decode, with matching frame counts and size")
    print("PASS: local data smoke test. No model loading, raw-data rebuild or GPU training performed.")


if __name__ == "__main__":
    main()
