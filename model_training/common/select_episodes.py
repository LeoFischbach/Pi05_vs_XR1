"""Select recordings, retained frame ranges and correction repeat counts.

Reads collect_data/ plus optional manual cuts. Rejects unusable recordings and bad
timing, removes pauses, and weights correction data. Writes one episode-list JSON
for both model pipelines; an optional reference file checks the expected selection.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common import dataset_settings as settings
from common.nova2_kinematics import gripper_centre_poses
from common.raw_recordings import list_recordings, load_recording, timing_problems, usable_reason

# Per arm: the slice of its 6 joints and the index of its gripper in the 14-D vectors.
ARMS = ((slice(0, 6), 6), (slice(7, 13), 13))


# --------------------------------------------------------------------------- automatic pause cut

def net_speed(values: np.ndarray, fps: float, window_s: float) -> np.ndarray:
    """Per frame: |x(t + w/2) - x(t - w/2)| / w (vector norm for 2-D input), window clamped at
    the ends. Jitter goes back and forth and cancels; a slow deliberate move adds up."""
    n = len(values)
    half = max(1, int(round(window_s * fps / 2)))
    before = np.clip(np.arange(n) - half, 0, n - 1)
    after = np.clip(np.arange(n) + half, 0, n - 1)
    difference = values[after] - values[before]
    distance = np.linalg.norm(difference, axis=1) if difference.ndim > 1 else np.abs(difference)
    return distance * fps / (2 * half)


def moving_frames(joints14: np.ndarray, fps: float) -> np.ndarray:
    """(T,) bool: the 14-D trajectory moves at that frame (either arm's gripper centre, any
    joint or a gripper opening faster than the pause thresholds)."""
    joints14 = np.asarray(joints14, dtype=np.float64)
    window = settings.PAUSE_WINDOW_S
    moving = np.zeros(len(joints14), dtype=bool)
    for joint_slice, gripper in ARMS:
        centre_mm = gripper_centre_poses(joints14[:, joint_slice])[0] * 1000.0
        fastest_joint = np.max([net_speed(joints14[:, j], fps, window)
                                for j in range(joint_slice.start, joint_slice.stop)], axis=0)
        moving |= net_speed(centre_mm, fps, window) >= settings.PAUSE_STILL_MM_S
        moving |= fastest_joint >= settings.PAUSE_STILL_RAD_S
        moving |= net_speed(joints14[:, gripper], fps, window) >= settings.PAUSE_GRIPPER_STEADY_S
    return moving


def automatic_pause_cut(joint_state: np.ndarray, joint_action: np.ndarray, fps: float) -> list[list[int]]:
    """[start, end) frame ranges kept once the pauses are cut out.

    Still = neither the measured state nor the command moves, and no gripper command changed
    in the second before (the gripper is still closing; the policy has to learn to wait). A
    pause in the middle keeps PAUSE_KEEP_S at each end, so the policy sees the arm come to rest
    and set off again; a pause at the start keeps only its last PAUSE_KEEP_S (a long wait
    teaches a policy to stay at the start pose); one at the end keeps PAUSE_KEEP_END_S.
    """
    still = ~(moving_frames(joint_state, fps) | moving_frames(joint_action, fps))
    guard = int(round(settings.PAUSE_GRIPPER_GUARD_S * fps))
    for _, gripper in ARMS:
        steps = np.flatnonzero(np.abs(np.diff(joint_action[:, gripper])) > settings.PAUSE_GRIPPER_STEP) + 1
        for t in steps:
            still[t:t + guard] = False

    n = len(still)
    keep = int(round(settings.PAUSE_KEEP_S * fps))
    keep_end = int(round(settings.PAUSE_KEEP_END_S * fps))
    dropped = []                                   # [start, end) of the frames cut out
    t = 0
    while t < n:
        if not still[t]:
            t += 1
            continue
        start = t
        while t < n and still[t]:
            t += 1
        end = t
        if end - start < settings.PAUSE_MIN_S * fps:
            continue                               # too short to be a pause
        if end == n:
            low, high = start + keep_end, n        # the hold at the end
        elif start == 0:
            low, high = 0, end - keep              # the wait at the start
        else:
            low, high = start + keep, end - keep
        if high > low:
            dropped.append((low, high))

    kept, t = [], 0
    for low, high in dropped:
        if low > t:
            kept.append([t, low])
        t = high
    if t < n:
        kept.append([t, n])
    return kept


# --------------------------------------------------------------------------- correction weights

def compute_repeats(episodes: list[dict]) -> float:
    """Set episode["repeats"]: corrections k times, everything else once; returns k.

    k = (frames of normal episodes) / (frames of corrections), rounded to 3 decimals, so both
    groups give the same number of samples per epoch. A fractional k is realised with whole
    copies: every correction int(k) times, and one more copy for a fixed random draw of them
    (seed 0) until their extra frames reach the fractional part. Both models use these counts.
    """
    corrections = sorted(e["index"] for e in episodes if e["is_correction"])
    frames = {e["index"]: e["kept_frames"] for e in episodes}
    correction_frames = sum(frames[i] for i in corrections)
    normal_frames = sum(frames.values()) - correction_frames
    for e in episodes:
        e["repeats"] = 1
    if not corrections:
        return 1.0
    factor = round(normal_frames / correction_frames, 3)
    repeats = {i: int(factor) for i in corrections}
    target = (factor - int(factor)) * correction_frames
    added = 0
    for position in np.random.default_rng(0).permutation(len(corrections)):
        if added >= target:
            break
        repeats[corrections[position]] += 1
        added += frames[corrections[position]]
    for e in episodes:
        e["repeats"] = repeats.get(e["index"], 1)
    return factor


# --------------------------------------------------------------------------- selection

def select_episodes(raw_project_dir: Path, manual_cuts: dict) -> tuple[list[dict], dict]:
    collect_dir = Path(raw_project_dir) / "collect_data"
    names = list_recordings(collect_dir)
    print(f"{len(names)} recording folders in {collect_dir}")

    usable = []
    for name in names:
        reason = usable_reason(collect_dir / name)
        if reason:
            print(f"  left out {name}: {reason}")
        else:
            usable.append(name)
    # Stable position among usable recordings, used in XR-1 episode filenames.
    numbered = list(enumerate(usable))

    episodes = []
    counts = {"unusable": len(names) - len(usable), "skipped": 0, "timing": 0, "nothing kept": 0}
    for number, name in numbered:
        if name.startswith(settings.SKIP_RECORDINGS):
            counts["skipped"] += 1
            continue
        recording = load_recording(collect_dir / name)
        frames = len(recording["timestamps"])
        problems = timing_problems(recording["timestamps"], recording["camera_frame_numbers"], settings.FPS)
        if problems:
            print(f"  rejected {name}: {'; '.join(problems)}")
            counts["timing"] += 1
            continue
        if name in manual_cuts:
            ranges = [[start, min(end, frames)] for start, end in manual_cuts[name] if min(end, frames) > start]
            cut = "manual"
        else:
            ranges = automatic_pause_cut(recording["joint_state"], recording["joint_action"], settings.FPS)
            cut = "automatic"
        kept = sum(end - start for start, end in ranges)
        if kept == 0:
            print(f"  left out {name}: nothing kept by the cut")
            counts["nothing kept"] += 1
            continue
        episodes.append({
            "index": len(episodes),
            "recording": name,
            "recording_number": number,
            "frames": frames,
            "kept_ranges": ranges,
            "kept_frames": kept,
            "cut": cut,
            "is_correction": name.startswith(settings.CORRECTION_PREFIXES),
        })
        if len(episodes) % 50 == 0:
            print(f"  ... {len(episodes)} episodes")
    return episodes, counts


def compare_with_reference(episodes: list[dict], reference_path: Path) -> list[str]:
    """Compare with the saved episode selection ([] = identical)."""
    reference = json.loads(Path(reference_path).read_text())["episodes"]
    differences = []
    if len(reference) != len(episodes):
        differences.append(f"{len(episodes)} episodes, the reference has {len(reference)}")
    for ours, theirs in zip(episodes, reference):
        for key in ("recording", "kept_ranges", "repeats"):
            if ours[key] != theirs[key]:
                differences.append(f"episode {ours['index']} {key}: {ours[key]} vs {theirs[key]}")
                break
    return differences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-project-dir", required=True, type=Path, help="folder that holds collect_data/")
    parser.add_argument("--manual-cuts", type=Path, default=None,
                        help="JSON {'cuts': {recording: [[start, end], ...]}}: frames kept per recording")
    parser.add_argument("--output", required=True, type=Path, help="episodes.json to write (must not exist)")
    parser.add_argument("--reference", type=Path, default=None,
                        help="expected episode selection: stop if the result differs")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; delete it to select again")

    manual_cuts = json.loads(args.manual_cuts.read_text())["cuts"] if args.manual_cuts else {}
    episodes, counts = select_episodes(args.raw_project_dir, manual_cuts)
    factor = compute_repeats(episodes)

    kept = sum(e["kept_frames"] for e in episodes)
    recorded = sum(e["frames"] for e in episodes)
    corrections = [e for e in episodes if e["is_correction"]]
    print(f"\n{len(episodes)} episodes, {kept} frames kept of {recorded} ({recorded - kept} pause frames cut); "
          f"left out: {counts}")
    print(f"corrections: {len(corrections)} episodes x{factor} -> repeats "
          f"{ {r: sum(e['repeats'] == r for e in corrections) for r in sorted({e['repeats'] for e in corrections})} }, "
          f"{sum(e['kept_frames'] * e['repeats'] for e in episodes)} samples per epoch")

    if args.reference:
        differences = compare_with_reference(episodes, args.reference)
        if differences:
            print("\n".join(["DIFFERENT from the reference:"] + differences[:20]))
            raise SystemExit("the selection differs from the reference (see above); nothing written")
        print(f"identical to the reference {args.reference.name}: same recordings, frames and repeats")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "dataset": settings.DATASET_NAME, "task": settings.TASK, "fps": settings.FPS,
        "raw_project_dir": str(Path(args.raw_project_dir).resolve()),
        "correction_factor": factor, "episodes": episodes,
    }, indent=1) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
