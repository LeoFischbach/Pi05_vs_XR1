"""Validation only: LeRobot's TrainPipelineConfig from train_pi05.sh's arguments vs the reference run's."""
import json, sys
from pathlib import Path
import draccus
from lerobot.scripts import lerobot_train  # registers the policy types
from lerobot.configs.train import TrainPipelineConfig
args = [a for a in Path("_validation/pi05_args.txt").read_text().splitlines()
        if a and not a.startswith(("--fixed_camera_jitter", "--episode_repeats", "--save_steps"))]
cfg = draccus.parse(TrainPipelineConfig, args=args)
cfg.validate()
ours = json.loads(json.dumps(draccus.encode(cfg)))
ref = json.loads(Path("/home/anton/dobot/dobot_xtrainer/ckpt/bocconi_clean4_20261115_from_base/pi05rel_full_ANTON_X_BOCCONI_clean4/checkpoints/017324/pretrained_model/train_config.json").read_text())
def flat(d, p=""):
    out = {}
    for k, v in (d or {}).items():
        out.update(flat(v, p + k + ".") if isinstance(v, dict) else {p + k: v})
    return out
a, b = flat(ours), flat(ref)
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k):
        va, vb = a.get(k), b.get(k)
        short = lambda v: (f"<list of {len(v)}>" if isinstance(v, list) and len(v) > 8 else v)
        print(f"{k}: ours={short(va)!r}  reference={short(vb)!r}")
