# Data checks (no GPU training)

## Quick local check

From `model_training/`, with Python 3 and FFmpeg installed:

```bash
python3 _validation/smoke_test.py
```

This read-only check validates source syntax, the supplied episode metadata, XR-1 annotations,
and sample video decoding. XR-1's workstation video paths are resolved against the downloaded
sample folder for this check only. It does not load a model, use the model data loaders,
rebuild raw recordings, or execute an optimisation step.

## Workstation checks

`print_pi05_args.sh` now forwards to `pi05/train_pi05.sh --print-args`;
`run_overnight_no_training.sh` forwards to `run_overnight.sh --prepare-only` and puts its
outputs under `_validation/overnight_flow/`. Both wrappers resolve the pipeline root from
their own location and contain no copy of the training arguments or orchestration logic.

The files below record earlier comparisons against the original workstation data. Some scripts
use absolute workstation paths and require the model environments; they are not portable local
tests. Existing logs are evidence of those checks, not a new run on this computer.

| check | script / command | result log |
|---|---|---|
| step 1 on all 716 recordings vs reference | `python -m common.select_episodes ... --reference ...` | `select_episodes.log`, `episodes.json` |
| π0.5 dataset, 3 episodes vs historical | `compare_lerobot.py` | `compare_lerobot.log` |
| relative-action stats on a copy of the full historical dataset | `python -m pi05.compute_relative_action_stats` (copy deleted afterwards) | `relative_stats_full.log` |
| XR-1 dataset, 3 episodes vs historical | inline check | `compare_xr1.log` |
| XR-1 stats over the 1256 historical JSON files | `xr1/compute_xr1_stats.py` | `xr1_stats_full.log`, `xr1_configs/` |
| XR-1 Hydra config vs reference `config.yaml` | `XR1_DRY_RUN=1 bash xr1/train_xr1.sh ...` | `xr1_dry_run.log` |
| π0.5 LeRobot config vs reference `train_config.json` | `print_pi05_args.sh`, `compare_pi05_config.py` | `pi05_config_diff.txt` |
| π0.5 dataset / sampler / spawned data loader (CPU) | `check_pi05_training_data.py` | `check_pi05_training_data.log` |
| earlier overnight check with training replaced by a print | now replaced by the preparation-only wrapper above | `overnight_flow/`, `overnight_error/` (archived logs) |
