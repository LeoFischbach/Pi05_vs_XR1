"""Shared data rules for episode selection and camera preprocessing.

Both dataset builders use these settings. Paths and training parameters are in config.sh.
"""

DATASET_NAME = "ANTON_X_BOCCONI_clean4"

# The language instruction stored with every frame (pi0.5) / every episode (XR-1).
TASK = ("Pick up the pin with the right arm from the cardboard box, give it to the left arm "
        "and place it in the next free slot in the black tray")

# Nominal frame rate of recording, datasets and training. A LeRobot dataset places frame i at
# i / FPS, so recordings whose measured timing strays too far from it are rejected (below).
FPS = 30

# Camera name in the datasets -> image folder in a raw recording.
CAMERA_FOLDERS = {"top": "topImg", "left_wrist": "leftImg", "right_wrist": "rightImg"}

# Image view. The top camera is cropped to the desk plus the grippers above it; every camera is
# then resized (stretched) to a square, so pi0.5's 224x224 letterbox uses all its image tokens.
# Box = (x0, y0, x1, y1) in pixels of the recorded 640x480 image. Wrist cameras: no crop.
CROP_BOXES = {"top": (136, 132, 488, 396)}
IMAGE_SIZE = (480, 480)            # (width, height) after the crop

# Recordings left out by name prefix: the 50 slow demonstrations of the first session (slow_*)
# and #591 (20261102134006, the 7th pin went into the 8th pin's pocket).
SKIP_RECORDINGS = ("slow_", "20261102134006")

# Recordings counted as corrections: interventions recorded while a policy ran, and the short
# correction snippets recorded by teleoperation. They are repeated so that they make up 50 %
# of the training samples (select_episodes.compute_repeats).
CORRECTION_PREFIXES = ("intervention_", "correction_")

# Timing quality check of a whole recording (measured per-frame timestamps vs FPS).
FPS_TOLERANCE = 0.05               # mean rate within +-5 % of FPS
MAX_GAP_PERIODS = 3.0              # no frame interval longer than 3 periods (100 ms)
MAX_LATE_FRACTION = 0.05           # at most 5 % of intervals longer than 1.5 periods
MAX_CAMERA_REPEAT_FRACTION = 0.02  # at most 2 % of steps may reuse a camera's previous frame

# Automatic pause cut (for recordings without a manual cut): stretches where both arms stand
# still for at least PAUSE_MIN_S are cut out, except PAUSE_KEEP_S at each end. "Still" is
# measured as net displacement over PAUSE_WINDOW_S (the master hands' jitter cancels out):
PAUSE_WINDOW_S = 0.4               # window of the net displacement
PAUSE_STILL_MM_S = 15.0            # gripper centre slower than this (mm/s), per arm
PAUSE_STILL_RAD_S = 0.06           # every joint slower than this (rad/s)
PAUSE_GRIPPER_STEADY_S = 0.15      # gripper opening (0..1) changes slower than this per second
PAUSE_GRIPPER_STEP = 0.02          # a gripper command step larger than this ...
PAUSE_GRIPPER_GUARD_S = 1.0        # ... marks the next second as moving (the gripper still closes)
PAUSE_MIN_S = 1.5                  # shorter stops are no pause
PAUSE_KEEP_S = 0.2                 # frames kept at each end of a cut pause
PAUSE_KEEP_END_S = 1.0             # a pause at the very end keeps its first second (the hold)
