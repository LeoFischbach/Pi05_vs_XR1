"""Nova 2 forward kinematics: joint angles in radians -> gripper pose in metres.

Used for pause detection and XR-1 action targets. Poses are expressed in each arm's
own base frame; XR-1 derives relative actions within that frame. Attribution: NOTICE.md.
"""

import numpy as np

# (d, a, alpha, theta offset) per joint, metres and radians.
NOVA2_DH_PARAMETERS = [
    (0.2234, 0.0, np.pi / 2, 0.0),
    (0.0, -0.280, 0.0, -np.pi / 2),
    (0.0, -0.225, 0.0, 0.0),
    (0.1175, 0.0, np.pi / 2, -np.pi / 2),
    (0.120, 0.0, -np.pi / 2, 0.0),
    (0.088, 0.0, 0.0, 0.0),
]
# Gripper centre = midpoint between the fingertips, 0.2 m along the flange z axis.
GRIPPER_CENTRE_Z = 0.2


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """4x4 transform of one joint (standard Denavit-Hartenberg convention)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ])


def gripper_centre_pose(joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """6 joint angles (rad) -> (position (3,) in metres, rotation matrix (3, 3))."""
    transform = np.eye(4)
    for angle, (d, a, alpha, offset) in zip(joint_angles, NOVA2_DH_PARAMETERS):
        transform = transform @ dh_transform(angle + offset, d, a, alpha)
    tool = np.eye(4)
    tool[2, 3] = GRIPPER_CENTRE_Z
    transform = transform @ tool
    return transform[:3, 3].copy(), transform[:3, :3].copy()


def gripper_centre_poses(joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T, 6) joint angles -> (T, 3) positions and (T, 3, 3) rotation matrices (float64)."""
    poses = [gripper_centre_pose(row) for row in joint_angles]
    positions = np.stack([position for position, _ in poses])
    rotations = np.stack([rotation for _, rotation in poses])
    return positions, rotations
