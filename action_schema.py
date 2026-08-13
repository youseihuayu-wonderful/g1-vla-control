"""Frozen G1 EDU dual-Dex1 policy action representation.

The project contract defines 16 absolute targets: left pelvis-frame EEF pose,
right pelvis-frame EEF pose, then two Dex1 motor values.  Quaternions are xyzw.
Conversion to world coordinates and MuJoCo wxyz happens only at the simulator
boundary; the future G1 adapter must implement the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

ACTION_DIM = 16
LEFT_POS = slice(0, 3)
LEFT_QUAT = slice(3, 7)
RIGHT_POS = slice(7, 10)
RIGHT_QUAT = slice(10, 14)
GRIPPERS = slice(14, 16)


def mujoco_wxyz_to_vla_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q[[1, 2, 3, 0]]


def vla_xyzw_to_mujoco_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q[[3, 0, 1, 2]]


def quaternion_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for MuJoCo-order quaternions."""
    aw, ax, ay, az = normalize_quaternion(a)
    bw, bx, by, bz = normalize_quaternion(b)
    return normalize_quaternion(np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]))


def quaternion_to_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        raise ValueError("Quaternion norm is zero")
    return q / norm


def pelvis_vla_action_to_world_mujoco(
    action: np.ndarray,
    pelvis_position: np.ndarray,
    pelvis_quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    """Convert pelvis-frame xyzw EEF targets into world-frame wxyz targets."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"action must have shape ({ACTION_DIM},)")
    pelvis_position = np.asarray(pelvis_position, dtype=np.float64)
    rotation = quaternion_to_matrix_wxyz(pelvis_quaternion_wxyz)
    result = action.copy()
    for position_slice, quaternion_slice in (
        (LEFT_POS, LEFT_QUAT), (RIGHT_POS, RIGHT_QUAT)
    ):
        result[position_slice] = pelvis_position + rotation @ action[position_slice]
        relative = vla_xyzw_to_mujoco_wxyz(action[quaternion_slice])
        result[quaternion_slice] = quaternion_multiply_wxyz(
            pelvis_quaternion_wxyz, relative
        )
    return result


def slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-path spherical interpolation between xyzw quaternions."""
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        return normalize_quaternion(q0 + fraction * (q1 - q0))
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (
        np.sin((1.0 - fraction) * theta) / sin_theta * q0
        + np.sin(fraction * theta) / sin_theta * q1
    )


@dataclass(frozen=True)
class EEFActionChunk:
    timestamps: np.ndarray
    actions: np.ndarray

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=np.float64)
        actions = np.asarray(self.actions, dtype=np.float64)
        if timestamps.ndim != 1 or len(timestamps) < 2:
            raise ValueError("timestamps must contain at least two samples")
        if actions.shape != (len(timestamps), ACTION_DIM):
            raise ValueError(f"actions must have shape (T, {ACTION_DIM})")
        if not np.all(np.diff(timestamps) > 0):
            raise ValueError("timestamps must be strictly increasing")
        actions = actions.copy()
        for index in range(len(actions)):
            for quaternion_slice in (LEFT_QUAT, RIGHT_QUAT):
                quaternion = actions[index, quaternion_slice]
                norm = float(np.linalg.norm(quaternion))
                if norm < 1e-9:
                    raise ValueError("Quaternion norm is zero")
                # Preserve already-canonical samples byte-for-byte. This makes
                # repeated chunk construction idempotent while still
                # canonicalizing genuinely non-unit inputs at the boundary.
                if abs(norm - 1.0) > 1e-12:
                    actions[index, quaternion_slice] = quaternion / norm
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "actions", actions)

    def sample(self, timestamp: float) -> np.ndarray:
        """Interpolate an action, using SLERP for both orientations."""
        t = float(np.clip(timestamp, self.timestamps[0], self.timestamps[-1]))
        upper = int(np.searchsorted(self.timestamps, t, side="right"))
        if upper == 0:
            return self.actions[0].copy()
        if upper >= len(self.timestamps):
            return self.actions[-1].copy()
        lower = upper - 1
        fraction = (t - self.timestamps[lower]) / (
            self.timestamps[upper] - self.timestamps[lower]
        )
        result = (1.0 - fraction) * self.actions[lower] + fraction * self.actions[upper]
        result[LEFT_QUAT] = slerp(
            self.actions[lower, LEFT_QUAT], self.actions[upper, LEFT_QUAT], fraction
        )
        result[RIGHT_QUAT] = slerp(
            self.actions[lower, RIGHT_QUAT], self.actions[upper, RIGHT_QUAT], fraction
        )
        return result
