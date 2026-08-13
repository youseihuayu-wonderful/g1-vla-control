#!/usr/bin/env python3
"""Forensic audit of the public Stack-the-cubes action/state contract."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

import mujoco
import numpy as np
import pyarrow.parquet as pq

from stack_scene import build_model, reset_to_stand

DATASET = "LGG100/Stack-the-cubes"
MODEL_REPO = "LGG100/stack-cube-eef-24k"
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = Path.home() / ".cache" / "g1_vla_control" / "stack_the_cubes"
ARM_JOINTS = tuple(
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw",
    )
)


def _download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)
    return path


def download_public_assets(episodes: int = 100) -> tuple[list[Path], Path]:
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            f"https://huggingface.co/datasets/{DATASET}/resolve/main/"
            f"data/chunk-000/episode_{index:06d}.parquet",
            CACHE / f"episode_{index:06d}.parquet",
        )
        for index in range(episodes)
    ]
    with ThreadPoolExecutor(max_workers=12) as pool:
        paths = list(pool.map(lambda job: _download(*job), jobs))
    norm_path = _download(
        f"https://huggingface.co/{MODEL_REPO}/resolve/main/"
        "assets/stack-cube-eef/norm_stats.json",
        CACHE / "norm_stats.json",
    )
    return paths, norm_path


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _fk_transform(values: np.ndarray, eef_offset_m: float = 0.050) -> np.ndarray:
    """Reconstruct a candidate pelvis-frame EEF transform from raw arm joints."""
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_stand(model, data)
    addresses = np.asarray([
        model.jnt_qposadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )]
        for name in ARM_JOINTS
    ])
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    wrist_ids = [
        mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link"
        )
        for side in ("left", "right")
    ]
    output = np.empty_like(values, dtype=np.float64)
    if not np.isfinite(eef_offset_m):
        raise ValueError("eef_offset_m must be finite")
    offset = np.array([eef_offset_m, 0.0, 0.0])
    for row, source in zip(output, values, strict=True):
        data.qpos[addresses] = source[:14]
        mujoco.mj_kinematics(model, data)
        pelvis_rotation = data.xmat[pelvis_id].reshape(3, 3)
        pelvis_position = data.xpos[pelvis_id]
        cursor = 0
        for wrist_id in wrist_ids:
            wrist_rotation = data.xmat[wrist_id].reshape(3, 3)
            point_world = data.xpos[wrist_id] + wrist_rotation @ offset
            point_pelvis = pelvis_rotation.T @ (point_world - pelvis_position)
            rotation_pelvis = pelvis_rotation.T @ wrist_rotation
            quaternion_wxyz = np.empty(4)
            mujoco.mju_mat2Quat(quaternion_wxyz, rotation_pelvis.ravel())
            row[cursor : cursor + 3] = point_pelvis
            row[cursor + 3 : cursor + 7] = quaternion_wxyz[[1, 2, 3, 0]]
            cursor += 7
        row[14:] = source[14:]
    return output


def _best_action_state_lag(episodes: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    all_actions = np.concatenate([action for _, action in episodes])
    scale = np.maximum(all_actions.std(axis=0), 1e-6)
    scores = []
    for lag in range(31):
        squared_error = 0.0
        count = 0
        for state, action in episodes:
            if lag == 0:
                lhs, rhs = state, action
            else:
                lhs, rhs = state[lag:], action[:-lag]
            squared_error += float(np.square((lhs - rhs) / scale).sum())
            count += lhs.size
        scores.append(float(np.sqrt(squared_error / count)))
    best = int(np.argmin(scores))
    return {"best_lag_frames": best, "best_lag_seconds": best / 30.0,
            "normalized_rmse_by_lag_0_to_30": scores}


def _motion_envelope(
    raw_episodes: list[tuple[np.ndarray, np.ndarray]],
    eef_actions: np.ndarray,
) -> dict:
    translation_speeds = []
    angular_speeds = []
    translation_accelerations = []
    translation_jerks = []
    joint_speeds = []
    joint_accelerations = []
    joint_jerks = []
    per_joint_speeds = []
    per_joint_accelerations = []
    per_joint_jerks = []
    gripper_speeds = []
    cursor = 0
    for _, raw_action in raw_episodes:
        count = len(raw_action)
        action = eef_actions[cursor : cursor + count]
        cursor += count
        left_velocity = np.diff(action[:, 0:3], axis=0) * 30.0
        right_velocity = np.diff(action[:, 7:10], axis=0) * 30.0
        translation_speeds.extend(np.maximum(
            np.linalg.norm(left_velocity, axis=1),
            np.linalg.norm(right_velocity, axis=1),
        ))
        if len(left_velocity) > 1:
            left_accel = np.diff(left_velocity, axis=0) * 30.0
            right_accel = np.diff(right_velocity, axis=0) * 30.0
            translation_accelerations.extend(np.maximum(
                np.linalg.norm(left_accel, axis=1),
                np.linalg.norm(right_accel, axis=1),
            ))
            if len(left_accel) > 1:
                left_jerk = np.diff(left_accel, axis=0) * 30.0
                right_jerk = np.diff(right_accel, axis=0) * 30.0
                translation_jerks.extend(np.maximum(
                    np.linalg.norm(left_jerk, axis=1),
                    np.linalg.norm(right_jerk, axis=1),
                ))
        per_side_angular = []
        for quaternion_slice in (slice(3, 7), slice(10, 14)):
            q0 = action[:-1, quaternion_slice]
            q1 = action[1:, quaternion_slice]
            dot = np.clip(np.abs(np.sum(q0 * q1, axis=1)), 0.0, 1.0)
            per_side_angular.append(2.0 * np.arccos(dot) * 30.0)
        angular_speeds.extend(np.maximum(*per_side_angular))
        joint_velocity = np.diff(raw_action[:, :14], axis=0) * 30.0
        per_joint_speeds.append(np.abs(joint_velocity))
        joint_speeds.extend(np.max(np.abs(joint_velocity), axis=1))
        if len(joint_velocity) > 1:
            joint_acceleration = np.diff(joint_velocity, axis=0) * 30.0
            per_joint_accelerations.append(np.abs(joint_acceleration))
            joint_accelerations.extend(np.max(np.abs(joint_acceleration), axis=1))
            if len(joint_acceleration) > 1:
                joint_jerk = np.diff(joint_acceleration, axis=0) * 30.0
                per_joint_jerks.append(np.abs(joint_jerk))
                joint_jerks.extend(np.max(np.abs(joint_jerk), axis=1))
        gripper_speeds.extend(np.max(np.abs(np.diff(raw_action[:, 14:], axis=0) * 30.0), axis=1))

    def quantiles(values) -> dict:
        values = np.asarray(values)
        return {
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
        }

    return {
        "eef_translation_speed_m_s": quantiles(translation_speeds),
        "eef_angular_speed_rad_s": quantiles(angular_speeds),
        "eef_translation_acceleration_m_s2": quantiles(translation_accelerations),
        "eef_translation_jerk_m_s3": quantiles(translation_jerks),
        "raw_joint_target_speed_rad_s": quantiles(joint_speeds),
        "raw_joint_target_acceleration_rad_s2": quantiles(joint_accelerations),
        "raw_joint_target_jerk_rad_s3": quantiles(joint_jerks),
        "per_joint_target_p99": {
            name: {
                "speed_rad_s": float(speed),
                "acceleration_rad_s2": float(acceleration),
                "jerk_rad_s3": float(jerk),
            }
            for name, speed, acceleration, jerk in zip(
                ARM_JOINTS,
                np.quantile(np.concatenate(per_joint_speeds), 0.99, axis=0),
                np.quantile(np.concatenate(per_joint_accelerations), 0.99, axis=0),
                np.quantile(np.concatenate(per_joint_jerks), 0.99, axis=0),
                strict=True,
            )
        },
        "gripper_target_speed_rad_s": quantiles(gripper_speeds),
        "warning": (
            "Finite differences of recorded targets, not authoritative hardware limits. "
            "Maxima may include target discontinuities or noise."
        ),
    }


def _compare(actual: dict, expected: dict) -> dict:
    output = {}
    for key in ("mean", "std", "q01", "q99"):
        delta = np.asarray(actual[key]) - np.asarray(expected[key])
        output[key] = {
            "max_abs": float(np.max(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    paths, norm_path = download_public_assets(args.episodes)

    episodes = []
    timestamps = []
    for path in paths:
        table = pq.read_table(
            path, columns=["observation.state", "action", "timestamp"]
        )
        state = np.asarray(table.column("observation.state").to_pylist())
        action = np.asarray(table.column("action").to_pylist())
        timestamp = np.asarray(table.column("timestamp"))
        episodes.append((state, action))
        timestamps.append(timestamp)
    raw_state = np.concatenate([state for state, _ in episodes])
    raw_action = np.concatenate([action for _, action in episodes])
    eef_state = _fk_transform(raw_state)
    eef_action = _fk_transform(raw_action)

    published = json.loads(norm_path.read_text())["norm_stats"]
    computed_state = _stats(eef_state)
    computed_action = _stats(eef_action)
    quaternion_norms = np.concatenate([
        np.linalg.norm(eef_state[:, 3:7], axis=1),
        np.linalg.norm(eef_state[:, 10:14], axis=1),
        np.linalg.norm(eef_action[:, 3:7], axis=1),
        np.linalg.norm(eef_action[:, 10:14], axis=1),
    ])
    dts = np.concatenate([np.diff(t) for t in timestamps])
    lag = _best_action_state_lag(episodes)
    correlation = [
        float(np.corrcoef(raw_state[:, index], raw_action[:, index])[0, 1])
        for index in range(16)
    ]
    state_compare = _compare(computed_state, published["state"])
    action_compare = _compare(computed_action, published["actions"])
    max_mean_error = max(
        state_compare["mean"]["max_abs"], action_compare["mean"]["max_abs"]
    )
    max_stat_error = max(
        metric[stat]["max_abs"]
        for metric in (state_compare, action_compare)
        for stat in ("mean", "std", "q01", "q99")
    )
    report = {
        "dataset": DATASET,
        "episodes": len(episodes),
        "frames": int(len(raw_state)),
        "dimensions": int(raw_state.shape[1]),
        "timestamp": {
            "median_dt_s": float(np.median(dts)),
            "effective_fps": float(1.0 / np.median(dts)),
            "max_dt_error_s": float(np.max(np.abs(dts - 1 / 30))),
        },
        "raw_contract": {
            "metadata_representation": "14 arm joint positions + 2 Dex1 motor angles",
            "state_min": raw_state.min(axis=0).tolist(),
            "state_max": raw_state.max(axis=0).tolist(),
            "action_min": raw_action.min(axis=0).tolist(),
            "action_max": raw_action.max(axis=0).tolist(),
            "same_frame_state_action_correlation": correlation,
            "interpretation": "absolute joint targets, not EEF and not deltas",
        },
        "action_state_timing": lag,
        "training_motion_envelope": _motion_envelope(episodes, eef_action),
        "reconstructed_training_transform": {
            "representation": (
                "[left xyz, left xyzw, right xyz, right xyzw, left grip, right grip]"
            ),
            "frame": "pelvis",
            "eef_point": "wrist_yaw_link + [0.050, 0, 0] m in wrist frame",
            "quaternion_order": "xyzw",
            "quaternion_norm_max_error": float(
                np.max(np.abs(quaternion_norms - 1.0))
            ),
            "published_norm_stats_comparison": {
                "state": state_compare,
                "actions": action_compare,
                "max_mean_error": max_mean_error,
                "max_any_stat_error": max_stat_error,
            },
            "exactly_confirmed": bool(max_stat_error < 2e-4),
            "strongly_supported": bool(
                max_mean_error < 5e-3 and max_stat_error < 1.5e-2
            ),
        },
        "gripper": {
            "motor_convention": (
                "larger is more open: episode 0 frame 0 is visibly open at "
                "approximately [5.368, 5.383] rad; URDF joint mapping is reversed"
            ),
            "state_q01_q50_q99": np.quantile(
                raw_state[:, 14:], [0.01, 0.5, 0.99], axis=0
            ).tolist(),
            "action_q01_q50_q99": np.quantile(
                raw_action[:, 14:], [0.01, 0.5, 0.99], axis=0
            ).tolist(),
        },
        "verdict": (
            "The public parquet stores absolute joint-space commands. Pelvis-frame "
            "G1 FK with a 50 mm wrist offset strongly explains the published EEF "
            "statistics, but residual differences mean the author's exact transform "
            "is not confirmed."
        ),
    }
    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / "dataset_contract_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(output)
    print(json.dumps({
        "episodes": report["episodes"], "frames": report["frames"],
        "best_lag_frames": lag["best_lag_frames"],
        "max_norm_stat_error": max_stat_error,
        "transform_strongly_supported": report["reconstructed_training_transform"]["strongly_supported"],
        "transform_exactly_confirmed": report["reconstructed_training_transform"]["exactly_confirmed"],
    }, indent=2))


if __name__ == "__main__":
    main()
