#!/usr/bin/env python3
"""Explicit zero-waist policy-frame to measured-waist IK-frame adapter.

The frozen policy action remains unchanged and separately hashable. This module
produces a downstream kinematic target by applying the common torso transform
caused by measured waist joints. Physical frame calibration is still required
before hardware use.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from action_schema import (
    LEFT_POS,
    LEFT_QUAT,
    RIGHT_POS,
    RIGHT_QUAT,
    normalize_quaternion,
    vla_xyzw_to_mujoco_wxyz,
    mujoco_wxyz_to_vla_xyzw,
)


@dataclass(frozen=True)
class WaistCompensation:
    pelvis_delta_transform: np.ndarray
    zero_torso_transform: np.ndarray
    measured_torso_transform: np.ndarray


def _body_transform_in_pelvis(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> np.ndarray:
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if pelvis_id < 0 or body_id < 0:
        raise ValueError("pelvis or torso body is missing")

    def world_transform(body: int) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = data.xmat[body].reshape(3, 3)
        transform[:3, 3] = data.xpos[body]
        return transform

    return np.linalg.inv(world_transform(pelvis_id)) @ world_transform(body_id)


def compute_waist_compensation(
    model: mujoco.MjModel,
    zero_waist_data: mujoco.MjData,
    measured_waist_data: mujoco.MjData,
) -> WaistCompensation:
    zero = _body_transform_in_pelvis(model, zero_waist_data, "torso_link")
    measured = _body_transform_in_pelvis(model, measured_waist_data, "torso_link")
    delta = measured @ np.linalg.inv(zero)
    if not np.allclose(delta[:3, :3].T @ delta[:3, :3], np.eye(3), atol=1e-9):
        raise RuntimeError("waist compensation rotation is not orthonormal")
    return WaistCompensation(delta, zero, measured)


def apply_waist_compensation(
    canonical_actions: np.ndarray,
    compensation: WaistCompensation,
) -> np.ndarray:
    actions = np.asarray(canonical_actions, dtype=np.float64)
    single = actions.ndim == 1
    if single:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != 16 or not np.all(np.isfinite(actions)):
        raise ValueError("canonical_actions must be finite [T,16] or [16]")
    result = actions.copy()
    rotation = compensation.pelvis_delta_transform[:3, :3]
    translation = compensation.pelvis_delta_transform[:3, 3]
    for index in range(len(result)):
        for position_slice, quaternion_slice in (
            (LEFT_POS, LEFT_QUAT),
            (RIGHT_POS, RIGHT_QUAT),
        ):
            result[index, position_slice] = (
                rotation @ actions[index, position_slice] + translation
            )
            source_wxyz = vla_xyzw_to_mujoco_wxyz(
                normalize_quaternion(actions[index, quaternion_slice])
            )
            source_matrix = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(source_matrix, source_wxyz)
            target_matrix = rotation @ source_matrix.reshape(3, 3)
            target_wxyz = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(target_wxyz, target_matrix.ravel())
            result[index, quaternion_slice] = mujoco_wxyz_to_vla_xyzw(target_wxyz)
    # Gripper channels are not part of the rigid frame transform.
    if not np.array_equal(result[:, 14:16], actions[:, 14:16]):
        raise RuntimeError("waist compensation changed gripper channels")
    return result[0] if single else result
