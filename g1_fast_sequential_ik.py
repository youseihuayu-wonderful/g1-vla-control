#!/usr/bin/env python3
"""Warm-started, tolerance-aware sequential MuJoCo IK for offline preflight.

Unlike the legacy fixed-250-iteration loop, each target stops immediately after
meeting the unchanged 5 mm / 3 degree Gate. This module has no hardware I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import mujoco
import numpy as np

from action_schema import pelvis_vla_action_to_world_mujoco
from g1_dual_arm_ik import G1DualArmIK, orientation_error


REGISTERED_POSITION_TOLERANCE_M = 0.005
REGISTERED_ORIENTATION_TOLERANCE_RAD = np.deg2rad(3.0)
# Stop inside the registered boundary to retain numerical/replay margin.
DEFAULT_EARLY_STOP_POSITION_M = 0.004
DEFAULT_EARLY_STOP_ORIENTATION_RAD = np.deg2rad(2.5)


@dataclass(frozen=True)
class FastIKConfig:
    damping: float = 0.03
    iteration_step_s: float = 0.04
    maximum_iterations: int = 60
    max_joint_speed_rad_s: float = 2.0
    position_tolerance_m: float = DEFAULT_EARLY_STOP_POSITION_M
    orientation_tolerance_rad: float = DEFAULT_EARLY_STOP_ORIENTATION_RAD


@dataclass(frozen=True)
class SequentialIKTarget:
    index: int
    iterations: int
    joint_target_rad: np.ndarray
    maximum_position_error_m: float
    maximum_orientation_error_rad: float
    reachable: bool
    joint_limits_ok: bool


@dataclass(frozen=True)
class SequentialIKResult:
    accepted: bool
    targets: tuple[SequentialIKTarget, ...]
    elapsed_ms: float
    total_iterations: int
    reason: str


def _errors(solver: G1DualArmIK, target: np.ndarray) -> tuple[float, float]:
    left_position, left_quaternion = solver.pose("left")
    right_position, right_quaternion = solver.pose("right")
    position = float(max(
        np.linalg.norm(left_position - target[:3]),
        np.linalg.norm(right_position - target[7:10]),
    ))
    orientation = float(max(
        np.linalg.norm(orientation_error(target[3:7], left_quaternion)),
        np.linalg.norm(orientation_error(target[10:14], right_quaternion)),
    ))
    return position, orientation


def solve_sequential_ik(
    model: mujoco.MjModel,
    source: mujoco.MjData,
    pelvis_actions_xyzw: np.ndarray,
    config: FastIKConfig = FastIKConfig(),
) -> SequentialIKResult:
    actions = np.asarray(pelvis_actions_xyzw, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 16 or not np.all(np.isfinite(actions)):
        raise ValueError("pelvis_actions_xyzw must be finite [T,16]")
    if config.maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")

    data = mujoco.MjData(model)
    data.qpos[:] = source.qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = source.ctrl
    mujoco.mj_forward(model, data)
    solver = G1DualArmIK(
        model,
        data,
        damping=config.damping,
        max_joint_speed=config.max_joint_speed_rad_s,
    )
    solver.reset()
    arm_qpos = np.concatenate((solver.left["qpos"], solver.right["qpos"]))
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    records = []
    started = time.perf_counter_ns()
    failure_reason = "accepted"
    for index, action in enumerate(actions):
        target = pelvis_vla_action_to_world_mujoco(
            action, data.xpos[pelvis], data.xquat[pelvis]
        )
        iterations = 0
        position_error, orientation_error_rad = _errors(solver, target)
        while not (
            position_error <= config.position_tolerance_m
            and orientation_error_rad <= config.orientation_tolerance_rad
        ) and iterations < config.maximum_iterations:
            solver.step(
                target[:7], target[7:14], config.iteration_step_s
            )
            data.qpos[arm_qpos] = solver.q_target[arm_qpos]
            data.qvel[:] = 0.0
            mujoco.mj_fwdPosition(model, data)
            iterations += 1
            position_error, orientation_error_rad = _errors(solver, target)

        joint_limits_ok = True
        for info in (solver.left, solver.right):
            for joint_id, qpos_index in zip(info["joint"], info["qpos"], strict=True):
                if model.jnt_limited[joint_id]:
                    low, high = model.jnt_range[joint_id]
                    joint_limits_ok &= bool(
                        low - 1e-9 <= data.qpos[qpos_index] <= high + 1e-9
                    )
        reachable = bool(
            position_error <= config.position_tolerance_m
            and orientation_error_rad <= config.orientation_tolerance_rad
        )
        records.append(SequentialIKTarget(
            index=index,
            iterations=iterations,
            joint_target_rad=data.qpos[arm_qpos].copy(),
            maximum_position_error_m=position_error,
            maximum_orientation_error_rad=orientation_error_rad,
            reachable=reachable,
            joint_limits_ok=joint_limits_ok,
        ))
        if not reachable:
            failure_reason = "ik_tolerance_not_reached"
            break
        if not joint_limits_ok:
            failure_reason = "joint_limits_failed"
            break

    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    accepted = len(records) == len(actions) and failure_reason == "accepted"
    return SequentialIKResult(
        accepted=accepted,
        targets=tuple(records),
        elapsed_ms=elapsed_ms,
        total_iterations=sum(record.iterations for record in records),
        reason=failure_reason,
    )
