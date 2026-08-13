"""Build conservative adaptive-speed context from live G1 MuJoCo state.

The future hardware implementation must replace MuJoCo geometry/contact access
with calibrated perception and controller feedback while preserving the same
fail-closed AdaptiveSafetyContext contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from adaptive_speed_context import AdaptiveSafetyContext
from g1_dual_arm_ik import LEFT_JOINTS, RIGHT_JOINTS


@dataclass(frozen=True)
class SimulationContextConfig:
    approach_clearance_m: float = 0.060
    gripper_transition_rad: float = 0.15
    maximum_distance_query_m: float = 2.0
    nominal_ik_margin_cap_rad: float = 0.20


@dataclass(frozen=True)
class SimulationContextEvidence:
    task_phase: str
    minimum_dex_cube_clearance_m: float
    dex_cube_contact: bool
    gripper_error_rad: tuple[float, float]
    joint_limit_margin_rad: float
    pelvis_stability: float


def _body_name(model: mujoco.MjModel, geom_id: int) -> str:
    body_id = int(model.geom_bodyid[geom_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"


def _task_geometries(model: mujoco.MjModel) -> tuple[list[int], list[int]]:
    dex_geoms: list[int] = []
    cube_geoms: list[int] = []
    for geom_id in range(model.ngeom):
        body = _body_name(model, geom_id)
        if "dex1" in body:
            dex_geoms.append(geom_id)
        if body.endswith("_cube"):
            cube_geoms.append(geom_id)
    if not dex_geoms or len(cube_geoms) != 3:
        raise ValueError("Expected Dex1 and three cube collision geometries")
    return dex_geoms, cube_geoms


def minimum_dex_cube_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    maximum_distance_m: float = 2.0,
) -> float:
    dex_geoms, cube_geoms = _task_geometries(model)
    closest = np.inf
    fromto = np.empty(6, dtype=np.float64)
    for dex_geom in dex_geoms:
        for cube_geom in cube_geoms:
            distance = float(mujoco.mj_geomDistance(
                model, data, dex_geom, cube_geom,
                maximum_distance_m, fromto,
            ))
            closest = min(closest, max(distance, 0.0))
    if not np.isfinite(closest):
        raise RuntimeError("Could not compute Dex1-to-cube clearance")
    return float(closest)


def dex_cube_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    for index in range(data.ncon):
        contact = data.contact[index]
        body1 = _body_name(model, int(contact.geom1))
        body2 = _body_name(model, int(contact.geom2))
        if (
            ("dex1" in body1 and body2.endswith("_cube"))
            or ("dex1" in body2 and body1.endswith("_cube"))
        ):
            return True
    return False


def joint_limit_margin(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    margins: list[float] = []
    for name in LEFT_JOINTS + RIGHT_JOINTS:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or not model.jnt_limited[joint]:
            continue
        qpos = int(model.jnt_qposadr[joint])
        low, high = model.jnt_range[joint]
        margins.append(float(min(data.qpos[qpos] - low, high - data.qpos[qpos])))
    if not margins:
        raise ValueError("No limited G1 arm joints found")
    return float(max(0.0, min(margins)))


def pelvis_stability(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    rotation = data.xmat[pelvis].reshape(3, 3)
    world_up_in_pelvis = rotation.T @ np.array([0.0, 0.0, 1.0])
    tilt = float(np.arccos(np.clip(world_up_in_pelvis[2], -1.0, 1.0)))
    root_angular_speed = float(np.linalg.norm(data.qvel[3:6])) if model.nv >= 6 else 0.0
    return float(np.clip(1.0 - tilt / np.deg2rad(20.0) - root_angular_speed / 2.0, 0.0, 1.0))


def build_simulation_speed_context(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    commanded_grippers_rad: np.ndarray,
    measured_grippers_rad: np.ndarray,
    eef_tracking_error_m: float,
    observation_age_ms: float,
    policy_response_age_ms: float,
    preflight_passed: bool,
    collision_free: bool,
    command_limits_passed: bool,
    gripper_tracking_error_rad: float | None = None,
    network_timeout: bool = False,
    config: SimulationContextConfig | None = None,
) -> tuple[AdaptiveSafetyContext, SimulationContextEvidence]:
    cfg = config or SimulationContextConfig()
    commanded = np.asarray(commanded_grippers_rad, dtype=np.float64)
    measured = np.asarray(measured_grippers_rad, dtype=np.float64)
    if commanded.shape != (2,) or measured.shape != (2,):
        raise ValueError("commanded and measured grippers must have shape (2,)")
    mujoco.mj_forward(model, data)
    clearance = minimum_dex_cube_clearance(
        model, data, maximum_distance_m=cfg.maximum_distance_query_m
    )
    contact = dex_cube_contact(model, data)
    gripper_error = commanded - measured
    measured_tracking_error = (
        float(np.max(np.abs(gripper_error)))
        if gripper_tracking_error_rad is None
        else float(gripper_tracking_error_rad)
    )
    if not np.isfinite(measured_tracking_error) or measured_tracking_error < 0.0:
        raise ValueError("gripper_tracking_error_rad must be finite and non-negative")
    closing = bool(np.any(gripper_error < -cfg.gripper_transition_rad))
    opening = bool(np.any(gripper_error > cfg.gripper_transition_rad))
    if contact and opening:
        phase = "place"
    elif contact or closing:
        phase = "grasp"
    elif clearance <= cfg.approach_clearance_m:
        phase = "approach"
    else:
        phase = "free_space"
    margin = joint_limit_margin(model, data)
    stability = pelvis_stability(model, data)
    context = AdaptiveSafetyContext(
        task_phase=phase,
        distance_to_goal_m=clearance,
        minimum_clearance_m=clearance,
        eef_tracking_error_m=float(eef_tracking_error_m),
        observation_age_ms=float(observation_age_ms),
        policy_response_age_ms=float(policy_response_age_ms),
        ik_margin_rad=min(margin, cfg.nominal_ik_margin_cap_rad),
        joint_limit_margin_rad=margin,
        pelvis_stability=stability,
        gripper_tracking_error_rad=measured_tracking_error,
        contact=contact,
        network_timeout=bool(network_timeout),
        hard_safety_gate_passed=bool(preflight_passed),
        ik_reachable=bool(preflight_passed),
        collision_free=bool(collision_free),
        command_limits_passed=bool(command_limits_passed),
    )
    evidence = SimulationContextEvidence(
        task_phase=phase,
        minimum_dex_cube_clearance_m=clearance,
        dex_cube_contact=contact,
        gripper_error_rad=(float(gripper_error[0]), float(gripper_error[1])),
        joint_limit_margin_rad=margin,
        pelvis_stability=stability,
    )
    return context, evidence
