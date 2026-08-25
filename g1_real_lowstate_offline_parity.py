#!/usr/bin/env python3
"""Zero-motion MuJoCo FK/IK diagnostic using a captured real G1 LowState sample.

This program has no Unitree SDK import, network access, publisher, or command
path. It consumes a saved waist+arm sample and checks whether the frozen
MuJoCo joint mapping can represent it and round-trip its current EEF pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from g1_dual_arm_ik import G1DualArmIK, LEFT_JOINTS, RIGHT_JOINTS, orientation_error
from g1_mujoco_bridge import policy_action_to_mujoco_world, policy_state_from_mujoco
from g1_policy_contract import (
    CONTRACT_ID, CONTRACT_SHA256, CONTRACT_VERSION, POLICY_RATE_HZ,
)
from g1_unitree_lowstate import CONTRACT_WAIST_JOINTS
from stack_scene import build_model, reset_to_reference_pose

POSITION_TOLERANCE_M = 0.005
ORIENTATION_TOLERANCE_RAD = float(np.deg2rad(3.0))
ARM_JOINTS = tuple(LEFT_JOINTS + RIGHT_JOINTS)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _joint_qpos(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"MuJoCo model is missing captured G1 joint: {name}")
    return joint_id, int(model.jnt_qposadr[joint_id])


def _load_sample(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    expected_waist = set(CONTRACT_WAIST_JOINTS)
    expected_arms = set(ARM_JOINTS)
    if set(payload.get("waist_q_rad", {})) != expected_waist:
        raise ValueError("captured waist joint order/names do not match the frozen mapping")
    if set(payload.get("arm_q_rad", {})) != expected_arms:
        raise ValueError("captured arm joint order/names do not match the frozen mapping")
    values = np.asarray([
        *payload["waist_q_rad"].values(),
        *payload["arm_q_rad"].values(),
    ], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("captured current-pose sample contains NaN/Inf")
    safety = payload.get("safety", {})
    if (
        safety.get("subscriber_only_source") is not True
        or safety.get("publisher_created") is not False
        or safety.get("robot_command_sent") is not False
    ):
        raise ValueError("current-pose fixture is not bound to subscriber-only evidence")
    return payload


def _pose_errors(solver: G1DualArmIK, target: np.ndarray) -> dict[str, float | bool]:
    left_position, left_quaternion = solver.pose("left")
    right_position, right_quaternion = solver.pose("right")
    position_errors = (
        float(np.linalg.norm(left_position - target[0:3])),
        float(np.linalg.norm(right_position - target[7:10])),
    )
    orientation_errors = (
        float(np.linalg.norm(orientation_error(target[3:7], left_quaternion))),
        float(np.linalg.norm(orientation_error(target[10:14], right_quaternion))),
    )
    maximum_position = max(position_errors)
    maximum_orientation = max(orientation_errors)
    return {
        "left_position_error_m": position_errors[0],
        "right_position_error_m": position_errors[1],
        "maximum_position_error_m": maximum_position,
        "left_orientation_error_rad": orientation_errors[0],
        "right_orientation_error_rad": orientation_errors[1],
        "maximum_orientation_error_rad": maximum_orientation,
        "maximum_orientation_error_deg": float(np.rad2deg(maximum_orientation)),
        "within_registered_tolerance": bool(
            maximum_position <= POSITION_TOLERANCE_M
            and maximum_orientation <= ORIENTATION_TOLERANCE_RAD
        ),
    }


def run_diagnostic(
    sample_path: Path,
    *,
    maximum_iterations: int = 250,
    seed_perturbation_rad: float = 0.02,
) -> dict[str, Any]:
    if not 1 <= maximum_iterations <= 1000:
        raise ValueError("maximum_iterations must be between 1 and 1000")
    if not 0.0 < seed_perturbation_rad <= 0.1:
        raise ValueError("seed_perturbation_rad must be in (0, 0.1]")
    sample = _load_sample(sample_path)
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)

    captured_q = {**sample["waist_q_rad"], **sample["arm_q_rad"]}
    range_records = []
    qpos_by_name: dict[str, int] = {}
    for name, value in captured_q.items():
        joint_id, qpos_id = _joint_qpos(model, name)
        qpos_by_name[name] = qpos_id
        value = float(value)
        limited = bool(model.jnt_limited[joint_id])
        low, high = (float(x) for x in model.jnt_range[joint_id])
        in_range = not limited or low <= value <= high
        range_records.append({
            "joint": name,
            "captured_q_rad": value,
            "model_range_rad": [low, high],
            "limited": limited,
            "in_range": in_range,
        })
        data.qpos[qpos_id] = value
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    solver = G1DualArmIK(model, data, damping=0.06, max_joint_speed=2.0)
    left_position, left_quaternion = solver.pose("left")
    right_position, right_quaternion = solver.pose("right")
    target_world = np.concatenate((
        left_position, left_quaternion, right_position, right_quaternion,
    ))
    pelvis_state = policy_state_from_mujoco(
        model, data, np.zeros(2, dtype=np.float64)
    ).astype(np.float64)
    world_roundtrip = policy_action_to_mujoco_world(model, data, pelvis_state)
    frame_position_error = max(
        float(np.linalg.norm(world_roundtrip[0:3] - target_world[0:3])),
        float(np.linalg.norm(world_roundtrip[7:10] - target_world[7:10])),
    )
    frame_orientation_error = max(
        float(np.linalg.norm(orientation_error(
            target_world[3:7], world_roundtrip[3:7]
        ))),
        float(np.linalg.norm(orientation_error(
            target_world[10:14], world_roundtrip[10:14]
        ))),
    )

    measured_arm_q = np.asarray(
        [float(sample["arm_q_rad"][name]) for name in ARM_JOINTS],
        dtype=np.float64,
    )
    perturbation = seed_perturbation_rad * np.where(
        np.arange(len(ARM_JOINTS)) % 2 == 0, 1.0, -1.0
    )
    for name, value in zip(ARM_JOINTS, measured_arm_q + perturbation, strict=True):
        joint_id, qpos_id = _joint_qpos(model, name)
        if model.jnt_limited[joint_id]:
            low, high = model.jnt_range[joint_id]
            value = float(np.clip(value, low, high))
        data.qpos[qpos_id] = value
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    solver = G1DualArmIK(model, data, damping=0.06, max_joint_speed=2.0)
    solver.reset()
    arm_qpos = np.concatenate((solver.left["qpos"], solver.right["qpos"]))
    initial_errors = _pose_errors(solver, target_world)
    first_passing_iteration = 0 if initial_errors["within_registered_tolerance"] else None
    final_errors = initial_errors
    executed_iterations = 0
    for iteration in range(1, maximum_iterations + 1):
        solver.step(target_world[0:7], target_world[7:14], 0.02)
        data.qpos[arm_qpos] = solver.q_target[arm_qpos]
        data.qvel[:] = 0.0
        mujoco.mj_fwdPosition(model, data)
        final_errors = _pose_errors(solver, target_world)
        executed_iterations = iteration
        if (
            first_passing_iteration is None
            and final_errors["within_registered_tolerance"]
        ):
            first_passing_iteration = iteration

    final_arm_q = data.qpos[arm_qpos].copy()
    model_mapping_passed = bool(all(item["in_range"] for item in range_records))
    frame_roundtrip_passed = bool(
        frame_position_error <= 1e-6 and frame_orientation_error <= 1e-6
    )
    ik_roundtrip_passed = bool(final_errors["within_registered_tolerance"])
    return {
        "schema_version": "g1_real_lowstate_offline_parity_v1",
        "scope": (
            "Zero-motion offline MuJoCo FK/frame/IK diagnostic seeded by one "
            "subscriber-only real G1 LowState sample."
        ),
        "input": {
            "sample_file": sample_path.name,
            "sample_sha256": _sha256(sample_path),
            "source_raw_capture_sha256": sample["source_raw_capture_sha256"],
            "tick": int(sample["tick"]),
            "mode_pr": int(sample["mode_pr"]),
            "mode_machine": int(sample["mode_machine"]),
            "mode_semantics_verified": bool(sample["mode_semantics_verified"]),
            "imu_used_for_pelvis_frame": False,
            "gripper_state_available": False,
        },
        "contract": {
            "id": CONTRACT_ID,
            "version": CONTRACT_VERSION,
            "sha256": CONTRACT_SHA256,
            "policy_rate_hz": POLICY_RATE_HZ,
        },
        "model": {
            "name": "MuJoCo Menagerie Unitree G1 29-DOF with frozen +X 0.05 m EEF sites",
            "captured_waist_joint_count": len(CONTRACT_WAIST_JOINTS),
            "captured_arm_joint_count": len(ARM_JOINTS),
            "joint_ranges": range_records,
            "all_captured_joints_present_and_in_model_range": model_mapping_passed,
        },
        "fk": {
            "pelvis_frame_eef_state_xyzw_without_grippers": pelvis_state[0:14].tolist(),
            "finite": bool(np.all(np.isfinite(pelvis_state[0:14]))),
            "physical_eef_measurement_available": False,
            "physical_eef_parity_verified": False,
        },
        "pelvis_world_frame_roundtrip": {
            "maximum_position_error_m": frame_position_error,
            "maximum_orientation_error_rad": frame_orientation_error,
            "passed": frame_roundtrip_passed,
        },
        "ik_current_pose_roundtrip": {
            "seed_perturbation_rad": seed_perturbation_rad,
            "maximum_iterations": maximum_iterations,
            "executed_iterations": executed_iterations,
            "first_passing_iteration": first_passing_iteration,
            "initial_errors": initial_errors,
            "final_errors": final_errors,
            "maximum_final_joint_difference_from_captured_rad": float(
                np.max(np.abs(final_arm_q - measured_arm_q))
            ),
            "passed": ik_roundtrip_passed,
        },
        "safety": {
            "offline_only": True,
            "unitree_sdk_imported": False,
            "network_accessed": False,
            "publisher_created": False,
            "robot_command_sent": False,
            "mode_change_requested": False,
        },
        "decision": {
            "real_lowstate_to_mujoco_joint_mapping_passed": model_mapping_passed,
            "numeric_fk_frame_roundtrip_passed": frame_roundtrip_passed,
            "numeric_current_pose_ik_roundtrip_passed": ik_roundtrip_passed,
            "physical_eef_parity_verified": False,
            "dds_freshness_fully_qualified": False,
            "zero_motion_policy_shadow_eligible": False,
            "robot_motion_allowed": False,
            "reason": (
                "Real waist/arm values are representable by the MuJoCo mapping and "
                "numeric FK/IK can be assessed, but no physical EEF measurement, "
                "camera synchronization, DDS freshness qualification, or command "
                "authority is established."
            ),
        },
        "g1_execution_enabled": False,
        "hardware_execution_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-iterations", type=int, default=250)
    parser.add_argument("--seed-perturbation-rad", type=float, default=0.02)
    args = parser.parse_args()
    report = run_diagnostic(
        args.sample,
        maximum_iterations=args.maximum_iterations,
        seed_perturbation_rad=args.seed_perturbation_rad,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    print(json.dumps({
        "mapping_passed": report["decision"]["real_lowstate_to_mujoco_joint_mapping_passed"],
        "frame_roundtrip_passed": report["decision"]["numeric_fk_frame_roundtrip_passed"],
        "ik_roundtrip_passed": report["decision"]["numeric_current_pose_ik_roundtrip_passed"],
        "final_errors": report["ik_current_pose_roundtrip"]["final_errors"],
        "robot_motion_allowed": report["decision"]["robot_motion_allowed"],
    }, indent=2))


if __name__ == "__main__":
    main()
