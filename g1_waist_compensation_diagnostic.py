#!/usr/bin/env python3
"""Offline regression for zero-waist policy to measured-waist IK adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from g1_fast_sequential_ik import solve_sequential_ik
from g1_mujoco_bridge import policy_state_from_mujoco
from g1_policy_contract import ACTION_HORIZON
from g1_waist_compensation import apply_waist_compensation, compute_waist_compensation
from safety_governor import manipulator_contact_violations
from stack_scene import build_model, reset_to_reference_pose


MAXIMUM_MAPPING_POSITION_ERROR_M = 1e-7
MAXIMUM_MAPPING_ORIENTATION_ERROR_DEG = 1e-5


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _set_sample(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sample: dict[str, Any],
    *,
    measured_waist: bool,
) -> None:
    reset_to_reference_pose(model, data)
    for name, value in sample["arm_q_rad"].items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value
    for name, value in sample["waist_q_rad"].items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value if measured_waist else 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _orientation_error_deg(left: np.ndarray, right: np.ndarray) -> float:
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    return float(np.rad2deg(2.0 * np.arccos(np.clip(abs(left @ right), 0.0, 1.0))))


def run_diagnostic(sample_path: Path) -> dict[str, Any]:
    sample = json.loads(sample_path.read_text())
    model = build_model()
    zero = mujoco.MjData(model)
    measured = mujoco.MjData(model)
    _set_sample(model, zero, sample, measured_waist=False)
    _set_sample(model, measured, sample, measured_waist=True)
    grippers = np.zeros(2, dtype=np.float64)
    zero_state = policy_state_from_mujoco(model, zero, grippers).astype(np.float64)
    measured_state = policy_state_from_mujoco(model, measured, grippers).astype(np.float64)
    canonical = np.repeat(zero_state[None, :], ACTION_HORIZON, axis=0)
    canonical_before = canonical.copy()
    canonical_hash = _hash_array(canonical)
    compensation = compute_waist_compensation(model, zero, measured)
    ik_targets = apply_waist_compensation(canonical, compensation)

    position_errors = []
    orientation_errors = []
    for target in ik_targets:
        position_errors.extend((
            float(np.linalg.norm(target[:3] - measured_state[:3])),
            float(np.linalg.norm(target[7:10] - measured_state[7:10])),
        ))
        orientation_errors.extend((
            _orientation_error_deg(target[3:7], measured_state[3:7]),
            _orientation_error_deg(target[10:14], measured_state[10:14]),
        ))
    uncompensated = solve_sequential_ik(model, measured, canonical)
    compensated = solve_sequential_ik(model, measured, ik_targets)
    initial_collision_reasons = manipulator_contact_violations(
        model, measured, "free_space"
    )
    source_joint_count = len(sample["waist_q_rad"]) + len(sample["arm_q_rad"])
    return {
        "schema_version": "g1_waist_compensation_diagnostic_v1",
        "scope": (
            "Offline numeric adapter regression using one saved subscriber-only "
            "waist/arm sample; no physical EEF or full-body collision qualification."
        ),
        "input": {
            "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
            "captured_joint_count": source_joint_count,
            "required_full_body_joint_count": 29,
            "full_body_joint_state_available": source_joint_count == 29,
        },
        "hashes": {
            "canonical_policy_action_sha256": canonical_hash,
            "canonical_policy_action_after_adapter_sha256": _hash_array(canonical),
            "kinematic_ik_target_sha256": _hash_array(ik_targets),
            "canonical_input_mutated": not np.array_equal(canonical, canonical_before),
            "ik_target_is_separate_artifact": True,
        },
        "transform": {
            "pelvis_delta_transform": compensation.pelvis_delta_transform.tolist(),
            "maximum_position_mapping_error_m": max(position_errors),
            "maximum_orientation_mapping_error_deg": max(orientation_errors),
            "maximum_allowed_mapping_position_error_m": MAXIMUM_MAPPING_POSITION_ERROR_M,
            "maximum_allowed_mapping_orientation_error_deg": MAXIMUM_MAPPING_ORIENTATION_ERROR_DEG,
            "gripper_channels_preserved": bool(
                np.array_equal(ik_targets[:, 14:16], canonical[:, 14:16])
            ),
        },
        "uncompensated_ik": {
            "accepted": uncompensated.accepted,
            "reason": uncompensated.reason,
            "checked_targets": len(uncompensated.targets),
            "total_iterations": uncompensated.total_iterations,
            "elapsed_ms": uncompensated.elapsed_ms,
            "final_position_error_m": (
                uncompensated.targets[-1].maximum_position_error_m
                if uncompensated.targets else None
            ),
            "final_orientation_error_deg": (
                float(np.rad2deg(
                    uncompensated.targets[-1].maximum_orientation_error_rad
                )) if uncompensated.targets else None
            ),
        },
        "compensated_ik": {
            "accepted": compensated.accepted,
            "reason": compensated.reason,
            "checked_targets": len(compensated.targets),
            "total_iterations": compensated.total_iterations,
            "elapsed_ms": compensated.elapsed_ms,
            "maximum_position_error_m": max(
                target.maximum_position_error_m for target in compensated.targets
            ),
            "maximum_orientation_error_deg": float(np.rad2deg(max(
                target.maximum_orientation_error_rad for target in compensated.targets
            ))),
        },
        "collision": {
            "initial_model_collision_reasons": list(initial_collision_reasons),
            "full_body_state_available": source_joint_count == 29,
            "initial_collision_physically_qualified": False,
            "reason": (
                "Saved state contains only waist and arms; hip/hand collision cannot "
                "be qualified without the real 12 leg joints and physical inspection."
            ),
        },
        "decision": {
            "waist_transform_numeric_regression_passed": bool(
                max(position_errors) <= MAXIMUM_MAPPING_POSITION_ERROR_M
                and max(orientation_errors) <= MAXIMUM_MAPPING_ORIENTATION_ERROR_DEG
                and compensated.targets[-1].maximum_position_error_m <= 0.004
                and compensated.accepted
                and np.array_equal(canonical, canonical_before)
            ),
            "uncompensated_zero_waist_target_passed": uncompensated.accepted,
            "compensated_hold_target_passed": compensated.accepted,
            "physical_eef_parity_verified": False,
            "initial_collision_resolved": False,
            "live_full_body_capture_required": True,
            "robot_motion_allowed": False,
        },
        "safety": {
            "offline_only": True,
            "publisher_created": False,
            "robot_command_sent": False,
        },
        "g1_execution_enabled": False,
        "hardware_execution_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_diagnostic(args.sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["decision"]["waist_transform_numeric_regression_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
