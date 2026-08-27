#!/usr/bin/env python3
"""Deterministic near-limit/collision-boundary fuzzing for G1 preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from action_schema import EEFActionChunk, slerp
from g1_dual_arm_ik import LEFT_JOINTS, RIGHT_JOINTS
from g1_fast_preflight_correctness_corpus import _result_payload, _timed_check
from g1_fast_swept_path_preflight import G1FastSweptPathPreflight
from g1_mujoco_bridge import policy_state_from_mujoco
from safety_governor import manipulator_contact_violations
from stack_scene import build_model, reset_to_reference_pose
from swept_path_preflight import G1SweptPathPreflight


SEED = 20260827
CASE_COUNT = 30


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _trajectory(start: np.ndarray, end: np.ndarray) -> EEFActionChunk:
    actions = np.empty((32, 16), dtype=np.float64)
    for index, fraction in enumerate(np.linspace(0.0, 1.0, 32)):
        actions[index] = (1.0 - fraction) * start + fraction * end
        actions[index, 3:7] = slerp(start[3:7], end[3:7], fraction)
        actions[index, 10:14] = slerp(start[10:14], end[10:14], fraction)
    return EEFActionChunk(np.arange(32, dtype=np.float64) / 15.0, actions)


def run_fuzz() -> dict[str, Any]:
    model = build_model()
    source = mujoco.MjData(model)
    reset_to_reference_pose(model, source)
    start = policy_state_from_mujoco(
        model, source, np.zeros(2, dtype=np.float64)
    ).astype(np.float64)
    names = LEFT_JOINTS + RIGHT_JOINTS
    joint_ids = np.asarray([
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in names
    ], dtype=np.int32)
    qpos_indices = model.jnt_qposadr[joint_ids]
    rng = np.random.default_rng(SEED)
    cases = []
    for case_index in range(CASE_COUNT):
        target_data = mujoco.MjData(model)
        target_data.qpos[:] = source.qpos
        target_data.qvel[:] = 0.0
        if case_index < 20:
            category = "near_limit_two_joint"
            selected = rng.choice(len(names), size=2, replace=False)
            lower_fraction, upper_fraction = 0.02, 0.12
        else:
            category = "boundary_one_joint"
            selected = rng.choice(len(names), size=1, replace=False)
            lower_fraction, upper_fraction = 0.20, 0.30
        selected_payload = []
        for joint_index in selected:
            joint_id = joint_ids[joint_index]
            low, high = model.jnt_range[joint_id]
            toward_upper = bool(rng.integers(0, 2))
            if toward_upper:
                fraction = float(rng.uniform(1.0 - upper_fraction, 1.0 - lower_fraction))
            else:
                fraction = float(rng.uniform(lower_fraction, upper_fraction))
            value = float(low + fraction * (high - low))
            target_data.qpos[qpos_indices[joint_index]] = value
            selected_payload.append({
                "joint": names[joint_index],
                "fraction_of_joint_range": fraction,
                "target_rad": value,
                "range_rad": [float(low), float(high)],
            })
        mujoco.mj_forward(model, target_data)
        end = policy_state_from_mujoco(
            model, target_data, np.zeros(2, dtype=np.float64)
        ).astype(np.float64)
        chunk = _trajectory(start, end)
        target_collision = manipulator_contact_violations(
            model, target_data, "free_space"
        )
        legacy = G1SweptPathPreflight(
            model,
            position_tolerance_m=0.005,
            orientation_tolerance_rad=np.deg2rad(3.0),
            ik_iterations=250,
            maximum_arm_interpolation_step_rad=0.01,
            maximum_finger_interpolation_step_m=0.001,
        )
        fast = G1FastSweptPathPreflight(
            model,
            maximum_arm_interpolation_step_rad=0.01,
            maximum_finger_interpolation_step_m=0.001,
        )
        legacy_ms, legacy_result = _timed_check(legacy, source, chunk)
        fast_ms, fast_result = _timed_check(fast, source, chunk)
        criteria = {
            "acceptance_concordant": legacy_result.accepted == fast_result.accepted,
            "no_dangerous_fast_accept": not (
                fast_result.accepted and not legacy_result.accepted
            ),
            "accepted_fast_residual_within_internal_gate": bool(
                not fast_result.accepted
                or (
                    fast_result.maximum_position_error_m <= 0.004
                    and fast_result.maximum_orientation_error_rad <= np.deg2rad(2.5)
                )
            ),
        }
        cases.append({
            "case": case_index,
            "category": category,
            "selected_joints": selected_payload,
            "target_qpos_sha256": _hash_array(target_data.qpos),
            "action_sha256": _hash_array(chunk.actions),
            "timestamp_sha256": _hash_array(chunk.timestamps),
            "target_configuration_collision_reasons": list(target_collision),
            "legacy": _result_payload(legacy_ms, legacy_result),
            "fast": _result_payload(fast_ms, fast_result),
            "criteria": criteria,
            "passed": all(criteria.values()),
        })

    pass_count = sum(case["passed"] for case in cases)
    dangerous = sum(
        case["fast"]["accepted"] and not case["legacy"]["accepted"]
        for case in cases
    )
    fast_elapsed = [case["fast"]["elapsed_ms"] for case in cases]
    return {
        "schema_version": "g1_fast_preflight_near_limit_fuzz_v1",
        "scope": (
            "Thirty deterministic FK-generated near-limit/boundary trajectories. "
            "Software-model fuzzing only; not physical limits, randomized property "
            "proof, torque/current qualification, or hardware evidence."
        ),
        "input": {
            "seed": SEED,
            "case_count": CASE_COUNT,
            "near_limit_two_joint_cases": 20,
            "boundary_one_joint_cases": 10,
        },
        "comparison_contract": {
            "registered_position_tolerance_m": 0.005,
            "registered_orientation_tolerance_deg": 3.0,
            "fast_internal_position_stop_m": 0.004,
            "fast_internal_orientation_stop_deg": 2.5,
            "maximum_arm_interpolation_step_rad": 0.01,
            "maximum_finger_interpolation_step_m": 0.001,
            "collision_classifier_identical": True,
            "threshold_relaxed": False,
        },
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "passed_count": pass_count,
            "legacy_accepted_count": sum(case["legacy"]["accepted"] for case in cases),
            "fast_accepted_count": sum(case["fast"]["accepted"] for case in cases),
            "acceptance_concordance_rate": sum(
                case["legacy"]["accepted"] == case["fast"]["accepted"]
                for case in cases
            ) / len(cases),
            "dangerous_fast_accept_count": dangerous,
            "fast_elapsed_ms_p50": float(np.percentile(fast_elapsed, 50)),
            "fast_elapsed_ms_p95": float(np.percentile(fast_elapsed, 95)),
            "all_cases_passed": pass_count == len(cases),
            "accepted_case_coverage_observed": any(
                case["legacy"]["accepted"] for case in cases
            ),
        },
        "decision": {
            "deterministic_near_limit_fuzz_completed": True,
            "deterministic_near_limit_rejection_concordance_passed": pass_count == len(cases),
            "accepted_near_limit_coverage_passed": any(
                case["legacy"]["accepted"] for case in cases
            ),
            "dangerous_fast_accept_observed": dangerous > 0,
            "official_hardware_limits_qualified": False,
            "physical_collision_qualified": False,
            "real_robot_compute_benchmark_completed": False,
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_fuzz()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["summary"]["all_cases_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
