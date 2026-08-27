#!/usr/bin/env python3
"""Offline correctness corpus for legacy and fast G1 swept preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np

from action_schema import EEFActionChunk
from g1_adaptive_phase_validation import FAR_LIFT_OFFSET_M
from g1_fast_swept_path_preflight import G1FastSweptPathPreflight
from g1_mujoco_bridge import policy_state_from_mujoco
from g1_waist_compensation_diagnostic import _set_sample
from run_simulation import build_contract_fixture
from stack_scene import build_model, reset_to_reference_pose
from swept_path_preflight import G1SweptPathPreflight


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _timed_check(gate, source, chunk):
    started = time.perf_counter_ns()
    result = gate.check(source, chunk, phase="free_space")
    return (time.perf_counter_ns() - started) / 1_000_000.0, result


def _result_payload(elapsed_ms: float, result) -> dict[str, Any]:
    return {
        "elapsed_ms": elapsed_ms,
        "accepted": bool(result.accepted),
        "reason": result.reason,
        "collision_reasons": list(result.collision_reasons),
        "checked_targets": int(result.checked_targets),
        "checked_interpolated_configurations": int(
            result.checked_interpolated_configurations
        ),
        "maximum_position_error_m": float(result.maximum_position_error_m),
        "maximum_orientation_error_rad": float(result.maximum_orientation_error_rad),
    }


def run_corpus(sample_path: Path) -> dict[str, Any]:
    model = build_model()
    reference = mujoco.MjData(model)
    reset_to_reference_pose(model, reference)
    base, _, _ = build_contract_fixture(FAR_LIFT_OFFSET_M)
    hold_actions = np.repeat(base.actions[0:1], len(base.actions), axis=0)
    cases: list[tuple[str, mujoco.MjData, EEFActionChunk, bool, str]] = [
        (
            "reachable_hold", reference,
            EEFActionChunk(base.timestamps.copy(), hold_actions.copy()),
            True, "reference hold must pass",
        ),
        ("reachable_far_lift", reference, base, True, "validated reach must pass"),
    ]
    gripper_actions = hold_actions.copy()
    gripper_actions[:, 14:16] = 5.5
    cases.append((
        "reachable_gripper_transition", reference,
        EEFActionChunk(base.timestamps.copy(), gripper_actions),
        True, "finger interpolation must remain collision-free",
    ))
    for name, delta in (
        ("unreachable_forward", np.array([1.0, 0.0, 0.0])),
        ("unreachable_downward", np.array([0.0, 0.0, -0.4])),
        ("unreachable_cross_body", np.array([0.0, -0.4, 0.0])),
    ):
        actions = base.actions.copy()
        actions[:, :3] += delta
        actions[:, 7:10] += delta
        cases.append((
            name, reference,
            EEFActionChunk(base.timestamps.copy(), actions),
            False, "unreachable EEF target must reject",
        ))

    saved_sample = json.loads(sample_path.read_text())
    incomplete_real_pose = mujoco.MjData(model)
    _set_sample(model, incomplete_real_pose, saved_sample, measured_waist=True)
    measured_hold = np.repeat(
        policy_state_from_mujoco(
            model, incomplete_real_pose, np.zeros(2, dtype=np.float64)
        )[None, :],
        len(base.actions),
        axis=0,
    )
    cases.append((
        "saved_incomplete_pose_initial_collision",
        incomplete_real_pose,
        EEFActionChunk(base.timestamps.copy(), measured_hold),
        False,
        "both implementations must reject the modeled initial collision; this does not physically qualify it",
    ))

    records = []
    for name, source, chunk, expected, expectation in cases:
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
        legacy_payload = _result_payload(legacy_ms, legacy_result)
        fast_payload = _result_payload(fast_ms, fast_result)
        criteria = {
            "legacy_matches_expected_acceptance": legacy_result.accepted == expected,
            "fast_matches_expected_acceptance": fast_result.accepted == expected,
            "acceptance_concordant": legacy_result.accepted == fast_result.accepted,
            "fast_does_not_accept_legacy_rejection": not (
                fast_result.accepted and not legacy_result.accepted
            ),
            "accepted_fast_residual_within_internal_4mm_2_5deg": bool(
                not fast_result.accepted
                or (
                    fast_result.maximum_position_error_m <= 0.004
                    and fast_result.maximum_orientation_error_rad <= np.deg2rad(2.5)
                )
            ),
        }
        if name == "saved_incomplete_pose_initial_collision":
            criteria["initial_collision_reason_concordant"] = bool(
                legacy_result.reason == fast_result.reason
                == "initial_configuration_collision"
                and legacy_result.collision_reasons == fast_result.collision_reasons
            )
        records.append({
            "name": name,
            "expected_accepted": expected,
            "expectation": expectation,
            "action_sha256": _hash_array(chunk.actions),
            "timestamp_sha256": _hash_array(chunk.timestamps),
            "legacy": legacy_payload,
            "fast": fast_payload,
            "criteria": criteria,
            "passed": all(criteria.values()),
        })

    schema_faults = []
    valid_actions = hold_actions.copy()
    for name, mutate, message in (
        ("nan_action", lambda t, a: a.__setitem__((0, 0), np.nan), "actions must be finite"),
        ("inf_action", lambda t, a: a.__setitem__((0, 0), np.inf), "actions must be finite"),
        ("nan_timestamp", lambda t, a: t.__setitem__(1, np.nan), "timestamps must be finite"),
    ):
        timestamps = base.timestamps.copy()
        actions = valid_actions.copy()
        mutate(timestamps, actions)
        exception = None
        try:
            EEFActionChunk(timestamps, actions)
        except Exception as error:  # boundary must reject before any solver runs
            exception = {"type": type(error).__name__, "message": str(error)}
        schema_faults.append({
            "name": name,
            "rejected": exception is not None,
            "exception": exception,
            "expected_message": message,
            "passed": bool(exception and message in exception["message"]),
        })

    case_passes = sum(record["passed"] for record in records)
    fault_passes = sum(fault["passed"] for fault in schema_faults)
    fast_elapsed = [record["fast"]["elapsed_ms"] for record in records]
    return {
        "schema_version": "g1_fast_preflight_correctness_corpus_v1",
        "scope": (
            "Seven deterministic MuJoCo verdict cases plus three schema faults; "
            "development corpus, not physical collision or task evidence."
        ),
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
        "cases": records,
        "schema_faults": schema_faults,
        "summary": {
            "verdict_case_count": len(records),
            "verdict_case_pass_count": case_passes,
            "schema_fault_count": len(schema_faults),
            "schema_fault_pass_count": fault_passes,
            "dangerous_fast_accept_count": sum(
                record["fast"]["accepted"] and not record["legacy"]["accepted"]
                for record in records
            ),
            "acceptance_concordance_rate": sum(
                record["legacy"]["accepted"] == record["fast"]["accepted"]
                for record in records
            ) / len(records),
            "fast_elapsed_ms_p50": float(np.percentile(fast_elapsed, 50)),
            "fast_elapsed_ms_p95": float(np.percentile(fast_elapsed, 95)),
            "development_corpus_passed": (
                case_passes == len(records) and fault_passes == len(schema_faults)
            ),
        },
        "decision": {
            "development_correctness_corpus_passed": (
                case_passes == len(records) and fault_passes == len(schema_faults)
            ),
            "minimum_30_trajectory_corpus_completed": False,
            "saved_initial_collision_physically_qualified": False,
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
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_corpus(args.sample)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["summary"]["development_corpus_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
