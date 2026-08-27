#!/usr/bin/env python3
"""Thirty-trajectory offline verdict corpus for Legacy/Fast G1 preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from action_schema import EEFActionChunk
from g1_fast_preflight_correctness_corpus import _result_payload, _timed_check
from g1_fast_swept_path_preflight import G1FastSweptPathPreflight
from g1_mujoco_bridge import policy_state_from_mujoco
from g1_waist_compensation_diagnostic import _set_sample
from run_simulation import build_contract_fixture
from stack_scene import build_model, reset_to_reference_pose
from swept_path_preflight import G1SweptPathPreflight


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run_corpus(sample_path: Path) -> dict[str, Any]:
    model = build_model()
    reference = mujoco.MjData(model)
    reset_to_reference_pose(model, reference)
    timestamps = np.arange(32, dtype=np.float64) / 15.0
    cases: list[dict[str, Any]] = []

    # Eighteen reachable vertical trajectories cover hold through 20 cm lift.
    for index, offset in enumerate(np.linspace(0.0, 0.20, 18)):
        chunk, _, _ = build_contract_fixture(
            np.array([0.0, 0.0, float(offset)], dtype=np.float64)
        )
        cases.append({
            "name": f"reachable_vertical_{index:02d}",
            "category": "reachable",
            "parameter": {"vertical_offset_m": float(offset)},
            "source": reference,
            "chunk": chunk,
            "expected_accepted": True,
        })

    base, _, _ = build_contract_fixture(np.array([0.0, 0.0, 0.17]))
    unreachable_deltas = (
        ("forward_080", np.array([0.80, 0.0, 0.0])),
        ("forward_100", np.array([1.00, 0.0, 0.0])),
        ("forward_120", np.array([1.20, 0.0, 0.0])),
        ("down_030", np.array([0.0, 0.0, -0.30])),
        ("down_050", np.array([0.0, 0.0, -0.50])),
        ("up_070", np.array([0.0, 0.0, 0.70])),
        ("cross_positive", np.array([0.0, 0.50, 0.0])),
        ("cross_negative", np.array([0.0, -0.50, 0.0])),
    )
    for name, delta in unreachable_deltas:
        actions = base.actions.copy()
        actions[:, :3] += delta
        actions[:, 7:10] += delta
        cases.append({
            "name": f"unreachable_{name}",
            "category": "unreachable",
            "parameter": {"eef_delta_m": delta.tolist()},
            "source": reference,
            "chunk": EEFActionChunk(base.timestamps.copy(), actions),
            "expected_accepted": False,
        })

    hold_actions = np.repeat(base.actions[0:1], 32, axis=0)
    for value in (1.0, 3.0, 5.5):
        actions = hold_actions.copy()
        actions[:, 14:16] = value
        cases.append({
            "name": f"reachable_gripper_{value:.1f}",
            "category": "gripper_transition",
            "parameter": {"gripper_target_rad": value},
            "source": reference,
            "chunk": EEFActionChunk(timestamps.copy(), actions),
            "expected_accepted": True,
        })

    saved_sample = json.loads(sample_path.read_text())
    incomplete_pose = mujoco.MjData(model)
    _set_sample(model, incomplete_pose, saved_sample, measured_waist=True)
    measured_hold = np.repeat(policy_state_from_mujoco(
        model, incomplete_pose, np.zeros(2, dtype=np.float64)
    )[None, :], 32, axis=0)
    cases.append({
        "name": "modeled_initial_collision_incomplete_real_pose",
        "category": "initial_collision",
        "parameter": {"full_body_state_available": False},
        "source": incomplete_pose,
        "chunk": EEFActionChunk(timestamps.copy(), measured_hold),
        "expected_accepted": False,
    })
    if len(cases) != 30:
        raise RuntimeError(f"expected 30 cases, got {len(cases)}")

    records = []
    for case in cases:
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
        legacy_ms, legacy_result = _timed_check(
            legacy, case["source"], case["chunk"]
        )
        fast_ms, fast_result = _timed_check(
            fast, case["source"], case["chunk"]
        )
        expected = case["expected_accepted"]
        criteria = {
            "legacy_matches_expected": legacy_result.accepted == expected,
            "fast_matches_expected": fast_result.accepted == expected,
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
        records.append({
            "name": case["name"],
            "category": case["category"],
            "parameter": case["parameter"],
            "expected_accepted": expected,
            "action_sha256": _hash_array(case["chunk"].actions),
            "timestamp_sha256": _hash_array(case["chunk"].timestamps),
            "legacy": _result_payload(legacy_ms, legacy_result),
            "fast": _result_payload(fast_ms, fast_result),
            "criteria": criteria,
            "passed": all(criteria.values()),
        })

    fast_elapsed = [record["fast"]["elapsed_ms"] for record in records]
    category_counts = {
        category: sum(record["category"] == category for record in records)
        for category in sorted({record["category"] for record in records})
    }
    pass_count = sum(record["passed"] for record in records)
    return {
        "schema_version": "g1_fast_preflight_30_trajectory_corpus_v1",
        "scope": (
            "Thirty deterministic synthetic/reference trajectories: reachable "
            "vertical, unreachable, gripper, and one unqualified saved-pose collision. "
            "No physical, neural-task, randomized near-limit, or hardware evidence."
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
        "category_counts": category_counts,
        "cases": records,
        "summary": {
            "trajectory_count": len(records),
            "passed_count": pass_count,
            "acceptance_concordance_rate": sum(
                record["legacy"]["accepted"] == record["fast"]["accepted"]
                for record in records
            ) / len(records),
            "dangerous_fast_accept_count": sum(
                record["fast"]["accepted"] and not record["legacy"]["accepted"]
                for record in records
            ),
            "fast_elapsed_ms_p50": float(np.percentile(fast_elapsed, 50)),
            "fast_elapsed_ms_p95": float(np.percentile(fast_elapsed, 95)),
            "all_30_passed": pass_count == len(records),
        },
        "decision": {
            "minimum_30_trajectory_development_corpus_completed": len(records) >= 30,
            "development_corpus_passed": pass_count == len(records),
            "randomized_near_limit_coverage_completed": False,
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
    return 0 if report["summary"]["all_30_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
