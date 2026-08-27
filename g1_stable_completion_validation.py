#!/usr/bin/env python3
"""Validate timestamp-only speed with a stable task-completion definition.

A run completes only after the final action is active and both EEFs remain
within the unchanged 5 mm / 3 degree Gate for 250 ms. This replaces the prior
path-duration-plus-fixed-settle proxy for task-time decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_retimer import AdaptiveRetimer, RetimerConfig
from retiming_safety_validation import _run_scale
from run_simulation import build_contract_fixture, preflight_contract_chunk
from stack_scene import REFERENCE_EP0_STATE


LIFT_OFFSETS_M = (0.10, 0.13, 0.15, 0.17, 0.19)
POSITION_TOLERANCE_M = 0.005
ORIENTATION_TOLERANCE_RAD = np.deg2rad(3.0)
COMPLETION_HOLD_S = 0.250
MAXIMUM_SETTLE_S = 3.0
MINIMUM_DURATION_REDUCTION = 0.10


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _stable_run(chunk):
    return _run_scale(
        chunk,
        REFERENCE_EP0_STATE[14:16],
        scale=1.0,
        use_filter=True,
        use_joint_filter=True,
        stop_when_settled=True,
        completion_position_tolerance_m=POSITION_TOLERANCE_M,
        completion_orientation_tolerance_rad=ORIENTATION_TOLERANCE_RAD,
        completion_hold_s=COMPLETION_HOLD_S,
        maximum_settle_s=MAXIMUM_SETTLE_S,
    )


def run_validation(optimizer_report_path: Path) -> dict[str, Any]:
    optimizer_bytes = optimizer_report_path.read_bytes()
    optimizer = json.loads(optimizer_bytes)
    selected = optimizer.get("selected")
    if not selected:
        raise ValueError("optimizer report has no selected candidate")
    config = RetimerConfig(**selected["config"])
    pairs = []
    for pair_index, lift_z in enumerate(LIFT_OFFSETS_M):
        chunk, left_target, right_target = build_contract_fixture(
            np.array([0.0, 0.0, lift_z], dtype=np.float64)
        )
        preflight = preflight_contract_chunk(chunk)
        retimed = AdaptiveRetimer(config).retime(chunk, left_target, right_target)
        baseline = _stable_run(chunk)
        adaptive = _stable_run(retimed.chunk)
        if baseline["task_completed"] and adaptive["task_completed"]:
            reduction = (
                1.0
                - adaptive["task_completion_time_s"]
                / baseline["task_completion_time_s"]
            )
            speedup = (
                baseline["task_completion_time_s"]
                / adaptive["task_completion_time_s"] - 1.0
            )
        else:
            reduction = None
            speedup = None
        baseline_jerk = baseline["maxima"]["actual_joint_jerk_rad_s3"]
        adaptive_jerk = adaptive["maxima"]["actual_joint_jerk_rad_s3"]
        jerk_regression = adaptive_jerk / baseline_jerk - 1.0
        criteria = {
            "preflight_all_accepted": preflight["all_accepted"],
            "action_samples_byte_identical": bool(
                np.array_equal(chunk.actions, retimed.chunk.actions)
            ),
            "baseline_completed_stable_gate": baseline["task_completed"],
            "adaptive_completed_stable_gate": adaptive["task_completed"],
            "stable_completion_time_reduction_at_least_10_percent": bool(
                reduction is not None and reduction >= MINIMUM_DURATION_REDUCTION
            ),
            "actual_joint_jerk_noninferior": jerk_regression <= 1e-12,
            "baseline_hard_command_limits_pass": baseline["hard_command_limits_pass"],
            "adaptive_hard_command_limits_pass": adaptive["hard_command_limits_pass"],
            "baseline_no_manipulation_contact": (
                baseline["manipulation_contact_step_rate"] == 0.0
            ),
            "adaptive_no_manipulation_contact": (
                adaptive["manipulation_contact_step_rate"] == 0.0
            ),
            "baseline_finite": baseline["finite"],
            "adaptive_finite": adaptive["finite"],
        }
        pairs.append({
            "pair": pair_index,
            "lift_offset_m": lift_z,
            "action_sha256_off": _hash_array(chunk.actions),
            "action_sha256_on": _hash_array(retimed.chunk.actions),
            "timestamp_sha256_off": _hash_array(chunk.timestamps),
            "timestamp_sha256_on": _hash_array(retimed.chunk.timestamps),
            "nominal_path_duration_off_s": baseline["nominal_path_duration_s"],
            "nominal_path_duration_on_s": adaptive["nominal_path_duration_s"],
            "stable_completion_time_off_s": baseline["task_completion_time_s"],
            "stable_completion_time_on_s": adaptive["task_completion_time_s"],
            "stable_completion_duration_reduction_fraction": reduction,
            "stable_completion_effective_speedup_fraction": speedup,
            "baseline_endpoint_position_error_m": baseline["endpoint_error_m"],
            "adaptive_endpoint_position_error_m": adaptive["endpoint_error_m"],
            "baseline_endpoint_orientation_error_deg": float(np.rad2deg(
                baseline["endpoint_orientation_error_rad"]
            )),
            "adaptive_endpoint_orientation_error_deg": float(np.rad2deg(
                adaptive["endpoint_orientation_error_rad"]
            )),
            "actual_joint_jerk_regression_fraction": jerk_regression,
            "criteria": criteria,
            "passed": all(criteria.values()),
        })
    passed = sum(pair["passed"] for pair in pairs)
    reductions = [
        pair["stable_completion_duration_reduction_fraction"]
        for pair in pairs
        if pair["stable_completion_duration_reduction_fraction"] is not None
    ]
    return {
        "schema_version": "g1_stable_completion_validation_v1",
        "scope": (
            "Five deterministic distance pairs with stable 5 mm / 3 degree "
            "hold completion; not neural LGG100 or hardware evidence."
        ),
        "candidate": {
            "optimizer_report_sha256": hashlib.sha256(optimizer_bytes).hexdigest(),
            "config": selected["config"],
            "selection_metric_was_fixed_settle_proxy": True,
        },
        "completion_contract": {
            "position_tolerance_m": POSITION_TOLERANCE_M,
            "orientation_tolerance_deg": float(np.rad2deg(ORIENTATION_TOLERANCE_RAD)),
            "continuous_hold_s": COMPLETION_HOLD_S,
            "maximum_settle_s": MAXIMUM_SETTLE_S,
            "position_or_orientation_tolerance_relaxed": False,
            "clock": "MuJoCo simulation time",
        },
        "pairs": pairs,
        "summary": {
            "pair_count": len(pairs),
            "passed_pair_count": passed,
            "all_pairs_passed": passed == len(pairs),
            "all_actions_byte_identical": all(
                pair["action_sha256_off"] == pair["action_sha256_on"]
                for pair in pairs
            ),
            "stable_completion_duration_reduction_range_fraction": [
                min(reductions), max(reductions)
            ],
            "stable_completion_duration_reduction_median_fraction": float(
                np.median(reductions)
            ),
        },
        "decision": {
            "fixed_settle_duration_is_accepted_task_speed_metric": False,
            "stable_completion_metric_frozen_for_future_search": True,
            "current_candidate_stable_completion_gate_passed": passed == len(pairs),
            "multi_scenario_reoptimization_required": passed != len(pairs),
            "real_lgg100_task_speedup_passed": False,
            "production_adaptive_enabled": False,
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
    parser.add_argument("--optimizer-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_validation(args.optimizer_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
