#!/usr/bin/env python3
"""Diagnose the 13 cm stable-completion bottleneck without hardware I/O."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_retimer import AdaptiveRetimer, RetimerConfig
from g1_stable_completion_validation import _stable_run
from run_simulation import build_contract_fixture


OFFSET_M = 0.13


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _run_payload(chunk) -> dict[str, Any]:
    run = _stable_run(chunk)
    return {
        "action_sha256": _hash_array(chunk.actions),
        "timestamp_sha256": _hash_array(chunk.timestamps),
        "nominal_path_duration_s": run["nominal_path_duration_s"],
        "task_completed": run["task_completed"],
        "task_completion_time_s": run["task_completion_time_s"],
        "settling_after_path_s": (
            run["task_completion_time_s"] - run["nominal_path_duration_s"]
            if run["task_completion_time_s"] is not None else None
        ),
        "first_tolerance_entry_s": run["first_completion_tolerance_entry_s"],
        "first_tolerance_entry_after_path_s": (
            run["first_completion_tolerance_entry_s"]
            - run["nominal_path_duration_s"]
            if run["first_completion_tolerance_entry_s"] is not None else None
        ),
        "completion_hold_reset_count": run["completion_hold_reset_count"],
        "maximum_consecutive_completion_hold_s": run[
            "maximum_consecutive_completion_hold_s"
        ],
        "endpoint_position_error_m": run["endpoint_error_m"],
        "endpoint_orientation_error_deg": float(np.rad2deg(
            run["endpoint_orientation_error_rad"]
        )),
        "actual_joint_jerk_rad_s3": run["maxima"]["actual_joint_jerk_rad_s3"],
        "maximum_actual_to_filtered_position_error_m": run["maxima"][
            "actual_to_filtered_position_error_m"
        ],
        "maximum_desired_to_filtered_position_lag_m": run["maxima"][
            "desired_to_filtered_position_lag_m"
        ],
        "hard_command_limits_pass": run["hard_command_limits_pass"],
        "manipulation_contact_step_rate": run["manipulation_contact_step_rate"],
        "worst_actual_joint_jerk_event": run["worst_actual_joint_jerk_event"],
    }


def run_diagnostic(optimizer_report_path: Path, stable_screen_path: Path) -> dict[str, Any]:
    optimizer = json.loads(optimizer_report_path.read_text())
    screen = json.loads(stable_screen_path.read_text())
    fixture, left_target, right_target = build_contract_fixture(
        np.array([0.0, 0.0, OFFSET_M])
    )
    previous_config = RetimerConfig(**optimizer["selected"]["config"])
    observed_best = max(
        screen["evaluated_candidates"],
        key=lambda item: item["screen"]["duration_reduction_fraction"],
    )
    best_config = RetimerConfig(**observed_best["config"])
    previous_retimed = AdaptiveRetimer(previous_config).retime(
        fixture, left_target, right_target
    )
    best_retimed = AdaptiveRetimer(best_config).retime(
        fixture, left_target, right_target
    )
    baseline = _run_payload(fixture)
    previous = _run_payload(previous_retimed.chunk)
    best = _run_payload(best_retimed.chunk)
    previous_reduction = (
        1.0 - previous["task_completion_time_s"] / baseline["task_completion_time_s"]
    )
    best_reduction = (
        1.0 - best["task_completion_time_s"] / baseline["task_completion_time_s"]
    )
    return {
        "schema_version": "g1_settling_bottleneck_diagnostic_v1",
        "scope": (
            "Offline 13 cm deterministic tracking/settling diagnostic; no neural "
            "policy, physical dynamics, or hardware evidence."
        ),
        "completion_contract": {
            "position_tolerance_m": 0.005,
            "orientation_tolerance_deg": 3.0,
            "continuous_hold_s": 0.250,
            "threshold_relaxed": False,
        },
        "offset_m": OFFSET_M,
        "baseline": baseline,
        "previous_single_fixture_candidate": {
            "config": optimizer["selected"]["config"],
            "run": previous,
            "stable_duration_reduction_fraction": previous_reduction,
        },
        "best_observed_30_screen_candidate": {
            "config": observed_best["config"],
            "run": best,
            "stable_duration_reduction_fraction": best_reduction,
        },
        "comparison": {
            "previous_action_samples_match_baseline": (
                previous["action_sha256"] == baseline["action_sha256"]
            ),
            "best_action_samples_match_baseline": (
                best["action_sha256"] == baseline["action_sha256"]
            ),
            "previous_timestamp_changed": (
                previous["timestamp_sha256"] != baseline["timestamp_sha256"]
            ),
            "best_timestamp_changed": (
                best["timestamp_sha256"] != baseline["timestamp_sha256"]
            ),
            "previous_path_time_reduction_s": (
                baseline["nominal_path_duration_s"]
                - previous["nominal_path_duration_s"]
            ),
            "previous_settling_time_increase_s": (
                previous["settling_after_path_s"] - baseline["settling_after_path_s"]
            ),
            "best_path_time_reduction_s": (
                baseline["nominal_path_duration_s"] - best["nominal_path_duration_s"]
            ),
            "best_settling_time_increase_s": (
                best["settling_after_path_s"] - baseline["settling_after_path_s"]
            ),
        },
        "decision": {
            "bottleneck_is_schedule_duration_only": False,
            "settling_aware_optimization_required": True,
            "previous_candidate_stable_10_percent_gate_passed": previous_reduction >= 0.10,
            "best_observed_candidate_stable_10_percent_gate_passed": best_reduction >= 0.10,
            "mujoco_result_requires_real_shadow_validation": True,
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
    parser.add_argument("--stable-screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_diagnostic(args.optimizer_report, args.stable_screen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
