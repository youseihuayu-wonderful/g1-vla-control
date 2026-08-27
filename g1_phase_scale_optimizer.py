#!/usr/bin/env python3
"""Constrained offline phase-scale search for the deterministic G1 fixture.

The search changes timestamps only. It first ranks candidates analytically,
then runs the existing MuJoCo command pipeline for the best candidates. This is
not neural-policy or hardware task evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_retimer import AdaptiveRetimer, RetimerConfig
from g1_adaptive_phase_validation import FAR_LIFT_OFFSET_M
from g1_policy_contract import CONTRACT_ID, CONTRACT_SHA256, CONTRACT_VERSION, POLICY_RATE_HZ
from retiming_safety_validation import _run_scale
from run_simulation import build_contract_fixture, preflight_contract_chunk
from stack_scene import REFERENCE_EP0_STATE


MINIMUM_COMPLETE_DURATION_REDUCTION = 0.10
MAXIMUM_ENDPOINT_REGRESSION_M = 0.005
MAXIMUM_JERK_REGRESSION_FRACTION = 0.0


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def optimize_phase_scales(maximum_dynamics_candidates: int = 12) -> dict[str, Any]:
    chunk, left_target, right_target = build_contract_fixture(FAR_LIFT_OFFSET_M)
    preflight = preflight_contract_chunk(chunk)
    baseline = _run_scale(
        chunk,
        REFERENCE_EP0_STATE[14:16],
        scale=1.0,
        use_filter=True,
        use_joint_filter=True,
    )
    grid = itertools.product(
        (0.85, 0.875, 0.90, 0.925, 0.95),
        (1.45, 1.55, 1.65),
        (0.015, 0.025),
        (0.08, 0.10, 0.13),
        (2.0, 4.0, 6.0),
    )
    ranked = []
    for minimum, maximum, near, far, rate in grid:
        config = RetimerConfig(
            min_scale=minimum,
            max_scale=maximum,
            near_distance=near,
            far_distance=far,
            max_scale_rate=rate,
            max_eef_speed=0.65,
        )
        retimed = AdaptiveRetimer(config).retime(chunk, left_target, right_target)
        path_identical = np.array_equal(retimed.chunk.actions, chunk.actions)
        ranked.append({
            "config": config,
            "retimed": retimed,
            "path_actions_byte_identical": bool(path_identical),
            "nominal_duration_reduction_fraction": (
                1.0 - retimed.retimed_duration / retimed.original_duration
            ),
        })
    ranked.sort(
        key=lambda item: item["nominal_duration_reduction_fraction"],
        reverse=True,
    )

    evaluated = []
    baseline_jerk = baseline["maxima"]["actual_joint_jerk_rad_s3"]
    for rank, candidate in enumerate(ranked[:maximum_dynamics_candidates], start=1):
        retimed = candidate["retimed"]
        dynamics = _run_scale(
            retimed.chunk,
            REFERENCE_EP0_STATE[14:16],
            scale=1.0,
            use_filter=True,
            use_joint_filter=True,
        )
        duration_reduction = (
            1.0 - dynamics["simulated_duration_s"] / baseline["simulated_duration_s"]
        )
        speedup = (
            baseline["simulated_duration_s"] / dynamics["simulated_duration_s"] - 1.0
        )
        endpoint_regression = dynamics["endpoint_error_m"] - baseline["endpoint_error_m"]
        jerk_regression_fraction = (
            dynamics["maxima"]["actual_joint_jerk_rad_s3"] / baseline_jerk - 1.0
        )
        scale = retimed.scale_profile
        criteria = {
            "preflight_all_accepted": preflight["all_accepted"],
            "path_actions_byte_identical": candidate["path_actions_byte_identical"],
            "near_remains_slower_than_baseline": float(scale.min()) < 1.0,
            "near_not_slower_than_0_85": float(scale.min()) >= 0.85,
            "far_accelerates": float(scale.max()) > 1.0,
            "hard_command_limits_pass": dynamics["hard_command_limits_pass"],
            "finite": dynamics["finite"],
            "no_manipulation_contact": dynamics["manipulation_contact_step_rate"] == 0.0,
            "endpoint_regression_within_5mm": endpoint_regression <= MAXIMUM_ENDPOINT_REGRESSION_M,
            "actual_joint_jerk_noninferior": (
                jerk_regression_fraction <= MAXIMUM_JERK_REGRESSION_FRACTION + 1e-12
            ),
            "complete_duration_reduction_at_least_10_percent": (
                duration_reduction >= MINIMUM_COMPLETE_DURATION_REDUCTION
            ),
        }
        evaluated.append({
            "analytic_rank": rank,
            "config": asdict(candidate["config"]),
            "scale_min": float(scale.min()),
            "scale_median": float(np.median(scale)),
            "scale_max": float(scale.max()),
            "nominal_retimed_duration_s": retimed.retimed_duration,
            "simulated_duration_s": dynamics["simulated_duration_s"],
            "duration_reduction_fraction": duration_reduction,
            "effective_speedup_fraction": speedup,
            "endpoint_error_m": dynamics["endpoint_error_m"],
            "endpoint_regression_m": endpoint_regression,
            "actual_joint_jerk_rad_s3": dynamics["maxima"]["actual_joint_jerk_rad_s3"],
            "actual_joint_jerk_regression_fraction": jerk_regression_fraction,
            "criteria": criteria,
            "passed": all(criteria.values()),
        })
    passing = [item for item in evaluated if item["passed"]]
    selected = max(
        passing,
        key=lambda item: item["duration_reduction_fraction"],
        default=None,
    )
    return {
        "schema_version": "g1_phase_scale_optimizer_v1",
        "scope": (
            "Constrained timestamp-only search on a deterministic G1 far-to-near "
            "fixture; not LGG100 task-success or hardware evidence."
        ),
        "contract": {
            "id": CONTRACT_ID,
            "version": CONTRACT_VERSION,
            "sha256": CONTRACT_SHA256,
            "policy_rate_hz": POLICY_RATE_HZ,
        },
        "input": {
            "trajectory_source": "deterministic_g1_contract_fixture_not_lgg100",
            "action_sha256": _hash_array(chunk.actions),
            "action_shape": list(chunk.actions.shape),
            "grid_candidate_count": len(ranked),
            "dynamics_candidate_count": len(evaluated),
        },
        "constraints": {
            "minimum_complete_duration_reduction_fraction": MINIMUM_COMPLETE_DURATION_REDUCTION,
            "maximum_endpoint_regression_m": MAXIMUM_ENDPOINT_REGRESSION_M,
            "maximum_actual_joint_jerk_regression_fraction": MAXIMUM_JERK_REGRESSION_FRACTION,
            "position_or_orientation_tolerance_relaxed": False,
            "collision_or_command_limit_relaxed": False,
        },
        "baseline": {
            "simulated_duration_s": baseline["simulated_duration_s"],
            "endpoint_error_m": baseline["endpoint_error_m"],
            "actual_joint_jerk_rad_s3": baseline_jerk,
            "hard_command_limits_pass": baseline["hard_command_limits_pass"],
            "manipulation_contact_step_rate": baseline["manipulation_contact_step_rate"],
        },
        "evaluated_candidates": evaluated,
        "selected": selected,
        "decision": {
            "offline_phase_scale_candidate_found": selected is not None,
            "complete_deterministic_fixture_speedup_passed": selected is not None,
            "real_lgg100_task_speedup_passed": False,
            "production_adaptive_enabled": False,
            "robot_motion_allowed": False,
        },
        "safety": {
            "offline_only": True,
            "action_samples_modified": False,
            "publisher_created": False,
            "robot_command_sent": False,
        },
        "g1_execution_enabled": False,
        "hardware_execution_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-dynamics-candidates", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.maximum_dynamics_candidates <= 100:
        raise SystemExit("maximum dynamics candidates must be in [1,100]")
    report = optimize_phase_scales(args.maximum_dynamics_candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["decision"]["offline_phase_scale_candidate_found"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
