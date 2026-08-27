#!/usr/bin/env python3
"""Deterministic multi-scenario screen using stable task completion time."""

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
from g1_stable_completion_validation import (
    COMPLETION_HOLD_S,
    LIFT_OFFSETS_M,
    MINIMUM_DURATION_REDUCTION,
    _stable_run,
)
from run_simulation import build_contract_fixture, preflight_contract_chunk


SCREEN_OFFSET_M = 0.13
MAXIMUM_ENDPOINT_ERROR_M = 0.005
MAXIMUM_ORIENTATION_ERROR_RAD = np.deg2rad(3.0)
MAXIMUM_JERK_REGRESSION_FRACTION = 0.0


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _config_key(config: RetimerConfig) -> tuple[float, ...]:
    return (
        config.min_scale,
        config.max_scale,
        config.near_distance,
        config.far_distance,
        config.max_scale_rate,
    )


def _evaluate_pair(config, fixture, baseline) -> dict[str, Any]:
    chunk, left_target, right_target = fixture
    retimed = AdaptiveRetimer(config).retime(chunk, left_target, right_target)
    adaptive = _stable_run(retimed.chunk)
    reduction = None
    speedup = None
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
    jerk_regression = (
        adaptive["maxima"]["actual_joint_jerk_rad_s3"]
        / baseline["maxima"]["actual_joint_jerk_rad_s3"] - 1.0
    )
    criteria = {
        "action_samples_byte_identical": bool(
            np.array_equal(chunk.actions, retimed.chunk.actions)
        ),
        "baseline_completed": bool(baseline["task_completed"]),
        "adaptive_completed": bool(adaptive["task_completed"]),
        "stable_completion_reduction_at_least_10_percent": bool(
            reduction is not None and reduction >= MINIMUM_DURATION_REDUCTION
        ),
        "endpoint_position_within_5mm": bool(
            adaptive["endpoint_error_m"] <= MAXIMUM_ENDPOINT_ERROR_M
        ),
        "endpoint_orientation_within_3deg": bool(
            adaptive["endpoint_orientation_error_rad"] <= MAXIMUM_ORIENTATION_ERROR_RAD
        ),
        "actual_joint_jerk_noninferior": bool(
            jerk_regression <= MAXIMUM_JERK_REGRESSION_FRACTION + 1e-12
        ),
        "hard_command_limits_pass": bool(adaptive["hard_command_limits_pass"]),
        "no_manipulation_contact": bool(
            adaptive["manipulation_contact_step_rate"] == 0.0
        ),
        "finite": bool(adaptive["finite"]),
    }
    return {
        "action_sha256_off": _hash_array(chunk.actions),
        "action_sha256_on": _hash_array(retimed.chunk.actions),
        "timestamp_sha256_off": _hash_array(chunk.timestamps),
        "timestamp_sha256_on": _hash_array(retimed.chunk.timestamps),
        "nominal_path_duration_s": retimed.retimed_duration,
        "baseline_completion_time_s": baseline["task_completion_time_s"],
        "adaptive_completion_time_s": adaptive["task_completion_time_s"],
        "duration_reduction_fraction": reduction,
        "effective_speedup_fraction": speedup,
        "endpoint_position_error_m": adaptive["endpoint_error_m"],
        "endpoint_orientation_error_rad": adaptive["endpoint_orientation_error_rad"],
        "actual_joint_jerk_regression_fraction": jerk_regression,
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def optimize_stable_completion(maximum_dynamics_candidates: int = 30) -> dict[str, Any]:
    if not 10 <= maximum_dynamics_candidates <= 100:
        raise ValueError("maximum_dynamics_candidates must be in [10,100]")
    fixtures = {
        offset: build_contract_fixture(np.array([0.0, 0.0, offset]))
        for offset in LIFT_OFFSETS_M
    }
    preflights = {
        offset: preflight_contract_chunk(fixture[0])
        for offset, fixture in fixtures.items()
    }
    baselines = {
        offset: _stable_run(fixture[0])
        for offset, fixture in fixtures.items()
    }

    configs = []
    for minimum, maximum, near, far, rate in itertools.product(
        (0.50, 0.70, 0.90, 0.95, 1.00),
        (1.40, 1.45, 1.60, 1.65, 1.80, 2.00),
        (0.015, 0.025, 0.050),
        (0.080, 0.100, 0.130),
        (1.0, 2.0, 4.0),
    ):
        if near >= far:
            continue
        configs.append(RetimerConfig(
            min_scale=minimum,
            max_scale=maximum,
            near_distance=near,
            far_distance=far,
            max_scale_rate=rate,
            max_eef_speed=0.65,
        ))
    # Include the prior selected config even if a future grid changes.
    configs.append(RetimerConfig(
        min_scale=0.95, max_scale=1.65,
        near_distance=0.015, far_distance=0.080,
        max_scale_rate=6.0, max_eef_speed=0.65,
    ))
    unique = {_config_key(config): config for config in configs}

    analytically_feasible = []
    for config in unique.values():
        retimed_durations = {}
        theoretical_reductions = {}
        actions_identical = True
        for offset, fixture in fixtures.items():
            chunk, left_target, right_target = fixture
            retimed = AdaptiveRetimer(config).retime(
                chunk, left_target, right_target
            )
            retimed_durations[str(offset)] = retimed.retimed_duration
            theoretical_earliest = retimed.retimed_duration + COMPLETION_HOLD_S
            theoretical_reductions[str(offset)] = (
                1.0
                - theoretical_earliest
                / baselines[offset]["task_completion_time_s"]
            )
            actions_identical &= np.array_equal(chunk.actions, retimed.chunk.actions)
        minimum_theoretical = min(theoretical_reductions.values())
        if minimum_theoretical >= MINIMUM_DURATION_REDUCTION:
            analytically_feasible.append({
                "config": config,
                "retimed_durations_s": retimed_durations,
                "theoretical_earliest_reductions": theoretical_reductions,
                "minimum_theoretical_reduction": minimum_theoretical,
                "actions_byte_identical": bool(actions_identical),
            })
    analytically_feasible.sort(
        key=lambda item: item["minimum_theoretical_reduction"], reverse=True
    )
    if len(analytically_feasible) < maximum_dynamics_candidates:
        selected_for_dynamics = analytically_feasible
    else:
        # Cover both aggressive and conservative feasible schedules instead of
        # overfitting to only the fastest nominal profiles. Retain four explicit
        # anchors from the prior single-fixture result and exploratory stable
        # completion diagnostics so the deterministic rank sampling cannot omit
        # already-observed local optima.
        anchor_keys = {
            (0.95, 1.65, 0.015, 0.080, 6.0),
            (0.90, 1.80, 0.050, 0.130, 4.0),
            (0.90, 1.60, 0.050, 0.130, 4.0),
            (0.70, 2.00, 0.025, 0.130, 4.0),
        }
        anchors = [
            item for item in analytically_feasible
            if _config_key(item["config"]) in anchor_keys
        ]
        remainder = [
            item for item in analytically_feasible
            if _config_key(item["config"]) not in anchor_keys
        ]
        remaining_count = maximum_dynamics_candidates - len(anchors)
        indices = np.unique(np.rint(np.linspace(
            0, len(remainder) - 1, remaining_count
        )).astype(int))
        selected_for_dynamics = anchors + [remainder[index] for index in indices]

    evaluated = []
    full_scenario_candidates = []
    for analytic in selected_for_dynamics:
        config = analytic["config"]
        screen = _evaluate_pair(
            config, fixtures[SCREEN_OFFSET_M], baselines[SCREEN_OFFSET_M]
        )
        record = {
            "config": asdict(config),
            "minimum_theoretical_reduction": analytic[
                "minimum_theoretical_reduction"
            ],
            "screen_offset_m": SCREEN_OFFSET_M,
            "screen": screen,
            "all_offsets": None,
            "passed": False,
        }
        if screen["passed"]:
            all_offsets = {
                str(offset): (
                    screen if offset == SCREEN_OFFSET_M else _evaluate_pair(
                        config, fixtures[offset], baselines[offset]
                    )
                )
                for offset in LIFT_OFFSETS_M
            }
            record["all_offsets"] = all_offsets
            record["passed"] = all(result["passed"] for result in all_offsets.values())
            full_scenario_candidates.append(record)
        evaluated.append(record)

    passing = [record for record in evaluated if record["passed"]]
    selected = max(
        passing,
        key=lambda record: min(
            result["duration_reduction_fraction"]
            for result in record["all_offsets"].values()
        ),
        default=None,
    )
    return {
        "schema_version": "g1_stable_completion_optimizer_v1",
        "scope": (
            "Deterministic timestamp-only development screen using stable 5 mm / "
            "3 degree continuous-250-ms completion; not exhaustive, neural, or hardware evidence."
        ),
        "completion_contract": {
            "position_tolerance_m": MAXIMUM_ENDPOINT_ERROR_M,
            "orientation_tolerance_deg": 3.0,
            "continuous_hold_s": COMPLETION_HOLD_S,
            "minimum_duration_reduction_fraction": MINIMUM_DURATION_REDUCTION,
            "maximum_actual_joint_jerk_regression_fraction": MAXIMUM_JERK_REGRESSION_FRACTION,
            "threshold_relaxed": False,
        },
        "input": {
            "lift_offsets_m": list(LIFT_OFFSETS_M),
            "grid_candidate_count": len(unique),
            "analytically_feasible_candidate_count": len(analytically_feasible),
            "dynamics_candidate_count": len(evaluated),
            "screen_offset_m": SCREEN_OFFSET_M,
            "selection": "even deterministic coverage over maximin theoretical rank plus feasible prior anchors",
            "preflight_all_offsets_accepted": all(
                result["all_accepted"] for result in preflights.values()
            ),
        },
        "baselines": {
            str(offset): {
                "task_completed": baseline["task_completed"],
                "completion_time_s": baseline["task_completion_time_s"],
                "endpoint_position_error_m": baseline["endpoint_error_m"],
                "endpoint_orientation_error_rad": baseline[
                    "endpoint_orientation_error_rad"
                ],
                "actual_joint_jerk_rad_s3": baseline["maxima"][
                    "actual_joint_jerk_rad_s3"
                ],
            }
            for offset, baseline in baselines.items()
        },
        "evaluated_candidates": evaluated,
        "summary": {
            "screen_pass_count": sum(record["screen"]["passed"] for record in evaluated),
            "maximum_observed_screen_duration_reduction_fraction": max(
                record["screen"]["duration_reduction_fraction"]
                for record in evaluated
                if record["screen"]["duration_reduction_fraction"] is not None
            ),
            "full_scenario_evaluated_count": len(full_scenario_candidates),
            "full_scenario_pass_count": len(passing),
            "selected": selected,
        },
        "decision": {
            "stable_multi_scenario_candidate_found": selected is not None,
            "minimum_30_candidate_development_screen_completed": len(evaluated) >= 30,
            "search_is_exhaustive": False,
            "production_adaptive_enabled": False,
            "real_lgg100_task_speedup_passed": False,
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
    parser.add_argument("--maximum-dynamics-candidates", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = optimize_stable_completion(args.maximum_dynamics_candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
