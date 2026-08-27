#!/usr/bin/env python3
"""Development paired A/B over deterministic G1 lift-distance fixtures.

Adaptive-OFF and Adaptive-ON use identical action samples; ON uses the selected
timestamp-only candidate. This is robustness screening, not neural task or
hardware speed evidence.
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
MINIMUM_DURATION_REDUCTION_FRACTION = 0.10
MAXIMUM_ENDPOINT_REGRESSION_M = 0.005
MAXIMUM_JERK_REGRESSION_FRACTION = 0.0


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run_paired_ab(optimizer_report_path: Path) -> dict[str, Any]:
    optimizer_bytes = optimizer_report_path.read_bytes()
    optimizer = json.loads(optimizer_bytes)
    selected = optimizer.get("selected")
    if not selected or selected.get("passed") is not True:
        raise ValueError("optimizer report has no passing selected candidate")
    config = RetimerConfig(**selected["config"])
    pairs = []
    for pair, lift_z in enumerate(LIFT_OFFSETS_M):
        chunk, left_target, right_target = build_contract_fixture(
            np.array([0.0, 0.0, lift_z], dtype=np.float64)
        )
        preflight = preflight_contract_chunk(chunk)
        retimed = AdaptiveRetimer(config).retime(chunk, left_target, right_target)
        baseline = _run_scale(
            chunk, REFERENCE_EP0_STATE[14:16],
            scale=1.0, use_filter=True, use_joint_filter=True,
        )
        adaptive = _run_scale(
            retimed.chunk, REFERENCE_EP0_STATE[14:16],
            scale=1.0, use_filter=True, use_joint_filter=True,
        )
        duration_reduction = (
            1.0 - adaptive["simulated_duration_s"] / baseline["simulated_duration_s"]
        )
        speedup = baseline["simulated_duration_s"] / adaptive["simulated_duration_s"] - 1.0
        endpoint_regression = adaptive["endpoint_error_m"] - baseline["endpoint_error_m"]
        jerk_regression = (
            adaptive["maxima"]["actual_joint_jerk_rad_s3"]
            / baseline["maxima"]["actual_joint_jerk_rad_s3"] - 1.0
        )
        criteria = {
            "preflight_all_accepted": preflight["all_accepted"],
            "action_samples_byte_identical": bool(
                np.array_equal(chunk.actions, retimed.chunk.actions)
            ),
            "duration_reduction_at_least_10_percent": (
                duration_reduction >= MINIMUM_DURATION_REDUCTION_FRACTION
            ),
            "endpoint_regression_within_5mm": (
                endpoint_regression <= MAXIMUM_ENDPOINT_REGRESSION_M
            ),
            "actual_joint_jerk_noninferior": (
                jerk_regression <= MAXIMUM_JERK_REGRESSION_FRACTION + 1e-12
            ),
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
            "pair": pair,
            "lift_offset_m": lift_z,
            "action_sha256_off": _hash_array(chunk.actions),
            "action_sha256_on": _hash_array(retimed.chunk.actions),
            "timestamps_sha256_off": _hash_array(chunk.timestamps),
            "timestamps_sha256_on": _hash_array(retimed.chunk.timestamps),
            "scale_min": float(retimed.scale_profile.min()),
            "scale_max": float(retimed.scale_profile.max()),
            "baseline_duration_s": baseline["simulated_duration_s"],
            "adaptive_duration_s": adaptive["simulated_duration_s"],
            "duration_reduction_fraction": duration_reduction,
            "effective_speedup_fraction": speedup,
            "baseline_endpoint_error_m": baseline["endpoint_error_m"],
            "adaptive_endpoint_error_m": adaptive["endpoint_error_m"],
            "endpoint_regression_m": endpoint_regression,
            "actual_joint_jerk_regression_fraction": jerk_regression,
            "criteria": criteria,
            "passed": all(criteria.values()),
        })

    passing = sum(pair["passed"] for pair in pairs)
    return {
        "schema_version": "g1_deterministic_speed_paired_ab_v1",
        "scope": (
            "Five-pair deterministic distance-robustness screen using identical "
            "actions and timestamp-only adaptation; not LGG100 task evidence."
        ),
        "candidate": {
            "optimizer_report_sha256": hashlib.sha256(optimizer_bytes).hexdigest(),
            "config": selected["config"],
            "selected_single_fixture_passed": selected["passed"],
        },
        "preregistered_criteria": {
            "minimum_duration_reduction_fraction": MINIMUM_DURATION_REDUCTION_FRACTION,
            "maximum_endpoint_regression_m": MAXIMUM_ENDPOINT_REGRESSION_M,
            "maximum_actual_joint_jerk_regression_fraction": MAXIMUM_JERK_REGRESSION_FRACTION,
            "action_samples_must_be_byte_identical": True,
        },
        "pairs": pairs,
        "summary": {
            "development_pair_count": len(pairs),
            "passed_pair_count": passing,
            "all_pairs_passed": passing == len(pairs),
            "duration_reduction_range_fraction": [
                min(pair["duration_reduction_fraction"] for pair in pairs),
                max(pair["duration_reduction_fraction"] for pair in pairs),
            ],
            "effective_speedup_range_fraction": [
                min(pair["effective_speedup_fraction"] for pair in pairs),
                max(pair["effective_speedup_fraction"] for pair in pairs),
            ],
            "all_action_hashes_paired": all(
                pair["action_sha256_off"] == pair["action_sha256_on"]
                for pair in pairs
            ),
        },
        "decision": {
            "development_paired_ab_executed": True,
            "single_fixture_candidate_generalized": passing == len(pairs),
            "formal_minimum_30_pair_ab_completed": False,
            "real_lgg100_task_speedup_passed": False,
            "production_adaptive_enabled": False,
            "robot_motion_allowed": False,
            "reason": (
                "The selected candidate is retained only if all distance pairs "
                "pass. Any duration, endpoint, or jerk failure requires a robust "
                "multi-scenario re-optimization before formal A/B."
            ),
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
    report = run_paired_ab(args.optimizer_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    # The runner succeeds when evidence is complete; generalization is a result,
    # not a reason to suppress the fail-closed report.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
