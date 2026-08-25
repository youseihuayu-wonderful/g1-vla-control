#!/usr/bin/env python3
"""Run offline fault injection against the mock-only G1 safety adapter."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from g1_fail_closed_adapter import (
    FailClosedMockAdapter,
    MockSuggestionSink,
    ShadowSafetyInputs,
    ShadowSafetyThresholds,
)


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run_fault_injection() -> dict[str, Any]:
    thresholds = ShadowSafetyThresholds()
    base_action = np.zeros((32, 16), dtype=np.float64)
    base_target = np.zeros(14, dtype=np.float64)
    nominal = ShadowSafetyInputs(
        lowstate_age_ms=2.0,
        unique_tick_age_ms=2.0,
        camera_age_ms={
            "head_left": 10.0,
            "left_wrist": 11.0,
            "right_wrist": 12.0,
        },
        camera_frozen={
            "head_left": False,
            "left_wrist": False,
            "right_wrist": False,
        },
        policy_latency_ms=80.0,
        canonical_action=base_action,
        ik_success=True,
        collision_free=True,
        joint_limits_ok=True,
        dds_connected=True,
        waist_divergence_m=0.0,
        waist_divergence_deg=0.0,
    )
    nan_action = base_action.copy()
    nan_action[0, 0] = np.nan
    inf_action = base_action.copy()
    inf_action[0, 1] = np.inf
    cases = [
        ("lowstate_stop", replace(nominal, lowstate_age_ms=51.0), "lowstate_stale"),
        (
            "camera_freeze",
            replace(
                nominal,
                camera_frozen={**nominal.camera_frozen, "head_left": True},
            ),
            "camera_frozen:head_left",
        ),
        ("policy_timeout", replace(nominal, policy_latency_ms=100.0), "policy_timeout"),
        ("action_nan", replace(nominal, canonical_action=nan_action), "action_nonfinite"),
        ("action_inf", replace(nominal, canonical_action=inf_action), "action_nonfinite"),
        ("ik_failure", replace(nominal, ik_success=False), "ik_failure"),
        (
            "collision_failure",
            replace(nominal, collision_free=False),
            "collision_failure",
        ),
        (
            "dds_disconnect",
            replace(nominal, dds_connected=False),
            "dds_disconnected",
        ),
    ]

    sink = MockSuggestionSink()
    adapter = FailClosedMockAdapter(sink, thresholds)
    nominal_record = adapter.commit_to_mock(base_target, nominal)
    results = []
    for name, inputs, expected_reason in cases:
        record = adapter.commit_to_mock(base_target, inputs)
        decision = record["decision"]
        passed = bool(
            expected_reason in decision["reasons"]
            and decision["publish_allowed"] is False
            and decision["hold"] is True
            and decision["robot_command_sent"] is False
            and decision["publisher_created"] is False
        )
        results.append({
            "name": name,
            "expected_reason": expected_reason,
            "observed_reasons": decision["reasons"],
            "publish_allowed": decision["publish_allowed"],
            "hold": decision["hold"],
            "publisher_created": decision["publisher_created"],
            "robot_command_sent": decision["robot_command_sent"],
            "passed": passed,
        })

    nominal_decision = nominal_record["decision"]
    all_faults_passed = all(item["passed"] for item in results)
    return {
        "schema_version": "g1_fail_closed_mock_fault_injection_v1",
        "scope": "Offline mock-only fault injection; no DDS or robot transport exists.",
        "thresholds": {
            "lowstate_stale_ms": thresholds.lowstate_stale_ms,
            "camera_stale_ms": thresholds.camera_stale_ms,
            "policy_timeout_ms": thresholds.policy_timeout_ms,
            "maximum_waist_divergence_m": thresholds.maximum_waist_divergence_m,
            "maximum_waist_divergence_deg": thresholds.maximum_waist_divergence_deg,
        },
        "input_hashes": {
            "nominal_action_sha256": _array_hash(base_action),
            "suggested_joint_target_sha256": _array_hash(base_target),
        },
        "nominal_mock_case": {
            "safety_gate_passed": nominal_decision["safety_gate_passed"],
            "hold": nominal_decision["hold"],
            "publish_allowed": nominal_decision["publish_allowed"],
            "hardware_transport_present": nominal_decision["hardware_transport_present"],
            "robot_command_sent": nominal_decision["robot_command_sent"],
            "interpretation": (
                "A nominal suggestion may be accepted by the evidence sink, but "
                "cannot be published because the adapter has no hardware transport."
            ),
        },
        "fault_cases": results,
        "summary": {
            "required_fault_case_count": len(cases),
            "passed_fault_case_count": sum(item["passed"] for item in results),
            "all_faults_fail_closed": all_faults_passed,
            "all_publish_allowed_false": all(
                item["publish_allowed"] is False for item in results
            ),
            "all_hold_true": all(item["hold"] is True for item in results),
            "all_robot_command_sent_false": all(
                item["robot_command_sent"] is False for item in results
            ),
        },
        "decision": {
            "offline_h5_mock_fault_suite_passed": all_faults_passed,
            "integrated_real_shadow_watchdog_validated": False,
            "hardware_adapter_reviewed": False,
            "hardware_publish_capability_present": False,
            "robot_motion_allowed": False,
            "reason": (
                "All required offline faults fail closed in a transport-free mock "
                "adapter. H5 cannot be promoted to integrated pass until the same "
                "suite is bound to a qualified real-observation H4 Shadow."
            ),
        },
        "safety": {
            "offline_only": True,
            "unitree_sdk_imported": False,
            "network_accessed": False,
            "publisher_created": False,
            "robot_command_sent": False,
            "mode_change_requested": False,
        },
        "g1_execution_enabled": False,
        "hardware_execution_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_fault_injection()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["summary"]["all_faults_fail_closed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
