#!/usr/bin/env python3
"""Offline Shadow replay with separate canonical actions and waist-aware IK targets."""

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
from g1_camera_observation import build_three_camera_observation
from g1_fail_closed_adapter import FailClosedMockAdapter, MockSuggestionSink, ShadowSafetyInputs
from g1_fast_sequential_ik import solve_sequential_ik
from g1_fast_swept_path_preflight import G1FastSweptPathPreflight
from g1_mujoco_bridge import policy_state_from_mujoco
from g1_offline_policy_shadow_replay import (
    _hash_array,
    _observation_hash,
    _set_real_sample,
    _synthetic_frames,
)
from g1_policy_contract import (
    ACTION_HORIZON,
    CONTRACT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    POLICY_RATE_HZ,
    canonicalize_policy_action_chunk,
)
from g1_waist_compensation import apply_waist_compensation, compute_waist_compensation
from stack_scene import build_model


def run_replay(sample_path: Path, fk_report_path: Path, q0_report_path: Path) -> dict[str, Any]:
    sample = json.loads(sample_path.read_text())
    fk_report = json.loads(fk_report_path.read_text())
    q0_report = json.loads(q0_report_path.read_text())
    model = build_model()
    measured_source = mujoco.MjData(model)
    zero_waist_source = mujoco.MjData(model)
    _set_real_sample(model, measured_source, sample, measured_waist=True)
    _set_real_sample(model, zero_waist_source, sample, measured_waist=False)

    policy_state = policy_state_from_mujoco(
        model, zero_waist_source, np.zeros(2, dtype=np.float64)
    ).astype(np.float32)
    observation, camera_metadata = build_three_camera_observation(
        _synthetic_frames(), policy_state, "stack the blue cube on the green cube"
    )
    raw = np.repeat(policy_state[None, :], ACTION_HORIZON, axis=0).astype(np.float64)
    raw[:, 3:7] *= 0.995
    raw[:, 10:14] *= 0.995
    canonical = canonicalize_policy_action_chunk(raw, expected_horizon=ACTION_HORIZON)
    canonical_before = canonical.copy()
    canonical_hash = _hash_array(canonical)
    timestamps = np.arange(ACTION_HORIZON, dtype=np.float64) / POLICY_RATE_HZ

    transform_started = time.perf_counter_ns()
    compensation = compute_waist_compensation(
        model, zero_waist_source, measured_source
    )
    ik_targets = apply_waist_compensation(canonical, compensation)
    transform_elapsed_ms = (time.perf_counter_ns() - transform_started) / 1_000_000.0

    ik = solve_sequential_ik(model, measured_source, ik_targets)
    preflight_started = time.perf_counter_ns()
    swept = G1FastSweptPathPreflight(model).check(
        measured_source,
        EEFActionChunk(timestamps, ik_targets),
        phase="free_space",
    )
    preflight_elapsed_ms = (time.perf_counter_ns() - preflight_started) / 1_000_000.0

    ik_records = []
    for target in ik.targets:
        ik_records.append({
            "index": target.index,
            "suggested_joint_target_rad": target.joint_target_rad.tolist(),
            "suggested_joint_target_sha256": _hash_array(target.joint_target_rad),
            "iterations": target.iterations,
            "position_error_m": target.maximum_position_error_m,
            "orientation_error_rad": target.maximum_orientation_error_rad,
            "within_internal_tolerance": bool(
                target.reachable and target.joint_limits_ok
            ),
        })

    waist_position = float(
        fk_report["decision"]["maximum_waist_induced_position_difference_m"]
    )
    waist_orientation = float(
        fk_report["decision"]["maximum_waist_induced_orientation_difference_deg"]
    )
    adapter_inputs = ShadowSafetyInputs(
        lowstate_age_ms=2.0,
        unique_tick_age_ms=2.0,
        camera_age_ms={"head_left": 10.0, "left_wrist": 11.0, "right_wrist": 12.0},
        camera_frozen={"head_left": False, "left_wrist": False, "right_wrist": False},
        policy_latency_ms=float(q0_report["output_only_probe"]["latency_ms"]["p50"]),
        canonical_action=canonical,
        ik_success=ik.accepted,
        collision_free=swept.accepted,
        joint_limits_ok=ik.accepted,
        dds_connected=True,
        waist_divergence_m=waist_position,
        waist_divergence_deg=waist_orientation,
    )
    sink = MockSuggestionSink()
    adapter = FailClosedMockAdapter(sink)
    mock_records = [
        adapter.commit_to_mock(target.joint_target_rad, adapter_inputs)
        for target in ik.targets
    ]
    decisions = [record["decision"] for record in mock_records]
    q0_p50_ms = float(q0_report["output_only_probe"]["latency_ms"]["p50"])
    current_safety_processing_ms = transform_elapsed_ms + ik.elapsed_ms + preflight_elapsed_ms

    return {
        "schema_version": "g1_offline_compensated_shadow_replay_v1",
        "scope": (
            "Saved waist/arm LowState, synthetic cameras, deterministic hold-policy "
            "fixture, waist-aware Fast IK/swept preflight, and mock sink. Not real "
            "camera, frozen-checkpoint inference, full-body collision, or live Shadow."
        ),
        "contract": {
            "id": CONTRACT_ID,
            "version": CONTRACT_VERSION,
            "sha256": CONTRACT_SHA256,
            "policy_rate_hz": POLICY_RATE_HZ,
            "action_horizon": ACTION_HORIZON,
            "prefetch_lead_steps": 5,
            "prefetch_window_ms": 5.0 / POLICY_RATE_HZ * 1000.0,
        },
        "inputs": {
            "lowstate_sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
            "captured_joint_count": len(sample["arm_q_rad"]) + len(sample["waist_q_rad"]),
            "full_body_joint_count_required": 29,
            "full_body_state_available": False,
            "camera_source": "deterministic_synthetic_fixture_not_real_camera",
            "policy_source": "deterministic_hold_fixture_not_lgg100_inference",
        },
        "observation": {
            "sha256": _observation_hash(observation),
            "state_sha256": _hash_array(observation["observation/state"]),
            "state_fk_view": "yuhao_zero_waist_for_contract_parity",
            "camera": camera_metadata,
        },
        "action_boundary": {
            "raw_sha256": _hash_array(raw),
            "canonical_policy_action_sha256": canonical_hash,
            "canonical_after_adapter_sha256": _hash_array(canonical),
            "canonical_input_mutated": not np.array_equal(canonical, canonical_before),
            "kinematic_ik_target_sha256": _hash_array(ik_targets),
            "ik_target_is_separate_artifact": True,
            "gripper_channels_preserved": bool(
                np.array_equal(canonical[:, 14:16], ik_targets[:, 14:16])
            ),
        },
        "waist_compensation": {
            "pelvis_delta_transform": compensation.pelvis_delta_transform.tolist(),
            "elapsed_ms": transform_elapsed_ms,
            "physical_eef_parity_verified": False,
        },
        "fast_sequential_ik": {
            "accepted": ik.accepted,
            "reason": ik.reason,
            "checked_targets": len(ik.targets),
            "total_iterations": ik.total_iterations,
            "elapsed_ms": ik.elapsed_ms,
            "maximum_position_error_m": max(
                target.maximum_position_error_m for target in ik.targets
            ),
            "maximum_orientation_error_rad": max(
                target.maximum_orientation_error_rad for target in ik.targets
            ),
            "targets": ik_records,
        },
        "fast_swept_path": {
            "accepted": swept.accepted,
            "reason": swept.reason,
            "collision_reasons": list(swept.collision_reasons),
            "checked_targets": swept.checked_targets,
            "checked_interpolated_configurations": swept.checked_interpolated_configurations,
            "elapsed_ms": preflight_elapsed_ms,
            "initial_collision_physically_qualified": False,
        },
        "latency": {
            "lgg100_inference_executed": False,
            "replayed_q0_inference_p50_ms": q0_p50_ms,
            "replayed_q0_latency_is_current_run_measurement": False,
            "current_run_safety_processing_ms": current_safety_processing_ms,
            "serial_q0_p50_plus_current_safety_processing_ms": (
                q0_p50_ms + current_safety_processing_ms
            ),
            "serial_estimate_within_333ms": (
                q0_p50_ms + current_safety_processing_ms
                <= 5.0 / POLICY_RATE_HZ * 1000.0
            ),
        },
        "mock_sink": {
            "record_count": len(mock_records),
            "all_publish_allowed_false": all(
                decision["publish_allowed"] is False for decision in decisions
            ),
            "all_robot_command_sent_false": all(
                decision["robot_command_sent"] is False for decision in decisions
            ),
            "all_hold_true": all(decision["hold"] is True for decision in decisions),
            "hold_reason_union": sorted({
                reason for decision in decisions for reason in decision["reasons"]
            }),
        },
        "decision": {
            "offline_compensated_shadow_replay_completed": True,
            "canonical_action_identity_preserved": bool(
                _hash_array(canonical) == canonical_hash
            ),
            "compensated_hold_ik_passed": ik.accepted,
            "swept_path_passed": swept.accepted,
            "initial_collision_resolved": False,
            "real_three_camera_observation_used": False,
            "real_streaming_lowstate_used": False,
            "frozen_lgg100_inference_executed": False,
            "real_15_hz_policy_shadow_passed": False,
            "integrated_shadow_eligible": False,
            "robot_motion_allowed": False,
            "reason": (
                "Waist adaptation resolves numeric IK while preserving the canonical "
                "action, but missing leg joints leave initial collision unresolved; "
                "synthetic cameras and hold-policy fixtures cannot qualify live Shadow."
            ),
        },
        "safety": {
            "offline_only": True,
            "network_accessed": False,
            "publisher_created": False,
            "hardware_transport_present": False,
            "robot_command_sent": False,
            "mode_change_requested": False,
        },
        "g1_execution_enabled": False,
        "hardware_execution_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--fk-report", type=Path, required=True)
    parser.add_argument("--q0-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_replay(args.sample, args.fk_report, args.q0_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["decision"]["offline_compensated_shadow_replay_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
