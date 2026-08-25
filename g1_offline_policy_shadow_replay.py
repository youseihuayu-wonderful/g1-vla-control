#!/usr/bin/env python3
"""Offline 15 Hz replay of the real-G1 Policy Shadow dataflow.

This uses one saved subscriber-only LowState sample, deterministic synthetic
camera fixtures, and a deterministic hold-policy fixture. It exercises the
observation boundary, official quaternion-only post-processing, measured-waist
sequential IK, swept-path checking, and mock sink. It does not execute LGG100
inference and must not be represented as a real-observation Policy Shadow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np

from action_schema import EEFActionChunk, pelvis_vla_action_to_world_mujoco
from g1_camera_observation import BGRFrame, build_three_camera_observation
from g1_dual_arm_ik import G1DualArmIK, LEFT_JOINTS, RIGHT_JOINTS, orientation_error
from g1_fail_closed_adapter import (
    FailClosedMockAdapter,
    MockSuggestionSink,
    ShadowSafetyInputs,
)
from g1_mujoco_bridge import policy_state_from_mujoco
from g1_policy_contract import (
    ACTION_HORIZON,
    CONTRACT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
    POLICY_RATE_HZ,
    canonicalize_policy_action_chunk,
)
from safety_governor import manipulator_contact_violations
from stack_scene import build_model, reset_to_reference_pose
from swept_path_preflight import G1SweptPathPreflight


def _hash_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _observation_hash(observation: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        digest.update(key.encode())
        value = observation[key]
        if isinstance(value, str):
            digest.update(value.encode())
        else:
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _set_real_sample(model: mujoco.MjModel, data: mujoco.MjData, sample: dict[str, Any], *, measured_waist: bool) -> None:
    reset_to_reference_pose(model, data)
    for name, value in sample["arm_q_rad"].items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = value
    for name, value in sample["waist_q_rad"].items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = value if measured_waist else 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _synthetic_frames() -> list[BGRFrame]:
    height, width = 480, 640
    y, x = np.indices((height, width), dtype=np.uint16)

    def pattern(seed: int) -> np.ndarray:
        return np.stack((
            (x + seed) % 256,
            (y + 2 * seed) % 256,
            (x + y + 3 * seed) % 256,
        ), axis=-1).astype(np.uint8)

    head_left = pattern(17)
    head_right = pattern(91)
    return [
        BGRFrame(
            "head_left", np.concatenate((head_left, head_right), axis=1),
            received_monotonic_ns=1_000_000_000, binocular=True,
        ),
        BGRFrame(
            "left_wrist", pattern(37),
            received_monotonic_ns=1_005_000_000,
        ),
        BGRFrame(
            "right_wrist", pattern(73),
            received_monotonic_ns=1_010_000_000,
        ),
    ]


def _sequential_ik(
    model: mujoco.MjModel,
    source: mujoco.MjData,
    canonical: np.ndarray,
) -> tuple[list[dict[str, Any]], bool, bool, bool, float]:
    data = mujoco.MjData(model)
    data.qpos[:] = source.qpos
    data.qvel[:] = 0.0
    data.ctrl[:] = source.ctrl
    mujoco.mj_forward(model, data)
    solver = G1DualArmIK(model, data, damping=0.06, max_joint_speed=2.0)
    solver.reset()
    arm_qpos = np.concatenate((solver.left["qpos"], solver.right["qpos"]))
    original = data.qpos[arm_qpos].copy()
    previous = original.copy()
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    records = []
    all_reachable = True
    all_collision_free = True
    all_joint_limits_ok = True
    started = time.perf_counter_ns()
    for index, action in enumerate(canonical):
        target = pelvis_vla_action_to_world_mujoco(
            action, data.xpos[pelvis], data.xquat[pelvis]
        )
        for _ in range(250):
            solver.step(target[:7], target[7:14], 0.02)
            data.qpos[arm_qpos] = solver.q_target[arm_qpos]
            data.qvel[:] = 0.0
            mujoco.mj_fwdPosition(model, data)
        left_pose = solver.pose("left")
        right_pose = solver.pose("right")
        position_error = float(max(
            np.linalg.norm(left_pose[0] - target[:3]),
            np.linalg.norm(right_pose[0] - target[7:10]),
        ))
        orientation_error_rad = float(max(
            np.linalg.norm(orientation_error(target[3:7], left_pose[1])),
            np.linalg.norm(orientation_error(target[10:14], right_pose[1])),
        ))
        joint_limits_ok = True
        for info in (solver.left, solver.right):
            for joint_id, qpos_index in zip(info["joint"], info["qpos"], strict=True):
                if model.jnt_limited[joint_id]:
                    low, high = model.jnt_range[joint_id]
                    joint_limits_ok &= bool(low <= data.qpos[qpos_index] <= high)
        collision_reasons = manipulator_contact_violations(model, data, "free_space")
        reachable = position_error <= 0.005 and orientation_error_rad <= np.deg2rad(3.0)
        solved = data.qpos[arm_qpos].copy()
        records.append({
            "index": index,
            "suggested_joint_target_rad": solved.tolist(),
            "suggested_joint_target_sha256": _hash_array(solved),
            "maximum_delta_from_current_rad": float(np.max(np.abs(solved - original))),
            "maximum_delta_from_previous_rad": float(np.max(np.abs(solved - previous))),
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error_rad,
            "reachable": reachable,
            "joint_limits_ok": joint_limits_ok,
            "collision_free": not collision_reasons,
            "collision_reasons": list(collision_reasons),
        })
        previous = solved
        all_reachable &= reachable
        all_joint_limits_ok &= joint_limits_ok
        all_collision_free &= not collision_reasons
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return records, all_reachable, all_collision_free, all_joint_limits_ok, latency_ms


def run_replay(sample_path: Path, fk_report_path: Path, q0_report_path: Path) -> dict[str, Any]:
    sample = json.loads(sample_path.read_text())
    fk_report = json.loads(fk_report_path.read_text())
    q0_report = json.loads(q0_report_path.read_text())
    model = build_model()
    measured_source = mujoco.MjData(model)
    zero_waist_source = mujoco.MjData(model)
    _set_real_sample(model, measured_source, sample, measured_waist=True)
    _set_real_sample(model, zero_waist_source, sample, measured_waist=False)

    gripper_fixture = np.zeros(2, dtype=np.float64)
    policy_state = policy_state_from_mujoco(
        model, zero_waist_source, gripper_fixture
    ).astype(np.float32)
    observation, camera_metadata = build_three_camera_observation(
        _synthetic_frames(), policy_state, "stack the blue cube on the green cube"
    )

    raw = np.repeat(policy_state[None, :], ACTION_HORIZON, axis=0).astype(np.float64)
    raw[:, 3:7] *= 0.995
    raw[:, 10:14] *= 0.995
    canonical = canonicalize_policy_action_chunk(raw, expected_horizon=ACTION_HORIZON)
    non_quaternion = np.ones(16, dtype=bool)
    non_quaternion[3:7] = False
    non_quaternion[10:14] = False
    non_quaternion_preserved = np.array_equal(
        raw[:, non_quaternion], canonical[:, non_quaternion]
    )
    timestamps = np.arange(ACTION_HORIZON, dtype=np.float64) / POLICY_RATE_HZ

    ik_records, ik_success, collision_free, joint_limits_ok, ik_latency_ms = (
        _sequential_ik(model, measured_source, canonical)
    )
    swept_started = time.perf_counter_ns()
    swept = G1SweptPathPreflight(model).check(
        measured_source,
        EEFActionChunk(timestamps, canonical),
        phase="free_space",
    )
    swept_latency_ms = (time.perf_counter_ns() - swept_started) / 1_000_000.0

    waist_position = float(
        fk_report["decision"]["maximum_waist_induced_position_difference_m"]
    )
    waist_orientation = float(
        fk_report["decision"]["maximum_waist_induced_orientation_difference_deg"]
    )
    adapter_inputs = ShadowSafetyInputs(
        lowstate_age_ms=2.0,
        unique_tick_age_ms=2.0,
        camera_age_ms={
            "head_left": 10.0, "left_wrist": 11.0, "right_wrist": 12.0,
        },
        camera_frozen={
            "head_left": False, "left_wrist": False, "right_wrist": False,
        },
        policy_latency_ms=float(q0_report["output_only_probe"]["latency_ms"]["p50"]),
        canonical_action=canonical,
        ik_success=ik_success,
        collision_free=bool(collision_free and swept.accepted),
        joint_limits_ok=joint_limits_ok,
        dds_connected=True,
        waist_divergence_m=waist_position,
        waist_divergence_deg=waist_orientation,
    )
    sink = MockSuggestionSink()
    adapter = FailClosedMockAdapter(sink)
    mock_records = [
        adapter.commit_to_mock(np.asarray(record["suggested_joint_target_rad"]), adapter_inputs)
        for record in ik_records
    ]
    decisions = [record["decision"] for record in mock_records]

    return {
        "schema_version": "g1_offline_policy_shadow_replay_v1",
        "scope": (
            "Offline dataflow replay using saved real joints, synthetic cameras, "
            "a deterministic hold-policy fixture, measured-waist IK, and mock sink."
        ),
        "contract": {
            "id": CONTRACT_ID,
            "version": CONTRACT_VERSION,
            "sha256": CONTRACT_SHA256,
            "policy_rate_hz": POLICY_RATE_HZ,
            "action_horizon": ACTION_HORIZON,
        },
        "inputs": {
            "lowstate_sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
            "fk_report_sha256": hashlib.sha256(fk_report_path.read_bytes()).hexdigest(),
            "camera_source": "deterministic_synthetic_fixture_not_real_camera",
            "policy_source": "deterministic_hold_fixture_not_lgg100_inference",
            "gripper_source": "zero_fixture_not_real_dex1_state",
        },
        "observation": {
            "sha256": _observation_hash(observation),
            "state_sha256": _hash_array(observation["observation/state"]),
            "state_fk_view": "yuhao_zero_waist_for_contract_parity",
            "camera": camera_metadata,
        },
        "action": {
            "raw_sha256": _hash_array(raw),
            "canonical_sha256": _hash_array(canonical),
            "raw_shape": list(raw.shape),
            "canonical_shape": list(canonical.shape),
            "quaternion_only_postprocessing": non_quaternion_preserved,
            "raw_and_canonical_different": _hash_array(raw) != _hash_array(canonical),
        },
        "latency": {
            "lgg100_inference_executed": False,
            "replayed_q0_inference_p50_ms": float(
                q0_report["output_only_probe"]["latency_ms"]["p50"]
            ),
            "replayed_q0_latency_is_current_run_measurement": False,
            "sequential_ik_total_ms": ik_latency_ms,
            "sequential_ik_per_target_mean_ms": ik_latency_ms / ACTION_HORIZON,
            "swept_path_ms": swept_latency_ms,
        },
        "schedule": {
            "timestamps_s": timestamps.tolist(),
            "sample_interval_ms": 1000.0 / POLICY_RATE_HZ,
            "chunk_span_s": (ACTION_HORIZON - 1) / POLICY_RATE_HZ,
            "exec_steps": 0,
            "prefetch_lead_steps": 5,
            "prefetch_boundary_index": ACTION_HORIZON - 5,
            "prefetch_boundary_time_s": (ACTION_HORIZON - 5) / POLICY_RATE_HZ,
            "prefetch_window_ms": 5 / POLICY_RATE_HZ * 1000.0,
            "blend_steps": 5,
            "time_alignment": True,
        },
        "sequential_ik": {
            "all_reachable": ik_success,
            "all_collision_free_at_targets": collision_free,
            "all_joint_limits_ok": joint_limits_ok,
            "targets": ik_records,
        },
        "swept_path": {
            "accepted": swept.accepted,
            "checked_targets": swept.checked_targets,
            "checked_interpolated_configurations": swept.checked_interpolated_configurations,
            "maximum_position_error_m": swept.maximum_position_error_m,
            "maximum_orientation_error_rad": swept.maximum_orientation_error_rad,
            "reason": swept.reason,
            "collision_reasons": list(swept.collision_reasons),
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
            "offline_h4_replay_harness_completed": True,
            "real_three_camera_observation_used": False,
            "real_streaming_lowstate_used": False,
            "frozen_lgg100_inference_executed": False,
            "real_15_hz_policy_shadow_passed": False,
            "waist_divergence_hold_observed": all(
                "waist_divergence" in decision["reasons"] for decision in decisions
            ),
            "integrated_shadow_eligible": False,
            "robot_motion_allowed": False,
            "reason": (
                "The offline replay plumbing is complete, but synthetic images and "
                "a deterministic policy fixture cannot qualify real Policy Shadow. "
                "Measured waist divergence correctly forces every mock decision to hold."
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
    return 0 if report["decision"]["offline_h4_replay_harness_completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
