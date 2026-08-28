#!/usr/bin/env python3
"""Analyze a saved subscriber-only full-29-joint G1 LowState capture offline."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from g1_unitree_lowstate import FULL_BODY_INDICES, FULL_BODY_JOINTS
from safety_governor import manipulator_contact_violations
from stack_scene import build_model, reset_to_reference_pose


PROVISIONAL_WARNING_MS = 20.0
PROVISIONAL_STALE_HOLD_MS = 50.0
PROVISIONAL_DISCONNECT_MS = 200.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _body_label(model: mujoco.MjModel, geom_id: int) -> str:
    body_id = int(model.geom_bodyid[geom_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"


def analyze_capture(
    raw_path: Path,
    *,
    adapter_path: Path,
    inventory_path: Path,
    console_log_path: Path,
    operator_safety_confirmed: bool,
) -> dict[str, Any]:
    raw = json.loads(raw_path.read_text())
    samples = raw["samples"]
    if not samples:
        raise ValueError("capture contains no samples")
    expected_names = list(FULL_BODY_JOINTS)
    expected_indices = list(FULL_BODY_INDICES)
    full_body_complete = all(
        [motor["name"] for motor in sample["full_body_motor_state"]] == expected_names
        and [motor["index"] for motor in sample["full_body_motor_state"]] == expected_indices
        for sample in samples
    )
    all_motor_values_finite = all(
        np.all(np.isfinite([
            motor["q"], motor["dq"], motor["ddq"], motor["tau_est"], motor["vol"]
        ]))
        for sample in samples
        for motor in sample["full_body_motor_state"]
    )
    ticks = np.asarray([sample["tick"] for sample in samples], dtype=np.int64)
    tick_delta = np.diff(ticks)
    monotonic_ns = np.asarray(
        [sample["received_monotonic_ns"] for sample in samples], dtype=np.int64
    )
    gaps_ms = np.diff(monotonic_ns) / 1_000_000.0
    q = np.asarray([
        [motor["q"] for motor in sample["full_body_motor_state"]]
        for sample in samples
    ], dtype=np.float64)
    dq = np.asarray([
        [motor["dq"] for motor in sample["full_body_motor_state"]]
        for sample in samples
    ], dtype=np.float64)
    tau = np.asarray([
        [motor["tau_est"] for motor in sample["full_body_motor_state"]]
        for sample in samples
    ], dtype=np.float64)
    temperatures = np.asarray([
        [max(motor["temperature"]) for motor in sample["full_body_motor_state"]]
        for sample in samples
    ], dtype=np.int64)
    reader_error_count = console_log_path.read_text(errors="replace").count(
        "[Reader] take sample error"
    )

    model = build_model()
    data = mujoco.MjData(model)
    missing_joints = []
    joint_ids = []
    qpos_indices = []
    for name in FULL_BODY_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            missing_joints.append(name)
        else:
            joint_ids.append(joint_id)
            qpos_indices.append(int(model.jnt_qposadr[joint_id]))
    pair_distances: dict[str, list[float]] = defaultdict(list)
    free_space_violation_count = 0
    representative_indices = {0, len(samples) // 2, len(samples) - 1}
    representatives = []
    for index, sample in enumerate(samples):
        reset_to_reference_pose(model, data)
        for qpos_index, motor in zip(qpos_indices, sample["full_body_motor_state"], strict=True):
            data.qpos[qpos_index] = motor["q"]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        violations = manipulator_contact_violations(model, data, "free_space")
        free_space_violation_count += int(bool(violations))
        contact_pairs = []
        for contact in data.contact:
            pair = " <-> ".join(sorted((
                _body_label(model, contact.geom1),
                _body_label(model, contact.geom2),
            )))
            pair_distances[pair].append(float(contact.dist))
            contact_pairs.append(pair)
        if index in representative_indices:
            representatives.append({
                "sample_index": index,
                "tick": int(sample["tick"]),
                "contact_count": int(data.ncon),
                "free_space_violations": list(violations),
                "contact_body_pairs": sorted(set(contact_pairs)),
            })
    pair_summary = {
        pair: {
            "contact_count": len(distances),
            "minimum_distance_mm": min(distances) * 1000.0,
            "median_distance_mm": float(np.median(distances)) * 1000.0,
            "maximum_distance_mm": max(distances) * 1000.0,
        }
        for pair, distances in sorted(pair_distances.items())
    }
    maximum_gap_ms = float(np.max(gaps_ms))
    unique_tick_count = int(len(np.unique(ticks)))
    duplicate_count = len(ticks) - unique_tick_count
    stale_gap_count = int(np.count_nonzero(gaps_ms > PROVISIONAL_STALE_HOLD_MS))
    return {
        "schema_version": "g1_full29_readonly_analysis_v1",
        "scope": (
            "Offline analysis of one real subscriber-only 29-joint LowState capture "
            "and software-model contact reconstruction. No physical EEF/contact "
            "qualification and no command path."
        ),
        "input": {
            "raw_capture_sha256": _sha256(raw_path),
            "adapter_sha256": _sha256(adapter_path),
            "ssh_inventory_sha256": _sha256(inventory_path),
            "console_log_sha256": _sha256(console_log_path),
            "raw_capture_committed_to_public_git": False,
            "operator_damping_support_estop_confirmed": operator_safety_confirmed,
        },
        "capture": {
            "success": bool(raw["success"]),
            "requested_samples": int(raw["requested_samples"]),
            "captured_samples": len(samples),
            "full_body_joint_count": len(FULL_BODY_JOINTS),
            "full_body_complete_all_samples": full_body_complete,
            "all_motor_values_finite": all_motor_values_finite,
            "q29_sha256": hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest(),
            "q_min_rad": float(np.min(q)),
            "q_max_rad": float(np.max(q)),
            "maximum_abs_dq_rad_s": float(np.max(np.abs(dq))),
            "maximum_abs_tau_est": float(np.max(np.abs(tau))),
            "maximum_temperature": int(np.max(temperatures)),
            "mode_pr_values": sorted({int(sample["mode_pr"]) for sample in samples}),
            "mode_machine_values": sorted({int(sample["mode_machine"]) for sample in samples}),
        },
        "dds": {
            "reader_take_sample_error_count": reader_error_count,
            "tick_first": int(ticks[0]),
            "tick_last": int(ticks[-1]),
            "unique_tick_count": unique_tick_count,
            "duplicate_tick_count": duplicate_count,
            "tick_decrease_count": int(np.count_nonzero(tick_delta < 0)),
            "positive_tick_increment_min": int(np.min(tick_delta[tick_delta > 0])),
            "positive_tick_increment_max": int(np.max(tick_delta[tick_delta > 0])),
            "receive_gap_ms": {
                "minimum": float(np.min(gaps_ms)),
                "median": float(np.median(gaps_ms)),
                "p95": float(np.percentile(gaps_ms, 95)),
                "p99": float(np.percentile(gaps_ms, 99)),
                "maximum": maximum_gap_ms,
            },
            "provisional_thresholds_ms": {
                "warning": PROVISIONAL_WARNING_MS,
                "stale_hold": PROVISIONAL_STALE_HOLD_MS,
                "disconnect": PROVISIONAL_DISCONNECT_MS,
            },
            "gaps_exceeding_provisional_stale_hold": stale_gap_count,
        },
        "collision_model": {
            "missing_model_joints": missing_joints,
            "samples_with_free_space_violation": free_space_violation_count,
            "sample_count": len(samples),
            "all_samples_initial_collision_free": free_space_violation_count == 0,
            "body_pair_distance_summary": pair_summary,
            "representatives": representatives,
            "physical_contact_verified": False,
        },
        "decision": {
            "subscriber_only_full29_capture_passed": bool(
                raw["success"] and len(samples) == raw["requested_samples"]
                and full_body_complete and all_motor_values_finite
            ),
            "missing_leg_state_blocker_resolved": full_body_complete,
            "initial_model_collision_resolved": free_space_violation_count == 0,
            "initial_collision_allowlist_authorized": False,
            "physical_contact_inspection_required": free_space_violation_count > 0,
            "dds_freshness_fully_qualified": bool(
                reader_error_count == 0
                and duplicate_count == 0
                and not np.any(tick_delta < 0)
                and stale_gap_count == 0
            ),
            "provisional_50ms_stale_hold_would_trigger": stale_gap_count > 0,
            "real_three_camera_passed": False,
            "real_15_hz_policy_shadow_passed": False,
            "live_gate_promoted": False,
            "robot_motion_allowed": False,
            "reason": (
                "Full 29-joint state is now available, but every reconstructed sample "
                "has non-adjacent hand/wrist-to-hip model contact and physical contact "
                "has not been inspected. DDS also has a Reader error, duplicate ticks, "
                "and a receive gap above the provisional stale-hold threshold."
            ),
        },
        "safety": {
            "subscriber_only": True,
            "send_channel_created": False,
            "publisher_created": False,
            "mode_change_requested": False,
            "robot_command_sent": False,
        },
        "g1_execution_enabled": False,
        "hardware_execution_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--console-log", type=Path, required=True)
    parser.add_argument("--operator-safety-confirmed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze_capture(
        args.raw,
        adapter_path=args.adapter,
        inventory_path=args.inventory,
        console_log_path=args.console_log,
        operator_safety_confirmed=args.operator_safety_confirmed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0 if report["decision"]["subscriber_only_full29_capture_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
