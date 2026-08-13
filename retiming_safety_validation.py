#!/usr/bin/env python3
"""Validate the simulation-only Gate-A safety governor on public episode 0."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

from action_schema import EEFActionChunk, pelvis_vla_action_to_world_mujoco
from dataset_contract_audit import ARM_JOINTS, CACHE, _fk_transform, download_public_assets
from dex1_gripper import Dex1Controller
from g1_dual_arm_ik import G1DualArmIK, orientation_error
from safety_governor import (
    G1TargetPreflight, JerkLimitedActionFilter, JerkLimitedJointFilter,
    JointMotionEnvelope, MotionEnvelope, manipulator_contact_violations,
)
from stack_scene import (
    build_model, reset_to_reference_pose, reset_to_stand, translate_cubes,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def _id(model, object_type, name: str) -> int:
    result = mujoco.mj_name2id(model, object_type, name)
    if result < 0:
        raise RuntimeError(f"Missing {name}")
    return result


def _load_episode_zero() -> tuple[np.ndarray, np.ndarray, np.ndarray, EEFActionChunk]:
    path = CACHE / "episode_000000.parquet"
    if not path.exists():
        download_public_assets()
    table = pq.read_table(path, columns=["observation.state", "action", "timestamp"])
    states = np.asarray(table.column("observation.state").to_pylist())
    raw_actions = np.asarray(table.column("action").to_pylist())
    timestamps = np.asarray(table.column("timestamp"), dtype=np.float64)
    timestamps -= timestamps[0]
    eef_actions = _fk_transform(raw_actions)
    return states, raw_actions, timestamps, EEFActionChunk(timestamps, eef_actions)


def _command_from_current(ik: G1DualArmIK, grippers: np.ndarray) -> np.ndarray:
    left = ik.pose("left")
    right = ik.pose("right")
    return np.r_[left[0], left[1], right[0], right[1], grippers]


def _quat_step_angle(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    return float(2 * np.arccos(np.clip(abs(np.dot(q0, q1)), 0.0, 1.0)))


def _run_scale(
    chunk: EEFActionChunk,
    initial_grippers: np.ndarray,
    scale: float,
    use_filter: bool,
    use_joint_filter: bool = False,
    phase_schedule: tuple[str, ...] | list[str] | None = None,
    abort_on_phase_aware_contact: bool = False,
    cube_translation_m: np.ndarray | None = None,
) -> dict:
    model = build_model()
    data = mujoco.MjData(model)
    reset_to_reference_pose(model, data)
    if cube_translation_m is not None:
        translate_cubes(model, data, cube_translation_m)
    hold_control = data.ctrl.copy()
    solver = G1DualArmIK(model, data)
    solver.reset()
    controller = Dex1Controller(model)
    envelope = MotionEnvelope()
    initial = _command_from_current(solver, initial_grippers)
    action_filter = JerkLimitedActionFilter(envelope)
    action_filter.reset(initial)
    joint_envelope = JointMotionEnvelope()
    joint_filter = JerkLimitedJointFilter(joint_envelope)
    pelvis = _id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    arm_dofs = np.concatenate((solver.left["dof"], solver.right["dof"]))
    arm_qpos = np.concatenate((solver.left["qpos"], solver.right["qpos"]))
    arm_actuators = np.concatenate((solver.left["actuator"], solver.right["actuator"]))
    joint_ids = np.concatenate((solver.left["joint"], solver.right["joint"]))
    lower_limits = model.jnt_range[joint_ids, 0]
    upper_limits = model.jnt_range[joint_ids, 1]
    joint_filter.reset(data.qpos[arm_qpos])
    dt = model.opt.timestep
    path_duration = float(chunk.timestamps[-1] / scale)
    total_duration = path_duration + 2.0
    if phase_schedule is not None:
        if len(phase_schedule) != len(chunk.timestamps):
            raise ValueError("phase_schedule must match the action horizon")
        unknown = set(phase_schedule) - {
            "free_space", "approach", "grasp", "lift", "place", "retreat"
        }
        if unknown:
            raise ValueError(f"unknown phase schedule entries: {sorted(unknown)}")

    previous_command = initial.copy()
    previous_velocity = np.zeros((2, 3))
    previous_acceleration = np.zeros((2, 3))
    previous_joint_command = data.qpos[arm_qpos].copy()
    previous_joint_command_velocity = np.zeros(14)
    previous_joint_command_acceleration = np.zeros(14)
    previous_arm_velocity = data.qvel[arm_dofs].copy()
    # mj_forward has already computed the acceleration implied by the initial
    # held pose. This avoids counting an artificial 0→qacc jump at frame zero.
    previous_arm_acceleration = data.qacc[arm_dofs].copy()
    maxima = {
        "command_speed_m_s": 0.0,
        "command_acceleration_m_s2": 0.0,
        "command_jerk_m_s3": 0.0,
        "command_angular_speed_rad_s": 0.0,
        "command_gripper_speed_rad_s": 0.0,
        "joint_command_speed_rad_s": 0.0,
        "joint_command_acceleration_rad_s2": 0.0,
        "joint_command_jerk_rad_s3": 0.0,
        "joint_command_speed_limit_ratio": 0.0,
        "joint_command_acceleration_limit_ratio": 0.0,
        "joint_command_jerk_limit_ratio": 0.0,
        "actual_joint_speed_rad_s": 0.0,
        "actual_joint_acceleration_rad_s2": 0.0,
        "actual_joint_jerk_rad_s3": 0.0,
        "desired_to_filtered_position_lag_m": 0.0,
        "actual_to_filtered_position_error_m": 0.0,
    }
    contact_steps_free_space = 0
    contact_steps_manipulation = 0
    free_space_contact_reasons: Counter = Counter()
    manipulation_contact_reasons: Counter = Counter()
    phase_aware_contact_steps = 0
    phase_aware_contact_reasons: Counter = Counter()
    phase_step_counts: Counter = Counter()
    first_phase_aware_contact_event: dict = {}
    aborted_on_phase_aware_contact = False
    minimum_pelvis = float(data.xpos[pelvis, 2])
    step_count = int(np.ceil(total_duration / dt))
    command = initial.copy()
    desired = initial.copy()
    worst_actual_jerk_event: dict = {}
    executed_step_count = 0
    for step in range(step_count):
        executed_step_count = step + 1
        elapsed = step * dt
        path_time = min(float(chunk.timestamps[-1]), elapsed * scale)
        desired_pelvis = chunk.sample(path_time)
        desired = pelvis_vla_action_to_world_mujoco(
            desired_pelvis, data.xpos[pelvis], data.xquat[pelvis]
        )
        if use_filter:
            command, telemetry = action_filter.step(desired, dt)
        else:
            command = desired.copy()
            telemetry = None

        positions = np.vstack((command[0:3], command[7:10]))
        previous_positions = np.vstack((previous_command[0:3], previous_command[7:10]))
        velocity = (positions - previous_positions) / dt
        acceleration = (velocity - previous_velocity) / dt
        jerk = (acceleration - previous_acceleration) / dt
        angular_speed = max(
            _quat_step_angle(previous_command[3:7], command[3:7]) / dt,
            _quat_step_angle(previous_command[10:14], command[10:14]) / dt,
        )
        gripper_speed = float(np.max(np.abs(
            command[14:] - previous_command[14:]
        )) / dt)
        maxima["command_speed_m_s"] = max(
            maxima["command_speed_m_s"], float(np.linalg.norm(velocity, axis=1).max())
        )
        maxima["command_acceleration_m_s2"] = max(
            maxima["command_acceleration_m_s2"], float(np.linalg.norm(acceleration, axis=1).max())
        )
        maxima["command_jerk_m_s3"] = max(
            maxima["command_jerk_m_s3"], float(np.linalg.norm(jerk, axis=1).max())
        )
        maxima["command_angular_speed_rad_s"] = max(
            maxima["command_angular_speed_rad_s"], angular_speed
        )
        maxima["command_gripper_speed_rad_s"] = max(
            maxima["command_gripper_speed_rad_s"], gripper_speed
        )
        desired_lag = max(
            np.linalg.norm(command[0:3] - desired[0:3]),
            np.linalg.norm(command[7:10] - desired[7:10]),
        )
        maxima["desired_to_filtered_position_lag_m"] = max(
            maxima["desired_to_filtered_position_lag_m"], float(desired_lag)
        )

        data.ctrl[:] = hold_control
        controller.set_motor_commands(data, command[14], command[15])
        solver.step(command[:7], command[7:14], dt)
        desired_joint_command = solver.q_target[arm_qpos].copy()
        if use_joint_filter:
            joint_command, _ = joint_filter.step(
                desired_joint_command, dt, lower_limits, upper_limits
            )
            data.ctrl[arm_actuators] = joint_command
            # Keep the IK proposal state separate from the governed actuator
            # target; feeding the governed value back here would apply the
            # position gain twice and make the arm artificially sluggish.
        else:
            joint_command = desired_joint_command
        joint_command_velocity = (joint_command - previous_joint_command) / dt
        joint_command_acceleration = (
            joint_command_velocity - previous_joint_command_velocity
        ) / dt
        joint_command_jerk = (
            joint_command_acceleration - previous_joint_command_acceleration
        ) / dt
        maxima["joint_command_speed_rad_s"] = max(
            maxima["joint_command_speed_rad_s"],
            float(np.abs(joint_command_velocity).max()),
        )
        maxima["joint_command_acceleration_rad_s2"] = max(
            maxima["joint_command_acceleration_rad_s2"],
            float(np.abs(joint_command_acceleration).max()),
        )
        maxima["joint_command_jerk_rad_s3"] = max(
            maxima["joint_command_jerk_rad_s3"],
            float(np.abs(joint_command_jerk).max()),
        )
        maxima["joint_command_speed_limit_ratio"] = max(
            maxima["joint_command_speed_limit_ratio"],
            float(np.max(np.abs(joint_command_velocity) / joint_filter.max_speed)),
        )
        maxima["joint_command_acceleration_limit_ratio"] = max(
            maxima["joint_command_acceleration_limit_ratio"],
            float(np.max(
                np.abs(joint_command_acceleration) / joint_filter.max_acceleration
            )),
        )
        maxima["joint_command_jerk_limit_ratio"] = max(
            maxima["joint_command_jerk_limit_ratio"],
            float(np.max(np.abs(joint_command_jerk) / joint_filter.max_jerk)),
        )
        mujoco.mj_step(model, data)
        left_actual = solver.pose("left")
        right_actual = solver.pose("right")
        actual_error = max(
            np.linalg.norm(left_actual[0] - command[0:3]),
            np.linalg.norm(right_actual[0] - command[7:10]),
        )
        maxima["actual_to_filtered_position_error_m"] = max(
            maxima["actual_to_filtered_position_error_m"], float(actual_error)
        )
        arm_velocity = data.qvel[arm_dofs].copy()
        arm_acceleration = (arm_velocity - previous_arm_velocity) / dt
        arm_jerk = (arm_acceleration - previous_arm_acceleration) / dt
        maxima["actual_joint_speed_rad_s"] = max(
            maxima["actual_joint_speed_rad_s"], float(np.abs(arm_velocity).max())
        )
        maxima["actual_joint_acceleration_rad_s2"] = max(
            maxima["actual_joint_acceleration_rad_s2"], float(np.abs(arm_acceleration).max())
        )
        actual_jerk_peak = float(np.abs(arm_jerk).max())
        if actual_jerk_peak > maxima["actual_joint_jerk_rad_s3"]:
            joint_index = int(np.argmax(np.abs(arm_jerk)))
            worst_actual_jerk_event = {
                "time_s": elapsed,
                "joint": ARM_JOINTS[joint_index],
                "jerk_rad_s3": actual_jerk_peak,
                "acceleration_rad_s2": float(arm_acceleration[joint_index]),
                "velocity_rad_s": float(arm_velocity[joint_index]),
                "free_space_contact_reasons": list(
                    manipulator_contact_violations(model, data, "free_space")
                ),
                "manipulation_contact_reasons": list(
                    manipulator_contact_violations(model, data, "grasp")
                ),
            }
        maxima["actual_joint_jerk_rad_s3"] = max(
            maxima["actual_joint_jerk_rad_s3"], actual_jerk_peak
        )
        free_space_violations = manipulator_contact_violations(
            model, data, "free_space"
        )
        manipulation_violations = manipulator_contact_violations(
            model, data, "grasp"
        )
        scheduled_phase = None
        phase_aware_violations: tuple[str, ...] = ()
        if phase_schedule is not None:
            phase_index = int(np.clip(
                np.searchsorted(chunk.timestamps, path_time, side="right") - 1,
                0,
                len(phase_schedule) - 1,
            ))
            scheduled_phase = phase_schedule[phase_index]
            collision_phase = {
                "free_space": "free_space",
                "approach": "free_space",
                "grasp": "grasp",
                "lift": "grasp",
                "place": "place",
                "retreat": "free_space",
            }[scheduled_phase]
            phase_aware_violations = manipulator_contact_violations(
                model, data, collision_phase
            )
            phase_step_counts[scheduled_phase] += 1
            if phase_aware_violations and not first_phase_aware_contact_event:
                first_phase_aware_contact_event = {
                    "time_s": elapsed,
                    "path_time_s": path_time,
                    "phase_index": phase_index,
                    "scheduled_phase": scheduled_phase,
                    "collision_phase": collision_phase,
                    "reasons": list(phase_aware_violations),
                }
            phase_aware_contact_steps += int(bool(phase_aware_violations))
            phase_aware_contact_reasons.update(phase_aware_violations)
        contact_steps_free_space += int(bool(free_space_violations))
        contact_steps_manipulation += int(bool(manipulation_violations))
        free_space_contact_reasons.update(free_space_violations)
        manipulation_contact_reasons.update(manipulation_violations)
        minimum_pelvis = min(minimum_pelvis, float(data.xpos[pelvis, 2]))
        previous_command = command.copy()
        previous_velocity = velocity
        previous_acceleration = acceleration
        previous_joint_command = joint_command
        previous_joint_command_velocity = joint_command_velocity
        previous_joint_command_acceleration = joint_command_acceleration
        previous_arm_velocity = arm_velocity
        previous_arm_acceleration = arm_acceleration
        if abort_on_phase_aware_contact and phase_aware_violations:
            aborted_on_phase_aware_contact = True
            break

    left_final = solver.pose("left")
    right_final = solver.pose("right")
    endpoint_error = float(max(
        np.linalg.norm(left_final[0] - desired[0:3]),
        np.linalg.norm(right_final[0] - desired[7:10]),
    ))
    hard_limits_pass = bool(
        maxima["command_speed_m_s"] <= envelope.max_eef_speed_m_s + 1e-6
        and maxima["command_acceleration_m_s2"] <= envelope.max_eef_acceleration_m_s2 + 1e-5
        and maxima["command_jerk_m_s3"] <= envelope.max_eef_jerk_m_s3 + 1e-3
        and maxima["command_angular_speed_rad_s"] <= envelope.max_angular_speed_rad_s + 1e-6
        and maxima["command_gripper_speed_rad_s"] <= envelope.max_gripper_speed_rad_s + 1e-5
    ) if use_filter else False
    joint_hard_limits_pass = bool(
        maxima["joint_command_speed_limit_ratio"] <= 1.0 + 1e-6
        and maxima["joint_command_acceleration_limit_ratio"] <= 1.0 + 1e-5
        and maxima["joint_command_jerk_limit_ratio"] <= 1.0 + 1e-4
    ) if use_joint_filter else False
    return {
        "scale": scale,
        "cube_translation_m": (
            [0.0, 0.0, 0.0]
            if cube_translation_m is None
            else np.asarray(cube_translation_m, dtype=np.float64).tolist()
        ),
        "filter_enabled": use_filter,
        "joint_filter_enabled": use_joint_filter,
        "nominal_path_duration_s": path_duration,
        "simulated_duration_s": executed_step_count * dt,
        "hard_command_limits_pass": bool(
            hard_limits_pass and (joint_hard_limits_pass if use_joint_filter else True)
        ),
        "eef_hard_limits_pass": hard_limits_pass,
        "joint_hard_limits_pass": joint_hard_limits_pass,
        "endpoint_error_m": endpoint_error,
        "minimum_pelvis_height_m": minimum_pelvis,
        "forbidden_contact_step_rate": contact_steps_manipulation / executed_step_count,
        "free_space_contact_step_rate": contact_steps_free_space / executed_step_count,
        "manipulation_contact_step_rate": contact_steps_manipulation / executed_step_count,
        "free_space_contact_reason_counts": dict(free_space_contact_reasons),
        "manipulation_contact_reason_counts": dict(manipulation_contact_reasons),
        "phase_schedule_enabled": phase_schedule is not None,
        "phase_aware_contact_step_rate": (
            phase_aware_contact_steps / executed_step_count
            if phase_schedule is not None else None
        ),
        "phase_aware_contact_reason_counts": dict(phase_aware_contact_reasons),
        "phase_step_counts": dict(phase_step_counts),
        "first_phase_aware_contact_event": first_phase_aware_contact_event,
        "abort_on_phase_aware_contact_enabled": abort_on_phase_aware_contact,
        "aborted_on_phase_aware_contact": aborted_on_phase_aware_contact,
        "executed_step_count": executed_step_count,
        "finite": bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))),
        "worst_actual_joint_jerk_event": worst_actual_jerk_event,
        "maxima": maxima,
    }


def _preflight_validation(
    states: np.ndarray,
    eef_actions: np.ndarray,
    sample_stride: int = 10,
    ood_samples: int = 100,
) -> dict:
    model = build_model()
    data = mujoco.MjData(model)
    gate = G1TargetPreflight(model)
    arm_qpos = np.asarray([
        model.jnt_qposadr[_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in ARM_JOINTS
    ])
    pelvis = _id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    phase_reasons = {phase: Counter() for phase in ("free_space", "grasp")}
    phase_collisions = {phase: Counter() for phase in ("free_space", "grasp")}
    checked = 0
    for index in range(0, len(states), sample_stride):
        reset_to_reference_pose(model, data)
        data.qpos[arm_qpos] = states[index, :14]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        target = pelvis_vla_action_to_world_mujoco(
            eef_actions[index], data.xpos[pelvis], data.xquat[pelvis]
        )
        for phase in phase_reasons:
            result = gate.check(data, target, phase=phase)
            phase_reasons[phase][result.reason] += 1
            phase_collisions[phase].update(result.collision_reasons)
        checked += 1

    reset_to_stand(model, data)
    ik = G1DualArmIK(model, data)
    left_q = ik.pose("left")[1]
    right_q = ik.pose("right")[1]
    rng = np.random.default_rng(20250811)
    ood_reasons = Counter()
    for _ in range(ood_samples):
        target = np.r_[
            [rng.uniform(0.10, 0.70), rng.uniform(0.02, 0.48), rng.uniform(0.62, 1.16)],
            left_q,
            [rng.uniform(0.10, 0.70), rng.uniform(-0.48, -0.02), rng.uniform(0.62, 1.16)],
            right_q,
            [5.5, 5.5],
        ]
        result = gate.check(data, target, phase="free_space")
        ood_reasons[result.reason] += 1
    selected_reasons = phase_reasons["grasp"]
    return {
        "in_distribution": {
            "checked_targets": checked,
            "selected_phase": "grasp_policy_view_without_ground_truth_phase_labels",
            "reason_counts": dict(selected_reasons),
            "collision_category_counts": dict(phase_collisions["grasp"]),
            "accepted_rate": selected_reasons["accepted"] / checked,
            "by_phase": {
                phase: {
                    "reason_counts": dict(phase_reasons[phase]),
                    "collision_category_counts": dict(phase_collisions[phase]),
                    "accepted_rate": phase_reasons[phase]["accepted"] / checked,
                }
                for phase in phase_reasons
            },
        },
        "out_of_distribution": {
            "checked_targets": ood_samples,
            "reason_counts": dict(ood_reasons),
            "rejected_rate": 1.0 - ood_reasons["accepted"] / ood_samples,
        },
        "rule": (
            "Free-space rejects Dex1-cube contact; grasp/place allows it. All phases "
            "reject table/floor, torso-elbow, cross-arm and non-adjacent self contact. "
            "The public-model torso/shoulder-yaw overlap is explicitly allowlisted."
        ),
    }


def main() -> None:
    states, raw_actions, _, chunk = _load_episode_zero()
    eef_actions = _fk_transform(raw_actions)
    envelope = MotionEnvelope()
    joint_envelope = JointMotionEnvelope()
    runs = [
        _run_scale(chunk, states[0, 14:], 1.0, False, False),
        _run_scale(chunk, states[0, 14:], 1.0, True, False),
    ]
    for scale in (0.50, 0.75, 1.00, 1.25, 1.50):
        runs.append(_run_scale(chunk, states[0, 14:], scale, True, True))
    report = {
        "scope": (
            "Gate-A.2 simulation-only validation using public episode-0 actions as "
            "recorded-policy proxies. No neural VLA checkpoint and no authoritative "
            "Unitree hardware limits are used."
        ),
        "motion_envelope": envelope.__dict__,
        "joint_motion_envelope": joint_envelope.__dict__,
        "episode_zero_scale_sweep": runs,
        "preflight_gate": _preflight_validation(states, eef_actions),
    }
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "retiming_safety_validation.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(path)
    print(json.dumps({
        "joint_filtered_runs": sum(run["joint_filter_enabled"] for run in runs),
        "all_joint_filtered_hard_limits_pass": all(
            run["hard_command_limits_pass"]
            for run in runs if run["joint_filter_enabled"]
        ),
        "endpoint_errors_m": {
            str(run["scale"]): run["endpoint_error_m"]
            for run in runs if run["joint_filter_enabled"]
        },
        "preflight": report["preflight_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
