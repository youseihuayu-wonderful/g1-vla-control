"""Persistent fail-closed MuJoCo executor for quarantined closed-loop studies.

This module is simulation-only. It preserves the EEF and joint jerk-limited
filter state across receding-horizon chunks and aborts immediately on phase-
invalid contact, excessive tracking error, instability, or non-finite state.
"""

from __future__ import annotations

from collections import Counter

import mujoco
import numpy as np

from action_schema import EEFActionChunk, pelvis_vla_action_to_world_mujoco
from dex1_gripper import Dex1Controller
from g1_dual_arm_ik import G1DualArmIK
from safety_governor import (
    JerkLimitedActionFilter, JerkLimitedJointFilter, JointMotionEnvelope,
    MotionEnvelope, manipulator_contact_violations,
)

_ALLOWED_PHASES = frozenset({
    "free_space", "approach", "grasp", "lift", "place", "retreat",
})
_COLLISION_PHASE = {
    "free_space": "free_space",
    "approach": "free_space",
    "grasp": "grasp",
    "lift": "grasp",
    "place": "place",
    "retreat": "free_space",
}


def _id(model: mujoco.MjModel, object_type, name: str) -> int:
    result = mujoco.mj_name2id(model, object_type, name)
    if result < 0:
        raise ValueError(f"missing MuJoCo object: {name}")
    return result


class QuarantinedChunkExecutor:
    """Execute short prefixes without resetting simulation or safety filters."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        hard_tracking_error_m: float = 0.050,
        minimum_pelvis_height_m: float = 0.70,
    ) -> None:
        self.model = model
        self.data = data
        self.hard_tracking_error_m = float(hard_tracking_error_m)
        self.minimum_pelvis_height_m = float(minimum_pelvis_height_m)
        if self.hard_tracking_error_m <= 0.0:
            raise ValueError("hard_tracking_error_m must be positive")
        mujoco.mj_forward(model, data)
        self.solver = G1DualArmIK(model, data)
        self.solver.reset()
        self.gripper = Dex1Controller(model)
        self.motion_envelope = MotionEnvelope()
        self.joint_envelope = JointMotionEnvelope()
        self.action_filter = JerkLimitedActionFilter(self.motion_envelope)
        self.joint_filter = JerkLimitedJointFilter(self.joint_envelope)
        self.pelvis = _id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.arm_qpos = np.concatenate((
            self.solver.left["qpos"], self.solver.right["qpos"],
        ))
        self.arm_actuators = np.concatenate((
            self.solver.left["actuator"], self.solver.right["actuator"],
        ))
        joint_ids = np.concatenate((
            self.solver.left["joint"], self.solver.right["joint"],
        ))
        self.lower_limits = model.jnt_range[joint_ids, 0].copy()
        self.upper_limits = model.jnt_range[joint_ids, 1].copy()
        self.base_control = data.ctrl.copy()
        self.total_simulated_time_s = 0.0
        self._reset_filters_to_measured_state()

    def _measured_world_command(self) -> np.ndarray:
        left = self.solver.pose("left")
        right = self.solver.pose("right")
        return np.r_[
            left[0], left[1], right[0], right[1],
            self.gripper.motor_states(self.data),
        ]

    def _reset_filters_to_measured_state(self) -> None:
        self.solver.reset()
        command = self._measured_world_command()
        self.action_filter.reset(command)
        self.joint_filter.reset(self.data.qpos[self.arm_qpos])
        self.last_filtered_command = command

    def current_eef_tracking_error_m(self) -> float:
        measured = self._measured_world_command()
        return float(max(
            np.linalg.norm(measured[0:3] - self.last_filtered_command[0:3]),
            np.linalg.norm(measured[7:10] - self.last_filtered_command[7:10]),
        ))

    def hold(self) -> None:
        """Replace outstanding commands with a measured-state hold target."""
        self._reset_filters_to_measured_state()
        self.data.ctrl[:] = self.base_control
        self.data.ctrl[self.arm_actuators] = self.data.qpos[self.arm_qpos]
        measured_grippers = self.gripper.motor_states(self.data)
        self.gripper.set_motor_commands(
            self.data, float(measured_grippers[0]), float(measured_grippers[1])
        )

    def execute_prefix(
        self,
        chunk: EEFActionChunk,
        phase_schedule: tuple[str, ...] | list[str],
        *,
        prefix_duration_s: float,
    ) -> dict:
        phases = tuple(phase_schedule)
        if len(phases) != len(chunk.timestamps):
            raise ValueError("phase schedule must match chunk horizon")
        unknown = set(phases) - _ALLOWED_PHASES
        if unknown:
            raise ValueError(f"unknown phases: {sorted(unknown)}")
        if not np.isfinite(prefix_duration_s) or prefix_duration_s <= 0.0:
            raise ValueError("prefix_duration_s must be finite and positive")
        prefix_duration_s = min(
            float(prefix_duration_s), float(chunk.timestamps[-1])
        )
        dt = float(self.model.opt.timestep)
        requested_steps = int(np.ceil(prefix_duration_s / dt))
        reasons: list[str] = []
        phase_steps: Counter = Counter()
        contact_reasons: Counter = Counter()
        maxima = {
            "eef_speed_m_s": 0.0,
            "eef_acceleration_m_s2": 0.0,
            "eef_jerk_m_s3": 0.0,
            "angular_speed_rad_s": 0.0,
            "gripper_speed_rad_s": 0.0,
            "joint_speed_ratio": 0.0,
            "joint_acceleration_ratio": 0.0,
            "joint_jerk_ratio": 0.0,
            "eef_tracking_error_m": 0.0,
        }
        minimum_pelvis = float(self.data.xpos[self.pelvis, 2])
        first_contact_event: dict = {}
        executed_steps = 0
        for step in range(requested_steps):
            executed_steps = step + 1
            path_time = min(step * dt, float(chunk.timestamps[-1]))
            phase_index = int(np.clip(
                np.searchsorted(chunk.timestamps, path_time, side="right") - 1,
                0,
                len(phases) - 1,
            ))
            scheduled_phase = phases[phase_index]
            phase_steps[scheduled_phase] += 1
            desired_pelvis = chunk.sample(path_time)
            desired_world = pelvis_vla_action_to_world_mujoco(
                desired_pelvis,
                self.data.xpos[self.pelvis],
                self.data.xquat[self.pelvis],
            )
            filtered, eef_telemetry = self.action_filter.step(desired_world, dt)
            self.data.ctrl[:] = self.base_control
            self.gripper.set_motor_commands(
                self.data, float(filtered[14]), float(filtered[15])
            )
            self.solver.step(filtered[:7], filtered[7:14], dt)
            desired_joints = self.solver.q_target[self.arm_qpos].copy()
            joint_command, joint_telemetry = self.joint_filter.step(
                desired_joints, dt, self.lower_limits, self.upper_limits
            )
            self.data.ctrl[self.arm_actuators] = joint_command
            mujoco.mj_step(self.model, self.data)
            self.last_filtered_command = filtered.copy()

            tracking_error = self.current_eef_tracking_error_m()
            maxima["eef_speed_m_s"] = max(
                maxima["eef_speed_m_s"],
                eef_telemetry.max_translation_speed_m_s,
            )
            maxima["eef_acceleration_m_s2"] = max(
                maxima["eef_acceleration_m_s2"],
                eef_telemetry.max_translation_acceleration_m_s2,
            )
            maxima["eef_jerk_m_s3"] = max(
                maxima["eef_jerk_m_s3"],
                eef_telemetry.max_translation_jerk_m_s3,
            )
            maxima["angular_speed_rad_s"] = max(
                maxima["angular_speed_rad_s"],
                eef_telemetry.max_angular_speed_rad_s,
            )
            maxima["gripper_speed_rad_s"] = max(
                maxima["gripper_speed_rad_s"],
                eef_telemetry.max_gripper_speed_rad_s,
            )
            maxima["joint_speed_ratio"] = max(
                maxima["joint_speed_ratio"], joint_telemetry.max_speed_ratio
            )
            maxima["joint_acceleration_ratio"] = max(
                maxima["joint_acceleration_ratio"],
                joint_telemetry.max_acceleration_ratio,
            )
            maxima["joint_jerk_ratio"] = max(
                maxima["joint_jerk_ratio"], joint_telemetry.max_jerk_ratio
            )
            maxima["eef_tracking_error_m"] = max(
                maxima["eef_tracking_error_m"], tracking_error
            )
            minimum_pelvis = min(
                minimum_pelvis, float(self.data.xpos[self.pelvis, 2])
            )
            violations = manipulator_contact_violations(
                self.model, self.data, _COLLISION_PHASE[scheduled_phase]
            )
            contact_reasons.update(violations)
            if violations and not first_contact_event:
                first_contact_event = {
                    "step": step,
                    "path_time_s": path_time,
                    "phase_index": phase_index,
                    "scheduled_phase": scheduled_phase,
                    "collision_phase": _COLLISION_PHASE[scheduled_phase],
                    "reasons": list(violations),
                }
            if violations:
                reasons.append("phase_invalid_contact")
            if tracking_error > self.hard_tracking_error_m:
                reasons.append("eef_tracking_error_above_hard_maximum")
            if minimum_pelvis < self.minimum_pelvis_height_m:
                reasons.append("pelvis_height_below_hard_minimum")
            if not (
                np.all(np.isfinite(self.data.qpos))
                and np.all(np.isfinite(self.data.qvel))
                and np.all(np.isfinite(self.data.ctrl))
            ):
                reasons.append("non_finite_simulation_state")
            if (
                eef_telemetry.max_translation_speed_m_s
                > self.motion_envelope.max_eef_speed_m_s + 1e-6
                or eef_telemetry.max_translation_acceleration_m_s2
                > self.motion_envelope.max_eef_acceleration_m_s2 + 1e-5
                or eef_telemetry.max_translation_jerk_m_s3
                > self.motion_envelope.max_eef_jerk_m_s3 + 1e-3
                or eef_telemetry.max_angular_speed_rad_s
                > self.motion_envelope.max_angular_speed_rad_s + 1e-6
                or eef_telemetry.max_gripper_speed_rad_s
                > self.motion_envelope.max_gripper_speed_rad_s + 1e-5
                or joint_telemetry.max_speed_ratio > 1.0 + 1e-6
                or joint_telemetry.max_acceleration_ratio > 1.0 + 1e-5
                or joint_telemetry.max_jerk_ratio > 1.0 + 1e-4
            ):
                reasons.append("filtered_command_limit_violation")
            if reasons:
                break

        elapsed = executed_steps * dt
        self.total_simulated_time_s += elapsed
        accepted = not reasons
        if not accepted:
            self.hold()
        return {
            "accepted": accepted,
            "hold_commanded": not accepted,
            "reasons": sorted(set(reasons)),
            "requested_prefix_duration_s": prefix_duration_s,
            "executed_duration_s": elapsed,
            "executed_steps": executed_steps,
            "phase_step_counts": dict(phase_steps),
            "phase_invalid_contact_reason_counts": dict(contact_reasons),
            "first_phase_invalid_contact_event": first_contact_event,
            "minimum_pelvis_height_m": minimum_pelvis,
            "maxima": maxima,
            "total_simulated_time_s": self.total_simulated_time_s,
            "finite": bool(
                np.all(np.isfinite(self.data.qpos))
                and np.all(np.isfinite(self.data.qvel))
            ),
        }
