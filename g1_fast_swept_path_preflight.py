#!/usr/bin/env python3
"""Tolerance-aware sequential IK plus swept collision preflight.

This combines early-stop warm-start IK with the existing phase-aware MuJoCo
collision classifier. Safety tolerances and interpolation resolution are not
relaxed. There is no hardware I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import time

import mujoco
import numpy as np

from action_schema import EEFActionChunk
from dex1_gripper import Dex1Controller, motor_radians_to_jaw_position
from g1_fast_sequential_ik import FastIKConfig, solve_sequential_ik
from safety_governor import manipulator_contact_violations


@dataclass(frozen=True)
class FastSweptPathResult:
    accepted: bool
    checked_targets: int
    checked_interpolated_configurations: int
    maximum_position_error_m: float
    maximum_orientation_error_rad: float
    total_ik_iterations: int
    ik_elapsed_ms: float
    collision_elapsed_ms: float
    total_elapsed_ms: float
    rejection_target_index: int | None
    rejection_substep: int | None
    reason: str
    collision_reasons: tuple[str, ...]


class G1FastSweptPathPreflight:
    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        ik_config: FastIKConfig = FastIKConfig(),
        maximum_arm_interpolation_step_rad: float = 0.01,
        maximum_finger_interpolation_step_m: float = 0.001,
    ) -> None:
        self.model = model
        self.ik_config = ik_config
        self.maximum_arm_interpolation_step_rad = maximum_arm_interpolation_step_rad
        self.maximum_finger_interpolation_step_m = maximum_finger_interpolation_step_m

    def check(
        self,
        source: mujoco.MjData,
        chunk: EEFActionChunk,
        *,
        phase: str | Sequence[str],
    ) -> FastSweptPathResult:
        started = time.perf_counter_ns()
        if isinstance(phase, str):
            phase_schedule = (phase,) * len(chunk.actions)
        else:
            phase_schedule = tuple(phase)
        if len(phase_schedule) != len(chunk.actions):
            raise ValueError("phase schedule must match the action horizon")
        phase_map = {
            "free_space": "free_space",
            "approach": "free_space",
            "grasp": "grasp",
            "lift": "grasp",
            "place": "place",
            "retreat": "free_space",
        }
        unknown = set(phase_schedule) - set(phase_map)
        if unknown:
            raise ValueError(f"unknown phase schedule entries: {sorted(unknown)}")
        collision_schedule = tuple(phase_map[item] for item in phase_schedule)

        data = mujoco.MjData(self.model)
        data.qpos[:] = source.qpos
        data.qvel[:] = 0.0
        data.ctrl[:] = source.ctrl
        mujoco.mj_forward(self.model, data)
        initial_violations = manipulator_contact_violations(
            self.model, data, collision_schedule[0]
        )
        if initial_violations:
            elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
            return FastSweptPathResult(
                False, 0, 1, 0.0, 0.0, 0, 0.0, elapsed, elapsed,
                0, 0, "initial_configuration_collision", initial_violations,
            )

        ik_result = solve_sequential_ik(
            self.model, source, chunk.actions, self.ik_config
        )
        maximum_position_error = max(
            (target.maximum_position_error_m for target in ik_result.targets),
            default=0.0,
        )
        maximum_orientation_error = max(
            (target.maximum_orientation_error_rad for target in ik_result.targets),
            default=0.0,
        )
        if not ik_result.accepted:
            elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
            rejection = ik_result.targets[-1].index if ik_result.targets else 0
            return FastSweptPathResult(
                False, len(ik_result.targets), 1,
                maximum_position_error, maximum_orientation_error,
                ik_result.total_iterations, ik_result.elapsed_ms,
                max(0.0, elapsed - ik_result.elapsed_ms), elapsed,
                rejection, None, ik_result.reason, (),
            )

        # Resolve arm and finger indices once; collision interpolation uses the
        # exact same 0.01 rad / 0.001 m resolution as the legacy preflight.
        from g1_dual_arm_ik import G1DualArmIK
        mapping = G1DualArmIK(self.model, data)
        arm_qpos = np.concatenate((mapping.left["qpos"], mapping.right["qpos"]))
        gripper = Dex1Controller(self.model)
        finger_qpos = np.concatenate((gripper.qpos["left"], gripper.qpos["right"]))
        previous_arm = data.qpos[arm_qpos].copy()
        previous_fingers = data.qpos[finger_qpos].copy()
        checked_configurations = 1
        collision_started = time.perf_counter_ns()

        for target_index, (action, solved) in enumerate(
            zip(chunk.actions, ik_result.targets, strict=True)
        ):
            solved_arm = solved.joint_target_rad
            target_fingers = np.array([
                motor_radians_to_jaw_position(action[14]),
                motor_radians_to_jaw_position(action[14]),
                motor_radians_to_jaw_position(action[15]),
                motor_radians_to_jaw_position(action[15]),
            ])
            arm_delta = float(np.max(np.abs(solved_arm - previous_arm)))
            finger_delta = float(np.max(np.abs(target_fingers - previous_fingers)))
            substeps = max(
                1,
                int(np.ceil(arm_delta / self.maximum_arm_interpolation_step_rad)),
                int(np.ceil(finger_delta / self.maximum_finger_interpolation_step_m)),
            )
            for substep in range(1, substeps + 1):
                fraction = substep / substeps
                data.qpos[arm_qpos] = previous_arm + fraction * (solved_arm - previous_arm)
                data.qpos[finger_qpos] = previous_fingers + fraction * (
                    target_fingers - previous_fingers
                )
                data.qvel[:] = 0.0
                mujoco.mj_fwdPosition(self.model, data)
                checked_configurations += 1
                violations = manipulator_contact_violations(
                    self.model, data, collision_schedule[target_index]
                )
                if violations:
                    collision_elapsed = (
                        time.perf_counter_ns() - collision_started
                    ) / 1_000_000.0
                    total_elapsed = (
                        time.perf_counter_ns() - started
                    ) / 1_000_000.0
                    return FastSweptPathResult(
                        False, target_index + 1, checked_configurations,
                        maximum_position_error, maximum_orientation_error,
                        ik_result.total_iterations, ik_result.elapsed_ms,
                        collision_elapsed, total_elapsed,
                        target_index, substep, "swept_collision", violations,
                    )
            previous_arm = solved_arm.copy()
            previous_fingers = target_fingers

        collision_elapsed = (
            time.perf_counter_ns() - collision_started
        ) / 1_000_000.0
        total_elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
        return FastSweptPathResult(
            True, len(chunk.actions), checked_configurations,
            maximum_position_error, maximum_orientation_error,
            ik_result.total_iterations, ik_result.elapsed_ms,
            collision_elapsed, total_elapsed,
            None, None, "accepted", (),
        )
