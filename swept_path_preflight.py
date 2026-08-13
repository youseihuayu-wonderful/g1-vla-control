"""Sequential swept-path IK/collision preflight for G1 dual-Dex1 chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import mujoco
import numpy as np

from action_schema import EEFActionChunk, pelvis_vla_action_to_world_mujoco
from dex1_gripper import Dex1Controller, motor_radians_to_jaw_position
from g1_dual_arm_ik import G1DualArmIK, orientation_error
from safety_governor import manipulator_contact_violations


@dataclass(frozen=True)
class SweptPathResult:
    accepted: bool
    checked_targets: int
    checked_interpolated_configurations: int
    maximum_position_error_m: float
    maximum_orientation_error_rad: float
    rejection_target_index: int | None
    rejection_substep: int | None
    reason: str
    collision_reasons: tuple[str, ...]


class G1SweptPathPreflight:
    """Check sequential IK and interpolated robot geometry with fixed objects.

    This is a conservative kinematic gate. It does not replace dynamic rollout,
    online contact monitoring, tracking feedback, or a hardware watchdog.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        position_tolerance_m: float = 0.005,
        orientation_tolerance_rad: float = np.deg2rad(3.0),
        ik_iterations: int = 250,
        maximum_arm_interpolation_step_rad: float = 0.01,
        maximum_finger_interpolation_step_m: float = 0.001,
    ):
        self.model = model
        self.position_tolerance_m = position_tolerance_m
        self.orientation_tolerance_rad = orientation_tolerance_rad
        self.ik_iterations = ik_iterations
        self.maximum_arm_interpolation_step_rad = (
            maximum_arm_interpolation_step_rad
        )
        self.maximum_finger_interpolation_step_m = (
            maximum_finger_interpolation_step_m
        )

    def check(
        self,
        source: mujoco.MjData,
        chunk: EEFActionChunk,
        *,
        phase: str | Sequence[str],
    ) -> SweptPathResult:
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
        collision_schedule = tuple(
            phase_map[item] for item in phase_schedule
        )
        data = mujoco.MjData(self.model)
        data.qpos[:] = source.qpos
        data.qvel[:] = 0.0
        data.ctrl[:] = source.ctrl
        mujoco.mj_forward(self.model, data)
        initial_violations = manipulator_contact_violations(
            self.model, data, collision_schedule[0]
        )
        if initial_violations:
            return SweptPathResult(
                False, 0, 1, 0.0, 0.0, 0, 0,
                "initial_configuration_collision", initial_violations,
            )

        solver = G1DualArmIK(
            self.model, data, damping=0.06, max_joint_speed=2.0
        )
        solver.reset()
        gripper = Dex1Controller(self.model)
        arm_qpos = np.concatenate((solver.left["qpos"], solver.right["qpos"]))
        finger_qpos = np.concatenate((
            gripper.qpos["left"], gripper.qpos["right"]
        ))
        pelvis = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        previous_arm = data.qpos[arm_qpos].copy()
        previous_fingers = data.qpos[finger_qpos].copy()
        checked_configurations = 1
        maximum_position_error = 0.0
        maximum_orientation_error = 0.0

        for target_index, action in enumerate(chunk.actions):
            target = pelvis_vla_action_to_world_mujoco(
                action, data.xpos[pelvis], data.xquat[pelvis]
            )
            data.qpos[arm_qpos] = previous_arm
            data.qpos[finger_qpos] = previous_fingers
            data.qvel[:] = 0.0
            mujoco.mj_fwdPosition(self.model, data)
            solver.q_target[arm_qpos] = previous_arm
            for _ in range(self.ik_iterations):
                solver.step(target[:7], target[7:14], 0.02)
                data.qpos[arm_qpos] = solver.q_target[arm_qpos]
                data.qvel[:] = 0.0
                mujoco.mj_fwdPosition(self.model, data)
            solved_arm = data.qpos[arm_qpos].copy()
            target_fingers = np.array([
                motor_radians_to_jaw_position(action[14]),
                motor_radians_to_jaw_position(action[14]),
                motor_radians_to_jaw_position(action[15]),
                motor_radians_to_jaw_position(action[15]),
            ])
            left_pose = solver.pose("left")
            right_pose = solver.pose("right")
            position_error = float(max(
                np.linalg.norm(left_pose[0] - target[:3]),
                np.linalg.norm(right_pose[0] - target[7:10]),
            ))
            orientation_error_value = float(max(
                np.linalg.norm(orientation_error(target[3:7], left_pose[1])),
                np.linalg.norm(orientation_error(target[10:14], right_pose[1])),
            ))
            maximum_position_error = max(maximum_position_error, position_error)
            maximum_orientation_error = max(
                maximum_orientation_error, orientation_error_value
            )
            if (
                position_error > self.position_tolerance_m
                or orientation_error_value > self.orientation_tolerance_rad
            ):
                return SweptPathResult(
                    False, target_index + 1, checked_configurations,
                    maximum_position_error, maximum_orientation_error,
                    target_index, None, "unreachable", (),
                )

            arm_delta = float(np.max(np.abs(solved_arm - previous_arm)))
            finger_delta = float(np.max(np.abs(target_fingers - previous_fingers)))
            substeps = max(
                1,
                int(np.ceil(
                    arm_delta / self.maximum_arm_interpolation_step_rad
                )),
                int(np.ceil(
                    finger_delta / self.maximum_finger_interpolation_step_m
                )),
            )
            for substep in range(1, substeps + 1):
                fraction = substep / substeps
                data.qpos[arm_qpos] = (
                    previous_arm + fraction * (solved_arm - previous_arm)
                )
                data.qpos[finger_qpos] = (
                    previous_fingers
                    + fraction * (target_fingers - previous_fingers)
                )
                data.qvel[:] = 0.0
                mujoco.mj_fwdPosition(self.model, data)
                checked_configurations += 1
                violations = manipulator_contact_violations(
                    self.model, data, collision_schedule[target_index]
                )
                if violations:
                    return SweptPathResult(
                        False, target_index + 1, checked_configurations,
                        maximum_position_error, maximum_orientation_error,
                        target_index, substep, "swept_collision", violations,
                    )
            previous_arm = solved_arm
            previous_fingers = target_fingers

        return SweptPathResult(
            True, len(chunk.actions), checked_configurations,
            maximum_position_error, maximum_orientation_error,
            None, None, "accepted", (),
        )
