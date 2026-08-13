import sys
from pathlib import Path
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptive_speed_context import decide_speed
from dex1_gripper import Dex1Controller
from g1_sim_speed_context import build_simulation_speed_context
from stack_scene import build_model, reset_to_reference_pose


class G1SimulationSpeedContextTests(unittest.TestCase):
    def setUp(self):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        reset_to_reference_pose(self.model, self.data)
        self.measured = Dex1Controller(self.model).motor_states(self.data)

    def context(self, commanded=None, **changes):
        values = {
            "commanded_grippers_rad": self.measured if commanded is None else commanded,
            "measured_grippers_rad": self.measured,
            "eef_tracking_error_m": 0.005,
            "observation_age_ms": 20.0,
            "policy_response_age_ms": 90.0,
            "preflight_passed": True,
            "collision_free": True,
            "command_limits_passed": True,
        }
        values.update(changes)
        return build_simulation_speed_context(
            self.model, self.data, **values
        )

    def test_reference_scene_is_near_objects_and_cannot_accelerate(self):
        context, evidence = self.context()
        self.assertEqual(evidence.task_phase, "approach")
        self.assertLess(evidence.minimum_dex_cube_clearance_m, 0.06)
        decision = decide_speed(context)
        self.assertFalse(decision.hold)
        self.assertFalse(decision.acceleration_allowed)
        self.assertLessEqual(decision.target_scale, 0.5)

    def test_moving_objects_far_produces_auditable_free_space_context(self):
        for color, y in (("red", 1.0), ("blue", 1.3), ("yellow", 1.6)):
            body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{color}_cube"
            )
            joint = int(self.model.body_jntadr[body])
            qpos = int(self.model.jnt_qposadr[joint])
            self.data.qpos[qpos:qpos + 3] = [1.5, y, 0.83]
        mujoco.mj_forward(self.model, self.data)
        context, evidence = self.context()
        self.assertEqual(evidence.task_phase, "free_space")
        self.assertGreater(evidence.minimum_dex_cube_clearance_m, 0.10)
        decision = decide_speed(context)
        self.assertFalse(decision.hold)
        self.assertTrue(decision.acceleration_allowed)
        self.assertGreater(decision.target_scale, 1.0)

    def test_closing_command_forces_grasp_and_slow_speed(self):
        commanded = self.measured - 0.2
        context, evidence = self.context(commanded=commanded)
        self.assertEqual(evidence.task_phase, "grasp")
        decision = decide_speed(context)
        self.assertFalse(decision.hold)
        self.assertLessEqual(decision.target_scale, 0.5)

    def test_excessive_gripper_tracking_error_holds(self):
        commanded = self.measured - 0.5
        context, evidence = self.context(commanded=commanded)
        self.assertEqual(evidence.task_phase, "grasp")
        decision = decide_speed(context)
        self.assertTrue(decision.hold)
        self.assertIn("gripper_error_above_hard_maximum", decision.reasons)

    def test_future_closing_intent_is_not_current_tracking_error(self):
        commanded = self.measured - 0.8
        context, evidence = self.context(
            commanded=commanded, gripper_tracking_error_rad=0.0
        )
        self.assertEqual(evidence.task_phase, "grasp")
        decision = decide_speed(context)
        self.assertFalse(decision.hold)
        self.assertLessEqual(decision.target_scale, 0.5)

    def test_failed_preflight_or_stale_context_holds(self):
        context, _ = self.context(preflight_passed=False)
        self.assertTrue(decide_speed(context).hold)
        context, _ = self.context(observation_age_ms=101.0)
        self.assertTrue(decide_speed(context).hold)


if __name__ == "__main__":
    unittest.main()
