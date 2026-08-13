import sys
from pathlib import Path
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from action_schema import EEFActionChunk
from dex1_gripper import Dex1Controller
from g1_mujoco_bridge import policy_state_from_mujoco
from quarantined_chunk_executor import QuarantinedChunkExecutor
from stack_scene import build_model, reset_to_reference_pose


class QuarantinedChunkExecutorTests(unittest.TestCase):
    def build_hold_chunk(self, model, data):
        state = policy_state_from_mujoco(
            model, data, Dex1Controller(model).motor_states(data)
        ).astype(np.float64)
        actions = np.repeat(state[None, :], 50, axis=0)
        return EEFActionChunk(np.arange(50) / 30.0, actions)

    def test_persistent_hold_prefix_stays_finite_and_within_limits(self):
        model = build_model()
        data = mujoco.MjData(model)
        reset_to_reference_pose(model, data)
        chunk = self.build_hold_chunk(model, data)
        executor = QuarantinedChunkExecutor(model, data)
        first = executor.execute_prefix(
            chunk, ["approach"] * 50, prefix_duration_s=0.04
        )
        second = executor.execute_prefix(
            chunk, ["approach"] * 50, prefix_duration_s=0.04
        )
        self.assertTrue(first["accepted"])
        self.assertTrue(second["accepted"])
        self.assertAlmostEqual(second["total_simulated_time_s"], 0.08)
        self.assertLessEqual(second["maxima"]["joint_jerk_ratio"], 1.0 + 1e-12)

    def test_phase_invalid_contact_aborts_and_commands_hold(self):
        model = build_model()
        data = mujoco.MjData(model)
        reset_to_reference_pose(model, data)
        left_site = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "left_eef"
        )
        red_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "red_cube_free"
        )
        qpos = int(model.jnt_qposadr[red_joint])
        data.qpos[qpos:qpos + 3] = data.site_xpos[left_site] + [0.025, 0.0, 0.0]
        data.qpos[qpos + 3:qpos + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(model, data)
        chunk = self.build_hold_chunk(model, data)
        executor = QuarantinedChunkExecutor(model, data)
        result = executor.execute_prefix(
            chunk, ["free_space"] * 50, prefix_duration_s=0.04
        )
        self.assertFalse(result["accepted"])
        self.assertTrue(result["hold_commanded"])
        self.assertIn("phase_invalid_contact", result["reasons"])
        self.assertTrue(result["first_phase_invalid_contact_event"])
        self.assertLess(result["executed_steps"], 20)


if __name__ == "__main__":
    unittest.main()
