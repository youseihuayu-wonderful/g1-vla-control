import sys
from pathlib import Path
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from action_schema import EEFActionChunk
from dex1_gripper import Dex1Controller
from g1_mujoco_bridge import policy_state_from_mujoco
from stack_scene import build_model, reset_to_reference_pose
from swept_path_preflight import G1SweptPathPreflight


class SweptPathPreflightTests(unittest.TestCase):
    def setUp(self):
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        reset_to_reference_pose(self.model, self.data)
        self.state = policy_state_from_mujoco(
            self.model,
            self.data,
            Dex1Controller(self.model).motor_states(self.data),
        ).astype(np.float64)
        self.gate = G1SweptPathPreflight(self.model)

    def test_hold_path_passes(self):
        actions = np.repeat(self.state[None, :], 3, axis=0)
        chunk = EEFActionChunk(np.arange(3) / 30.0, actions)
        result = self.gate.check(self.data, chunk, phase="free_space")
        self.assertTrue(result.accepted, result)

    def test_path_into_cube_is_rejected_before_execution(self):
        pelvis = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        blue = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "blue_cube"
        )
        target_world = self.data.xpos[blue].copy()
        target_pelvis = self.data.xmat[pelvis].reshape(3, 3).T @ (
            target_world - self.data.xpos[pelvis]
        )
        count = 20
        actions = np.repeat(self.state[None, :], count, axis=0)
        progress = np.linspace(0.0, 1.0, count)
        actions[:, 0:3] = (
            self.state[None, 0:3] * (1.0 - progress[:, None])
            + target_pelvis[None, :] * progress[:, None]
        )
        chunk = EEFActionChunk(np.arange(count) / 30.0, actions)
        result = self.gate.check(self.data, chunk, phase="free_space")
        self.assertFalse(result.accepted)
        self.assertIn(result.reason, {"swept_collision", "unreachable"})


if __name__ == "__main__":
    unittest.main()
