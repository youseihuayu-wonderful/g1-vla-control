import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from action_schema import (
    EEFActionChunk, LEFT_QUAT, RIGHT_QUAT, slerp,
    mujoco_wxyz_to_vla_xyzw, vla_xyzw_to_mujoco_wxyz,
    pelvis_vla_action_to_world_mujoco,
)
from adaptive_retimer import AdaptiveChunkExecutor, AdaptiveRetimer
from g1_contract_trajectory import make_reach_chunk


class ActionSchemaTests(unittest.TestCase):
    def test_pelvis_frame_action_to_world_boundary(self):
        action = np.zeros(16)
        action[0:3] = [0.2, 0.1, 0.3]
        action[3:7] = [0.0, 0.0, 0.0, 1.0]
        action[7:10] = [0.2, -0.1, 0.3]
        action[10:14] = [0.0, 0.0, 0.0, 1.0]
        world = pelvis_vla_action_to_world_mujoco(
            action,
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(world[0:3], [1.2, 2.1, 3.3])
        np.testing.assert_allclose(world[7:10], [1.2, 1.9, 3.3])
        np.testing.assert_allclose(world[3:7], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(world[10:14], [1.0, 0.0, 0.0, 0.0])

    def test_slerp_is_normalized_and_shortest_path(self):
        q = slerp(np.array([0.0, 0, 0, 1.0]), np.array([0, -0.707, 0, -0.707]), 0.5)
        self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=8)
        self.assertGreater(q[3], 0.0)

    def test_mujoco_quaternion_boundary_round_trip(self):
        wxyz = np.array([0.9, 0.1, -0.2, 0.3])
        np.testing.assert_allclose(
            vla_xyzw_to_mujoco_wxyz(mujoco_wxyz_to_vla_xyzw(wxyz)), wxyz
        )

    def test_chunk_rejects_bad_dimension(self):
        with self.assertRaises(ValueError):
            EEFActionChunk(np.array([0.0, 1.0]), np.zeros((2, 15)))

    def test_repeated_chunk_construction_preserves_canonical_actions_bytewise(self):
        timestamps = np.arange(3, dtype=np.float64) / 30.0
        actions = np.zeros((3, 16), dtype=np.float64)
        actions[:, 3:7] = np.array([0.1, -0.2, 0.3, 0.9])
        actions[:, 10:14] = np.array([-0.2, 0.3, 0.1, 0.9])
        first = EEFActionChunk(timestamps, actions)
        second = EEFActionChunk(first.timestamps, first.actions)
        np.testing.assert_array_equal(second.actions, first.actions)


class RetimerTests(unittest.TestCase):
    def setUp(self):
        self.left_start = np.array([0.0, 0.2, 0.7])
        self.right_start = np.array([0.0, -0.2, 0.7])
        self.left_target = np.array([0.32, 0.25, 0.8])
        self.right_target = np.array([0.22, -0.24, 0.76])
        quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        self.chunk = make_reach_chunk(
            self.left_start, quaternion, self.right_start, quaternion,
            self.left_target, self.right_target,
            gripper_state_rad=np.array([5.5, 5.5]),
        )

    def test_offline_retiming_preserves_path_and_quaternions(self):
        result = AdaptiveRetimer().retime(
            self.chunk, self.left_target, self.right_target
        )
        np.testing.assert_allclose(result.chunk.actions, self.chunk.actions)
        np.testing.assert_allclose(
            np.linalg.norm(result.chunk.actions[:, LEFT_QUAT], axis=1), 1.0
        )
        np.testing.assert_allclose(
            np.linalg.norm(result.chunk.actions[:, RIGHT_QUAT], axis=1), 1.0
        )
        self.assertGreater(result.scale_profile.max(), 1.0)
        self.assertLess(result.scale_profile[-1], 1.0)

    def test_online_governor_accelerates_far_and_slows_near(self):
        executor = AdaptiveChunkExecutor()
        executor.reset(self.chunk)
        far_scales = []
        for _ in range(100):
            _, scale, _ = executor.step(
                self.chunk, 0.01, self.left_start, self.right_start,
                self.left_target, self.right_target, stability=1.0,
            )
            far_scales.append(scale)
        self.assertGreater(far_scales[-1], 1.5)
        near_scales = []
        for _ in range(100):
            _, scale, _ = executor.step(
                self.chunk, 0.01, self.left_target, self.right_target,
                self.left_target, self.right_target, stability=1.0,
            )
            near_scales.append(scale)
        self.assertLess(near_scales[-1], 0.6)


if __name__ == "__main__":
    unittest.main()
