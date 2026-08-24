import json
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from action_schema import ACTION_DIM
from g1_contract_trajectory import make_reach_chunk
from g1_dual_arm_ik import LEFT_JOINTS, RIGHT_JOINTS, G1DualArmIK
from g1_mujoco_bridge import policy_action_to_mujoco_world, policy_state_from_mujoco
from g1_policy_contract import (
    ACTION_HORIZON, CONTRACT_ID, CONTRACT_PATH, IMAGE_KEYS,
    OFFICIAL_ACTION_POSTPROCESSING_SOURCE, POLICY_RATE_HZ,
    canonicalize_policy_action_chunk, contract_metadata, preprocess_rgb_image,
    require_verified_contract, validate_action_chunk, validate_observation,
    validate_raw_policy_action_chunk,
)
from stack_scene import CAMERA_NAMES, REFERENCE_EP0_STATE, build_model, reset_to_reference_pose


class G1PolicyContractTests(unittest.TestCase):
    def test_contract_is_single_source_for_scene_and_joint_order(self):
        parsed = json.loads(CONTRACT_PATH.read_text())
        self.assertEqual(parsed["contract_id"], CONTRACT_ID)
        self.assertEqual(ACTION_DIM, parsed["action"]["dimension"])
        self.assertEqual(list(CAMERA_NAMES), [key.split("/")[-1] for key in IMAGE_KEYS])
        self.assertEqual(
            parsed["joint_output"]["ordered_joints"], LEFT_JOINTS + RIGHT_JOINTS
        )
        self.assertEqual(ACTION_HORIZON, 32)
        self.assertEqual(POLICY_RATE_HZ, 15.0)
        self.assertTrue(parsed["timing"]["policy_rate_confirmed"])
        self.assertEqual(
            parsed["timing"]["deployment_receding_horizon"]["control_hz_resolved"],
            15.0,
        )
        self.assertEqual(
            parsed["production_policy"]["model"],
            "LGG100/stack-cube-eef-24k",
        )
        self.assertFalse(parsed["production_policy"]["simulation_execution_allowed"])
        self.assertTrue(
            parsed["production_policy"]["official_action_consumer_postprocessing_verified"]
        )
        self.assertIn("unitree_eef_policy.py", OFFICIAL_ACTION_POSTPROCESSING_SOURCE)
        self.assertFalse(parsed["robot"]["locomotion_controlled_by_policy"])
        self.assertFalse(parsed["robot"]["torso_controlled_by_policy"])
        with self.assertRaises(ValueError):
            require_verified_contract(contract_metadata(verified=False))

    def test_mujoco_state_and_action_share_pelvis_frame(self):
        model = build_model()
        data = mujoco.MjData(model)
        reset_to_reference_pose(model, data)
        state = policy_state_from_mujoco(model, data, REFERENCE_EP0_STATE[14:16])
        world = policy_action_to_mujoco_world(model, data, state)
        ik = G1DualArmIK(model, data)
        left_position, left_quaternion = ik.pose("left")
        right_position, right_quaternion = ik.pose("right")
        np.testing.assert_allclose(world[0:3], left_position, atol=1e-6)
        np.testing.assert_allclose(world[7:10], right_position, atol=1e-6)
        self.assertAlmostEqual(abs(float(np.dot(world[3:7], left_quaternion))), 1.0, places=6)
        self.assertAlmostEqual(abs(float(np.dot(world[10:14], right_quaternion))), 1.0, places=6)

    def test_contract_fixture_is_valid_16d_and_rejects_other_robot_shapes(self):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        chunk = make_reach_chunk(
            np.array([0.2, 0.2, 0.7]), q,
            np.array([0.2, -0.2, 0.7]), q,
            np.array([0.25, 0.2, 0.75]), np.array([0.25, -0.2, 0.75]),
            gripper_state_rad=np.array([5.0, 5.0]),
        )
        self.assertEqual(chunk.actions.shape, (ACTION_HORIZON, ACTION_DIM))
        validate_action_chunk(
            chunk.actions, chunk.timestamps,
            expected_horizon=ACTION_HORIZON,
            expected_rate_hz=POLICY_RATE_HZ,
        )
        with self.assertRaises(ValueError):
            validate_action_chunk(np.zeros((10, 8)))

    def test_official_consumer_normalizes_only_quaternions(self):
        actions = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float64)
        actions[:, 0:3] = [0.2, 0.1, 0.7]
        actions[:, 7:10] = [0.2, -0.1, 0.7]
        actions[:, 6] = 0.993
        actions[:, 13] = 1.007
        actions[:, 14:16] = [5.0, 4.0]
        raw = validate_raw_policy_action_chunk(
            actions, expected_horizon=ACTION_HORIZON
        )
        with self.assertRaisesRegex(ValueError, "quaternion norm error"):
            validate_action_chunk(raw, expected_horizon=ACTION_HORIZON)
        canonical = canonicalize_policy_action_chunk(
            raw, expected_horizon=ACTION_HORIZON
        )
        validate_action_chunk(canonical, expected_horizon=ACTION_HORIZON)
        np.testing.assert_array_equal(canonical[:, 0:3], raw[:, 0:3])
        np.testing.assert_array_equal(canonical[:, 7:10], raw[:, 7:10])
        np.testing.assert_array_equal(canonical[:, 14:16], raw[:, 14:16])
        np.testing.assert_allclose(
            np.linalg.norm(canonical[:, 3:7], axis=1), 1.0, atol=1e-12
        )
        np.testing.assert_allclose(
            np.linalg.norm(canonical[:, 10:14], axis=1), 1.0, atol=1e-12
        )

    def test_official_consumer_rejects_excessive_or_zero_quaternions(self):
        actions = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float64)
        actions[:, 6] = 0.95
        actions[:, 13] = 1.0
        with self.assertRaisesRegex(ValueError, "exceeds official-consumer"):
            canonicalize_policy_action_chunk(actions)
        actions[:, 3:7] = 0.0
        with self.assertRaisesRegex(ValueError, "zero or near zero"):
            canonicalize_policy_action_chunk(actions)

    def test_observation_rejects_camera_or_state_contract_drift(self):
        source = np.full((480, 640, 3), 123, dtype=np.uint8)
        processed = preprocess_rgb_image(source)
        self.assertEqual(processed.shape, (224, 224, 3))
        self.assertEqual(processed.dtype, np.uint8)
        self.assertTrue(np.all(processed[:28] == 0))
        self.assertTrue(np.all(processed[28:196] == 123))
        self.assertTrue(np.all(processed[196:] == 0))

        state = np.zeros(16, dtype=np.float32)
        state[6] = 1.0
        state[13] = 1.0
        state[14:16] = 5.0
        observation = {
            key: np.zeros((224, 224, 3), dtype=np.uint8) for key in IMAGE_KEYS
        }
        observation["observation/state"] = state
        observation["prompt"] = "stack the red, blue, and yellow blocks"
        validate_observation(observation)
        observation[IMAGE_KEYS[0]] = np.zeros((224, 224, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_observation(observation)


if __name__ == "__main__":
    unittest.main()
