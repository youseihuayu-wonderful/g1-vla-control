import sys
from pathlib import Path
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stack_scene import (
    build_model, reset_to_stand, reset_to_reference_pose, translate_cubes,
    CAMERA_NAMES, REFERENCE_EP0_STATE,
)
from dex1_gripper import (
    Dex1Controller, motor_radians_to_jaw_position,
    jaw_position_to_motor_radians, JAW_MIN_M, JAW_MAX_M,
)


class StackSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = build_model()
        cls.data = mujoco.MjData(cls.model)
        reset_to_stand(cls.model, cls.data)

    def test_cameras_and_cubes_exist(self):
        for name in CAMERA_NAMES:
            self.assertGreaterEqual(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name), 0
            )
        for side in ("left", "right"):
            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, f"cam_{side}_wrist"
            )
            parent_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.model.cam_bodyid[camera_id],
            )
            self.assertEqual(parent_name, f"{side}_dex1_base_link")
        for name in ("red_cube", "blue_cube", "yellow_cube"):
            self.assertGreaterEqual(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name), 0
            )

    def test_official_dex1_replaces_articulated_hands(self):
        self.assertEqual(
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "left_hand_thumb_0_link"
            ),
            -1,
        )
        for side in ("left", "right"):
            self.assertGreaterEqual(
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"{side}_dex1_base_link",
                ),
                0,
            )
        controller = Dex1Controller(self.model)
        self.assertEqual(len(controller.actuators["left"]), 2)
        self.assertEqual(len(controller.actuators["right"]), 2)

    def test_both_dex1_grippers_point_along_wrist_forward_axis(self):
        for side in ("left", "right"):
            wrist_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link"
            )
            base_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_dex1_base_link"
            )
            tips = []
            for link in ("Link1_3", "Link2_2"):
                body_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"{side}_dex1_{link}",
                )
                tips.append(self.data.xpos[body_id])
            finger_direction = np.mean(tips, axis=0) - self.data.xpos[base_id]
            wrist_forward = self.data.xmat[wrist_id].reshape(3, 3)[:, 0]
            self.assertGreater(float(finger_direction @ wrist_forward), 0.05)

    def test_eef_sites_are_50mm_along_wrist_forward(self):
        for side in ("left", "right"):
            wrist = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link"
            )
            site = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_eef"
            )
            offset = self.data.site_xpos[site] - self.data.xpos[wrist]
            forward = self.data.xmat[wrist].reshape(3, 3)[:, 0]
            np.testing.assert_allclose(offset, 0.05 * forward, atol=1e-9)

    def test_public_episode_zero_reference_pose(self):
        data = mujoco.MjData(self.model)
        reset_to_reference_pose(self.model, data)
        values = []
        for side in ("left", "right"):
            for joint in (
                "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                "wrist_roll", "wrist_pitch", "wrist_yaw",
            ):
                joint_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{side}_{joint}_joint",
                )
                values.append(data.qpos[self.model.jnt_qposadr[joint_id]])
        np.testing.assert_allclose(values, REFERENCE_EP0_STATE[:14], atol=1e-8)
        np.testing.assert_allclose(
            Dex1Controller(self.model).motor_states(data),
            REFERENCE_EP0_STATE[14:16],
            atol=1e-8,
        )

    def test_dex1_motor_mapping_matches_limits(self):
        self.assertAlmostEqual(motor_radians_to_jaw_position(0.0), JAW_MAX_M)
        self.assertAlmostEqual(motor_radians_to_jaw_position(5.5), JAW_MIN_M)
        self.assertAlmostEqual(motor_radians_to_jaw_position(-10.0), JAW_MAX_M)
        self.assertAlmostEqual(motor_radians_to_jaw_position(10.0), JAW_MIN_M)
        for command in np.linspace(0.0, 5.5, 12):
            self.assertAlmostEqual(
                jaw_position_to_motor_radians(
                    motor_radians_to_jaw_position(command)
                ),
                command,
            )

    def test_explicit_cube_translation_preserves_relative_layout(self):
        data = mujoco.MjData(self.model)
        reset_to_reference_pose(self.model, data)
        before = []
        for color in ("red", "blue", "yellow"):
            body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{color}_cube"
            )
            before.append(data.xpos[body].copy())
        translation = np.array([0.12, -0.03, 0.01])
        translate_cubes(self.model, data, translation)
        after = []
        for color in ("red", "blue", "yellow"):
            body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{color}_cube"
            )
            after.append(data.xpos[body].copy())
        np.testing.assert_allclose(np.asarray(after) - np.asarray(before), translation)

    def test_robot_and_cubes_remain_stable(self):
        reset_to_stand(self.model, self.data)
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        self.assertGreater(self.data.qpos[2], 0.75)
        for name in ("red_cube", "blue_cube", "yellow_cube"):
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            self.assertGreater(self.data.xpos[body, 2], 0.61)


if __name__ == "__main__":
    unittest.main()
