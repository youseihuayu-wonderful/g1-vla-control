import sys
from pathlib import Path
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgg100_quarantined_closed_loop import _cube_state
from stack_scene import build_model, reset_to_reference_pose


class LGG100QuarantinedClosedLoopTests(unittest.TestCase):
    def test_initial_scene_is_not_misreported_as_stacked(self):
        model = build_model()
        data = mujoco.MjData(model)
        reset_to_reference_pose(model, data)
        state = _cube_state(model, data)
        self.assertFalse(state["task_metrics"]["blue_on_red"])
        self.assertFalse(state["task_metrics"]["yellow_on_blue"])
        self.assertFalse(state["task_metrics"]["full_stack_geometric_success"])

    def test_geometric_stack_requires_both_pairs_and_low_speed(self):
        model = build_model()
        data = mujoco.MjData(model)
        reset_to_reference_pose(model, data)
        red_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "red_cube"
        )
        base = data.xpos[red_body].copy()
        for index, color in enumerate(("red", "blue", "yellow")):
            joint = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{color}_cube_free"
            )
            qpos = int(model.jnt_qposadr[joint])
            data.qpos[qpos:qpos + 3] = base + [0.0, 0.0, 0.08 * index]
            data.qpos[qpos + 3:qpos + 7] = [1.0, 0.0, 0.0, 0.0]
            dof = int(model.jnt_dofadr[joint])
            data.qvel[dof:dof + 3] = 0.0
        mujoco.mj_forward(model, data)
        state = _cube_state(model, data)
        self.assertTrue(state["task_metrics"]["blue_on_red"])
        self.assertTrue(state["task_metrics"]["yellow_on_blue"])
        self.assertTrue(state["task_metrics"]["full_stack_geometric_success"])
        blue_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "blue_cube_free"
        )
        data.qvel[int(model.jnt_dofadr[blue_joint])] = 0.2
        mujoco.mj_forward(model, data)
        self.assertFalse(
            _cube_state(model, data)["task_metrics"]["full_stack_geometric_success"]
        )


if __name__ == "__main__":
    unittest.main()
