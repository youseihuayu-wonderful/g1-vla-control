import sys
from pathlib import Path
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dex1_gripper import Dex1Controller
from g1_contract_trajectory import make_reach_chunk
from g1_mujoco_bridge import policy_state_from_mujoco
from retiming_safety_validation import _run_scale
from stack_scene import build_model, reset_to_reference_pose


class DynamicContactAbortTests(unittest.TestCase):
    def test_free_space_contact_aborts_shadow_rollout(self):
        model = build_model()
        data = mujoco.MjData(model)
        reset_to_reference_pose(model, data)
        grippers = Dex1Controller(model).motor_states(data)
        state = policy_state_from_mujoco(model, data, grippers).astype(np.float64)
        pelvis = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )
        blue = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "blue_cube"
        )
        target = data.xmat[pelvis].reshape(3, 3).T @ (
            data.xpos[blue] - data.xpos[pelvis]
        )
        chunk = make_reach_chunk(
            state[0:3], state[3:7], state[7:10], state[10:14],
            target, state[7:10], gripper_state_rad=grippers,
        )
        result = _run_scale(
            chunk,
            grippers,
            scale=1.0,
            use_filter=True,
            use_joint_filter=True,
            phase_schedule=["free_space"] * len(chunk.actions),
            abort_on_phase_aware_contact=True,
        )
        self.assertTrue(result["aborted_on_phase_aware_contact"])
        self.assertTrue(result["first_phase_aware_contact_event"])
        self.assertLess(
            result["executed_step_count"],
            int(np.ceil((chunk.timestamps[-1] + 2.0) / model.opt.timestep)),
        )


if __name__ == "__main__":
    unittest.main()
