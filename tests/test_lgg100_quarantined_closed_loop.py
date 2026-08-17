import sys
from pathlib import Path
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgg100_quarantined_closed_loop import (
    _cube_state,
    _realtime_watchdog_evidence,
)
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

    def test_watchdog_requires_fresh_execution_for_validation(self):
        evidence = _realtime_watchdog_evidence(
            [{
                "execution_performed": False,
                "observation_age_at_commit_ms": 120.0,
            }],
            paused_step_synchronous_diagnostic=False,
            maximum_observation_age_ms=100.0,
        )
        self.assertFalse(evidence["real_time_watchdog_validated"])
        self.assertTrue(evidence["stale_fail_closed_hold_observed"])
        self.assertFalse(evidence["unsafe_stale_execution_observed"])

    def test_watchdog_accepts_only_fresh_execution(self):
        fresh = _realtime_watchdog_evidence(
            [{
                "execution_performed": True,
                "observation_age_at_commit_ms": 95.0,
            }],
            paused_step_synchronous_diagnostic=False,
            maximum_observation_age_ms=100.0,
        )
        self.assertTrue(fresh["real_time_watchdog_validated"])
        stale = _realtime_watchdog_evidence(
            [{
                "execution_performed": True,
                "observation_age_at_commit_ms": 101.0,
            }],
            paused_step_synchronous_diagnostic=False,
            maximum_observation_age_ms=100.0,
        )
        self.assertFalse(stale["real_time_watchdog_validated"])
        self.assertTrue(stale["unsafe_stale_execution_observed"])

    def test_paused_execution_never_validates_realtime_watchdog(self):
        evidence = _realtime_watchdog_evidence(
            [{
                "execution_performed": True,
                "observation_age_at_commit_ms": 20.0,
            }],
            paused_step_synchronous_diagnostic=True,
            maximum_observation_age_ms=100.0,
        )
        self.assertFalse(evidence["real_time_watchdog_validated"])
        self.assertFalse(evidence["evaluated_in_real_time"])

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
