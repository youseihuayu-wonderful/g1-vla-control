import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from action_schema import EEFActionChunk
from adaptive_speed_context import (
    AdaptiveSafetyContext,
    ContextAwareRetimer,
    decide_speed,
)
from g1_contract_trajectory import make_reach_chunk
from safety_governor import MotionEnvelope


class ContextAwareRetimerTests(unittest.TestCase):
    def setUp(self):
        q = np.array([0.0, 0.0, 0.0, 1.0])
        self.chunk = make_reach_chunk(
            np.array([0.20, 0.20, 0.70]), q,
            np.array([0.20, -0.20, 0.70]), q,
            np.array([0.30, 0.20, 0.75]),
            np.array([0.30, -0.20, 0.75]),
            gripper_state_rad=np.array([5.2, 5.2]),
        )

    @staticmethod
    def context(**changes):
        values = {
            "task_phase": "free_space",
            "distance_to_goal_m": 0.20,
            "minimum_clearance_m": 0.20,
            "eef_tracking_error_m": 0.005,
            "observation_age_ms": 20.0,
            "policy_response_age_ms": 80.0,
            "ik_margin_rad": 0.20,
            "joint_limit_margin_rad": 0.20,
            "pelvis_stability": 1.0,
        }
        values.update(changes)
        return AdaptiveSafetyContext(**values)

    def test_safe_free_space_can_accelerate_without_changing_actions(self):
        envelope = MotionEnvelope()
        result = ContextAwareRetimer(envelope=envelope).plan(
            self.chunk, self.context()
        )
        self.assertTrue(result.accepted)
        self.assertFalse(result.hold)
        self.assertTrue(result.path_actions_byte_identical)
        self.assertIsNotNone(result.chunk)
        np.testing.assert_array_equal(result.chunk.actions, self.chunk.actions)
        self.assertGreater(result.scale_profile.max(), 1.0)
        self.assertLessEqual(result.metrics["eef_speed_m_s"], envelope.max_eef_speed_m_s * 1.00001)
        self.assertLessEqual(
            result.metrics["eef_acceleration_m_s2"],
            envelope.max_eef_acceleration_m_s2 * 1.00001,
        )
        self.assertLessEqual(result.metrics["eef_jerk_m_s3"], envelope.max_eef_jerk_m_s3 * 1.00001)

    def test_precision_phases_and_low_clearance_never_accelerate(self):
        for context in (
            self.context(task_phase="grasp"),
            self.context(task_phase="place"),
            self.context(minimum_clearance_m=0.03),
            self.context(contact=True),
        ):
            result = ContextAwareRetimer().plan(self.chunk, context)
            self.assertTrue(result.accepted)
            self.assertLessEqual(result.scale_profile.max(), 0.5 + 1e-12)

    def test_stale_unknown_or_failed_safety_gate_holds(self):
        contexts = (
            self.context(observation_age_ms=101.0),
            self.context(policy_response_age_ms=501.0),
            self.context(task_phase="unknown"),
            self.context(collision_free=False),
            self.context(ik_reachable=False),
            self.context(network_timeout=True),
        )
        for context in contexts:
            decision = decide_speed(context)
            self.assertTrue(decision.hold)
            result = ContextAwareRetimer().plan(self.chunk, context)
            self.assertTrue(result.hold)
            self.assertFalse(result.accepted)
            self.assertIsNone(result.chunk)

    def test_infeasible_discontinuity_is_rejected_not_misreported(self):
        timestamps = np.arange(50, dtype=np.float64) / 30.0
        actions = np.repeat(self.chunk.actions[:1], 50, axis=0)
        actions[25:, 0] += 0.30
        discontinuous = EEFActionChunk(timestamps, actions)
        result = ContextAwareRetimer().plan(discontinuous, self.context())
        self.assertFalse(result.accepted)
        self.assertTrue(result.hold)
        self.assertIn("motion_envelope_infeasible_at_safety_minimum_scale", result.reasons)

    def test_nonfinite_context_holds(self):
        result = ContextAwareRetimer().plan(
            self.chunk, self.context(distance_to_goal_m=np.nan)
        )
        self.assertTrue(result.hold)
        self.assertIn("sample_0:non_finite_context", result.reasons)


if __name__ == "__main__":
    unittest.main()
