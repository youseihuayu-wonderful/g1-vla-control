import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_policy_contract import ACTION_HORIZON
from neural_action_audit import audit_neural_action_chunk


class NeuralActionAuditTests(unittest.TestCase):
    @staticmethod
    def valid_actions():
        actions = np.zeros((ACTION_HORIZON, 16), dtype=np.float64)
        actions[:, 6] = 1.0
        actions[:, 13] = 1.0
        actions[:, 14:16] = 5.0
        return actions

    def test_exact_unit_chunk_needs_no_normalization(self):
        actions = self.valid_actions()
        audit = audit_neural_action_chunk(actions)
        self.assertTrue(audit.raw_quaternion_exact_unit_passed)
        self.assertTrue(audit.official_consumer_postprocess_passed)
        self.assertFalse(audit.normalization_applied)
        self.assertFalse(audit.simulation_eligible)
        np.testing.assert_array_equal(audit.official_postprocessed_actions, actions)

    def test_small_norm_error_uses_documented_consumer_postprocessing(self):
        actions = self.valid_actions()
        actions[:, 3:7] *= 0.9974
        raw_copy = actions.copy()
        audit = audit_neural_action_chunk(actions)
        self.assertFalse(audit.raw_quaternion_exact_unit_passed)
        self.assertTrue(audit.official_consumer_postprocess_passed)
        self.assertTrue(audit.normalization_applied)
        self.assertFalse(audit.simulation_eligible)
        self.assertIn(
            "official_consumer_quaternion_normalization_applied", audit.reasons
        )
        np.testing.assert_array_equal(audit.raw_actions, raw_copy)
        normalized = audit.official_postprocessed_actions
        np.testing.assert_allclose(np.linalg.norm(normalized[:, 3:7], axis=1), 1.0)
        np.testing.assert_array_equal(normalized[:, :3], raw_copy[:, :3])
        np.testing.assert_array_equal(normalized[:, 7:10], raw_copy[:, 7:10])
        np.testing.assert_array_equal(normalized[:, 14:16], raw_copy[:, 14:16])

    def test_large_error_zero_quaternion_and_bad_gripper_reject(self):
        large = self.valid_actions()
        large[:, 3:7] *= 0.95
        self.assertFalse(
            audit_neural_action_chunk(large).official_consumer_postprocess_passed
        )
        zero = self.valid_actions()
        zero[:, 3:7] = 0.0
        self.assertFalse(
            audit_neural_action_chunk(zero).official_consumer_postprocess_passed
        )
        gripper = self.valid_actions()
        gripper[:, 14] = 6.0
        self.assertFalse(
            audit_neural_action_chunk(gripper).official_consumer_postprocess_passed
        )

    def test_shape_or_nonfinite_rejects(self):
        self.assertFalse(
            audit_neural_action_chunk(
                np.zeros((ACTION_HORIZON, 8))
            ).finite_shape_passed
        )
        actions = self.valid_actions()
        actions[0, 0] = np.nan
        self.assertFalse(audit_neural_action_chunk(actions).finite_shape_passed)


if __name__ == "__main__":
    unittest.main()
