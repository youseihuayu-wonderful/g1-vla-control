import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_action_audit import audit_neural_action_chunk


class NeuralActionAuditTests(unittest.TestCase):
    @staticmethod
    def valid_actions():
        actions = np.zeros((50, 16), dtype=np.float64)
        actions[:, 6] = 1.0
        actions[:, 13] = 1.0
        actions[:, 14:16] = 5.0
        return actions

    def test_raw_contract_chunk_needs_no_normalization(self):
        actions = self.valid_actions()
        audit = audit_neural_action_chunk(actions)
        self.assertTrue(audit.raw_contract_passed)
        self.assertTrue(audit.bounded_quaternion_normalization_passed)
        self.assertFalse(audit.normalization_applied)
        self.assertFalse(audit.simulation_eligible)
        np.testing.assert_array_equal(audit.canonicalized_actions_for_analysis, actions)

    def test_small_norm_error_is_explicitly_quarantined(self):
        actions = self.valid_actions()
        actions[:, 3:7] *= 0.9974
        audit = audit_neural_action_chunk(actions)
        self.assertFalse(audit.raw_contract_passed)
        self.assertTrue(audit.bounded_quaternion_normalization_passed)
        self.assertTrue(audit.normalization_applied)
        self.assertFalse(audit.simulation_eligible)
        self.assertIn(
            "quaternion_normalized_for_quarantined_analysis_only", audit.reasons
        )
        normalized = audit.canonicalized_actions_for_analysis
        np.testing.assert_allclose(np.linalg.norm(normalized[:, 3:7], axis=1), 1.0)

    def test_large_error_zero_quaternion_and_bad_gripper_reject(self):
        large = self.valid_actions()
        large[:, 3:7] *= 0.95
        self.assertFalse(
            audit_neural_action_chunk(large).bounded_quaternion_normalization_passed
        )
        zero = self.valid_actions()
        zero[:, 3:7] = 0.0
        self.assertFalse(
            audit_neural_action_chunk(zero).bounded_quaternion_normalization_passed
        )
        gripper = self.valid_actions()
        gripper[:, 14] = 6.0
        self.assertFalse(
            audit_neural_action_chunk(gripper).bounded_quaternion_normalization_passed
        )

    def test_shape_or_nonfinite_rejects(self):
        self.assertFalse(
            audit_neural_action_chunk(np.zeros((50, 8))).finite_shape_passed
        )
        actions = self.valid_actions()
        actions[0, 0] = np.nan
        self.assertFalse(audit_neural_action_chunk(actions).finite_shape_passed)


if __name__ == "__main__":
    unittest.main()
