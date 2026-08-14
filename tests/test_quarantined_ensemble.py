import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_quarantined_ensemble import build_ensemble, quaternion_ensemble
from g1_policy_contract import ACTION_HORIZON


class QuarantinedEnsembleTests(unittest.TestCase):
    def test_quaternion_mean_is_sign_invariant_and_unit_length(self):
        q = np.array([0.1, -0.2, 0.3, 0.9])
        q /= np.linalg.norm(q)
        values = np.stack((q, -q, q))[..., None, :]
        result = quaternion_ensemble(values)
        np.testing.assert_allclose(result[0], q, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(result, axis=-1), 1.0)

    def test_chunk_ensemble_preserves_shape_and_unit_quaternions(self):
        base = np.zeros((3, ACTION_HORIZON, 16), dtype=np.float64)
        base[:, :, 3] = 1.0
        base[:, :, 10] = 1.0
        base[0, :, 0] = 0.1
        base[1, :, 0] = 0.2
        base[2, :, 0] = 0.3
        result = build_ensemble(base)
        self.assertEqual(result.shape, (ACTION_HORIZON, 16))
        np.testing.assert_allclose(result[:, 0], 0.2)
        np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=1), 1.0)
        np.testing.assert_allclose(np.linalg.norm(result[:, 10:14], axis=1), 1.0)

    def test_fewer_than_three_draws_is_rejected(self):
        draws = np.zeros((2, ACTION_HORIZON, 16))
        draws[:, :, 3] = 1.0
        draws[:, :, 10] = 1.0
        with self.assertRaises(ValueError):
            build_ensemble(draws)


if __name__ == "__main__":
    unittest.main()
