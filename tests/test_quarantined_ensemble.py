import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_quarantined_ensemble import build_ensemble, quaternion_ensemble
from g1_policy_contract import (
    ACTION_HORIZON,
    CONTRACT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
)
from lgg100_quarantined_preflight import _load_chunks


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

    def test_preflight_chunk_loader_requires_current_contract_binding(self):
        actions = np.zeros((1, ACTION_HORIZON, 16), dtype=np.float64)
        actions[:, :, 6] = 1.0
        actions[:, :, 13] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.npz"
            np.savez_compressed(
                path,
                actions=actions,
                g1_policy_contract_id=np.asarray(CONTRACT_ID),
                g1_policy_contract_version=np.asarray(CONTRACT_VERSION),
                g1_policy_contract_sha256=np.asarray(CONTRACT_SHA256),
                action_horizon=np.asarray(ACTION_HORIZON),
            )
            np.testing.assert_array_equal(_load_chunks(path), actions)
            np.savez_compressed(
                path,
                actions=actions,
                g1_policy_contract_id=np.asarray(CONTRACT_ID),
                g1_policy_contract_version=np.asarray(CONTRACT_VERSION),
                g1_policy_contract_sha256=np.asarray("wrong"),
                action_horizon=np.asarray(ACTION_HORIZON),
            )
            with self.assertRaises(ValueError):
                _load_chunks(path)


if __name__ == "__main__":
    unittest.main()
