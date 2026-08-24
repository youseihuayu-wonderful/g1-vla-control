import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_policy_contract import ACTION_HORIZON, CONTRACT_ID, CONTRACT_SHA256, POLICY_RATE_HZ
from lgg100_adaptive_ab import HF_REVISION, _load_verified_chunk
from lgg100_candidate_server import G1CandidateInputs, G1CandidateOutputs
from lgg100_sim_smoke import build_sim_observation


class LGG100CandidateContractTests(unittest.TestCase):
    def test_candidate_input_output_contract_is_strict(self):
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        state = np.zeros(16, dtype=np.float32)
        state[6] = 1.0
        state[13] = 1.0
        mapped = G1CandidateInputs()({
            "observation/cam_left_high": image,
            "observation/cam_left_wrist": image,
            "observation/cam_right_wrist": image,
            "observation/state": state,
            "prompt": "stack blocks",
        })
        self.assertEqual(set(mapped["image"]), {
            "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb",
        })
        self.assertEqual(mapped["state"].shape, (16,))
        actions = np.zeros((ACTION_HORIZON, 32), dtype=np.float32)
        self.assertEqual(
            G1CandidateOutputs()({"actions": actions})["actions"].shape,
            (ACTION_HORIZON, 16),
        )
        with self.assertRaises(ValueError):
            G1CandidateOutputs()({"actions": np.zeros((ACTION_HORIZON, 8))})

    def test_sim_observation_contains_real_three_camera_contract(self):
        observation, evidence = build_sim_observation()
        self.assertEqual(observation["observation/state"].shape, (16,))
        self.assertTrue(evidence["state_finite"])
        for key in (
            "observation/cam_left_high",
            "observation/cam_left_wrist",
            "observation/cam_right_wrist",
        ):
            self.assertEqual(observation[key].shape, (224, 224, 3))
            self.assertEqual(observation[key].dtype, np.uint8)

    def test_adaptive_ab_requires_passing_fingerprinted_neural_artifact(self):
        actions = np.zeros((ACTION_HORIZON, 16), dtype=np.float64)
        actions[:, 6] = 1.0
        actions[:, 13] = 1.0
        timestamps = np.arange(ACTION_HORIZON) / POLICY_RATE_HZ
        digest = hashlib.sha256(actions.tobytes()).hexdigest()
        report = {
            "summary": {"passed": True},
            "neural_vla_claimed": True,
            "g1_execution_enabled": False,
            "g1_sim_eligible": True,
            "g1_contract_verified": True,
            "g1_policy_contract_id": CONTRACT_ID,
            "g1_policy_contract_sha256": CONTRACT_SHA256,
            "checkpoint": {"revision": HF_REVISION},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "report.json"
            chunk_path = root / "chunk.npz"
            report_path.write_text(json.dumps(report))
            np.savez_compressed(
                chunk_path,
                actions=actions,
                timestamps=timestamps,
                action_sha256=np.asarray(digest),
                hf_revision=np.asarray(HF_REVISION),
                g1_policy_contract_id=np.asarray(CONTRACT_ID),
                g1_policy_contract_sha256=np.asarray(CONTRACT_SHA256),
                g1_sim_eligible=np.asarray(True),
            )
            chunk, actual_digest, norm_error = _load_verified_chunk(
                chunk_path, report_path, 0.001
            )
            self.assertEqual(chunk.actions.shape, (ACTION_HORIZON, 16))
            self.assertEqual(actual_digest, digest)
            self.assertAlmostEqual(norm_error, 0.0)

            np.savez_compressed(
                chunk_path,
                actions=actions,
                timestamps=timestamps,
                action_sha256=np.asarray("wrong"),
                hf_revision=np.asarray(HF_REVISION),
                g1_policy_contract_id=np.asarray(CONTRACT_ID),
                g1_policy_contract_sha256=np.asarray(CONTRACT_SHA256),
                g1_sim_eligible=np.asarray(True),
            )
            with self.assertRaises(ValueError):
                _load_verified_chunk(chunk_path, report_path, 0.001)


if __name__ == "__main__":
    unittest.main()
