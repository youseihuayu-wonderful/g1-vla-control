import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_policy_contract import (
    ACTION_HORIZON,
    CONTRACT_ID,
    CONTRACT_SHA256,
    CONTRACT_VERSION,
)
from phase_speed_sweep_validation import (
    NOMINAL_CHUNK_DURATION_S, validate_scenario,
)


FALSE_FLAGS = {
    "g1_contract_verified": False,
    "g1_sim_eligible": False,
    "g1_execution_enabled": False,
}
CONTRACT_BINDING = {
    "g1_policy_contract_id": CONTRACT_ID,
    "g1_policy_contract_version": CONTRACT_VERSION,
    "g1_policy_contract_sha256": CONTRACT_SHA256,
    "action_horizon": ACTION_HORIZON,
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PhaseSpeedSweepValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def scenario(self, role, offset, clearance, phases, scales, duration):
        artifact = self.root / f"{role}.npz"
        artifact.write_bytes((role + "-ensemble").encode())
        ensemble_path = self.root / f"{role}-ensemble.json"
        ensemble_path.write_text(json.dumps({
            "output": str(artifact),
            "output_sha256": sha256(artifact),
            "quarantined": True,
            **CONTRACT_BINDING,
            **FALSE_FLAGS,
        }))
        preflight_path = self.root / f"{role}-preflight.json"
        preflight_path.write_text(json.dumps({
            "cube_translation_m": [offset, 0.0, 0.0],
            "source_chunks_sha256": sha256(artifact),
            "observation_binding": {
                "accepted": True,
                "maximum_state_error": 0.0,
                "cube_translation_matches": True,
                "contract_matches": True,
            },
            "summary": {
                "bounded_analysis_chunks": 1,
                "inferred_schedule_swept_paths_accepted": 1,
            },
            **CONTRACT_BINDING,
            **FALSE_FLAGS,
        }))
        branch = {
            "hard_command_limits_pass": True,
            "finite": True,
            "aborted_on_phase_aware_contact": False,
            "phase_aware_contact_step_rate": 0.0,
        }
        dynamics_path = self.root / f"{role}-dynamics.json"
        dynamics_path.write_text(json.dumps({
            "cube_translation_m": [offset, 0.0, 0.0],
            "source_chunks_sha256": sha256(artifact),
            "preflight_report_sha256": sha256(preflight_path),
            "task_success_claimed": False,
            "summary": {
                "executed_diagnostic_chunks": 1,
                "candidate_passes": 1,
                "all_executed_candidates_passed": True,
            },
            "records": [{
                "execution_performed": True,
                "candidate_passed": True,
                "context": {
                    "minimum_clearance_m": clearance,
                    "phase_counts": phases,
                },
                "retiming": {
                    "scale_range": scales,
                    "path_actions_byte_identical": True,
                    "metrics": {"duration_s": duration},
                },
                "baseline": branch,
                "guarded": branch,
            }],
            **CONTRACT_BINDING,
            **FALSE_FLAGS,
        }))
        return validate_scenario(
            role, offset, ensemble_path, preflight_path, dynamics_path
        )

    def test_expected_near_mixed_and_far_behaviors_pass(self):
        near = self.scenario(
            "near", 0.08, 0.05, {"approach": 20, "grasp": 12},
            [0.5, 0.5], NOMINAL_CHUNK_DURATION_S + 0.01,
        )
        mixed = self.scenario(
            "mixed", 0.16, 0.12, {"free_space": 24, "grasp": 8},
            [0.4, 1.1], 2.05,
        )
        far = self.scenario(
            "far", 0.24, 0.20, {"free_space": ACTION_HORIZON},
            [0.66, 1.05], NOMINAL_CHUNK_DURATION_S - 0.01,
        )
        self.assertTrue(near["accepted"])
        self.assertTrue(mixed["accepted"])
        self.assertTrue(far["accepted"])

    def test_missing_observation_binding_fails_closed(self):
        record = self.scenario(
            "near", 0.08, 0.05, {"approach": ACTION_HORIZON},
            [0.5, 0.5], NOMINAL_CHUNK_DURATION_S + 0.01,
        )
        self.assertTrue(record["accepted"])
        preflight_path = self.root / "near-preflight.json"
        payload = json.loads(preflight_path.read_text())
        del payload["observation_binding"]
        preflight_path.write_text(json.dumps(payload))
        dynamics_path = self.root / "near-dynamics.json"
        dynamics = json.loads(dynamics_path.read_text())
        dynamics["preflight_report_sha256"] = sha256(preflight_path)
        dynamics_path.write_text(json.dumps(dynamics))
        record = validate_scenario(
            "near", 0.08, self.root / "near-ensemble.json",
            preflight_path, dynamics_path,
        )
        self.assertFalse(record["accepted"])
        self.assertIn(
            "preflight_observation_binding_missing_or_failed", record["reasons"]
        )

    def test_far_scenario_must_reduce_chunk_duration(self):
        far = self.scenario(
            "far", 0.24, 0.20, {"free_space": ACTION_HORIZON},
            [0.66, 1.05], NOMINAL_CHUNK_DURATION_S + 0.01,
        )
        self.assertFalse(far["accepted"])
        self.assertIn("far_chunk_not_faster_than_nominal", far["reasons"])

    def test_scene_mismatch_fails_closed(self):
        record = self.scenario(
            "near", 0.09, 0.05, {"approach": ACTION_HORIZON},
            [0.5, 0.5], NOMINAL_CHUNK_DURATION_S + 0.01,
        )
        self.assertTrue(record["accepted"])
        ensemble_path = self.root / "near-ensemble.json"
        preflight_path = self.root / "near-preflight.json"
        dynamics_path = self.root / "near-dynamics.json"
        record = validate_scenario(
            "near", 0.08, ensemble_path, preflight_path, dynamics_path
        )
        self.assertFalse(record["accepted"])
        self.assertIn("preflight_scene_translation_mismatch", record["reasons"])
        self.assertIn("dynamics_scene_translation_mismatch", record["reasons"])

    def test_contract_hash_mismatch_fails_closed(self):
        record = self.scenario(
            "near", 0.08, 0.05, {"approach": ACTION_HORIZON},
            [0.5, 0.5], NOMINAL_CHUNK_DURATION_S + 0.01,
        )
        self.assertTrue(record["accepted"])
        ensemble_path = self.root / "near-ensemble.json"
        payload = json.loads(ensemble_path.read_text())
        payload["g1_policy_contract_sha256"] = "wrong"
        ensemble_path.write_text(json.dumps(payload))
        record = validate_scenario(
            "near", 0.08, ensemble_path,
            self.root / "near-preflight.json",
            self.root / "near-dynamics.json",
        )
        self.assertFalse(record["accepted"])
        self.assertIn("ensemble_contract_binding_mismatch", record["reasons"])


if __name__ == "__main__":
    unittest.main()
