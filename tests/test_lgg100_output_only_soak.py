import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from g1_policy_contract import ACTION_HORIZON, CONTRACT_ID, CONTRACT_SHA256, CONTRACT_VERSION
from lgg100_output_only_soak import _load_observation, _quantiles


ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "lgg100_output_only_soak.py"
JOB = ROOT / "run_lgg100_slurm_q05_soak.sh"


class OutputOnlySoakTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_source = CLIENT.read_text()
        cls.job_source = JOB.read_text()

    def test_quantiles_include_tail_and_max(self):
        result = _quantiles([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(result["max"], 4.0)
        self.assertGreater(result["p99"], result["p95"])
        self.assertGreater(result["p95"], result["p50"])

    def test_observation_loader_enforces_contract_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observation.npz"
            np.savez_compressed(
                path,
                cam_left_high=np.zeros((2, 2, 3), dtype=np.uint8),
                cam_left_wrist=np.zeros((2, 2, 3), dtype=np.uint8),
                cam_right_wrist=np.zeros((2, 2, 3), dtype=np.uint8),
                state=np.zeros(16),
                prompt=np.asarray("test"),
                g1_policy_contract_id=np.asarray(CONTRACT_ID),
                g1_policy_contract_version=np.asarray(CONTRACT_VERSION),
                g1_policy_contract_sha256=np.asarray(CONTRACT_SHA256),
                action_horizon=np.asarray(ACTION_HORIZON),
            )
            loaded = _load_observation(path)
            self.assertEqual(loaded["observation/state"].shape, (16,))
            with np.load(path) as payload:
                altered = {key: payload[key] for key in payload.files}
            altered["g1_policy_contract_sha256"] = np.asarray("wrong")
            np.savez_compressed(path, **altered)
            with self.assertRaises(ValueError):
                _load_observation(path)

    def test_client_is_four_hour_multi_scenario_output_only(self):
        self.assertIn("default=14_400.0", self.client_source)
        self.assertIn('"scenario": scenario["name"]', self.client_source)
        self.assertIn('"raw_contract_passed": audit.raw_contract_passed', self.client_source)
        self.assertIn('"mujoco_dynamics_executed": False', self.client_source)
        self.assertIn('"hardware_execution_performed": False', self.client_source)
        self.assertNotIn("adaptive_retimer", self.client_source)

    def test_client_keeps_sparse_quarantined_samples_and_full_audit_records(self):
        self.assertIn("inference_records.jsonl.gz", self.client_source)
        self.assertIn("sampled_actions_quarantined.npz", self.client_source)
        self.assertIn("quarantined=np.asarray(True)", self.client_source)
        self.assertIn("unique_raw_chunk_hashes", self.client_source)

    def test_job_requests_exactly_one_l40s_for_four_hours_of_real_work(self):
        self.assertIn("#SBATCH --gres=gpu:l40s:1", self.job_source)
        self.assertIn("#SBATCH --time=04:20:00", self.job_source)
        self.assertIn("TARGET_DURATION_S=14400", self.job_source)
        self.assertIn("TARGET_RATE_HZ=5", self.job_source)
        self.assertNotIn("sleep 14400", self.job_source)

    def test_job_rotates_near_mixed_and_far(self):
        for scenario in ("near", "mixed", "far"):
            self.assertIn(f"--scenario {scenario}", self.job_source)
        self.assertIn("--sample-every 300", self.job_source)

    def test_job_rejects_unallocated_or_shared_gpu(self):
        self.assertIn("exactly_one_slurm_gpu_required", self.job_source)
        self.assertIn("allocated_gpu_has_preexisting_compute_process", self.job_source)
        self.assertIn("foreign_compute_process_appeared", self.job_source)
        self.assertIn("GPU_FREE_MIB >= 40000", self.job_source)

    def test_job_has_no_fixed_host_gpu_index_or_hardware_adapter(self):
        for forbidden in (
            "GPU_INDEX=", "CUDA_VISIBLE_DEVICES=5", "ChannelPublisher", "LowCmd",
            "MotionSwitcher", "g1_execution_enabled=true",
        ):
            self.assertNotIn(forbidden, self.job_source)
        self.assertIn("--host 127.0.0.1", self.job_source)

    def test_job_fails_before_side_effects_outside_slurm(self):
        env = dict(os.environ)
        env.pop("SLURM_JOB_ID", None)
        result = subprocess.run(
            [str(JOB)], cwd=ROOT, env=env, text=True, capture_output=True, timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must run inside a Slurm allocation", result.stderr)

    def test_both_entrypoints_are_syntactically_valid(self):
        subprocess.run(["bash", "-n", str(JOB)], check=True)
        subprocess.run(["python3", "-m", "py_compile", str(CLIENT)], check=True)


if __name__ == "__main__":
    unittest.main()
