import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_lgg100_slurm_q0.sh"


class SlurmQ0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text()

    def test_script_is_executable_and_valid_bash(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK))
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    def test_q0_requests_one_l40s_with_bounded_time(self):
        self.assertIn("#SBATCH --gres=gpu:l40s:1", self.source)
        self.assertIn("#SBATCH --time=01:00:00", self.source)
        self.assertNotIn("#SBATCH --gres=gpu:l40s:2", self.source)
        self.assertIn("exactly_one_slurm_gpu_required", self.source)

    def test_q0_has_no_fixed_host_or_physical_gpu_index(self):
        self.assertNotIn("hostname)", self.source)
        self.assertNotIn("GPU_INDEX=", self.source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=5", self.source)
        self.assertIn('GPU_TOKEN="${CUDA_VISIBLE_DEVICES:-}"', self.source)

    def test_q0_rejects_preexisting_or_foreign_compute_processes(self):
        self.assertIn("allocated_gpu_has_preexisting_compute_process", self.source)
        self.assertIn("foreign_compute_process_appeared", self.source)
        self.assertIn("allocated_gpu_not_fully_available", self.source)
        self.assertIn("GPU_FREE_MIB >= 40000", self.source)

    def test_q0_is_output_only_and_loopback_bound(self):
        self.assertIn("--host 127.0.0.1", self.source)
        self.assertIn("--draws 30", self.source)
        self.assertIn('"neural_training": False', self.source)
        self.assertIn('"mujoco_dynamics_executed": False', self.source)
        self.assertIn('"hardware_execution_performed": False', self.source)
        for forbidden in ("ChannelPublisher", "LowCmd", "MotionSwitcher", "g1_execution_enabled=true"):
            self.assertNotIn(forbidden, self.source)

    def test_q0_fails_before_side_effects_outside_slurm(self):
        env = dict(os.environ)
        env.pop("SLURM_JOB_ID", None)
        result = subprocess.run(
            [str(SCRIPT)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must run inside a Slurm allocation", result.stderr)

    def test_q0_has_verified_cleanup_and_atomic_status(self):
        self.assertIn("Refusing to stop unverified process", self.source)
        self.assertIn("target.with_suffix(target.suffix + \".tmp\")", self.source)
        self.assertIn("temp.replace(target)", self.source)
        self.assertIn("#SBATCH --signal=B:TERM@120", self.source)


if __name__ == "__main__":
    unittest.main()
