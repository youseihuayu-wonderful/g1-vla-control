import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "lgg100_slurm_q0_output_only_20260824.json"


class SlurmQ0ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads(RESULT.read_text())

    def test_scheduler_allocated_one_gpu_and_released_it_early(self):
        scheduler = self.result["scheduler"]
        self.assertEqual(scheduler["requested_gpu_count"], 1)
        self.assertEqual(scheduler["final_state"], "COMPLETED")
        self.assertEqual(scheduler["exit_code"], 0)
        self.assertLess(scheduler["actual_runtime_s"], scheduler["maximum_wall_time_s"])
        self.assertTrue(scheduler["released_early"])

    def test_gpu_was_fully_idle_before_restore(self):
        gpu = self.result["gpu_preflight"]
        self.assertEqual(gpu["model"], "NVIDIA L40S")
        self.assertEqual(gpu["preexisting_compute_process_count"], 0)
        self.assertEqual(gpu["memory_used_mib"], 0)
        self.assertGreater(gpu["memory_free_mib"], 40000)
        self.assertTrue(gpu["slurm_allocation_verified"])

    def test_frozen_checkpoint_inference_passed(self):
        probe = self.result["output_only_probe"]
        self.assertEqual(probe["formal_draws"], 30)
        self.assertEqual(probe["finite_shape_passes"], 30)
        self.assertTrue(probe["finite_shape_qualification_passed"])
        self.assertLess(probe["latency_ms"]["p95"], 100)
        self.assertGreater(probe["peak_memory_used_mib"], 0)
        self.assertFalse(self.result["checkpoint"]["weights_modified"])
        self.assertFalse(self.result["checkpoint"]["neural_training"])

    def test_official_consumer_contract_is_distinct_from_raw_exact_unit_metric(self):
        probe = self.result["output_only_probe"]
        self.assertEqual(probe["raw_quaternion_exact_unit_passes"], 0)
        self.assertEqual(probe["official_consumer_postprocess_passes"], 30)
        self.assertTrue(probe["official_consumer_postprocess_qualification_passed"])
        self.assertFalse(probe["raw_quaternion_exact_unit_qualification_passed"])
        self.assertFalse(probe["raw_exact_unit_required_by_official_consumer"])
        self.assertTrue(self.result["decision"]["g1_contract_verified"])
        self.assertFalse(self.result["decision"]["quaternion_gap_blocks_adaptive_off"])
        self.assertFalse(self.result["decision"]["g1_sim_eligible"])
        self.assertFalse(self.result["decision"]["adaptive_off_closed_loop_authorized_by_this_result"])

    def test_failed_portability_attempts_did_not_begin_gpu_compute(self):
        attempts = self.result["scheduler"]["initial_portability_attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertTrue(all("before_gpu_compute" in item["result"] for item in attempts))
        self.assertIn("non_shared_workspace", self.result["scheduler"]["resolution"])

    def test_cleanup_and_hardware_gates_remain_safe(self):
        self.assertTrue(self.result["cleanup"]["policy_server_stopped"])
        self.assertTrue(self.result["cleanup"]["allocation_released"])
        self.assertFalse(self.result["cleanup"]["other_gpu_process_modified"])
        self.assertFalse(self.result["mujoco_dynamics_executed"])
        self.assertFalse(self.result["robot_command_sent"])
        self.assertFalse(self.result["g1_execution_enabled"])
        self.assertFalse(self.result["hardware_execution_performed"])


if __name__ == "__main__":
    unittest.main()
