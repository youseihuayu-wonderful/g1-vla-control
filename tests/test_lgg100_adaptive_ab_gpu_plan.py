import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_JSON = ROOT / "results" / "lgg100_adaptive_ab_gpu_experiment_plan.json"
PLAN_MD = ROOT / "LGG100_ADAPTIVE_AB_GPU_EXECUTION_PLAN_CN.md"


class AdaptiveABGPUPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(PLAN_JSON.read_text())
        cls.markdown = PLAN_MD.read_text()

    def test_plan_is_inference_only_and_keeps_hardware_closed(self):
        self.assertFalse(self.plan["neural_training"])
        self.assertTrue(self.plan["checkpoint_weights_frozen"])
        self.assertFalse(self.plan["robot_command_allowed"])
        self.assertFalse(self.plan["hardware_execution_performed"])
        self.assertFalse(self.plan["g1_execution_enabled"])

    def test_retimer_can_only_modify_timestamps(self):
        contract = self.plan["adaptive_contract"]
        self.assertFalse(contract["retimer_may_modify_action_samples"])
        self.assertTrue(contract["single_chunk_byte_identity_required"])
        self.assertIn("may diverge", contract["closed_loop_identity_boundary"])

    def test_gpu_request_is_bounded_and_scheduler_managed(self):
        gpu = self.plan["gpu_plan"]
        self.assertEqual(gpu["qualification"]["gpu_count"], 1)
        self.assertEqual(gpu["pilot"]["gpu_count"], 1)
        self.assertLessEqual(gpu["formal"]["maximum_parallel_gpu_count"], 2)
        self.assertEqual(gpu["formal"]["gpu_per_array_task"], 1)
        self.assertFalse(gpu["dummy_gpu_occupancy_allowed"])
        self.assertTrue(gpu["scheduler_required"])
        self.assertTrue(gpu["release_early_on_completion_or_failure"])

    def test_every_stage_is_fail_closed_and_ordered(self):
        stages = self.plan["stages"]
        self.assertEqual([stage["id"] for stage in stages], [f"G{i}" for i in range(8)])
        self.assertTrue(all(stage["fail_closed"] for stage in stages))
        self.assertEqual(stages[0]["permission_after_pass"], "submit_Q0")
        self.assertEqual(stages[-1]["permission_after_pass"], "simulation_result_only")

    def test_plan_separates_gpu_inference_from_local_simulation(self):
        split = self.plan["compute_split"]
        self.assertIn("output_only_vla_inference", split["gpu"])
        self.assertIn("mujoco_observation_and_dynamics", split["local"])
        self.assertIn("adaptive_timestamp_retiming", split["local"])
        self.assertNotIn("neural_training", split["gpu"])

    def test_formal_ab_and_react_result_contract_are_explicit(self):
        self.assertGreaterEqual(self.plan["formal_matrix"]["minimum_paired_seeds_per_scenario"], 30)
        self.assertTrue(self.plan["formal_matrix"]["all_failures_retained"])
        app = self.plan["result_app"]
        self.assertEqual(app["framework"], "React_TypeScript_static_build")
        self.assertTrue(app["public_redaction_required"])
        self.assertFalse(app["direct_internal_api_access"])
        self.assertIn("trial_explorer", app["sections"])

    def test_markdown_rejects_training_and_gpu_squatting(self):
        self.assertIn("这不是神经网络训练任务", self.markdown)
        self.assertIn("不使用 cron 抢卡", self.markdown)
        self.assertIn("不训练或更新 VLA 权重", self.markdown)
        self.assertIn("下一次 GPU 作业：只有完成前置 Gate 后才允许提交 **Q1 Adaptive-OFF Pilot**", self.markdown)


if __name__ == "__main__":
    unittest.main()
