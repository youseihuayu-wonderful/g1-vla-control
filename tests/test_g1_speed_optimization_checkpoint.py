import ast
import json
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_adaptive_phase_validation import FAR_LIFT_OFFSET_M
from g1_fast_sequential_ik import solve_sequential_ik
from g1_fast_swept_path_preflight import G1FastSweptPathPreflight
from retiming_safety_validation import _run_scale
from run_simulation import build_contract_fixture
from stack_scene import REFERENCE_EP0_STATE, build_model, reset_to_reference_pose


class G1SpeedOptimizationCheckpointTests(unittest.TestCase):
    def test_new_speed_modules_have_no_robot_sdk_or_controller_import(self):
        files = [
            "g1_phase_scale_optimizer.py",
            "g1_fast_sequential_ik.py",
            "g1_fast_swept_path_preflight.py",
            "g1_fast_preflight_benchmark.py",
            "g1_waist_compensation.py",
            "g1_waist_compensation_diagnostic.py",
            "g1_deterministic_speed_paired_ab.py",
            "g1_stable_completion_validation.py",
            "g1_fast_preflight_correctness_corpus.py",
            "g1_yuhao_fast_layered_benchmark.py",
            "g1_offline_compensated_shadow_replay.py",
            "g1_stable_completion_optimizer.py",
            "g1_settling_bottleneck_diagnostic.py",
            "g1_fast_preflight_30_trajectory_corpus.py",
            "g1_fast_preflight_near_limit_fuzz.py",
            "trajectory_analyzer.py",
            "speed_scheduler.py",
        ]
        forbidden = ("unitree_sdk", "arm_controller", "gripper_controller")
        for filename in files:
            tree = ast.parse((ROOT / filename).read_text())
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertFalse(
                any(token in name.lower() for name in imports for token in forbidden),
                filename,
            )

    def test_fast_ik_stops_inside_stricter_internal_tolerance(self):
        model = build_model()
        source = mujoco.MjData(model)
        reset_to_reference_pose(model, source)
        chunk, _, _ = build_contract_fixture(FAR_LIFT_OFFSET_M)
        result = solve_sequential_ik(model, source, chunk.actions)
        self.assertTrue(result.accepted)
        self.assertEqual(len(result.targets), 32)
        self.assertLess(result.total_iterations, 32 * 60)
        self.assertLessEqual(
            max(target.maximum_position_error_m for target in result.targets),
            0.004,
        )
        self.assertLessEqual(
            max(target.maximum_orientation_error_rad for target in result.targets),
            np.deg2rad(2.5),
        )

    def test_fast_swept_preflight_preserves_interpolation_resolution(self):
        model = build_model()
        source = mujoco.MjData(model)
        reset_to_reference_pose(model, source)
        chunk, _, _ = build_contract_fixture(FAR_LIFT_OFFSET_M)
        result = G1FastSweptPathPreflight(model).check(
            source, chunk, phase="free_space"
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.checked_targets, 32)
        self.assertEqual(result.checked_interpolated_configurations, 146)
        self.assertLessEqual(result.maximum_position_error_m, 0.004)
        self.assertLessEqual(result.maximum_orientation_error_rad, np.deg2rad(2.5))

    def test_stable_completion_requires_continuous_unchanged_gate(self):
        chunk, _, _ = build_contract_fixture(FAR_LIFT_OFFSET_M)
        result = _run_scale(
            chunk,
            REFERENCE_EP0_STATE[14:16],
            scale=1.0,
            use_filter=True,
            use_joint_filter=True,
            stop_when_settled=True,
        )
        self.assertTrue(result["completion_gate_enabled"])
        self.assertTrue(result["task_completed"])
        self.assertIsNotNone(result["task_completion_time_s"])
        self.assertEqual(result["completion_position_tolerance_m"], 0.005)
        self.assertAlmostEqual(
            np.rad2deg(result["completion_orientation_tolerance_rad"]), 3.0
        )
        self.assertEqual(result["completion_hold_s"], 0.250)
        self.assertIsNotNone(result["first_completion_tolerance_entry_s"])
        self.assertGreaterEqual(result["maximum_consecutive_completion_hold_s"], 0.250)

    def test_formal_microbenchmark_passes_but_is_not_yuhao_or_task_evidence(self):
        report = json.loads((
            ROOT / "results" / "g1_fast_preflight_paired_benchmark_20260827.json"
        ).read_text())
        self.assertEqual(report["input"]["formal_pairs"], 50)
        self.assertTrue(report["summary"]["benchmark_passed"])
        self.assertLess(report["summary"]["fast_total_ms"]["p95"], 333.334)
        self.assertGreater(
            report["summary"]["paired_median_speedup_95ci"][0], 4.1
        )
        self.assertFalse(report["comparison_contract"]["legacy_is_yuhao_pinocchio"])
        self.assertFalse(report["decision"]["task_level_speedup_passed"])
        self.assertFalse(report["decision"]["robot_motion_allowed"])
        self.assertEqual(len(report["runs"]), 50)

    def test_single_fixture_scale_candidate_is_action_preserving(self):
        report = json.loads((
            ROOT / "results" / "g1_phase_scale_optimizer_20260827.json"
        ).read_text())
        selected = report["selected"]
        self.assertTrue(selected["passed"])
        self.assertGreaterEqual(selected["duration_reduction_fraction"], 0.10)
        self.assertTrue(selected["criteria"]["path_actions_byte_identical"])
        self.assertFalse(report["decision"]["real_lgg100_task_speedup_passed"])
        self.assertFalse(report["decision"]["production_adaptive_enabled"])

    def test_waist_adapter_resolves_numeric_hold_target_not_physical_collision(self):
        report = json.loads((
            ROOT / "results" / "g1_waist_compensation_diagnostic_20260827.json"
        ).read_text())
        self.assertTrue(report["decision"]["waist_transform_numeric_regression_passed"])
        self.assertFalse(report["decision"]["uncompensated_zero_waist_target_passed"])
        self.assertTrue(report["decision"]["compensated_hold_target_passed"])
        self.assertFalse(report["hashes"]["canonical_input_mutated"])
        self.assertFalse(report["collision"]["full_body_state_available"])
        self.assertFalse(report["decision"]["initial_collision_resolved"])
        self.assertFalse(report["decision"]["robot_motion_allowed"])

    def test_stable_completion_rejects_fixed_settle_speed_proxy(self):
        report = json.loads((
            ROOT / "results" / "g1_stable_completion_validation_20260827.json"
        ).read_text())
        self.assertEqual(report["summary"]["pair_count"], 5)
        self.assertEqual(report["summary"]["passed_pair_count"], 1)
        self.assertTrue(report["summary"]["all_actions_byte_identical"])
        self.assertFalse(
            report["decision"]["fixed_settle_duration_is_accepted_task_speed_metric"]
        )
        self.assertTrue(
            report["decision"]["stable_completion_metric_frozen_for_future_search"]
        )
        self.assertFalse(
            report["decision"]["current_candidate_stable_completion_gate_passed"]
        )
        self.assertFalse(report["decision"]["robot_motion_allowed"])

    def test_development_correctness_corpus_has_no_dangerous_fast_accept(self):
        report = json.loads((
            ROOT / "results" / "g1_fast_preflight_correctness_corpus_20260827.json"
        ).read_text())
        self.assertEqual(report["summary"]["verdict_case_pass_count"], 7)
        self.assertEqual(report["summary"]["schema_fault_pass_count"], 3)
        self.assertEqual(report["summary"]["dangerous_fast_accept_count"], 0)
        self.assertEqual(report["summary"]["acceptance_concordance_rate"], 1.0)
        self.assertTrue(report["summary"]["development_corpus_passed"])
        self.assertFalse(
            report["decision"]["minimum_30_trajectory_corpus_completed"]
        )
        self.assertFalse(report["decision"]["robot_motion_allowed"])

    def test_stable_optimizer_reports_no_passing_candidate_without_relaxation(self):
        report = json.loads((
            ROOT / "results" / "g1_stable_completion_optimizer_20260827.json"
        ).read_text())
        self.assertEqual(report["input"]["grid_candidate_count"], 811)
        self.assertEqual(report["input"]["dynamics_candidate_count"], 30)
        self.assertEqual(report["summary"]["screen_pass_count"], 0)
        self.assertLess(
            report["summary"][
                "maximum_observed_screen_duration_reduction_fraction"
            ],
            0.10,
        )
        self.assertIsNone(report["summary"]["selected"])
        self.assertFalse(
            report["decision"]["stable_multi_scenario_candidate_found"]
        )
        self.assertTrue(
            report["decision"][
                "minimum_30_candidate_development_screen_completed"
            ]
        )
        self.assertFalse(report["decision"]["search_is_exhaustive"])
        self.assertFalse(report["decision"]["production_adaptive_enabled"])
        self.assertFalse(report["decision"]["robot_motion_allowed"])

    def test_30_trajectory_fast_preflight_corpus_has_no_dangerous_accept(self):
        report = json.loads((
            ROOT / "results" / "g1_fast_preflight_30_trajectory_corpus_20260827.json"
        ).read_text())
        self.assertEqual(report["summary"]["trajectory_count"], 30)
        self.assertEqual(report["summary"]["passed_count"], 30)
        self.assertEqual(report["summary"]["acceptance_concordance_rate"], 1.0)
        self.assertEqual(report["summary"]["dangerous_fast_accept_count"], 0)
        self.assertTrue(report["summary"]["all_30_passed"])
        self.assertTrue(
            report["decision"]["minimum_30_trajectory_development_corpus_completed"]
        )
        self.assertFalse(
            report["decision"]["randomized_near_limit_coverage_completed"]
        )
        self.assertFalse(report["decision"]["robot_motion_allowed"])

    def test_near_limit_fuzz_is_rejection_only_and_not_hardware_qualification(self):
        report = json.loads((
            ROOT / "results" / "g1_fast_preflight_near_limit_fuzz_20260827.json"
        ).read_text())
        self.assertEqual(report["summary"]["case_count"], 30)
        self.assertEqual(report["summary"]["passed_count"], 30)
        self.assertEqual(report["summary"]["dangerous_fast_accept_count"], 0)
        self.assertEqual(report["summary"]["legacy_accepted_count"], 0)
        self.assertFalse(report["summary"]["accepted_case_coverage_observed"])
        self.assertTrue(
            report["decision"][
                "deterministic_near_limit_rejection_concordance_passed"
            ]
        )
        self.assertFalse(
            report["decision"]["accepted_near_limit_coverage_passed"]
        )
        self.assertFalse(report["decision"]["official_hardware_limits_qualified"])
        self.assertFalse(report["decision"]["robot_motion_allowed"])

    def test_settling_diagnostic_explains_schedule_gain_loss(self):
        report = json.loads((
            ROOT / "results" / "g1_settling_bottleneck_diagnostic_20260827.json"
        ).read_text())
        comparison = report["comparison"]
        previous = report["previous_single_fixture_candidate"]
        best = report["best_observed_30_screen_candidate"]
        self.assertTrue(comparison["previous_action_samples_match_baseline"])
        self.assertTrue(comparison["best_action_samples_match_baseline"])
        self.assertGreater(comparison["previous_path_time_reduction_s"], 0.0)
        self.assertGreater(
            comparison["previous_settling_time_increase_s"],
            comparison["previous_path_time_reduction_s"],
        )
        self.assertLess(previous["stable_duration_reduction_fraction"], 0.0)
        self.assertGreater(best["stable_duration_reduction_fraction"], 0.0)
        self.assertLess(best["stable_duration_reduction_fraction"], 0.10)
        self.assertTrue(
            report["decision"]["settling_aware_optimization_required"]
        )
        self.assertFalse(report["decision"]["production_adaptive_enabled"])
        self.assertFalse(report["decision"]["robot_motion_allowed"])

    def test_compensated_shadow_preserves_canonical_action_and_holds_on_collision(self):
        report = json.loads((
            ROOT / "results" / "g1_offline_compensated_shadow_replay_20260827.json"
        ).read_text())
        boundary = report["action_boundary"]
        self.assertEqual(
            boundary["canonical_policy_action_sha256"],
            boundary["canonical_after_adapter_sha256"],
        )
        self.assertFalse(boundary["canonical_input_mutated"])
        self.assertTrue(boundary["ik_target_is_separate_artifact"])
        self.assertTrue(report["fast_sequential_ik"]["accepted"])
        self.assertEqual(report["fast_sequential_ik"]["checked_targets"], 32)
        self.assertFalse(report["fast_swept_path"]["accepted"])
        self.assertEqual(
            report["fast_swept_path"]["reason"],
            "initial_configuration_collision",
        )
        self.assertTrue(report["mock_sink"]["all_hold_true"])
        self.assertTrue(report["mock_sink"]["all_robot_command_sent_false"])
        self.assertFalse(report["decision"]["real_15_hz_policy_shadow_passed"])
        self.assertFalse(report["decision"]["robot_motion_allowed"])

    def test_yuhao_comparison_is_layered_and_does_not_claim_total_ratio(self):
        report = json.loads((
            ROOT / "results" / "g1_yuhao_fast_layered_benchmark_20260827.json"
        ).read_text())
        yuhao = report["layers"]["yuhao_pinocchio_ik_only"]
        fast = report["layers"]["project_fast_mujoco_ik_plus_swept"]
        self.assertEqual(report["input"]["pairs"], 50)
        self.assertTrue(yuhao["all_runs_pass_project_5mm_3deg_outer_gate"])
        self.assertFalse(yuhao["swept_collision_included"])
        self.assertTrue(fast["swept_collision_included"])
        self.assertTrue(fast["prefetch_gate_passed"])
        self.assertTrue(
            report["decision"]["same_input_solver_layer_comparison_completed"]
        )
        self.assertFalse(
            report["decision"][
                "yuhao_ik_only_to_project_complete_preflight_ratio_claimed"
            ]
        )
        self.assertFalse(report["decision"]["yuhao_production_loop_benchmarked"])
        self.assertFalse(report["decision"]["task_level_speedup_passed"])
        self.assertFalse(report["decision"]["robot_motion_allowed"])

    def test_five_distance_ab_rejects_non_generalizing_candidate(self):
        report = json.loads((
            ROOT / "results" / "g1_deterministic_speed_paired_ab_20260827.json"
        ).read_text())
        self.assertEqual(report["summary"]["development_pair_count"], 5)
        self.assertEqual(report["summary"]["passed_pair_count"], 2)
        self.assertTrue(report["summary"]["all_action_hashes_paired"])
        self.assertFalse(report["summary"]["all_pairs_passed"])
        self.assertFalse(report["decision"]["single_fixture_candidate_generalized"])
        self.assertFalse(report["decision"]["formal_minimum_30_pair_ab_completed"])
        self.assertFalse(report["decision"]["production_adaptive_enabled"])

    def test_retained_method_and_hardware_boundary_are_explicit(self):
        plan = (ROOT / "G1_SPEED_OPTIMIZATION_AND_HARDWARE_VALIDATION_PLAN_CN.md").read_text()
        for text in (
            "MuJoCo：继续用于算法优化和安全筛选",
            "真机只读Shadow：联网后尽快开始",
            "真机动作：H1–H5全部通过后才开始",
            "任务速度结论：最终必须以真机paired A/B为准",
        ):
            self.assertIn(text, plan)
        status = json.loads((
            ROOT / "results" / "g1_speed_optimization_checkpoint_20260827.json"
        ).read_text())
        self.assertTrue(status["decision"]["offline_speed_checkpoint_ready_for_commit"])
        self.assertFalse(status["decision"]["multi_scenario_adaptive_speed_gate_passed"])
        self.assertFalse(status["decision"]["real_task_speedup_passed"])
        self.assertFalse(status["decision"]["robot_motion_allowed"])
        self.assertFalse(status["hardware_execution_performed"])


if __name__ == "__main__":
    unittest.main()
