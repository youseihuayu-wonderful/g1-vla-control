import ast
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_camera_observation import BGRFrame, bgr_frame_to_rgb224
from g1_dds_freshness import LowStateFreshnessWatchdog
from g1_fail_closed_adapter import (
    FailClosedMockAdapter,
    MockSuggestionSink,
    ShadowSafetyInputs,
)
from g1_fail_closed_fault_injection import run_fault_injection


class OfflineFiveStepPlanTests(unittest.TestCase):
    def test_new_offline_modules_have_no_unitree_or_controller_import(self):
        paths = [
            ROOT / "g1_camera_observation.py",
            ROOT / "g1_dds_freshness.py",
            ROOT / "g1_fail_closed_adapter.py",
            ROOT / "g1_fail_closed_fault_injection.py",
            ROOT / "g1_offline_policy_shadow_replay.py",
            ROOT / "g1_yuhao_fk_source_parity.py",
            ROOT / "g1_offline_five_step_completion.py",
        ]
        forbidden = ("unitree_sdk", "arm_controller", "gripper_controller")
        for path in paths:
            tree = ast.parse(path.read_text())
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertFalse(
                any(token in name.lower() for name in imports for token in forbidden),
                path.name,
            )

    def test_camera_boundary_crops_head_left_and_converts_bgr_to_rgb(self):
        left = np.zeros((480, 640, 3), dtype=np.uint8)
        left[..., 0] = 11
        left[..., 1] = 22
        left[..., 2] = 33
        right = np.full((480, 640, 3), 199, dtype=np.uint8)
        image, metadata = bgr_frame_to_rgb224(BGRFrame(
            "head_left",
            np.concatenate((left, right), axis=1),
            received_monotonic_ns=1,
            binocular=True,
        ))
        self.assertEqual(image.shape, (224, 224, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertTrue(metadata["head_left_crop_applied"])
        self.assertEqual(image[112, 112].tolist(), [33, 22, 11])

    def test_lowstate_watchdog_uses_unique_tick_age(self):
        watchdog = LowStateFreshnessWatchdog()
        self.assertEqual(
            watchdog.observe(received_monotonic_ns=0, tick=10), "first"
        )
        self.assertTrue(watchdog.evaluate(now_monotonic_ns=1_000_000)["hold"])
        self.assertEqual(
            watchdog.observe(received_monotonic_ns=2_000_000, tick=11), "forward"
        )
        self.assertFalse(watchdog.evaluate(now_monotonic_ns=3_000_000)["hold"])
        self.assertEqual(
            watchdog.observe(received_monotonic_ns=40_000_000, tick=11), "duplicate"
        )
        stale = watchdog.evaluate(now_monotonic_ns=53_000_000)
        self.assertTrue(stale["hold"])
        self.assertEqual(stale["reason"], "lowstate_stale")
        self.assertFalse(stale["publish_allowed"])
        self.assertFalse(stale["robot_command_sent"])
        watchdog.observe(received_monotonic_ns=54_000_000, tick=12)
        self.assertTrue(watchdog.evaluate(now_monotonic_ns=55_000_000)["hold"])
        watchdog.observe(received_monotonic_ns=56_000_000, tick=13)
        self.assertFalse(watchdog.evaluate(now_monotonic_ns=57_000_000)["hold"])

    def test_lowstate_watchdog_handles_uint32_rollover_as_forward(self):
        watchdog = LowStateFreshnessWatchdog()
        watchdog.observe(received_monotonic_ns=0, tick=(1 << 32) - 1)
        relation = watchdog.observe(received_monotonic_ns=1, tick=0)
        self.assertEqual(relation, "forward")
        self.assertEqual(watchdog.last_tick, 0)

    def test_fault_injection_all_cases_fail_closed(self):
        report = run_fault_injection()
        self.assertEqual(report["summary"]["required_fault_case_count"], 8)
        self.assertTrue(report["summary"]["all_faults_fail_closed"])
        self.assertTrue(report["summary"]["all_publish_allowed_false"])
        self.assertTrue(report["summary"]["all_hold_true"])
        self.assertTrue(report["summary"]["all_robot_command_sent_false"])
        self.assertFalse(report["decision"]["integrated_real_shadow_watchdog_validated"])
        self.assertFalse(report["decision"]["robot_motion_allowed"])

    def test_mock_adapter_nominal_case_still_cannot_publish(self):
        action = np.zeros((32, 16), dtype=np.float64)
        action[:, 6] = 1.0
        action[:, 13] = 1.0
        inputs = ShadowSafetyInputs(
            lowstate_age_ms=1.0,
            unique_tick_age_ms=1.0,
            camera_age_ms={"head_left": 1.0, "left_wrist": 1.0, "right_wrist": 1.0},
            camera_frozen={"head_left": False, "left_wrist": False, "right_wrist": False},
            policy_latency_ms=80.0,
            canonical_action=action,
            ik_success=True,
            collision_free=True,
            joint_limits_ok=True,
            dds_connected=True,
            waist_divergence_m=0.0,
            waist_divergence_deg=0.0,
        )
        adapter = FailClosedMockAdapter(MockSuggestionSink())
        decision = adapter.evaluate(inputs)
        self.assertTrue(decision["safety_gate_passed"])
        self.assertFalse(decision["hold"])
        self.assertFalse(decision["publish_allowed"])
        self.assertFalse(decision["hardware_transport_present"])
        self.assertFalse(decision["robot_command_sent"])

    def test_fk_source_parity_rejects_zero_waist_as_current_pose_approximation(self):
        report = json.loads((
            ROOT / "results" / "g1_yuhao_pinocchio_mujoco_fk_source_parity_20260824.json"
        ).read_text())
        decision = report["decision"]
        self.assertTrue(decision["yuhao_project_zero_waist_source_parity_passed"])
        self.assertTrue(decision["offline_h2_core_completed"])
        self.assertFalse(decision["zero_waist_is_safe_approximation_at_current_pose"])
        self.assertGreater(decision["maximum_waist_induced_position_difference_m"], 0.005)
        self.assertTrue(decision["waist_divergence_requires_hold"])
        self.assertFalse(decision["robot_motion_allowed"])

    def test_offline_replay_is_not_promoted_to_real_shadow(self):
        report = json.loads((
            ROOT / "results" / "g1_offline_policy_shadow_replay_20260824.json"
        ).read_text())
        self.assertTrue(report["decision"]["offline_h4_replay_harness_completed"])
        self.assertFalse(report["decision"]["real_15_hz_policy_shadow_passed"])
        self.assertFalse(report["decision"]["frozen_lgg100_inference_executed"])
        self.assertFalse(report["swept_path"]["accepted"])
        self.assertEqual(report["mock_sink"]["record_count"], 32)
        self.assertTrue(report["mock_sink"]["all_publish_allowed_false"])
        self.assertTrue(report["mock_sink"]["all_hold_true"])
        self.assertTrue(report["mock_sink"]["all_robot_command_sent_false"])
        self.assertFalse(report["decision"]["robot_motion_allowed"])

    def test_h6_checklist_has_no_implicit_authorization(self):
        report = json.loads((
            ROOT / "results" / "g1_first_motion_review_checklist_20260824.json"
        ).read_text())
        self.assertTrue(report["offline_checklist_preparation_completed"])
        self.assertFalse(report["decision"]["motion_review_passed"])
        self.assertFalse(report["decision"]["motion_authorized"])
        self.assertFalse(report["decision"]["publisher_creation_allowed"])
        self.assertTrue(all(
            value is False for value in report["required_authorization"].values()
        ))
        self.assertFalse(report["hardware_execution_performed"])

    def test_five_step_completion_aggregate_keeps_live_gates_closed(self):
        report = json.loads((
            ROOT / "results" / "g1_offline_five_step_completion_20260824.json"
        ).read_text())
        self.assertEqual(report["summary"]["offline_steps_completed"], 5)
        self.assertTrue(report["summary"]["all_offline_steps_completed"])
        self.assertEqual(report["summary"]["live_gates_completed"], 0)
        self.assertTrue(all(not step["live_gate_promoted"] for step in report["steps"]))
        self.assertFalse(report["decision"]["motion_authorized"])
        self.assertFalse(report["safety"]["publisher_created"])
        self.assertFalse(report["hardware_execution_performed"])

    def test_tracker_distinguishes_offline_work_from_live_gate_pass(self):
        status = json.loads((
            ROOT / "results" / "g1_six_gate_execution_status_20260824.json"
        ).read_text())
        self.assertEqual(status["offline_tasks_completed"], 5)
        self.assertEqual(status["offline_tasks_total"], 5)
        self.assertEqual(status["resolved_gate_count"], 0)
        self.assertTrue(status["offline_tasks_do_not_promote_live_gates"])
        self.assertTrue(all(not gate["completed"] for gate in status["gates"]))
        self.assertFalse(status["decision"]["robot_motion_allowed"])


if __name__ == "__main__":
    unittest.main()
