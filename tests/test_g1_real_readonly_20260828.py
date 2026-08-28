import ast
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


class G1RealReadonly20260828Tests(unittest.TestCase):
    def test_full29_capture_passes_without_promoting_collision_or_dds(self):
        report = json.loads((RESULTS / "g1_full29_readonly_analysis_20260828.json").read_text())
        self.assertTrue(report["capture"]["success"])
        self.assertEqual(report["capture"]["captured_samples"], 1000)
        self.assertEqual(report["capture"]["full_body_joint_count"], 29)
        self.assertTrue(report["capture"]["full_body_complete_all_samples"])
        self.assertEqual(report["dds"]["duplicate_tick_count"], 39)
        self.assertEqual(report["dds"]["reader_take_sample_error_count"], 1)
        self.assertGreater(report["dds"]["receive_gap_ms"]["maximum"], 50.0)
        self.assertEqual(
            report["collision_model"]["samples_with_free_space_violation"], 1000
        )
        self.assertFalse(
            report["collision_model"]["all_samples_initial_collision_free"]
        )
        self.assertTrue(report["decision"]["subscriber_only_full29_capture_passed"])
        self.assertTrue(report["decision"]["missing_leg_state_blocker_resolved"])
        self.assertFalse(report["decision"]["initial_model_collision_resolved"])
        self.assertFalse(report["decision"]["dds_freshness_fully_qualified"])
        self.assertFalse(report["decision"]["live_gate_promoted"])
        self.assertFalse(report["decision"]["robot_motion_allowed"])
        self.assertFalse(report["safety"]["publisher_created"])
        self.assertFalse(report["safety"]["robot_command_sent"])

    def test_camera_single_frames_pass_but_continuous_h1_does_not(self):
        probe = json.loads((RESULTS / "g1_three_camera_readonly_probe_20260828.json").read_text())
        status = json.loads((RESULTS / "g1_camera_server_readonly_status_20260828.json").read_text())
        self.assertTrue(probe["success"])
        self.assertTrue(probe["all_three_frames_available"])
        self.assertFalse(probe["true_simultaneous_snapshot"])
        for frame in probe["frames"].values():
            self.assertTrue(frame["success"])
            self.assertEqual(frame["shape"], [480, 640, 3])
            self.assertEqual(frame["dtype"], "uint8")
            self.assertTrue(frame["finite"])
        self.assertTrue(status["start"]["uvc_driver_reload_skipped"])
        self.assertFalse(status["start"]["sudo_invoked"])
        self.assertFalse(status["start"]["unitree_dds_imported"])
        self.assertFalse(status["start"]["robot_command_publisher_created"])
        self.assertTrue(status["decision"]["single_frame_camera_core_passed"])
        self.assertFalse(status["decision"]["h1_continuous_freshness_passed"])
        self.assertFalse(status["decision"]["next_connection_cleanup_verified"])
        self.assertFalse(status["decision"]["live_gate_promoted"])

    def test_ssh_inventory_never_ran_upstream_privileged_server(self):
        report = json.loads((RESULTS / "g1_robot_ssh_inventory_20260828.json").read_text())
        self.assertTrue(report["decision"]["ssh_inventory_completed"])
        self.assertTrue(report["camera"]["upstream_image_server_attempts_sudo_uvc_reload"])
        self.assertFalse(report["camera"]["upstream_autostart_script_executed"])
        self.assertFalse(report["safety"]["sudo_invoked"])
        self.assertFalse(report["safety"]["dds_participant_created"])
        self.assertFalse(report["safety"]["robot_command_publisher_created"])
        self.assertFalse(report["decision"]["live_gate_promoted"])

    def test_next_connection_interlock_completed_without_unlocking_motion(self):
        interlock = json.loads((
            RESULTS / "g1_next_connection_mandatory_interlock_20260828.json"
        ).read_text())
        cleanup = json.loads((
            RESULTS / "g1_next_connection_cleanup_20260828.json"
        ).read_text())
        shutdown = json.loads((
            RESULTS / "g1_unattended_shutdown_status_20260828.json"
        ).read_text())
        self.assertEqual(interlock["status"], "COMPLETE")
        self.assertTrue(interlock["current"]["cleanup_verified"])
        self.assertFalse(
            interlock["decision"]["next_hardware_test_blocked_until_cleanup"]
        )
        self.assertTrue(cleanup["decision"]["cleanup_verified"])
        self.assertEqual(cleanup["postcheck"]["camera_wrapper_process_count"], 0)
        self.assertEqual(cleanup["postcheck"]["lowstate_capture_process_count"], 0)
        self.assertEqual(cleanup["postcheck"]["camera_listener_ports"], [])
        self.assertFalse(interlock["decision"]["real_policy_shadow_allowed"])
        self.assertFalse(interlock["decision"]["robot_motion_allowed"])
        self.assertTrue(shutdown["actions"]["new_hardware_work_stopped"])
        self.assertTrue(
            shutdown["actions"]["lowstate_subscriber_had_already_closed_after_fixed_capture"]
        )
        self.assertFalse(
            shutdown["actions"]["camera_only_server_termination_verified"]
        )
        self.assertFalse(shutdown["decision"]["unattended_hardware_work_allowed"])
        plan = (ROOT / "G1_NEXT_CONNECTION_INTERLOCK_CN.md").read_text()
        runbook = (ROOT / "CONNECTION_RUNBOOK.md").read_text()
        self.assertIn("cleanup_verified=true", plan)
        self.assertIn("cleanup_verified=true", runbook)

    def test_reconnect_camera_freshness_passes_and_cleanup_is_verified(self):
        soak = json.loads((
            RESULTS / "g1_three_camera_freshness_soak_10s_15hz_20260828.json"
        ).read_text())
        session = json.loads((
            RESULTS / "g1_camera_readonly_reconnect_session_20260828.json"
        ).read_text())
        self.assertTrue(soak["success"])
        self.assertEqual(soak["observed"]["cycles"], 150)
        self.assertLessEqual(
            soak["observed"]["cycle_gap_ms"]["maximum"],
            soak["thresholds"]["maximum_cycle_gap_ms"],
        )
        for summary in soak["camera_summary"].values():
            self.assertEqual(summary["availability_rate"], 1.0)
            self.assertEqual(summary["unique_hash_fraction"], 1.0)
            self.assertEqual(summary["max_consecutive_identical_hashes"], 1)
            self.assertTrue(summary["all_expected_shape"])
            self.assertTrue(summary["all_uint8"])
            self.assertTrue(summary["all_finite"])
        self.assertTrue(session["shutdown"]["cleanup_verified"])
        self.assertEqual(session["shutdown"]["camera_process_count_after"], 0)
        self.assertEqual(session["shutdown"]["listener_ports_after"], [])
        self.assertTrue(session["decision"]["h1_real_three_camera_completed"])
        self.assertFalse(session["decision"]["real_policy_shadow_allowed"])
        self.assertFalse(session["decision"]["robot_motion_allowed"])

    def test_reconnect_dds_repeat_remains_fail_closed(self):
        report = json.loads((
            RESULTS / "g1_full29_dds_with_camera_load_20260828.json"
        ).read_text())
        self.assertTrue(report["capture"]["success"])
        self.assertEqual(report["capture"]["captured_samples"], 1000)
        self.assertEqual(report["tick"]["duplicate_transition_count"], 37)
        self.assertEqual(report["tick"]["duplicate_tick_q29_identical_count"], 0)
        self.assertEqual(report["tick"]["duplicate_tick_q29_changed_count"], 37)
        self.assertFalse(report["tick"]["tick_is_unique_sample_identity"])
        self.assertGreater(report["receive_gap_ms"]["maximum"], 50.0)
        self.assertEqual(report["receive_gap_ms"]["over_provisional_50ms_count"], 1)
        self.assertEqual(report["reader"]["reader_error_line_count"], 0)
        self.assertFalse(report["decision"]["dds_freshness_fully_qualified"])
        self.assertFalse(report["decision"]["real_policy_shadow_allowed"])
        self.assertFalse(report["decision"]["robot_motion_allowed"])
        self.assertFalse(report["safety"]["send_channel_created"])
        self.assertFalse(report["safety"]["robot_command_sent"])

    def test_offline_full29_analyzer_has_no_unitree_or_command_imports(self):
        path = ROOT / "g1_full29_readonly_analysis.py"
        tree = ast.parse(path.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertFalse(any("unitree_sdk" in name for name in imports))
        source = path.read_text()
        for token in ("ChannelPublisher", "LowCmd_", "MotionSwitcher", "SportClient"):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
