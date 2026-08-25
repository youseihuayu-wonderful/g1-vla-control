import ast
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_real_lowstate_offline_parity import run_diagnostic


CAMERA_PROBE = ROOT / "g1_camera_readonly_probe.py"
PARITY_SOURCE = ROOT / "g1_real_lowstate_offline_parity.py"
CURRENT_SAMPLE = ROOT / "results" / "g1_lowstate_current_pose_sample_20260824.json"
CAMERA_RESULT = ROOT / "results" / "g1_three_camera_readonly_probe_20260824.json"
GATE_STATUS = ROOT / "results" / "g1_six_gate_execution_status_20260824.json"


class RealG1SixGateTests(unittest.TestCase):
    def test_camera_probe_has_no_robot_control_import_or_call(self):
        tree = ast.parse(CAMERA_PROBE.read_text())
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        self.assertFalse(any("unitree" in name.lower() for name in imports))
        self.assertFalse(any("arm_controller" in name for name in imports))
        self.assertFalse(any("gripper_controller" in name for name in imports))
        forbidden_calls = {
            "ChannelPublisher", "Write", "MotionSwitcher", "SportClient",
        }
        calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertTrue(forbidden_calls.isdisjoint(calls))

    def test_real_sample_offline_roundtrip_is_zero_motion(self):
        report = run_diagnostic(CURRENT_SAMPLE, maximum_iterations=100)
        decision = report["decision"]
        self.assertTrue(decision["real_lowstate_to_mujoco_joint_mapping_passed"])
        self.assertTrue(decision["numeric_fk_frame_roundtrip_passed"])
        self.assertTrue(decision["numeric_current_pose_ik_roundtrip_passed"])
        self.assertFalse(decision["physical_eef_parity_verified"])
        self.assertFalse(decision["robot_motion_allowed"])
        self.assertFalse(report["safety"]["publisher_created"])
        self.assertFalse(report["safety"]["robot_command_sent"])
        self.assertFalse(report["hardware_execution_performed"])

    def test_failed_camera_probe_is_not_promoted(self):
        report = json.loads(CAMERA_RESULT.read_text())
        self.assertTrue(all(
            camera["enable_zmq"] for camera in report["config"].values()
        ))
        self.assertFalse(report["all_three_frames_available"])
        self.assertFalse(report["success"])
        self.assertFalse(report["decision"]["image_server_runtime_verified"])
        self.assertFalse(report["decision"]["robot_motion_allowed"])

    def test_six_gates_are_strictly_ordered_and_motion_locked(self):
        status = json.loads(GATE_STATUS.read_text())
        self.assertEqual(status["strict_order"], [f"H{i}" for i in range(1, 7)])
        self.assertEqual(status["resolved_gate_count"], 0)
        self.assertEqual(status["current_gate"], "H1_REAL_THREE_CAMERA")
        self.assertEqual(status["gates"][0]["status"], "BLOCKED_IMAGE_SERVER_NOT_RUNNING")
        self.assertTrue(all(not gate["completed"] for gate in status["gates"]))
        self.assertFalse(status["decision"]["robot_motion_allowed"])
        self.assertFalse(status["safety_baseline"]["publisher_created"])
        self.assertFalse(status["hardware_execution_performed"])


if __name__ == "__main__":
    unittest.main()
