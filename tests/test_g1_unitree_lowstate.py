import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from g1_dual_arm_ik import LEFT_JOINTS, RIGHT_JOINTS
from g1_unitree_lowstate import (
    CONTRACT_ARM_INDICES,
    CONTRACT_ARM_JOINTS,
    CONTRACT_WAIST_INDICES,
    CONTRACT_WAIST_JOINTS,
    G1_29DOF_JOINT_INDEX,
    LOWSTATE_TOPIC,
    extract_lowstate_snapshot,
    validate_network_interface,
)


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "g1_unitree_lowstate.py"


def fake_message():
    motors = [
        SimpleNamespace(
            mode=1,
            q=index / 10.0,
            dq=index / 100.0,
            ddq=index / 1000.0,
            tau_est=index / 20.0,
            temperature=[30 + index, 31 + index],
            vol=48.0,
            motorstate=0,
        )
        for index in range(35)
    ]
    imu = SimpleNamespace(
        quaternion=[1.0, 0.0, 0.0, 0.0],
        gyroscope=[0.1, 0.2, 0.3],
        accelerometer=[0.0, 0.0, 9.81],
        rpy=[0.01, 0.02, 0.03],
        temperature=41,
    )
    return SimpleNamespace(
        version=[1, 2],
        mode_pr=0,
        mode_machine=3,
        tick=1234,
        imu_state=imu,
        motor_state=motors,
    )


class UnitreeLowStateTests(unittest.TestCase):
    def test_official_g1_arm_indices_match_frozen_contract_order(self):
        self.assertEqual(CONTRACT_WAIST_JOINTS, (
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"
        ))
        self.assertEqual(CONTRACT_WAIST_INDICES, (12, 13, 14))
        self.assertEqual(CONTRACT_ARM_JOINTS, tuple(LEFT_JOINTS + RIGHT_JOINTS))
        self.assertEqual(CONTRACT_ARM_INDICES, tuple(range(15, 29)))
        self.assertEqual(len(G1_29DOF_JOINT_INDEX), 29)
        self.assertEqual(LOWSTATE_TOPIC, "rt/lowstate")

    def test_extracts_arm_state_in_contract_order(self):
        snapshot = extract_lowstate_snapshot(
            fake_message(), received_unix_ns=10, received_monotonic_ns=20
        )
        self.assertEqual(snapshot.received_unix_ns, 10)
        self.assertEqual(snapshot.received_monotonic_ns, 20)
        self.assertEqual(snapshot.tick, 1234)
        self.assertEqual(snapshot.version, (1, 2))
        self.assertEqual(len(snapshot.waist_motor_state), 3)
        self.assertEqual(tuple(state.index for state in snapshot.waist_motor_state), (12, 13, 14))
        self.assertEqual(snapshot.waist_motor_state[0].q, 1.2)
        self.assertEqual(len(snapshot.arm_motor_state), 14)
        self.assertEqual(snapshot.arm_motor_state[0].index, 15)
        self.assertEqual(snapshot.arm_motor_state[0].q, 1.5)
        self.assertEqual(snapshot.arm_motor_state[-1].index, 28)
        self.assertEqual(snapshot.imu_quaternion, (1.0, 0.0, 0.0, 0.0))

    def test_nonfinite_motor_feedback_fails_closed(self):
        message = fake_message()
        message.motor_state[15].q = math.nan
        with self.assertRaisesRegex(ValueError, "must be finite"):
            extract_lowstate_snapshot(message)

    def test_short_motor_array_fails_closed(self):
        message = fake_message()
        message.motor_state = message.motor_state[:28]
        with self.assertRaisesRegex(ValueError, "at least 29 motors"):
            extract_lowstate_snapshot(message)

    def test_malformed_imu_fails_closed(self):
        message = fake_message()
        message.imu_state.rpy = [0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "must contain 3"):
            extract_lowstate_snapshot(message)

    def test_network_interface_validation(self):
        self.assertEqual(validate_network_interface("enp3s0"), "enp3s0")
        self.assertEqual(validate_network_interface("eth0.123"), "eth0.123")
        for bad in ("", "eth0;reboot", "eth0 space", "x" * 65):
            with self.assertRaises(ValueError):
                validate_network_interface(bad)

    def test_module_has_no_robot_send_or_mode_switch_identifiers(self):
        tree = ast.parse(MODULE.read_text())
        identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        forbidden = {
            "ChannelPublisher",
            "LowCmd_",
            "MotionSwitcherClient",
            "SportClient",
            "Write",
        }
        self.assertFalse(forbidden & identifiers)
        self.assertFalse(forbidden & imported)
        self.assertFalse(forbidden & attributes)


if __name__ == "__main__":
    unittest.main()
