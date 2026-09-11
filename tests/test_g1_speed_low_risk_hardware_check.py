import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_speed_low_risk_hardware_check import evaluate_artifacts


class G1SpeedLowRiskHardwareCheckTests(unittest.TestCase):
    def test_artifact_checks_pass_but_motion_stays_locked(self):
        trajectory = {
            "decision": {
                "trajectory_analyzer_completed": True,
                "hardware_execution_performed": False,
            }
        }
        schedule = {
            "summary": {"all_episodes_accepted": True},
            "decision": {
                "speed_scheduler_completed": True,
                "path_actions_byte_identical_all": True,
                "hardware_execution_performed": False,
            },
        }
        replay = {
            "summary": {"all_episodes_accepted": True},
            "decision": {
                "offline_replay_comparison_completed": True,
                "risky_intervals_not_accelerated_all": True,
                "endpoints_preserved_all": True,
                "hardware_execution_performed": False,
            },
        }
        six_gate = {
            "resolved_gate_count": 1,
            "total_gate_count": 6,
            "safety_baseline": {
                "publisher_created": False,
                "robot_command_sent": False,
                "hardware_execution_performed": False,
            },
        }
        d0_d1 = {
            "d0": {"passed": True},
            "d1": {"passed": True},
            "safety": {
                "send_channel_created": False,
                "robot_command_publisher_created": False,
                "robot_command_sent": False,
                "hardware_motion_performed": False,
            },
            "hardware_execution_performed": False,
        }

        checks = evaluate_artifacts(
            trajectory_analysis=trajectory,
            speed_schedule=schedule,
            offline_replay=replay,
            six_gate=six_gate,
            d0_d1=d0_d1,
        )

        self.assertTrue(all(checks.values()))
        self.assertTrue(checks["six_gate_not_fully_resolved"])
        self.assertTrue(checks["no_prior_hardware_motion_claim"])

    def test_motion_claim_fails_artifact_checks(self):
        base = {
            "trajectory_analysis": {
                "decision": {
                    "trajectory_analyzer_completed": True,
                    "hardware_execution_performed": False,
                }
            },
            "speed_schedule": {
                "summary": {"all_episodes_accepted": True},
                "decision": {
                    "speed_scheduler_completed": True,
                    "path_actions_byte_identical_all": True,
                    "hardware_execution_performed": False,
                },
            },
            "offline_replay": {
                "summary": {"all_episodes_accepted": True},
                "decision": {
                    "offline_replay_comparison_completed": True,
                    "risky_intervals_not_accelerated_all": True,
                    "endpoints_preserved_all": True,
                    "hardware_execution_performed": False,
                },
            },
            "six_gate": {
                "resolved_gate_count": 1,
                "total_gate_count": 6,
                "safety_baseline": {
                    "publisher_created": False,
                    "robot_command_sent": False,
                    "hardware_execution_performed": True,
                },
            },
            "d0_d1": {
                "d0": {"passed": True},
                "d1": {"passed": True},
                "safety": {
                    "send_channel_created": False,
                    "robot_command_publisher_created": False,
                    "robot_command_sent": False,
                    "hardware_motion_performed": False,
                },
                "hardware_execution_performed": False,
            },
        }

        checks = evaluate_artifacts(**base)

        self.assertFalse(checks["six_gate_motion_locked"])


if __name__ == "__main__":
    unittest.main()
