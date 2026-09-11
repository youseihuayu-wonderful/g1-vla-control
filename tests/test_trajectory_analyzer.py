import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trajectory_analyzer import AnalyzerConfig, analyze_episode, compute_signals, derive_thresholds


class TrajectoryAnalyzerTests(unittest.TestCase):
    def test_constant_hold_is_marked_as_idle_wait(self):
        timestamps = np.arange(12, dtype=np.float64) / 10.0
        actions = np.zeros((12, 16), dtype=np.float64)
        signals = compute_signals(actions, timestamps, eef_actions=actions)
        thresholds = derive_thresholds([signals])

        report = analyze_episode(
            "hold",
            actions,
            timestamps,
            eef_actions=actions,
            thresholds=thresholds,
            config=AnalyzerConfig(minimum_segment_s=0.2),
        )

        self.assertEqual(report["counts"]["idle_interval_count"], 11)
        self.assertEqual(report["counts"]["high_speed_interval_count"], 0)
        self.assertEqual(report["counts"]["joint_jerk_spike_count"], 0)
        self.assertEqual(report["segments"]["idle_wait"][0]["start_frame"], 0)
        self.assertEqual(report["segments"]["idle_wait"][0]["end_frame"], 11)

    def test_joint_jump_is_marked_as_high_speed_and_jerk_spike(self):
        timestamps = np.arange(10, dtype=np.float64) / 10.0
        actions = np.zeros((10, 16), dtype=np.float64)
        actions[4:, 0] = 1.0
        eef_actions = actions.copy()
        thresholds = {
            "idle_joint_speed_rad_s": 0.02,
            "idle_gripper_speed_rad_s": 0.02,
            "gripper_activity_speed_rad_s": 0.25,
            "high_joint_speed_rad_s": 5.0,
            "high_eef_speed_m_s": 5.0,
            "high_gripper_speed_rad_s": 1.0,
            "joint_acceleration_spike_rad_s2": 50.0,
            "joint_jerk_spike_rad_s3": 500.0,
            "eef_acceleration_spike_m_s2": 50.0,
            "eef_jerk_spike_m_s3": 500.0,
            "fine_eef_speed_m_s": 0.01,
            "coarse_eef_speed_m_s": 0.1,
        }

        report = analyze_episode(
            "jump",
            actions,
            timestamps,
            eef_actions=eef_actions,
            thresholds=thresholds,
            config=AnalyzerConfig(minimum_segment_s=0.2),
        )

        self.assertGreater(report["counts"]["high_speed_interval_count"], 0)
        self.assertGreater(report["counts"]["joint_jerk_spike_count"], 0)
        self.assertTrue(report["segments"]["high_speed"])
        self.assertTrue(report["spikes"]["joint_jerk"])
        self.assertFalse(report["spikes"]["joint_jerk"][0]["value_rad_s3"] <= 500.0)


if __name__ == "__main__":
    unittest.main()
