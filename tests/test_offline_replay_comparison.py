import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline_replay_comparison import ReplayConfig, compare_episode, fixed_rate_times, sample_actions
from speed_scheduler import SpeedScheduleConfig


THRESHOLDS = {
    "idle_joint_speed_rad_s": 0.02,
    "idle_gripper_speed_rad_s": 0.02,
    "gripper_activity_speed_rad_s": 0.25,
    "high_joint_speed_rad_s": 1.0,
    "high_eef_speed_m_s": 1.0,
    "high_gripper_speed_rad_s": 1.0,
    "joint_jerk_spike_rad_s3": 1000.0,
    "eef_jerk_spike_m_s3": 1000.0,
    "fine_eef_speed_m_s": 0.05,
    "coarse_eef_speed_m_s": 0.20,
}


class OfflineReplayComparisonTests(unittest.TestCase):
    def test_fixed_rate_sampler_preserves_endpoints(self):
        timestamps = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        actions = np.zeros((3, 16), dtype=np.float64)
        actions[:, 0] = [0.0, 0.5, 1.0]

        query = fixed_rate_times(timestamps, 10.0)
        sampled = sample_actions(timestamps, actions, query)

        self.assertAlmostEqual(query[0], 0.0)
        self.assertAlmostEqual(query[-1], 1.0)
        self.assertAlmostEqual(sampled[0, 0], 0.0)
        self.assertAlmostEqual(sampled[-1, 0], 1.0)

    def test_replay_comparison_is_timestamp_only_and_reduces_idle_duration(self):
        timestamps = np.arange(12, dtype=np.float64) / 10.0
        actions = np.zeros((12, 16), dtype=np.float64)
        eef_actions = actions.copy()
        original = actions.copy()

        report = compare_episode(
            "idle",
            actions,
            timestamps,
            eef_actions=eef_actions,
            thresholds=THRESHOLDS,
            schedule_config=SpeedScheduleConfig(maximum_scale_increase_per_s=100.0),
            replay_config=ReplayConfig(replay_rate_hz=10.0),
        )

        self.assertTrue(np.array_equal(actions, original))
        self.assertTrue(report["accepted"])
        self.assertGreater(report["duration_reduction_fraction"], 0.0)
        self.assertGreater(report["replay_frame_reduction_fraction"], 0.0)
        self.assertEqual(report["endpoint_max_abs_delta"], 0.0)
        self.assertTrue(report["criteria"]["indexed_action_samples_byte_identical"])

    def test_replay_comparison_does_not_accelerate_risky_jump(self):
        timestamps = np.arange(8, dtype=np.float64) / 10.0
        actions = np.zeros((8, 16), dtype=np.float64)
        actions[4:, 0] = 1.0
        eef_actions = actions.copy()

        report = compare_episode(
            "jump",
            actions,
            timestamps,
            eef_actions=eef_actions,
            thresholds=THRESHOLDS,
            schedule_config=SpeedScheduleConfig(maximum_scale_increase_per_s=100.0),
            replay_config=ReplayConfig(replay_rate_hz=10.0),
        )

        self.assertTrue(report["accepted"])
        self.assertGreater(report["risk_interval_count"], 0)
        self.assertLessEqual(report["risk_scale_max"], 1.0)
        self.assertTrue(report["criteria"]["risky_intervals_not_accelerated"])


if __name__ == "__main__":
    unittest.main()
