import sys
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from speed_scheduler import SpeedScheduleConfig, schedule_timestamps


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


class SpeedSchedulerTests(unittest.TestCase):
    def test_idle_timestamps_speed_up_without_action_change(self):
        timestamps = np.arange(12, dtype=np.float64) / 10.0
        actions = np.zeros((12, 16), dtype=np.float64)
        eef_actions = actions.copy()
        original = actions.copy()

        result = schedule_timestamps(
            actions,
            timestamps,
            eef_actions=eef_actions,
            thresholds=THRESHOLDS,
            config=SpeedScheduleConfig(maximum_scale_increase_per_s=100.0),
        )

        self.assertTrue(np.array_equal(actions, original))
        self.assertTrue(np.all(np.diff(result.timestamps) > 0.0))
        self.assertLess(result.timestamps[-1] - result.timestamps[0], timestamps[-1] - timestamps[0])
        self.assertGreater(result.scale_profile.max(), 1.0)
        self.assertGreater(result.interval_labels["idle"].sum(), 0)

    def test_high_speed_jump_is_slowed_without_action_change(self):
        timestamps = np.arange(8, dtype=np.float64) / 10.0
        actions = np.zeros((8, 16), dtype=np.float64)
        actions[4:, 0] = 1.0
        eef_actions = actions.copy()
        original = actions.copy()
        config = SpeedScheduleConfig(
            speed_cap_margin=0.90,
            maximum_scale_increase_per_s=100.0,
        )

        result = schedule_timestamps(
            actions,
            timestamps,
            eef_actions=eef_actions,
            thresholds=THRESHOLDS,
            config=config,
        )

        self.assertTrue(np.array_equal(actions, original))
        self.assertTrue(result.interval_labels["high_speed"].any())
        self.assertLess(result.scale_profile.min(), 1.0)
        self.assertLess(
            result.retimed_signals["joint_speed"].max(),
            10.0,
        )


if __name__ == "__main__":
    unittest.main()
