from types import ModuleType, SimpleNamespace
from unittest import mock
import unittest

import numpy as np

from g1_camera_readonly_probe import probe_three_cameras


class _FakeImageClient:
    def __init__(self, **_kwargs):
        self.counts = {"head": 0, "left": 0, "right": 0}
        self.closed = False

    def get_cam_config(self):
        return {
            "head_camera": {"enable_zmq": True, "binocular": False},
            "left_wrist_camera": {"enable_zmq": True, "binocular": False},
            "right_wrist_camera": {"enable_zmq": True, "binocular": False},
        }

    def _frame(self, name):
        self.counts[name] += 1
        image = None
        if self.counts[name] >= 3:
            image = np.full((4, 5, 3), self.counts[name], dtype=np.uint8)
        return SimpleNamespace(bgr=image, sequence=self.counts[name])

    def get_head_frame(self):
        return self._frame("head")

    def get_left_wrist_frame(self):
        return self._frame("left")

    def get_right_wrist_frame(self):
        return self._frame("right")

    def close(self):
        self.closed = True


class G1CameraReadonlyProbeRetryTests(unittest.TestCase):
    def test_retries_initial_none_until_all_three_frames_arrive(self):
        module = ModuleType("teleimager")
        module.ImageClient = _FakeImageClient
        with mock.patch.dict("sys.modules", {"teleimager": module}):
            report = probe_three_cameras(
                "127.0.0.1", 60000,
                frame_timeout_s=1.0,
                retry_interval_s=0.001,
            )
        self.assertTrue(report["success"])
        self.assertTrue(report["all_three_frames_available"])
        for frame in report["frames"].values():
            self.assertTrue(frame["success"])
            self.assertEqual(frame["attempts"], 3)
            self.assertEqual(frame["shape"], [4, 5, 3])
            self.assertEqual(frame["dtype"], "uint8")
            self.assertTrue(frame["finite"])
            self.assertIsNotNone(frame["sha256"])

    def test_rejects_unbounded_retry_parameters(self):
        with self.assertRaises(ValueError):
            probe_three_cameras("127.0.0.1", 60000, frame_timeout_s=31.0)
        with self.assertRaises(ValueError):
            probe_three_cameras("127.0.0.1", 60000, retry_interval_s=0.0)


if __name__ == "__main__":
    unittest.main()
