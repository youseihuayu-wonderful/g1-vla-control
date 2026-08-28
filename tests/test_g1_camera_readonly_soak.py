import ast
from pathlib import Path
import unittest

from g1_camera_readonly_soak import EXPECTED_SHAPE, summarize_camera_samples


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "g1_camera_readonly_soak.py"


def _sample(index, *, available=True, digest=None):
    return {
        "available": available,
        "shape": EXPECTED_SHAPE if available else None,
        "dtype": "uint8" if available else None,
        "finite": available,
        "sha256": digest or f"hash-{index}",
        "source_fps": 30.0,
    }


class G1CameraReadonlySoakTests(unittest.TestCase):
    def test_summary_accepts_fresh_complete_camera_samples(self):
        summary = summarize_camera_samples([_sample(i) for i in range(100)])
        self.assertEqual(summary["attempted_samples"], 100)
        self.assertEqual(summary["available_samples"], 100)
        self.assertEqual(summary["availability_rate"], 1.0)
        self.assertTrue(summary["all_expected_shape"])
        self.assertTrue(summary["all_uint8"])
        self.assertTrue(summary["all_finite"])
        self.assertEqual(summary["unique_hash_count"], 100)
        self.assertEqual(summary["unique_hash_fraction"], 1.0)
        self.assertEqual(summary["max_consecutive_identical_hashes"], 1)
        self.assertEqual(summary["source_fps_median"], 30.0)

    def test_summary_exposes_missing_and_frozen_samples(self):
        samples = [_sample(i, digest="same") for i in range(5)]
        samples.append(_sample(5, available=False))
        summary = summarize_camera_samples(samples)
        self.assertEqual(summary["available_samples"], 5)
        self.assertEqual(summary["missing_samples"], 1)
        self.assertEqual(summary["unique_hash_count"], 1)
        self.assertEqual(summary["unique_hash_fraction"], 0.2)
        self.assertEqual(summary["max_consecutive_identical_hashes"], 5)

    def test_module_has_no_robot_or_process_execution_api(self):
        tree = ast.parse(MODULE.read_text())
        imports = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        attributes = {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        self.assertNotIn("subprocess", imports)
        self.assertNotIn("os", imports)
        self.assertFalse({
            "ChannelPublisher", "ChannelSubscriber", "LowCmd_",
            "MotionSwitcherClient", "SportClient", "system", "Popen",
        } & (identifiers | attributes))


if __name__ == "__main__":
    unittest.main()
