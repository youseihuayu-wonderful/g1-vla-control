import argparse
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_policy_contract import ACTION_HORIZON
from lgg100_ik_convergence_diagnostic import _load_action, _parse_checkpoints


class LGG100IKConvergenceDiagnosticTests(unittest.TestCase):
    def _artifact(self, directory: str, *, executable: bool = False) -> Path:
        path = Path(directory) / "chunks.npz"
        actions = np.zeros((1, ACTION_HORIZON, 16), dtype=np.float64)
        actions[..., 6] = 1.0
        actions[..., 13] = 1.0
        np.savez_compressed(
            path,
            canonicalized_actions_for_analysis=actions,
            cube_translation_m=np.array([0.08, 0.0, 0.0]),
            executable=np.asarray(executable),
            quarantined=np.asarray(True),
        )
        return path

    def test_checkpoint_parser_requires_zero(self):
        self.assertEqual(_parse_checkpoints("30,0,10,30"), (0, 10, 30))
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_checkpoints("10,30")

    def test_load_action_requires_bound_quarantined_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._artifact(directory)
            action = _load_action(
                path,
                cycle=0,
                target_index=0,
                expected_cube_translation=np.array([0.08, 0.0, 0.0]),
            )
            self.assertEqual(action.shape, (16,))
            with self.assertRaises(ValueError):
                _load_action(
                    path,
                    cycle=0,
                    target_index=0,
                    expected_cube_translation=np.array([0.16, 0.0, 0.0]),
                )

    def test_load_action_rejects_executable_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._artifact(directory, executable=True)
            with self.assertRaises(ValueError):
                _load_action(
                    path,
                    cycle=0,
                    target_index=0,
                    expected_cube_translation=np.array([0.08, 0.0, 0.0]),
                )


if __name__ == "__main__":
    unittest.main()
