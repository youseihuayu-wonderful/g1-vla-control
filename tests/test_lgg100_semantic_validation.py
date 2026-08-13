import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lgg100_semantic_validation import _hypotheses, _metrics


class LGG100SemanticValidationTests(unittest.TestCase):
    def test_absolute_xyzw_left_right_hypothesis_wins_golden_fixture(self):
        samples, horizon = 4, 50
        states = np.zeros((samples, 16), dtype=np.float64)
        states[:, 0:3] = [0.2, 0.2, 0.7]
        states[:, 7:10] = [0.25, -0.2, 0.75]
        states[:, 6] = 1.0
        states[:, 13] = 1.0
        states[:, 14:16] = 5.0
        reference = np.repeat(states[:, None, :], horizon, axis=1)
        progress = np.linspace(0.0, 1.0, horizon)
        reference[:, :, 0] += 0.08 * progress
        reference[:, :, 7] += 0.04 * progress
        reference[:, :, 14] -= 0.5 * progress
        reference[:, :, 15] -= 0.3 * progress
        candidates = dict(_hypotheses(reference.copy(), states))
        scores = {
            name: _metrics(actions, reference, states)["score"]
            for name, actions in candidates.items()
        }
        best = min(scores, key=scores.get)
        self.assertEqual(best, "absolute_xyzw_lr")
        self.assertAlmostEqual(scores[best], 0.0)
        self.assertGreater(scores["absolute_xyzw_swapped"], 0.0)
        self.assertGreater(
            scores["delta_xyzw_current_times_delta_lr"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
