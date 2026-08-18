from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is not installed")
class GainSummariesTest(unittest.TestCase):
    def test_phase_features_are_one_hot(self) -> None:
        import numpy as np

        from poe2_training.summaries import phase_features

        features = phase_features(np.array([0, 17, 49], dtype=np.uint8))
        self.assertEqual(features.shape, (3, 50))
        self.assertEqual(features.sum(axis=1).tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(features.argmax(axis=1).tolist(), [0, 17, 49])

    def test_phase_interpolation_and_interactions(self) -> None:
        import numpy as np

        from poe2_training.summaries import phase_interactions, phase_interpolation

        phase = phase_interpolation(np.array([0, 6, 12, 30, 49]), (0, 12, 49))
        self.assertTrue(np.allclose(phase.sum(axis=1), 1.0))
        self.assertEqual(phase[0].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(phase[2].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(phase[-1].tolist(), [0.0, 0.0, 1.0])
        self.assertTrue(np.allclose(phase[1], [0.5, 0.5, 0.0]))

        values = np.array([[2.0, 4.0], [3.0, 6.0]])
        interacted = phase_interactions(values, np.array([[1.0, 0.0], [0.25, 0.75]]))
        self.assertEqual(interacted.tolist(), [[2.0, 4.0, 0.0, 0.0],
                                                [0.75, 1.5, 2.25, 4.5]])

    def test_gain_summaries_capture_tops_counts_contest_and_denial(self) -> None:
        import numpy as np

        from poe2_training.summaries import GAIN_SUMMARY_NAMES, gain_summaries

        own = np.ones((2, 49), dtype=np.int16)
        opponent = np.ones((2, 49), dtype=np.int16)
        own[0, :4] = (10, 5, 3, 2)
        opponent[0, :4] = (9, 4, 3, 2)
        own[1, :3] = (7, 4, 2)
        opponent[1, :3] = (6, 6, 2)

        summaries = gain_summaries(own, opponent)
        encoded = [dict(zip(GAIN_SUMMARY_NAMES, row, strict=True)) for row in summaries]

        self.assertEqual(
            [encoded[0][f"own_gain_rank_{rank}"] for rank in range(1, 5)],
            [10.0, 5.0, 3.0, 2.0],
        )
        self.assertEqual(encoded[0]["own_count_ge_2"], 4.0)
        self.assertEqual(encoded[0]["own_count_ge_4"], 2.0)
        self.assertEqual(encoded[0]["own_count_ge_8"], 1.0)
        self.assertEqual(encoded[0]["contested_best_square"], 1.0)
        self.assertEqual(encoded[0]["opponent_unique_best_square"], 1.0)
        self.assertEqual(encoded[0]["denied_opponent_best_gain"], 5.0)
        self.assertEqual(encoded[1]["contested_best_square"], 1.0)
        self.assertEqual(encoded[1]["opponent_unique_best_square"], 0.0)
        self.assertEqual(encoded[1]["denied_opponent_best_gain"], 0.0)

    def test_rejects_different_occupancy_masks(self) -> None:
        import numpy as np

        from poe2_training.summaries import OCCUPIED_GAIN, gain_summaries

        own = np.ones((1, 49), dtype=np.int16)
        opponent = np.ones((1, 49), dtype=np.int16)
        own[0, 0] = OCCUPIED_GAIN
        with self.assertRaisesRegex(ValueError, "occupancy masks"):
            gain_summaries(own, opponent)

    def test_denial_is_invariant_when_closure_optimal_squares_tie(self) -> None:
        import numpy as np

        from poe2_training.summaries import GAIN_SUMMARY_NAMES, gain_summaries

        own = np.ones((2, 49), dtype=np.int16)
        opponent = np.ones((2, 49), dtype=np.int16)
        own[0, :2] = (10, 14)
        opponent[0, :2] = (10, 6)
        own[1] = own[0, ::-1]
        opponent[1] = opponent[0, ::-1]

        summaries = gain_summaries(own, opponent)
        denial = GAIN_SUMMARY_NAMES.index("denied_opponent_best_gain")
        self.assertEqual(summaries[:, denial].tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
