from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from fixture import FixtureRecord, write_feature_fixture


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is not installed")
class PatternRepresentationTest(unittest.TestCase):
    def test_frozen_pattern_gain_suite_matches_the_development_selection(self) -> None:
        from poe2_training.pattern_experiment import (
            FROZEN_PATTERN_GAIN_FRACTIONAL_BITS,
            FROZEN_PATTERN_GAIN_MODEL_ID,
            GAIN_KNOTS,
            frozen_pattern_gain_suite,
            line_pattern_audit_suite,
        )

        configs = frozen_pattern_gain_suite()
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].name, FROZEN_PATTERN_GAIN_MODEL_ID)
        self.assertEqual(FROZEN_PATTERN_GAIN_FRACTIONAL_BITS, 5)
        self.assertEqual(configs[0].line_knots, (0, 28, 49))
        self.assertEqual(configs[0].gain_knots, GAIN_KNOTS)
        self.assertEqual(configs[0].loss, "huber")
        self.assertEqual(configs[0].huber_delta, 8.0)
        self.assertEqual(configs[0].l2, 1.0e-4)
        line_configs = line_pattern_audit_suite()
        self.assertEqual(len(line_configs), 6)
        self.assertTrue(all(config.gain_knots is None for config in line_configs))

    def test_reversal_map_has_expected_orbits_and_ties_reversals(self) -> None:
        from poe2_training.patterns import (
            REVERSAL_ORBIT_COUNT,
            REVERSAL_ORBITS,
            reverse_ternary,
        )

        self.assertEqual(REVERSAL_ORBIT_COUNT, 1716)
        for length, offset in zip(range(2, 8), REVERSAL_ORBITS.pattern_offsets, strict=True):
            for code in range(3 ** length):
                reversed_code = reverse_ternary(code, length)
                self.assertEqual(
                    REVERSAL_ORBITS.raw_to_orbit[offset + code],
                    REVERSAL_ORBITS.raw_to_orbit[offset + reversed_code],
                )

    def test_pattern_counts_sum_to_scoring_line_count(self) -> None:
        import numpy as np

        from poe2_training.patterns import line_pattern_counts

        patterns = np.vstack((np.arange(36), np.arange(36, 72))).astype(np.int64)
        counts = line_pattern_counts(patterns)
        self.assertEqual(counts.shape, (2, 1716))
        self.assertEqual(counts.sum(axis=1).tolist(), [36.0, 36.0])

    def test_training_records_loss_trace_and_authenticated_chart(self) -> None:
        from poe2_training.pattern_experiment import (
            ModelConfig,
            open_pattern_report,
            run_pattern_experiment,
        )

        rows = tuple(
            FixtureRecord(
                key=index,
                split=1 if index <= 4 else 2,
                ply=index % 2,
                teacher_value=index + 2,
                two_ply_closure_value=index // 2,
            )
            for index in range(1, 7)
        )
        config = ModelConfig(
            name="tiny-loss-curve", line_knots=(0,), gain_knots=None,
            loss="mse", huber_delta=0.0, l2=1.0e-5,
            max_steps=2, evaluation_interval=1, patience_steps=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_feature_fixture(root / "features", rows=rows)
            output = root / "training"
            report = run_pattern_experiment(
                dataset, output, requested_device="cpu", configs=(config,))
            trace = report["models"][0]["trace"]
            self.assertEqual([point["step"] for point in trace], [0, 1, 2])
            self.assertTrue(all("training_loss" in point and "validation_loss" in point
                                for point in trace))
            opened = open_pattern_report(output)
            self.assertEqual(opened["models"][0]["trace"], trace)
            self.assertEqual(opened["attachments"], report["attachments"])
            self.assertTrue((output / "training-metrics.svg").is_file())


if __name__ == "__main__":
    unittest.main()
