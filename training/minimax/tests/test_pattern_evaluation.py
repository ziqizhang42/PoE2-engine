from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from fixture import FixtureRecord, write_feature_fixture


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is not installed")
class PatternEvaluationTest(unittest.TestCase):
    def test_predicts_with_validation_selected_phase_line_model(self) -> None:
        import numpy as np

        from poe2_training.dataset import MappedFeatureDataset
        from poe2_training.pattern_evaluation import (
            KEY_DTYPE,
            _known_key_mask,
            _predict_model,
            _predict_quantized_model,
            _selected_model,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = write_feature_fixture(
                root / "features",
                rows=(FixtureRecord(
                    key=1,
                    split=3,
                    ply=14,
                    teacher_value=70,
                    two_ply_closure_value=2,
                ),),
            )
            line_weights = np.zeros((3, 1716), dtype=np.float64)
            line_weights[:, 0] = (1.0, 2.0, 3.0)
            model = {
                "name": "selected",
                "config": {
                    "line_knots": [0, 28, 49],
                    "gain_knots": None,
                },
                "phase_weights": np.arange(50, dtype=np.float64).tolist(),
                "line_weights": line_weights.tolist(),
                "gain_weights": None,
            }
            experiment = {
                "selection": {"best_model": "selected"},
                "models": [model],
                "features": {
                    "line_center": np.zeros(1716).tolist(),
                    "line_scale": np.ones(1716).tolist(),
                },
            }
            self.assertIs(_selected_model(experiment), model)
            with MappedFeatureDataset(dataset_path) as dataset:
                batch = dataset.materialize("test")
            prediction = _predict_model(batch, experiment, model)
            self.assertIsNone(_predict_quantized_model(batch, model))
            # All 36 fixture line IDs are orbit zero. At ply 14 the first two
            # knot weights interpolate equally: 14 + 36 * (1 + 2) / 2 = 68.
            np.testing.assert_allclose(prediction, np.array([68.0]))

            candidates = np.array([(1, 2), (2, 1), (3, 4)], dtype=KEY_DTYPE)
            known = np.array([(1, 2), (3, 3), (3, 4)], dtype=KEY_DTYPE)
            np.testing.assert_array_equal(
                _known_key_mask(candidates, known),
                np.array([True, False, True]),
            )


if __name__ == "__main__":
    unittest.main()
