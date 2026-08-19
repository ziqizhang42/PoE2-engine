from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from fixture import FixtureRecord, write_feature_fixture


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is not installed")
class PatternQuantizationTest(unittest.TestCase):
    def test_frozen_model_uses_engine_scale_instead_of_validation_tie_break(self) -> None:
        import numpy as np

        from poe2_training.dataset import NumpyFeatureBatch
        from poe2_training.pattern_experiment import (
            FROZEN_PATTERN_GAIN_MODEL_ID,
            GAIN_KNOTS,
            _quantization_report,
        )

        records = 2
        zeros = np.zeros(records, dtype=np.int64)
        batch = NumpyFeatureBatch(
            indices=np.arange(records),
            line_patterns=np.zeros((records, 36), dtype=np.uint16),
            own_gains=np.ones((records, 49), dtype=np.int16),
            opponent_gains=np.ones((records, 49), dtype=np.int16),
            teacher_values=np.asarray((1, -1), dtype=np.int32),
            two_ply_closure_values=np.zeros(records, dtype=np.int32),
            residuals=np.asarray((1, -1), dtype=np.int32),
            plys=np.asarray((0, 49), dtype=np.uint8),
            completed_depths=zeros,
            deepest_completed_depths=zeros,
            terminal_depths=zeros,
            flags=np.asarray((1, 0), dtype=np.uint8),
            duplicate_counts=zeros,
            family_ids=zeros,
            trajectory_ids=zeros,
            parent_ids=zeros,
        )
        metadata = {
            "line_center": np.zeros(1716).tolist(),
            "line_scale": np.ones(1716).tolist(),
            "gain_center": np.zeros(19).tolist(),
            "gain_scale": np.ones(19).tolist(),
        }
        model = {
            "name": FROZEN_PATTERN_GAIN_MODEL_ID,
            "config": {"line_knots": [0, 28, 49], "gain_knots": list(GAIN_KNOTS)},
            "phase_weights": np.zeros(50).tolist(),
            "line_weights": np.zeros((3, 1716)).tolist(),
            "gain_weights": np.zeros((5, 19)).tolist(),
        }

        frozen = _quantization_report(batch, metadata, model, np.zeros(records))
        self.assertEqual(frozen["selection"], "fixed_scale_32_engine_contract")
        self.assertEqual(frozen["fractional_bits"], 5)
        self.assertEqual(frozen["scale"], 32)

        exploratory = _quantization_report(
            batch, metadata, {**model, "name": "exploratory"}, np.zeros(records))
        self.assertEqual(
            exploratory["selection"],
            "validation_mae_then_rmse_then_fractional_bits")
        self.assertEqual(exploratory["fractional_bits"], 4)

    def test_folded_and_quantized_lookup_predictions(self) -> None:
        import numpy as np

        from poe2_training.dataset import MappedFeatureDataset
        from poe2_training.pattern_evaluation import _predict_model
        from poe2_training.quantization import (
            fold_pattern_model,
            predict_folded,
            predict_quantized,
            predict_quantized_fixed,
            quantize_pattern_model,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = write_feature_fixture(
                root / "features",
                rows=(FixtureRecord(
                    key=1,
                    split=2,
                    ply=14,
                    teacher_value=70,
                    two_ply_closure_value=2,
                ),),
            )
            line_weights = np.zeros((3, 1716), dtype=np.float64)
            line_weights[:, 0] = (1.0, 2.0, 3.0)
            model = {
                "name": "selected",
                "config": {"line_knots": [0, 28, 49], "gain_knots": None},
                "phase_weights": np.arange(50, dtype=np.float64).tolist(),
                "line_weights": line_weights.tolist(),
                "gain_weights": None,
            }
            experiment = {
                "features": {
                    "line_center": np.zeros(1716).tolist(),
                    "line_scale": np.ones(1716).tolist(),
                },
            }
            with MappedFeatureDataset(dataset_path) as dataset:
                batch = dataset.materialize("validation")
            direct = _predict_model(batch, experiment, model)
            folded = fold_pattern_model(experiment, model)
            np.testing.assert_allclose(predict_folded(batch, folded), direct)
            quantized = quantize_pattern_model(folded, 8)
            np.testing.assert_array_equal(predict_quantized(batch, quantized), np.rint(direct))
            np.testing.assert_array_equal(
                predict_quantized_fixed(batch, quantized),
                np.rint(direct * quantized.scale).astype(np.int64),
            )

    def test_rejects_int16_table_overflow(self) -> None:
        import numpy as np

        from poe2_training.quantization import FoldedPatternModel, quantize_pattern_model

        model = FoldedPatternModel(
            name="overflow",
            line_knots=(0,),
            gain_knots=None,
            intercept=np.zeros(50),
            line_weights=np.full((1, 1716), 1000.0),
            gain_weights=None,
        )
        with self.assertRaises(OverflowError):
            quantize_pattern_model(model, 8)


if __name__ == "__main__":
    unittest.main()
