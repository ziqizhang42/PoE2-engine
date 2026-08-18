from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from fixture import write_feature_fixture


@unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy is not installed")
class MappedFeatureDatasetTest(unittest.TestCase):
    def test_maps_and_materializes_without_record_loop(self) -> None:
        from poe2_training.dataset import MappedFeatureDataset, RECORD_DTYPE

        with tempfile.TemporaryDirectory() as temporary:
            directory = write_feature_fixture(Path(temporary) / "data")
            with MappedFeatureDataset(directory) as dataset:
                self.assertEqual(RECORD_DTYPE.itemsize, 432)
                self.assertEqual(dataset.indices("train").tolist(), [0])
                self.assertEqual(dataset.indices("validation").tolist(), [])
                batch = dataset.materialize("train")
                self.assertEqual(batch.records, 1)
                self.assertEqual(batch.line_patterns.shape, (1, 36))
                self.assertEqual(batch.own_gains.shape, (1, 49))
                self.assertEqual(batch.teacher_values.tolist(), [8.0])
                self.assertEqual(batch.two_ply_closure_values.tolist(), [6.0])
                self.assertEqual(batch.residuals.tolist(), [2.0])


if __name__ == "__main__":
    unittest.main()
