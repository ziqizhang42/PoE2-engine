from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as element_tree
from pathlib import Path

from poe2_training.shared import (
    SharedArtifactError,
    complete_json_report,
    open_json_report,
    reserve_report_directory,
)
from poe2_training.training_visualization import write_training_metrics


class TrainingVisualizationTest(unittest.TestCase):
    def _report(self) -> dict:
        metrics = {"mae": 2.0, "rmse": 3.0, "sign_accuracy": 0.75}
        return {
            "schema": "visualization-test", "schema_version": 1,
            "models": [
                {
                    "name": "small",
                    "best_step": 1,
                    "trace": [
                        {"step": 0, "training_loss": 5.0, "validation_loss": 6.0,
                         "validation_mae": 3.0, "validation_rmse": 4.0},
                        {"step": 1, "training_loss": 2.0, "validation_loss": 2.5,
                         "validation_mae": 2.0, "validation_rmse": 3.0},
                    ],
                    "validation": {"overall": metrics},
                    "quantization": {"validation": {"overall": {
                        "mae": 2.125, "rmse": 3.125, "sign_accuracy": 0.75,
                    }}},
                },
                {
                    "name": "large",
                    "validation": {"overall": {
                        "mae": 2.5, "rmse": 3.5, "sign_accuracy": 0.7}},
                },
            ],
            "selection": {"best_model": "small"},
        }

    def test_svg_is_deterministic_and_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "report"
            reserve_report_directory(root, "visualization-test")
            report = self._report()
            report["attachments"] = {
                "training_metrics": write_training_metrics(
                    root, report, error_type=SharedArtifactError)
            }
            complete_json_report(root, "visualization-test", report)
            opened, _ = open_json_report(root, schema="visualization-test")
            self.assertEqual(opened, report)
            svg = root / "training-metrics.svg"
            element_tree.parse(svg)
            contents = svg.read_text(encoding="utf-8")
            self.assertIn("Selected-model loss", contents)
            self.assertIn("Validation MAE by model", contents)
            self.assertIn("quantized MAE", contents)
            self.assertIn("pink tick is quantized", contents)

            svg.write_text(contents + "<!-- changed -->\n", encoding="utf-8")
            with self.assertRaisesRegex(SharedArtifactError, "differs from metadata"):
                open_json_report(root, schema="visualization-test")


if __name__ == "__main__":
    unittest.main()
