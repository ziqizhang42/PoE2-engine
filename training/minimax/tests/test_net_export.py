from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fixture import REFERENCE_NET
from poe2_training.net_export import QUANTIZATION, export_net
from poe2_training.net_file import FRACTIONAL_BITS, NetFileError, decode_net, read_net
from poe2_training.pattern_experiment import FROZEN_PATTERN_GAIN_MODEL_ID, SCHEMA
from poe2_training.shared import complete_json_report, reserve_report_directory, sha256_file


def _rows(values: tuple[int, ...], width: int) -> list[list[int]]:
    return [list(values[index:index + width]) for index in range(0, len(values), width)]


class NetExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.parameters = read_net(REFERENCE_NET).parameters

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _experiment(self, name: str = "experiment", *, line_knots=(0, 28, 49)) -> Path:
        directory = self.root / name
        reserve_report_directory(directory, SCHEMA)
        parameters = self.parameters
        complete_json_report(directory, SCHEMA, {
            "schema": SCHEMA, "schema_version": 1,
            "input": {"corpus_id": "export-test", "corpus_digest": "sha256:" + "c" * 64,
                      "feature_binary_sha256": "f" * 64},
            "selection": {"best_model": FROZEN_PATTERN_GAIN_MODEL_ID},
            "models": [{
                "name": FROZEN_PATTERN_GAIN_MODEL_ID,
                "config": {"line_knots": list(line_knots),
                           "gain_knots": list(parameters.gain_knots)},
                "quantization": {
                    "definition": QUANTIZATION, "fractional_bits": FRACTIONAL_BITS,
                    "intercept": list(parameters.intercept),
                    "line_weights": _rows(parameters.line_weights, 1716),
                    "gain_weights": _rows(parameters.gain_weights, 19),
                },
            }],
        })
        return directory

    def test_export_holds_the_selected_parameters_and_its_provenance(self) -> None:
        experiment = self._experiment()
        data = export_net(experiment, name="run9/g002", teacher="run9/g001", parent="run9/g001",
                          created="2026-11-17")
        self.assertEqual(data, export_net(experiment, name="run9/g002", teacher="run9/g001",
                                          parent="run9/g001", created="2026-11-17"))
        net = decode_net(data)
        self.assertEqual(net.parameters, self.parameters)
        self.assertEqual(net.metadata, {
            "name": "run9/g002", "teacher": "run9/g001", "parent": "run9/g001",
            "created": "2026-11-17", "corpus": "export-test",
            "corpus_digest": "sha256:" + "c" * 64, "feature_binary_sha256": "f" * 64,
            "training_report_sha256": sha256_file(experiment / "report.json"),
        })
        self.assertEqual(net.net_id, net.sha256[:12])

    def test_export_refuses_a_model_of_another_design(self) -> None:
        experiment = self._experiment("other-knots", line_knots=(0, 24, 49))
        with self.assertRaisesRegex(NetFileError, "differ from design pattern-gain-v1"):
            export_net(experiment, name="run9/g002", teacher="two-ply-closure",
                       created="2026-11-17")


if __name__ == "__main__":
    unittest.main()
