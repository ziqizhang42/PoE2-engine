from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from fixture import write_feature_fixture
from test_training_data import learnable_rows
from poe2_training.pattern_experiment import (
    PatternExperimentError,
    _pattern_lookup,
    default_suite,
    frozen_pattern_gain_suite,
    run_pattern_experiment,
)
from poe2_training.patterns import REVERSAL_ORBIT_COUNT
from poe2_training.workflow_adapters import AdapterError, authenticate_training
from poe2_training.workflow_config import (
    ImportedDatasetConfig,
    IterationConfig,
    PatternTrainingConfig,
)

# Lookup and dense runs differ only in summation order; Adam turns that into small weight
# differences that grow with the steps taken. These bounds hold with a wide margin.
WEIGHT_TOLERANCE = 2.0e-3
METRIC_TOLERANCE = 1.0e-3


def _clean_provenance() -> dict[str, object]:
    return {"commit": "1" * 40, "dirty": False}


class LookupSumTest(unittest.TestCase):
    def test_sums_and_gradients_equal_dense_counts_and_repeat_exactly(self) -> None:
        random = np.random.default_rng(5)
        orbits = random.integers(0, REVERSAL_ORBIT_COUNT, (700, 36)).astype(np.int16)
        orbits[:, :20] = random.integers(0, 4, (700, 20))
        raw = REVERSAL_ORBIT_COUNT
        counts = torch.zeros((700, raw), dtype=torch.float64)
        counts.scatter_add_(1, torch.as_tensor(orbits, dtype=torch.int64),
                            torch.ones((700, 36), dtype=torch.float64))
        values = torch.as_tensor(random.normal(size=(raw, 3)), dtype=torch.float64)
        upstream = torch.as_tensor(random.normal(size=(700, 3)), dtype=torch.float64)
        devices = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]
        for device in devices:
            with self.subTest(device=device):
                lookup = _pattern_lookup(orbits, torch.device(device), plan=True)
                gradients = []
                for _ in range(2):
                    weights = values.to(device=device, dtype=torch.float32).requires_grad_()
                    sums = lookup.sums(weights)
                    (sums * upstream.to(device, torch.float32)).sum().backward()
                    gradients.append(weights.grad.cpu())
                torch.testing.assert_close(sums.detach().cpu().double(), counts @ values,
                                           rtol=1e-5, atol=1e-4)
                torch.testing.assert_close(gradients[0].double(), counts.T @ upstream,
                                           rtol=1e-5, atol=1e-3)
                self.assertTrue(torch.equal(gradients[0], gradients[1]))
                no_plan = _pattern_lookup(orbits, torch.device(device), plan=False)
                with self.assertRaisesRegex(PatternExperimentError, "no gradient plan"):
                    no_plan.sums(values.to(device, torch.float32).requires_grad_()).sum().backward()


class LookupTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.primary = write_feature_fixture(self.root / "primary",
                                             rows=learnable_rows(1500, 1), corpus_id="primary")
        self.extra = write_feature_fixture(
            self.root / "extra", rows=learnable_rows(900, 2, keys=range(1, 1801, 2)),
            corpus_id="extra")
        self.enterContext(mock.patch("poe2_training.pattern_experiment.git_provenance",
                                     _clean_provenance))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _train(self, name: str, configs, **options):
        return run_pattern_experiment(self.primary, self.root / name, requested_device="cpu",
                                      seed=3, configs=configs, **options)

    def test_lookup_trains_the_dense_model_up_to_rounding(self) -> None:
        configs = [replace(frozen_pattern_gain_suite()[0], max_steps=300),
                   replace(default_suite()[0], max_steps=300)]
        self.assertIsNone(configs[1].gain_knots)
        for label, options in (("single", {}), ("weighted", {"extra_datasets": [
                (self.extra, 0.5)]})):
            with self.subTest(label=label):
                dense = self._train(f"{label}-dense", configs, **options)
                lookup = self._train(f"{label}-lookup", configs, method="lookup", **options)
                self.assertEqual(lookup["features"], dense["features"])
                self.assertEqual(lookup["gain_control"], dense["gain_control"])
                for in_dense, in_lookup in zip(dense["models"], lookup["models"]):
                    self.assertEqual(in_lookup["best_step"], in_dense["best_step"])
                    for name in ("phase_weights", "line_weights", "gain_weights"):
                        if in_dense[name] is not None:
                            np.testing.assert_allclose(in_lookup[name], in_dense[name], rtol=0,
                                                       atol=WEIGHT_TOLERANCE, err_msg=name)
                    for split in ("train", "validation"):
                        for metric in ("mae", "rmse"):
                            self.assertAlmostEqual(
                                in_lookup[split]["overall"][metric],
                                in_dense[split]["overall"][metric], delta=METRIC_TOLERANCE)

    def test_lookup_is_recorded_only_when_used_and_excludes_pieces(self) -> None:
        config = [replace(frozen_pattern_gain_suite()[0], max_steps=50)]
        dense = self._train("dense/training", config)
        lookup = self._train("lookup/training", config, method="lookup")
        self.assertNotIn("method", dense["training"])
        self.assertEqual(lookup["training"]["method"], "lookup")
        self.assertNotIn("training_data", lookup)
        with self.assertRaisesRegex(PatternExperimentError, "only to the dense"):
            self._train("both", config, method="lookup", piece_records=100)
        with self.assertRaisesRegex(PatternExperimentError, "must be one of"):
            self._train("other", config, method="sparse")

        dataset = ImportedDatasetConfig(name="primary", kind="imported",
                                        feature_directory=self.primary)
        self.enterContext(mock.patch("poe2_training.workflow_adapters.pattern_suite",
                                     lambda _name: config))
        for recorded, configured in (("dense", "dense"), ("lookup", "lookup"),
                                     ("lookup", "dense"), ("dense", "lookup")):
            iteration = IterationConfig(
                name="one", dataset="primary", depends_on=(), root=self.root / recorded,
                training=PatternTrainingConfig(type="pattern", device="cpu", seed=3,
                                               suite="frozen-pattern-gain", method=configured),
                sealed_evaluation=None, candidate_validation=None, engine_gate=None)
            with self.subTest(recorded=recorded, configured=configured):
                if recorded == configured:
                    authenticate_training(dataset, iteration)
                else:
                    with self.assertRaisesRegex(AdapterError, "settings differ"):
                        authenticate_training(dataset, iteration)


if __name__ == "__main__":
    unittest.main()
