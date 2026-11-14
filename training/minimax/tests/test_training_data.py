from __future__ import annotations

import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from fixture import FixtureRecord, write_feature_fixture
from poe2_training.dataset import MappedFeatureDataset, TrainingSelection, _family_order
from poe2_training.pattern_experiment import (
    default_suite,
    frozen_pattern_gain_suite,
    run_pattern_experiment,
)
from poe2_training.patterns import REVERSAL_ORBITS


SIGNAL = np.random.default_rng(99).normal(0.0, 1.5, 1716)


def learnable_rows(count: int, seed: int, *, keys: range | None = None,
                   family_size: int = 6) -> list[FixtureRecord]:
    """Records whose residual is a fixed function of their line patterns plus noise."""
    random = np.random.default_rng(seed)
    rows = []
    for index, key in enumerate(keys or range(1, count + 1)):
        legal = random.random(49) < 0.7
        legal[random.integers(49)] = True
        closure = int(random.integers(-40, 40))
        patterns = random.integers(0, 3276, 36)
        signal = SIGNAL[REVERSAL_ORBITS.raw_to_orbit[patterns]].sum() + random.normal(0.0, 2.0)
        rows.append(FixtureRecord(
            key=key, split=int(random.choice([1, 1, 1, 2, 3])), ply=int(random.integers(0, 50)),
            teacher_value=closure + int(round(signal)), two_ply_closure_value=closure,
            own_gains=tuple(int(v) for v in np.where(legal, random.integers(1, 24, 49), -32768)),
            opponent_gains=tuple(
                int(v) for v in np.where(legal, random.integers(1, 24, 49), -32768)),
            family=index // family_size, line_patterns=tuple(int(v) for v in patterns)))
    return rows


def simple_row(key: int, split: int, family: int = 0) -> FixtureRecord:
    return FixtureRecord(key=key, split=split, ply=key % 50, teacher_value=key,
                         two_ply_closure_value=0, family=family)


class TrainingSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _dataset(self, name: str, rows: list[FixtureRecord]) -> MappedFeatureDataset:
        dataset = MappedFeatureDataset(write_feature_fixture(self.root / name, rows=rows,
                                                             corpus_id=name))
        self.addCleanup(dataset.close)
        return dataset

    def test_extra_train_records_lose_primary_holdout_and_newer_positions(self) -> None:
        primary = self._dataset("primary", [
            simple_row(1, 1), simple_row(2, 2), simple_row(3, 3), simple_row(4, 1)])
        newer = self._dataset("newer", [
            simple_row(2, 1), simple_row(4, 1), simple_row(5, 1), simple_row(6, 2)])
        older = self._dataset("older", [
            simple_row(3, 1), simple_row(5, 1), simple_row(7, 1), simple_row(8, 3)])
        selection = TrainingSelection(primary, [(newer, 0.5), (older, 0.25)])
        keys = [sorted(int(key) for key in part.dataset.records["key_low"][part.indices])
                for part in selection.parts]
        self.assertEqual(keys, [[1, 4], [5], [7]])
        accounting = [{name: entry[name] for name in (
            "train_records", "dropped_primary_holdout", "dropped_newer_duplicate",
            "dropped_cap", "records_used", "weight")} for entry in selection.describe()]
        self.assertEqual(accounting, [
            {"train_records": 2, "dropped_primary_holdout": 0, "dropped_newer_duplicate": 0,
             "dropped_cap": 0, "records_used": 2, "weight": 1.0},
            {"train_records": 3, "dropped_primary_holdout": 1, "dropped_newer_duplicate": 1,
             "dropped_cap": 0, "records_used": 1, "weight": 0.5},
            {"train_records": 3, "dropped_primary_holdout": 1, "dropped_newer_duplicate": 1,
             "dropped_cap": 0, "records_used": 1, "weight": 0.25},
        ])
        self.assertEqual(selection.describe()[1]["corpus_id"], "newer")
        self.assertFalse(selection.uniform)
        with self.assertRaisesRegex(ValueError, "positive"):
            TrainingSelection(primary, [(newer, 0.0)])

    def test_cap_keeps_datasets_newest_first_then_whole_games(self) -> None:
        primary = self._dataset("primary", [simple_row(key, 1, family=key) for key in range(1, 5)])
        extra_rows = [simple_row(key, 1, family=key // 3) for key in range(10, 40)]
        extra = self._dataset("extra", extra_rows)
        last = self._dataset("last", [simple_row(key, 1, family=key) for key in range(50, 53)])
        selection = TrainingSelection(primary, [(extra, 1.0), (last, 1.0)], max_records=15)
        used = [part.indices.size for part in selection.parts]
        self.assertEqual(used[0], 4)
        self.assertEqual(used[2], 0)
        self.assertLessEqual(sum(used), 15)
        self.assertGreater(used[1], 15 - 4 - 3)
        families = extra.records["family_id"]
        chosen = set(families[selection.parts[1].indices].tolist())
        whole = {int(family) for family in chosen
                 if np.sum(families == family) == np.sum(families[selection.parts[1].indices] ==
                                                          family)}
        self.assertEqual(chosen, whole)
        self.assertEqual([entry["dropped_cap"] for entry in selection.describe()],
                         [0, 30 - used[1], 3])
        again = TrainingSelection(primary, [(extra, 1.0), (last, 1.0)], max_records=15)
        np.testing.assert_array_equal(again.parts[1].indices, selection.parts[1].indices)
        self.assertNotEqual(sorted(chosen), list(range(len(chosen))))
        ids = np.arange(100, dtype=np.uint64)
        self.assertEqual(sorted(_family_order(ids).tolist()), list(range(100)))

    def test_pieces_equal_one_materialized_batch(self) -> None:
        primary = self._dataset("primary", learnable_rows(40, 1))
        extra = self._dataset("extra", learnable_rows(30, 2, keys=range(100, 130)))
        selection = TrainingSelection(primary, [(extra, 0.5)])
        batch, weights = selection.materialize()
        pieces = list(selection.pieces(7))
        self.assertTrue(all(piece.records <= 7 for piece, _ in pieces))
        for field in fields(batch):
            np.testing.assert_array_equal(
                np.concatenate([getattr(piece, field.name) for piece, _ in pieces]),
                getattr(batch, field.name))
        np.testing.assert_array_equal(np.concatenate([w for _, w in pieces]), weights)
        self.assertEqual(batch.records, selection.records)


class PatternTrainingDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.primary = write_feature_fixture(self.root / "primary",
                                             rows=learnable_rows(1500, 1), corpus_id="primary")
        self.extra = write_feature_fixture(
            self.root / "extra", rows=learnable_rows(900, 2, keys=range(1, 1801, 2)),
            corpus_id="extra")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _train(self, name: str, configs, **options):
        return run_pattern_experiment(self.primary, self.root / name, requested_device="cpu",
                                      seed=3, configs=configs, **options)

    def test_pieces_and_in_memory_training_are_the_same_optimisation(self) -> None:
        # Summation order differs between modes, and Adam rescales tiny gradients, so the two
        # are close rather than identical; CPU and GPU in-memory runs differ by as much.
        configs = [replace(frozen_pattern_gain_suite()[0], max_steps=300),
                   replace(default_suite()[0], max_steps=300)]
        for label, options in (("single", {}), ("weighted", {"extra_datasets": [
                (self.extra, 0.5)]})):
            with self.subTest(label=label):
                memory = self._train(f"{label}-memory", configs, **options)
                pieces = self._train(f"{label}-pieces", configs, piece_records=173, **options)
                for key in ("line_center", "line_scale", "gain_center", "gain_scale"):
                    self.assertEqual(memory["features"][key], pieces["features"][key], key)
                np.testing.assert_allclose(pieces["gain_control"]["weights"],
                                           memory["gain_control"]["weights"], atol=1e-12)
                for in_memory, in_pieces in zip(memory["models"], pieces["models"]):
                    for split in ("train", "validation"):
                        for metric in ("mae", "rmse"):
                            self.assertAlmostEqual(
                                in_pieces[split]["overall"][metric],
                                in_memory[split]["overall"][metric],
                                delta=0.01 * in_memory[split]["overall"][metric])
                    self.assertAlmostEqual(
                        in_pieces["quantization"]["validation"]["overall"]["mae"],
                        in_memory["quantization"]["validation"]["overall"]["mae"],
                        delta=0.02 * in_memory["quantization"]["validation"]["overall"]["mae"])
                self.assertEqual(pieces["training_data"]["piece_records"], 173)

    def test_extra_data_is_reported_and_absent_options_leave_the_report_unchanged(self) -> None:
        config = [replace(frozen_pattern_gain_suite()[0], max_steps=50)]
        plain = self._train("plain", config)
        self.assertNotIn("training_data", plain)
        mixed = self._train("mixed", config, extra_datasets=[(self.extra, 0.25)],
                            max_records=1400)
        data = mixed["training_data"]
        self.assertEqual((data["max_records"], data["piece_records"]), (1400, None))
        self.assertEqual([entry["corpus_id"] for entry in data["datasets"]], ["primary", "extra"])
        self.assertEqual([entry["weight"] for entry in data["datasets"]], [1.0, 0.25])
        self.assertEqual(data["records"], sum(entry["records_used"] for entry in data["datasets"]))
        self.assertLessEqual(data["records"], 1400)
        self.assertGreater(data["datasets"][1]["dropped_primary_holdout"], 0)
        self.assertEqual(mixed["input"], plain["input"])


if __name__ == "__main__":
    unittest.main()
