from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from fixture import FixtureRecord, write_feature_fixture


def _gains(signal: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    own = [1] * 49
    opponent = [1] * 49
    own[0] = signal + 2
    own[1] = 2
    opponent[0] = 4
    opponent[1] = signal + 1
    return tuple(own), tuple(opponent)


def _rows(*, test_offset: int = 0) -> tuple[FixtureRecord, ...]:
    rows: list[FixtureRecord] = []
    key = 1
    phase_effect = {4: 3, 8: -2}
    for split, repetitions in ((1, 4), (2, 2)):
        for _ in range(repetitions):
            for ply in (4, 8):
                for signal in range(4):
                    own, opponent = _gains(signal)
                    residual = phase_effect[ply] + 3 * signal
                    closure_value = -8 + signal * 3 + (2 if ply == 4 else -2)
                    rows.append(FixtureRecord(
                        key=key,
                        split=split,
                        ply=ply,
                        teacher_value=closure_value + residual,
                        two_ply_closure_value=closure_value,
                        own_gains=own,
                        opponent_gains=opponent,
                        flags=1 if key % 2 == 0 else 0,
                    ))
                    key += 1
    for signal in range(4):
        own, opponent = _gains(signal)
        rows.append(FixtureRecord(
            key=key,
            split=3,
            ply=12,
            teacher_value=1000 + test_offset + signal,
            two_ply_closure_value=-1000,
            own_gains=own,
            opponent_gains=opponent,
        ))
        key += 1
    return tuple(rows)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class BaselineExperimentTest(unittest.TestCase):
    def test_is_deterministic_and_gain_model_beats_nested_controls(self) -> None:
        from poe2_training.baseline import open_baseline_report, run_baseline_experiment

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_feature_fixture(root / "dataset", rows=_rows())
            first = root / "first"
            second = root / "second"
            report = run_baseline_experiment(dataset, first, requested_device="cpu")
            repeated = run_baseline_experiment(dataset, second, requested_device="cpu")

            self.assertEqual((first / "report.json").read_bytes(),
                             (second / "report.json").read_bytes())
            self.assertEqual(open_baseline_report(first), report)
            self.assertEqual(report, repeated)
            models = {model["name"]: model for model in report["models"]}
            closure_mae = models["two_ply_closure"]["validation"]["overall"]["mae"]
            phase_mae = models["phase"]["validation"]["overall"]["mae"]
            gain_mae = models["gain_summary"]["validation"]["overall"]["mae"]
            self.assertLess(phase_mae, closure_mae)
            self.assertLess(gain_mae, phase_mae)
            self.assertEqual(report["input"]["split_counts"]["test_held_out"], 4)
            self.assertEqual(report["training"]["test_policy"],
                             "excluded_from_fit_selection_and_metrics")

    def test_test_targets_cannot_change_model_or_validation_metrics(self) -> None:
        from poe2_training.baseline import run_baseline_experiment

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_dataset = write_feature_fixture(root / "dataset-one", rows=_rows())
            second_dataset = write_feature_fixture(
                root / "dataset-two", rows=_rows(test_offset=500_000)
            )
            first = run_baseline_experiment(
                first_dataset, root / "output-one", requested_device="cpu"
            )
            second = run_baseline_experiment(
                second_dataset, root / "output-two", requested_device="cpu"
            )
            self.assertEqual(first["models"], second["models"])
            self.assertEqual(first["comparisons"], second["comparisons"])

    def test_refuses_to_overwrite_and_detects_report_corruption(self) -> None:
        from poe2_training.baseline import (
            BaselineExperimentError,
            open_baseline_report,
            run_baseline_experiment,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = write_feature_fixture(root / "dataset", rows=_rows())
            output = root / "output"
            run_baseline_experiment(dataset, output, requested_device="cpu")
            with self.assertRaisesRegex(BaselineExperimentError, "reserve"):
                run_baseline_experiment(dataset, output, requested_device="cpu")
            report = output / "report.json"
            report.write_bytes(report.read_bytes() + b" ")
            with self.assertRaisesRegex(BaselineExperimentError, "digest differs"):
                open_baseline_report(output)


if __name__ == "__main__":
    unittest.main()
