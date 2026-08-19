"""Evaluate one validation-selected pattern model on its sealed test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dataset import MappedFeatureDataset, NumpyFeatureBatch, subset_batch
from .model_metrics import model_metrics
from .pattern_experiment import GAIN_KNOTS, open_pattern_report
from .patterns import line_pattern_counts, phase_basis
from .quantization import QuantizedPatternModel, predict_quantized
from .summaries import (
    gain_summaries,
    phase_features,
    phase_interactions,
    phase_interpolation,
)
from .shared import (
    complete_json_report,
    git_provenance,
    open_json_report,
    reserve_report_directory,
    runtime_provenance,
    sha256_file,
    training_source_digest,
)


SCHEMA = "poe2-minimax-pattern-evaluation"
SCHEMA_VERSION = 1
COMPLETE_HEADER = SCHEMA
KEY_DTYPE = np.dtype([("low", "<u8"), ("high", "<u8")])


class PatternEvaluationError(ValueError):
    """Raised when a sealed pattern-model evaluation is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PatternEvaluationError(message)


def open_pattern_evaluation(directory: Path | str) -> dict[str, Any]:
    """Authenticate a completed sealed-test evaluation report."""
    report, _ = open_json_report(
        directory, schema=SCHEMA, schema_version=SCHEMA_VERSION,
        error_type=PatternEvaluationError,
    )
    return report


def _selected_model(experiment: dict[str, Any]) -> dict[str, Any]:
    selected = experiment.get("selection", {}).get("best_model")
    models = experiment.get("models")
    _require(isinstance(selected, str) and isinstance(models, list),
             "pattern experiment has no selected model")
    matches = [model for model in models
               if isinstance(model, dict) and model.get("name") == selected]
    _require(len(matches) == 1, "pattern experiment selected model is missing or duplicated")
    return matches[0]


def _predict_model(batch: NumpyFeatureBatch, experiment: dict[str, Any],
                   model: dict[str, Any]) -> np.ndarray:
    features = experiment.get("features")
    config = model.get("config")
    _require(isinstance(features, dict) and isinstance(config, dict),
             "pattern experiment model metadata is incomplete")
    line_knots = tuple(config.get("line_knots", ()))
    gain_knots_value = config.get("gain_knots")
    gain_knots = tuple(gain_knots_value) if gain_knots_value is not None else None

    line_center = np.asarray(features.get("line_center"), dtype=np.float64)
    line_scale = np.asarray(features.get("line_scale"), dtype=np.float64)
    line_weights = np.asarray(model.get("line_weights"), dtype=np.float64)
    phase_weights = np.asarray(model.get("phase_weights"), dtype=np.float64)
    _require(line_center.shape == line_scale.shape == (1716,) and
             line_weights.shape == (len(line_knots), 1716) and
             phase_weights.shape == (50,) and bool((line_scale > 0.0).all()),
             "pattern experiment line-model dimensions are invalid")

    lines = line_pattern_counts(batch.line_patterns).astype(np.float64, copy=False)
    lines -= line_center
    lines /= line_scale
    line_phase = phase_basis(batch.plys, line_knots).astype(np.float64, copy=False)
    prediction = phase_weights[np.asarray(batch.plys, dtype=np.int64)]
    prediction = prediction + ((lines @ line_weights.T) * line_phase).sum(axis=1)

    gain_weights_value = model.get("gain_weights")
    if gain_knots is None:
        _require(gain_weights_value is None, "gain-free model unexpectedly contains gain weights")
    else:
        _require(gain_knots == GAIN_KNOTS, "pattern model uses unsupported gain knots")
        gain_center = np.asarray(features.get("gain_center"), dtype=np.float64)
        gain_scale = np.asarray(features.get("gain_scale"), dtype=np.float64)
        gain_weights = np.asarray(gain_weights_value, dtype=np.float64)
        _require(gain_center.shape == gain_scale.shape == (19,) and
                 gain_weights.shape == (len(gain_knots), 19) and
                 bool((gain_scale > 0.0).all()),
                 "pattern experiment gain-model dimensions are invalid")
        gains = gain_summaries(batch.own_gains, batch.opponent_gains)
        gains -= gain_center
        gains /= gain_scale
        gain_phase = phase_basis(batch.plys, gain_knots).astype(np.float64, copy=False)
        prediction = prediction + ((gains @ gain_weights.T) * gain_phase).sum(axis=1)
    return np.ascontiguousarray(prediction, dtype=np.float64)


def _predict_gain_control(batch: NumpyFeatureBatch,
                          experiment: dict[str, Any]) -> np.ndarray:
    features = experiment.get("features")
    control = experiment.get("gain_control")
    _require(isinstance(features, dict) and isinstance(control, dict),
             "pattern experiment gain control is missing")
    center = np.asarray(features.get("gain_center"), dtype=np.float64)
    scale = np.asarray(features.get("gain_scale"), dtype=np.float64)
    weights = np.asarray(control.get("weights"), dtype=np.float64)
    _require(center.shape == scale.shape == (19,) and weights.shape == (145,) and
             bool((scale > 0.0).all()), "pattern gain-control dimensions are invalid")
    gains = gain_summaries(batch.own_gains, batch.opponent_gains)
    gains -= center
    gains /= scale
    design = np.concatenate((
        phase_features(batch.plys),
        phase_interactions(gains, phase_interpolation(batch.plys, GAIN_KNOTS)),
    ), axis=1)
    return np.ascontiguousarray(design @ weights, dtype=np.float64)


def _predict_quantized_model(batch: NumpyFeatureBatch,
                             model: dict[str, Any]) -> np.ndarray | None:
    quantization = model.get("quantization")
    if quantization is None:
        return None
    config = model.get("config")
    _require(isinstance(quantization, dict) and isinstance(config, dict),
             "pattern quantization metadata is invalid")
    fractional_bits = quantization.get("fractional_bits")
    line_knots = tuple(config.get("line_knots", ()))
    gain_value = config.get("gain_knots")
    gain_knots = tuple(gain_value) if gain_value is not None else None
    _require(isinstance(fractional_bits, int) and 0 <= fractional_bits <= 20 and line_knots,
             "pattern quantization scale or phase knots are invalid")
    intercept = np.asarray(quantization.get("intercept"), dtype=np.int64)
    line_weights = np.asarray(quantization.get("line_weights"), dtype=np.int64)
    gain_weights_value = quantization.get("gain_weights")
    gain_weights = (np.asarray(gain_weights_value, dtype=np.int64)
                    if gain_weights_value is not None else None)
    _require(intercept.shape == (50,) and line_weights.shape == (len(line_knots), 1716) and
             bool((line_weights >= np.iinfo(np.int16).min).all()) and
             bool((line_weights <= np.iinfo(np.int16).max).all()) and
             bool((intercept >= np.iinfo(np.int32).min).all()) and
             bool((intercept <= np.iinfo(np.int32).max).all()),
             "pattern quantization table dimensions or ranges are invalid")
    if gain_knots is None:
        _require(gain_weights is None, "gain-free quantization contains gain weights")
    else:
        _require(gain_weights is not None and gain_weights.shape == (len(gain_knots), 19) and
                 bool((gain_weights >= np.iinfo(np.int16).min).all()) and
                 bool((gain_weights <= np.iinfo(np.int16).max).all()),
                 "quantized gain table dimensions or ranges are invalid")
    quantized = QuantizedPatternModel(
        name=str(model.get("name")),
        fractional_bits=fractional_bits,
        line_knots=line_knots,
        gain_knots=gain_knots,
        intercept=np.ascontiguousarray(intercept, dtype=np.int32),
        line_weights=np.ascontiguousarray(line_weights, dtype=np.int16),
        gain_weights=(np.ascontiguousarray(gain_weights, dtype=np.int16)
                      if gain_weights is not None else None),
    )
    _require(quantization.get("scale") == quantized.scale,
             "pattern quantization scale is inconsistent")
    return predict_quantized(batch, quantized)


def _reduction(reference: float, candidate: float) -> float | None:
    return None if reference == 0.0 else float(100.0 * (reference - candidate) / reference)


def _canonical_keys(dataset: MappedFeatureDataset,
                    indices: np.ndarray | None = None) -> np.ndarray:
    records = dataset.records if indices is None else dataset.records[indices]
    keys = np.empty(records.shape[0], dtype=KEY_DTYPE)
    keys["low"] = records["key_low"]
    keys["high"] = records["key_high"]
    return keys


def _known_key_mask(keys: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Return membership in a sorted unique canonical-key array."""
    candidates = np.asarray(keys, dtype=KEY_DTYPE)
    reference = np.asarray(known, dtype=KEY_DTYPE)
    if reference.size == 0:
        return np.zeros(candidates.size, dtype=np.bool_)
    offsets = np.searchsorted(reference, candidates)
    in_range = offsets < reference.size
    result = np.zeros(candidates.size, dtype=np.bool_)
    result[in_range] = reference[offsets[in_range]] == candidates[in_range]
    return result


def run_pattern_evaluation(
    dataset_directory: Path | str,
    experiment_directory: Path | str,
    output_directory: Path | str,
    *,
    exclude_dataset_directories: Sequence[Path | str] = (),
) -> dict[str, Any]:
    """Open the sealed test split once for a previously selected model."""
    import torch

    experiment_root = Path(experiment_directory).resolve()
    experiment = open_pattern_report(experiment_root)
    model = _selected_model(experiment)
    output = Path(output_directory).resolve()
    with MappedFeatureDataset(dataset_directory, verify_digest=True) as dataset:
        expected = experiment.get("input")
        _require(isinstance(expected, dict) and
                 expected.get("feature_binary_sha256") == dataset.artifact.binary_digest and
                 expected.get("feature_manifest_sha256") == dataset.artifact.manifest_digest,
                 "pattern experiment was trained on a different feature artifact")
        training = experiment.get("training")
        _require(isinstance(training, dict) and
                 training.get("test_policy") == "excluded_from_fit_selection_and_metrics",
                 "pattern experiment did not preserve the sealed test split")
        _require(expected.get("split_counts", {}).get("test_held_out") ==
                 dataset.artifact.split_counts[2],
                 "pattern experiment test count differs from the feature artifact")

        reserve_report_directory(output, SCHEMA, error_type=PatternEvaluationError)
        test = dataset.materialize("test")
        unfiltered_test_records = test.records
        exclusions: list[dict[str, Any]] = []
        known_keys: list[np.ndarray] = []
        for exclusion_directory in exclude_dataset_directories:
            with MappedFeatureDataset(exclusion_directory, verify_digest=True) as exclusion:
                known_keys.append(_canonical_keys(exclusion))
                exclusions.append({
                    "corpus_id": exclusion.artifact.manifest["corpus"]["id"],
                    "feature_binary_sha256": exclusion.artifact.binary_digest,
                    "feature_manifest_sha256": exclusion.artifact.manifest_digest,
                    "records": exclusion.artifact.record_count,
                })
        excluded_records = 0
        if known_keys:
            known = np.unique(np.concatenate(known_keys))
            overlap = _known_key_mask(_canonical_keys(dataset, test.indices), known)
            excluded_records = int(overlap.sum())
            test = subset_batch(test, ~overlap)
        _require(test.records > 0, "sealed test split is empty after overlap exclusion")
        closure_metrics = model_metrics(
            test, np.zeros(test.records, dtype=np.float64), 8,
            error_type=PatternEvaluationError)
        gain_metrics = model_metrics(
            test, _predict_gain_control(test, experiment), 8,
            error_type=PatternEvaluationError)
        selected_metrics = model_metrics(
            test, _predict_model(test, experiment, model), 8,
            error_type=PatternEvaluationError)
        quantized_prediction = _predict_quantized_model(test, model)
        quantized_metrics = (model_metrics(
            test, quantized_prediction, 8, error_type=PatternEvaluationError)
                             if quantized_prediction is not None else None)
        closure_overall = closure_metrics["overall"]
        gain_overall = gain_metrics["overall"]
        model_overall = selected_metrics["overall"]
        report = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "input": {
                "corpus_id": dataset.artifact.manifest["corpus"]["id"],
                "feature_binary_sha256": dataset.artifact.binary_digest,
                "feature_manifest_sha256": dataset.artifact.manifest_digest,
                "experiment_report_sha256": sha256_file(experiment_root / "report.json"),
                "test_records_before_exclusion": unfiltered_test_records,
                "development_overlap_records_excluded": excluded_records,
                "test_records": test.records,
                "excluded_datasets": exclusions,
            },
            "evaluation": {
                "split": "test",
                "selection": "preselected_by_training_report_validation_metrics",
                "model": model["name"],
                "two_ply_closure": closure_metrics,
                "gain_control": gain_metrics,
                "model_metrics": selected_metrics,
                "quantized_model_metrics": quantized_metrics,
                "comparison": {
                    "model_vs_two_ply_closure_mae_reduction_percent": _reduction(
                        closure_overall["mae"], model_overall["mae"]
                    ),
                    "model_vs_gain_mae_reduction_percent": _reduction(
                        gain_overall["mae"], model_overall["mae"]
                    ),
                    "model_vs_two_ply_closure_rmse_reduction_percent": _reduction(
                        closure_overall["rmse"], model_overall["rmse"]
                    ),
                    "model_vs_gain_rmse_reduction_percent": _reduction(
                        gain_overall["rmse"], model_overall["rmse"]
                    ),
                },
            },
            "provenance": {
                "evaluation_source_sha256": training_source_digest(),
                "git": git_provenance(),
                "runtime": runtime_provenance(torch, torch.device("cpu")),
            },
        }
        complete_json_report(output, SCHEMA, report, error_type=PatternEvaluationError)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="the same completed feature artifact used for training")
    parser.add_argument("--experiment", type=Path, required=True,
                        help="completed pattern experiment with a validation-selected model")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new create-only sealed-test evaluation directory")
    parser.add_argument("--exclude-dataset", type=Path, action="append", default=[],
                        help="authenticated development artifact whose canonical keys are excluded")
    arguments = parser.parse_args()
    try:
        report = run_pattern_evaluation(
            arguments.dataset,
            arguments.experiment,
            arguments.output_dir,
            exclude_dataset_directories=arguments.exclude_dataset,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"pattern evaluation failed: {error}", file=sys.stderr)
        return 1
    metrics = report["evaluation"]["model_metrics"]["overall"]
    quantized = report["evaluation"]["quantized_model_metrics"]
    quantized_text = (f" quantized_test_mae={quantized['overall']['mae']:.6f}"
                      if quantized is not None else "")
    print(
        "pattern_test_evaluation_complete"
        f" model={report['evaluation']['model']}"
        f" records={report['input']['test_records']}"
        f" test_mae={metrics['mae']:.6f}"
        f" test_rmse={metrics['rmse']:.6f}"
        f" test_sign_accuracy={metrics['sign_accuracy']:.6f}"
        f"{quantized_text}"
        f" output_dir={Path(arguments.output_dir).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
