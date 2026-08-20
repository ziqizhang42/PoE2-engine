"""Train deterministic two-ply-closure residual baselines without the test split."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dataset import MappedFeatureDataset
from .model_metrics import (
    DEFAULT_RIDGE_LAMBDAS,
    RidgeFit,
    fit_ridge,
    model_comparison,
    model_metrics,
)
from .shared import (
    complete_json_report,
    git_provenance,
    open_json_report,
    reserve_report_directory,
    runtime_provenance,
    training_source_digest,
)
from .summaries import (
    GAIN_SUMMARY_NAMES,
    PHASE_FEATURE_NAMES,
    fit_standardization,
    gain_summaries,
    phase_features,
    phase_interactions,
    phase_interpolation,
)
from .training_visualization import write_training_metrics


EXPERIMENT_SCHEMA = "poe2-minimax-baseline-experiment"
EXPERIMENT_SCHEMA_VERSION = 1
COMPLETE_HEADER = "poe2-minimax-baseline-experiment"
GAIN_PHASE_KNOTS = (
    (0, 49),
    (0, 24, 49),
    (0, 12, 24, 36, 49),
)


class BaselineExperimentError(ValueError):
    """Raised when a baseline experiment cannot be run or authenticated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineExperimentError(message)


def _device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected = requested
    if selected == "cuda" and not torch.cuda.is_available():
        raise BaselineExperimentError("CUDA/ROCm was requested but no device is available")
    return torch.device(selected)


def open_baseline_report(directory: Path | str) -> dict[str, Any]:
    """Authenticate a completed baseline report."""
    report, _ = open_json_report(
        directory, schema=EXPERIMENT_SCHEMA,
        schema_version=EXPERIMENT_SCHEMA_VERSION,
        error_type=BaselineExperimentError,
    )
    return report


def run_baseline_experiment(
    dataset_directory: Path | str,
    output_directory: Path | str,
    *,
    requested_device: str = "auto",
    ridge_lambdas: Sequence[float] = DEFAULT_RIDGE_LAMBDAS,
    close_margin: int = 8,
    seed: int = 20260818,
) -> dict[str, Any]:
    """Run train/validation model selection without fitting or scoring on test records."""
    import torch

    if close_margin < 0:
        raise BaselineExperimentError("close margin must be nonnegative")
    if seed < 0:
        raise BaselineExperimentError("seed must be nonnegative")
    device = _device(torch, requested_device)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    output = Path(output_directory).resolve()
    with MappedFeatureDataset(dataset_directory, verify_digest=True) as dataset:
        reserve_report_directory(
            output, EXPERIMENT_SCHEMA, error_type=BaselineExperimentError)
        train = dataset.materialize("train")
        validation = dataset.materialize("validation")
        if train.records == 0 or validation.records == 0:
            raise BaselineExperimentError("both train and validation splits must be nonempty")

        train_phase = phase_features(train.plys)
        validation_phase = phase_features(validation.plys)
        train_gain = gain_summaries(train.own_gains, train.opponent_gains)
        validation_gain = gain_summaries(validation.own_gains, validation.opponent_gains)
        standardization = fit_standardization(train_gain)
        train_gain_standardized = standardization.transform(train_gain)
        validation_gain_standardized = standardization.transform(validation_gain)
        train_combined = np.ascontiguousarray(
            np.concatenate((train_phase, train_gain_standardized), axis=1),
            dtype=np.float64,
        )
        validation_combined = np.ascontiguousarray(
            np.concatenate((validation_phase, validation_gain_standardized), axis=1),
            dtype=np.float64,
        )
        train_target = np.ascontiguousarray(train.residuals, dtype=np.float64)
        validation_target = np.ascontiguousarray(validation.residuals, dtype=np.float64)

        phase_fit = fit_ridge(
            train_phase, train_target, validation_phase, validation_target,
            ridge_lambdas, device, error_type=BaselineExperimentError,
        )
        gain_fit = fit_ridge(
            train_combined, train_target, validation_combined, validation_target,
            ridge_lambdas, device, error_type=BaselineExperimentError,
        )

        phase_aware_fits: list[tuple[tuple[int, ...], np.ndarray, np.ndarray, RidgeFit]] = []
        for knots in GAIN_PHASE_KNOTS:
            train_interactions = phase_interactions(
                train_gain_standardized, phase_interpolation(train.plys, knots)
            )
            validation_interactions = phase_interactions(
                validation_gain_standardized, phase_interpolation(validation.plys, knots)
            )
            train_design = np.ascontiguousarray(
                np.concatenate((train_phase, train_interactions), axis=1), dtype=np.float64
            )
            validation_design = np.ascontiguousarray(
                np.concatenate((validation_phase, validation_interactions), axis=1),
                dtype=np.float64,
            )
            fit = fit_ridge(
                train_design, train_target, validation_design, validation_target,
                ridge_lambdas, device, error_type=BaselineExperimentError,
            )
            phase_aware_fits.append((knots, train_design, validation_design, fit))

        zero_train = np.zeros(train.records, dtype=np.float64)
        zero_validation = np.zeros(validation.records, dtype=np.float64)
        phase_train_prediction = train_phase @ phase_fit.weights
        phase_validation_prediction = validation_phase @ phase_fit.weights
        gain_train_prediction = train_combined @ gain_fit.weights
        gain_validation_prediction = validation_combined @ gain_fit.weights

        closure_model = {
            "name": "two_ply_closure",
            "definition": "zero-residual-two-ply-closure-control-v1",
            "parameters": 0,
            "train": model_metrics(train, zero_train, close_margin,
                                   error_type=BaselineExperimentError),
            "validation": model_metrics(validation, zero_validation, close_margin,
                                        error_type=BaselineExperimentError),
        }
        phase_model = {
            "name": "phase",
            "definition": "one-hot-ply-residual-ridge-v1",
            "parameters": len(PHASE_FEATURE_NAMES),
            "feature_names": list(PHASE_FEATURE_NAMES),
            "weights": phase_fit.weights.tolist(),
            "selected_ridge_lambda": phase_fit.selected_lambda,
            "ridge_path": list(phase_fit.path),
            "train": model_metrics(train, phase_train_prediction, close_margin,
                                   error_type=BaselineExperimentError),
            "validation": model_metrics(validation, phase_validation_prediction, close_margin,
                                        error_type=BaselineExperimentError),
        }
        gain_model = {
            "name": "gain_summary",
            "definition": "one-hot-ply-plus-closure-gain-summary-residual-ridge-v1",
            "parameters": len(PHASE_FEATURE_NAMES) + len(GAIN_SUMMARY_NAMES),
            "feature_names": [*PHASE_FEATURE_NAMES, *GAIN_SUMMARY_NAMES],
            "gain_summary_names": list(GAIN_SUMMARY_NAMES),
            "gain_center": standardization.center.tolist(),
            "gain_scale": standardization.scale.tolist(),
            "weights": gain_fit.weights.tolist(),
            "selected_ridge_lambda": gain_fit.selected_lambda,
            "ridge_path": list(gain_fit.path),
            "train": model_metrics(train, gain_train_prediction, close_margin,
                                   error_type=BaselineExperimentError),
            "validation": model_metrics(validation, gain_validation_prediction, close_margin,
                                        error_type=BaselineExperimentError),
        }
        phase_aware_models: list[dict[str, Any]] = []
        for knots, train_design, validation_design, fit in phase_aware_fits:
            knot_names = [
                f"{name}@ply_{knot:02d}"
                for knot in knots
                for name in GAIN_SUMMARY_NAMES
            ]
            model = {
                "name": f"gain_summary_phase_{len(knots)}",
                "definition": (
                    "one-hot-ply-plus-interpolated-closure-gain-summary-residual-ridge-v1"
                ),
                "phase_knots": list(knots),
                "parameters": len(PHASE_FEATURE_NAMES) + len(knot_names),
                "feature_names": [*PHASE_FEATURE_NAMES, *knot_names],
                "gain_summary_names": list(GAIN_SUMMARY_NAMES),
                "gain_center": standardization.center.tolist(),
                "gain_scale": standardization.scale.tolist(),
                "weights": fit.weights.tolist(),
                "selected_ridge_lambda": fit.selected_lambda,
                "ridge_path": list(fit.path),
                "train": model_metrics(
                    train, train_design @ fit.weights, close_margin,
                    error_type=BaselineExperimentError),
                "validation": model_metrics(
                    validation, validation_design @ fit.weights, close_margin,
                    error_type=BaselineExperimentError,
                ),
            }
            phase_aware_models.append(model)

        manifest = dataset.artifact.manifest
        report = {
            "schema": EXPERIMENT_SCHEMA,
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "input": {
                "corpus_id": manifest["corpus"]["id"],
                "corpus_digest": manifest["corpus"]["digest"],
                "feature_binary_sha256": dataset.artifact.binary_digest,
                "feature_manifest_sha256": dataset.artifact.manifest_digest,
                "feature_definition": manifest["features"]["definition"],
                "records": dataset.artifact.record_count,
                "split_counts": {
                    "train": dataset.artifact.split_counts[0],
                    "validation": dataset.artifact.split_counts[1],
                    "test_held_out": dataset.artifact.split_counts[2],
                },
            },
            "training": {
                "target": "teacher_value_minus_two_ply_closure_value",
                "selection_metric": "validation_mae_then_rmse_then_lambda",
                "ridge_lambdas": [float(value) for value in ridge_lambdas],
                "close_margin": close_margin,
                "seed": seed,
                "duplicate_weighting": "one_vote_per_canonical_position",
                "test_policy": "excluded_from_fit_selection_and_metrics",
            },
            "provenance": {
                "training_source_sha256": training_source_digest(),
                "git": git_provenance(),
                "runtime": runtime_provenance(torch, device),
            },
            "models": [closure_model, phase_model, gain_model, *phase_aware_models],
            "comparisons": {
                "phase_vs_two_ply_closure": model_comparison(closure_model, phase_model),
                "gain_summary_vs_two_ply_closure": model_comparison(
                    closure_model, gain_model
                ),
                "gain_summary_vs_phase": model_comparison(phase_model, gain_model),
                **{
                    f"{model['name']}_vs_gain_summary": model_comparison(gain_model, model)
                    for model in phase_aware_models
                },
            },
        }
        report["attachments"] = {
            "training_metrics": write_training_metrics(
                output, report, error_type=BaselineExperimentError)
        }
        complete_json_report(
            output, EXPERIMENT_SCHEMA, report, error_type=BaselineExperimentError)
        return report


def _parse_lambdas(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("ridge lambdas must be comma-separated numbers") from error
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("ridge lambdas must be finite and positive")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="completed minimax feature artifact")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new create-only experiment directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="PyTorch compute device (ROCm uses the cuda device name)")
    parser.add_argument("--ridge-lambdas", type=_parse_lambdas,
                        default=DEFAULT_RIDGE_LAMBDAS,
                        help="comma-separated ridge path")
    parser.add_argument("--close-margin", type=int, default=8,
                        help="absolute doubled margin considered close")
    parser.add_argument("--seed", type=int, default=20260818,
                        help="deterministic PyTorch seed")
    arguments = parser.parse_args()

    try:
        report = run_baseline_experiment(
            arguments.dataset,
            arguments.output_dir,
            requested_device=arguments.device,
            ridge_lambdas=arguments.ridge_lambdas,
            close_margin=arguments.close_margin,
            seed=arguments.seed,
        )
    except (BaselineExperimentError, OSError, RuntimeError, ValueError) as error:
        print(f"baseline experiment failed: {error}", file=sys.stderr)
        return 1

    models = {model["name"]: model for model in report["models"]}
    best = min(report["models"], key=lambda model: model["validation"]["overall"]["mae"])
    print(
        "baseline_experiment_complete"
        f" train={report['input']['split_counts']['train']}"
        f" validation={report['input']['split_counts']['validation']}"
        f" test_held_out={report['input']['split_counts']['test_held_out']}"
        " two_ply_closure_validation_mae="
        f"{models['two_ply_closure']['validation']['overall']['mae']:.6f}"
        f" phase_validation_mae={models['phase']['validation']['overall']['mae']:.6f}"
        f" gain_validation_mae={models['gain_summary']['validation']['overall']['mae']:.6f}"
        f" best_model={best['name']}"
        f" best_validation_mae={best['validation']['overall']['mae']:.6f}"
        f" output_dir={Path(arguments.output_dir).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
