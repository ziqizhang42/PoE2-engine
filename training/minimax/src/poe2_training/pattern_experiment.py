"""Train reversal-tied phase-aware line-pattern residual models."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .model_metrics import (
    DEFAULT_RIDGE_LAMBDAS,
    fit_ridge,
    model_comparison,
    model_metrics,
)
from .dataset import MappedFeatureDataset, NumpyFeatureBatch
from .pattern_suites import SUITE_NAMES, pattern_suite
from .patterns import REVERSAL_ORBITS, line_pattern_counts, phase_basis
from .quantization import (
    fold_pattern_model,
    predict_folded,
    predict_quantized,
    quantize_pattern_model,
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
from .shared import (
    complete_json_report,
    git_provenance,
    open_json_report,
    reserve_report_directory,
    runtime_provenance,
    training_source_digest,
)


SCHEMA = "poe2-minimax-pattern-experiment"
SCHEMA_VERSION = 1
COMPLETE_HEADER = SCHEMA
GAIN_KNOTS = (0, 12, 24, 36, 49)
FROZEN_PATTERN_GAIN_MODEL_ID = "frozen_c_line_0_28_49_gain_phase_5_huber8"
FROZEN_PATTERN_GAIN_FRACTIONAL_BITS = 5


class PatternExperimentError(ValueError):
    """Raised when a pattern experiment cannot be completed or authenticated."""


@dataclass(frozen=True)
class ModelConfig:
    name: str
    line_knots: tuple[int, ...]
    gain_knots: tuple[int, ...] | None
    loss: str
    huber_delta: float
    l2: float
    learning_rate: float = 0.05
    max_steps: int = 2000
    evaluation_interval: int = 25
    patience_steps: int = 500


@dataclass(frozen=True)
class PatternFit:
    phase_weights: np.ndarray
    line_weights: np.ndarray
    gain_weights: np.ndarray | None
    train_prediction: np.ndarray
    validation_prediction: np.ndarray
    best_step: int
    trace: tuple[dict[str, float | int], ...]


def default_suite() -> tuple[ModelConfig, ...]:
    line_knots = ((0,), (0, 49), (0, 24, 49), (0, 12, 24, 36, 49))
    configs = [
        ModelConfig(
            name=f"line_phase_{len(knots)}_mse",
            line_knots=knots,
            gain_knots=None,
            loss="mse",
            huber_delta=0.0,
            l2=1.0e-5,
        )
        for knots in line_knots
    ]
    configs.extend(
        ModelConfig(
            name=f"line_phase_{len(knots)}_gain_phase_5_mse",
            line_knots=knots,
            gain_knots=GAIN_KNOTS,
            loss="mse",
            huber_delta=0.0,
            l2=1.0e-5,
        )
        for knots in line_knots
    )
    configs.extend((
        ModelConfig(
            name="line_phase_2_gain_phase_5_huber8",
            line_knots=(0, 49),
            gain_knots=GAIN_KNOTS,
            loss="huber",
            huber_delta=8.0,
            l2=1.0e-4,
        ),
        ModelConfig(
            name="line_phase_3_gain_phase_5_huber8",
            line_knots=(0, 24, 49),
            gain_knots=GAIN_KNOTS,
            loss="huber",
            huber_delta=8.0,
            l2=1.0e-4,
        ),
    ))
    return tuple(configs)


def frozen_pattern_gain_suite() -> tuple[ModelConfig, ...]:
    """Return the fixed model and quantization contract accepted by the engine."""
    return (ModelConfig(
        name=FROZEN_PATTERN_GAIN_MODEL_ID,
        line_knots=(0, 28, 49),
        gain_knots=GAIN_KNOTS,
        loss="huber",
        huber_delta=8.0,
        l2=1.0e-4,
    ),)


def line_pattern_audit_suite() -> tuple[ModelConfig, ...]:
    """Compare a bounded set of line-only pattern variants."""
    candidates = (
        ("line_0_24_49_mse", (0, 24, 49), "mse", 0.0, 1.0e-5),
        ("line_0_24_49_huber8", (0, 24, 49), "huber", 8.0, 1.0e-4),
        ("line_0_28_49_mse", (0, 28, 49), "mse", 0.0, 1.0e-5),
        ("line_0_28_49_huber8", (0, 28, 49), "huber", 8.0, 1.0e-4),
        ("line_0_32_49_huber8", (0, 32, 49), "huber", 8.0, 1.0e-4),
        ("line_0_16_32_49_huber8", (0, 16, 32, 49), "huber", 8.0, 1.0e-4),
    )
    return tuple(ModelConfig(
        name=name,
        line_knots=knots,
        gain_knots=None,
        loss=loss,
        huber_delta=delta,
        l2=l2,
    ) for name, knots, loss, delta, l2 in candidates)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PatternExperimentError(message)


def open_pattern_report(directory: Path | str) -> dict[str, Any]:
    """Authenticate a completed pattern experiment report."""
    report, _ = open_json_report(
        directory, schema=SCHEMA, schema_version=SCHEMA_VERSION,
        error_type=PatternExperimentError,
    )
    return report


def _standardize_counts(train: np.ndarray, validation: np.ndarray) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1.0e-6] = 1.0
    train -= center
    train /= scale
    validation -= center
    validation /= scale
    return train, validation, center, scale


def _phase_means(batch: NumpyFeatureBatch) -> np.ndarray:
    result = np.zeros(len(PHASE_FEATURE_NAMES), dtype=np.float32)
    for ply in range(result.size):
        selected = batch.plys == ply
        if bool(selected.any()):
            result[ply] = float(batch.residuals[selected].mean())
    return result


def _fit_gain_control(
    train: NumpyFeatureBatch,
    validation: NumpyFeatureBatch,
    train_gain: np.ndarray,
    validation_gain: np.ndarray,
    device: Any,
    ridge_lambdas: Sequence[float],
) -> tuple[dict[str, Any], np.ndarray]:
    train_phase = phase_features(train.plys)
    validation_phase = phase_features(validation.plys)
    train_interactions = phase_interactions(
        train_gain, phase_interpolation(train.plys, GAIN_KNOTS)
    )
    validation_interactions = phase_interactions(
        validation_gain, phase_interpolation(validation.plys, GAIN_KNOTS)
    )
    train_design = np.ascontiguousarray(
        np.concatenate((train_phase, train_interactions), axis=1), dtype=np.float64
    )
    validation_design = np.ascontiguousarray(
        np.concatenate((validation_phase, validation_interactions), axis=1), dtype=np.float64
    )
    fit = fit_ridge(
        train_design,
        np.asarray(train.residuals, dtype=np.float64),
        validation_design,
        np.asarray(validation.residuals, dtype=np.float64),
        ridge_lambdas,
        device,
        error_type=PatternExperimentError,
    )
    model = {
        "name": "gain_summary_phase_5_control",
        "parameters": int(fit.weights.size),
        "phase_knots": list(GAIN_KNOTS),
        "selected_ridge_lambda": fit.selected_lambda,
        "ridge_path": list(fit.path),
        "weights": fit.weights.tolist(),
        "train": model_metrics(train, train_design @ fit.weights, 8,
                               error_type=PatternExperimentError),
        "validation": model_metrics(validation, validation_design @ fit.weights, 8,
                                    error_type=PatternExperimentError),
    }
    return model, fit.weights.astype(np.float32)


def _train_model(
    config: ModelConfig,
    train: NumpyFeatureBatch,
    validation: NumpyFeatureBatch,
    train_lines: Any,
    validation_lines: Any,
    train_gain: Any,
    validation_gain: Any,
    train_target: Any,
    validation_target: Any,
    train_plys: Any,
    validation_plys: Any,
    gain_initial: np.ndarray,
    device: Any,
) -> PatternFit:
    import torch
    import torch.nn.functional as functional

    train_line_phase = torch.as_tensor(
        phase_basis(train.plys, config.line_knots), device=device
    )
    validation_line_phase = torch.as_tensor(
        phase_basis(validation.plys, config.line_knots), device=device
    )
    line_weights = torch.zeros(
        (len(config.line_knots), train_lines.shape[1]),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    if config.gain_knots is None:
        phase_weights = torch.tensor(
            _phase_means(train), dtype=torch.float32, device=device, requires_grad=True
        )
        gain_weights = None
        train_gain_phase = validation_gain_phase = None
        parameters = [line_weights, phase_weights]
    else:
        _require(config.gain_knots == GAIN_KNOTS, "only the calibrated five-knot gain basis is supported")
        phase_weights = torch.tensor(
            gain_initial[:len(PHASE_FEATURE_NAMES)],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        gain_weights = torch.tensor(
            gain_initial[len(PHASE_FEATURE_NAMES):].reshape(len(GAIN_KNOTS), -1),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        train_gain_phase = torch.as_tensor(phase_basis(train.plys, GAIN_KNOTS), device=device)
        validation_gain_phase = torch.as_tensor(
            phase_basis(validation.plys, GAIN_KNOTS), device=device
        )
        parameters = [line_weights, phase_weights, gain_weights]

    def predict(lines: Any, gains: Any, plys: Any, line_phase: Any,
                gain_phase: Any | None) -> Any:
        result = phase_weights[plys] + ((lines @ line_weights.T) * line_phase).sum(dim=1)
        if gain_weights is not None:
            result = result + ((gains @ gain_weights.T) * gain_phase).sum(dim=1)
        return result

    def data_loss(prediction: Any, target: Any) -> Any:
        if config.loss == "mse":
            return (prediction - target).square().mean()
        if config.loss == "huber":
            return functional.huber_loss(prediction, target, delta=config.huber_delta)
        raise PatternExperimentError(f"unknown pattern loss: {config.loss}")

    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    best_key = (math.inf, math.inf, math.inf)
    best_step = 0
    best_state: tuple[Any, Any, Any | None] | None = None
    trace: list[dict[str, float | int]] = []
    for step in range(config.max_steps + 1):
        if step > 0:
            optimizer.zero_grad(set_to_none=True)
            prediction = predict(
                train_lines, train_gain, train_plys, train_line_phase, train_gain_phase
            )
            measured_loss = data_loss(prediction, train_target)
            penalty = line_weights.square().sum()
            if gain_weights is not None:
                penalty = penalty + gain_weights.square().sum()
            loss = measured_loss + config.l2 * penalty
            loss.backward()
            optimizer.step()

        if step % config.evaluation_interval == 0:
            with torch.no_grad():
                train_checkpoint = predict(
                    train_lines,
                    train_gain,
                    train_plys,
                    train_line_phase,
                    train_gain_phase,
                )
                validation_checkpoint = predict(
                    validation_lines,
                    validation_gain,
                    validation_plys,
                    validation_line_phase,
                    validation_gain_phase,
                )
                training_loss = float(data_loss(train_checkpoint, train_target).cpu())
                validation_loss = float(
                    data_loss(validation_checkpoint, validation_target).cpu())
                error = validation_checkpoint - validation_target
                mae = float(error.abs().mean().cpu())
                rmse = float(error.square().mean().sqrt().cpu())
            key = (mae, rmse, float(step))
            trace.append({
                "step": step,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "validation_mae": mae,
                "validation_rmse": rmse,
            })
            if key < best_key:
                best_key = key
                best_step = step
                best_state = (
                    phase_weights.detach().clone(),
                    line_weights.detach().clone(),
                    gain_weights.detach().clone() if gain_weights is not None else None,
                )
            if step - best_step >= config.patience_steps:
                break

    if best_state is None:
        raise PatternExperimentError("pattern training produced no validation checkpoint")
    with torch.no_grad():
        phase_weights.copy_(best_state[0])
        line_weights.copy_(best_state[1])
        if gain_weights is not None:
            gain_weights.copy_(best_state[2])
        train_prediction = predict(
            train_lines, train_gain, train_plys, train_line_phase, train_gain_phase
        ).cpu().numpy()
        validation_prediction = predict(
            validation_lines,
            validation_gain,
            validation_plys,
            validation_line_phase,
            validation_gain_phase,
        ).cpu().numpy()
    return PatternFit(
        phase_weights=np.ascontiguousarray(best_state[0].cpu().numpy()),
        line_weights=np.ascontiguousarray(best_state[1].cpu().numpy()),
        gain_weights=(np.ascontiguousarray(best_state[2].cpu().numpy())
                      if best_state[2] is not None else None),
        train_prediction=np.ascontiguousarray(train_prediction),
        validation_prediction=np.ascontiguousarray(validation_prediction),
        best_step=best_step,
        trace=tuple(trace),
    )


def _quantization_report(
    validation: NumpyFeatureBatch,
    feature_metadata: dict[str, Any],
    model: dict[str, Any],
    float_prediction: np.ndarray,
) -> dict[str, Any]:
    folded = fold_pattern_model({"features": feature_metadata}, model)
    folded_prediction = predict_folded(validation, folded)
    maximum_difference = float(np.max(np.abs(folded_prediction - float_prediction)))
    _require(np.allclose(folded_prediction, float_prediction, rtol=2.0e-5, atol=2.0e-4),
             "folding standardization changed pattern predictions")

    candidates: list[tuple[tuple[float, float, int], Any, dict[str, Any]]] = []
    path: list[dict[str, Any]] = []
    for fractional_bits in range(4, 11):
        try:
            quantized = quantize_pattern_model(folded, fractional_bits)
        except OverflowError:
            path.append({"fractional_bits": fractional_bits, "status": "int16_overflow"})
            continue
        prediction = predict_quantized(validation, quantized)
        metrics = model_metrics(validation, prediction, 8,
                                error_type=PatternExperimentError)
        overall = metrics["overall"]
        entry = {
            "fractional_bits": fractional_bits,
            "status": "valid",
            "validation_mae": overall["mae"],
            "validation_rmse": overall["rmse"],
            "validation_sign_accuracy": overall["sign_accuracy"],
        }
        path.append(entry)
        candidates.append((
            (overall["mae"], overall["rmse"], fractional_bits),
            quantized,
            metrics,
        ))
    _require(bool(candidates), "no fixed-point scale fits the pattern tables")
    if model.get("name") == FROZEN_PATTERN_GAIN_MODEL_ID:
        deployment = [
            candidate for candidate in candidates
            if candidate[1].fractional_bits == FROZEN_PATTERN_GAIN_FRACTIONAL_BITS
        ]
        _require(len(deployment) == 1,
                 "frozen pattern/gain model does not fit the deployment scale")
        _, selected, selected_metrics = deployment[0]
        selection = "fixed_scale_32_engine_contract"
    else:
        candidates.sort(key=lambda candidate: candidate[0])
        _, selected, selected_metrics = candidates[0]
        selection = "validation_mae_then_rmse_then_fractional_bits"
    return {
        "definition": "int16-tables-int32-intercept-power-of-two-v1",
        "selection": selection,
        "folded_maximum_absolute_difference": maximum_difference,
        "path": path,
        "fractional_bits": selected.fractional_bits,
        "scale": selected.scale,
        "intercept": selected.intercept.tolist(),
        "line_weights": selected.line_weights.tolist(),
        "gain_weights": (selected.gain_weights.tolist()
                         if selected.gain_weights is not None else None),
        "validation": selected_metrics,
    }


def run_pattern_experiment(
    dataset_directory: Path | str,
    output_directory: Path | str,
    *,
    requested_device: str = "cuda",
    seed: int = 20260818,
    configs: Sequence[ModelConfig] | None = None,
) -> dict[str, Any]:
    """Train configured model candidates using train/validation only."""
    import torch

    if seed < 0:
        raise PatternExperimentError("seed must be nonnegative")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise PatternExperimentError("ROCm/CUDA was requested but unavailable")
    device = torch.device("cuda" if requested_device == "cuda" else "cpu")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    selected_configs = tuple(configs or default_suite())
    _require(bool(selected_configs), "pattern experiment requires at least one model")

    output = Path(output_directory).resolve()
    with MappedFeatureDataset(dataset_directory, verify_digest=True) as dataset:
        reserve_report_directory(output, SCHEMA, error_type=PatternExperimentError)
        train = dataset.materialize("train")
        validation = dataset.materialize("validation")
        _require(train.records > 0 and validation.records > 0,
                 "train and validation splits must be nonempty")

        raw_train_lines = line_pattern_counts(train.line_patterns)
        raw_validation_lines = line_pattern_counts(validation.line_patterns)
        train_line_values, validation_line_values, line_center, line_scale = _standardize_counts(
            raw_train_lines, raw_validation_lines
        )
        raw_train_gain = gain_summaries(train.own_gains, train.opponent_gains)
        raw_validation_gain = gain_summaries(validation.own_gains, validation.opponent_gains)
        gain_standardization = fit_standardization(raw_train_gain)
        train_gain_values = gain_standardization.transform(raw_train_gain)
        validation_gain_values = gain_standardization.transform(raw_validation_gain)
        feature_metadata = {
            "pattern_definition": "length-shared-reversal-tied-ternary-line-counts-v1",
            "raw_pattern_count": int(REVERSAL_ORBITS.raw_to_orbit.size),
            "reversal_orbit_count": REVERSAL_ORBITS.count,
            "raw_to_orbit": REVERSAL_ORBITS.raw_to_orbit.tolist(),
            "orbit_lengths": REVERSAL_ORBITS.orbit_lengths.tolist(),
            "line_center": line_center.tolist(),
            "line_scale": line_scale.tolist(),
            "gain_names": list(GAIN_SUMMARY_NAMES),
            "gain_center": gain_standardization.center.tolist(),
            "gain_scale": gain_standardization.scale.tolist(),
        }

        gain_control, gain_initial = _fit_gain_control(
            train,
            validation,
            train_gain_values,
            validation_gain_values,
            device,
            DEFAULT_RIDGE_LAMBDAS,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

        train_lines = torch.as_tensor(train_line_values, device=device)
        validation_lines = torch.as_tensor(validation_line_values, device=device)
        train_gain_tensor = torch.as_tensor(
            train_gain_values.astype(np.float32), device=device
        )
        validation_gain_tensor = torch.as_tensor(
            validation_gain_values.astype(np.float32), device=device
        )
        train_target = torch.as_tensor(train.residuals, dtype=torch.float32, device=device)
        validation_target = torch.as_tensor(
            validation.residuals, dtype=torch.float32, device=device
        )
        train_plys = torch.as_tensor(train.plys.astype(np.int64), device=device)
        validation_plys = torch.as_tensor(validation.plys.astype(np.int64), device=device)

        models: list[dict[str, Any]] = []
        for config in selected_configs:
            fit = _train_model(
                config,
                train,
                validation,
                train_lines,
                validation_lines,
                train_gain_tensor,
                validation_gain_tensor,
                train_target,
                validation_target,
                train_plys,
                validation_plys,
                gain_initial,
                device,
            )
            model = {
                "name": config.name,
                "definition": "reversal-tied-phase-line-residual-v1",
                "config": asdict(config),
                "parameters": int(
                    fit.phase_weights.size + fit.line_weights.size +
                    (fit.gain_weights.size if fit.gain_weights is not None else 0)
                ),
                "best_step": fit.best_step,
                "phase_weights": fit.phase_weights.tolist(),
                "line_weights": fit.line_weights.tolist(),
                "gain_weights": fit.gain_weights.tolist() if fit.gain_weights is not None else None,
                "trace": list(fit.trace),
                "train": model_metrics(train, fit.train_prediction, 8,
                                       error_type=PatternExperimentError),
                "validation": model_metrics(validation, fit.validation_prediction, 8,
                                            error_type=PatternExperimentError),
            }
            model["quantization"] = _quantization_report(
                validation, feature_metadata, model, fit.validation_prediction
            )
            models.append(model)
            print(
                "pattern_model_complete"
                f" name={config.name}"
                f" parameters={model['parameters']}"
                f" best_step={fit.best_step}"
                f" validation_mae={model['validation']['overall']['mae']:.6f}"
                f" validation_rmse={model['validation']['overall']['rmse']:.6f}",
                flush=True,
            )

        best = min(
            models,
            key=lambda model: (
                model["validation"]["overall"]["mae"],
                model["validation"]["overall"]["rmse"],
                model["parameters"],
            ),
        )
        manifest = dataset.artifact.manifest
        report = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "input": {
                "corpus_id": manifest["corpus"]["id"],
                "corpus_digest": manifest["corpus"]["digest"],
                "feature_binary_sha256": dataset.artifact.binary_digest,
                "feature_manifest_sha256": dataset.artifact.manifest_digest,
                "records": dataset.artifact.record_count,
                "split_counts": {
                    "train": dataset.artifact.split_counts[0],
                    "validation": dataset.artifact.split_counts[1],
                    "test_held_out": dataset.artifact.split_counts[2],
                },
            },
            "training": {
                "seed": seed,
                "target": "teacher_value_minus_two_ply_closure_value",
                "selection": "validation_mae_then_rmse_then_parameter_count",
                "test_policy": "excluded_from_fit_selection_and_metrics",
                "configs": [asdict(config) for config in selected_configs],
            },
            "features": feature_metadata,
            "provenance": {
                "training_source_sha256": training_source_digest(),
                "git": git_provenance(),
                "runtime": runtime_provenance(torch, device),
            },
            "gain_control": gain_control,
            "models": models,
            "selection": {
                "best_model": best["name"],
                "best_validation_mae": best["validation"]["overall"]["mae"],
                "best_validation_rmse": best["validation"]["overall"]["rmse"],
                "best_vs_gain_control": model_comparison(gain_control, best),
            },
        }
        report["attachments"] = {
            "training_metrics": write_training_metrics(
                output, report, error_type=PatternExperimentError)
        }
        complete_json_report(output, SCHEMA, report, error_type=PatternExperimentError)
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="completed minimax feature artifact")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new create-only pattern experiment directory")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--suite",
                        choices=SUITE_NAMES,
                        default="default",
                        help="model suite; frozen-pattern-gain is the fixed deployment contract")
    arguments = parser.parse_args()
    configs = pattern_suite(arguments.suite)
    try:
        report = run_pattern_experiment(
            arguments.dataset,
            arguments.output_dir,
            requested_device=arguments.device,
            seed=arguments.seed,
            configs=configs,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"pattern experiment failed: {error}", file=sys.stderr)
        return 1
    print(
        "pattern_experiment_complete"
        f" best_model={report['selection']['best_model']}"
        f" validation_mae={report['selection']['best_validation_mae']:.6f}"
        f" validation_rmse={report['selection']['best_validation_rmse']:.6f}"
        f" test_held_out={report['input']['split_counts']['test_held_out']}"
        f" output_dir={Path(arguments.output_dir).resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
