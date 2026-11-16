"""Train reversal-tied phase-aware line-pattern residual models."""

from __future__ import annotations

import argparse
import functools
import math
import sys
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .model_metrics import (
    DEFAULT_RIDGE_LAMBDAS,
    fit_ridge,
    fit_ridge_moments,
    model_comparison,
    model_metrics,
)
from .dataset import MappedFeatureDataset, NumpyFeatureBatch, TrainingSelection
from .memory_guard import GIB, check_memory, training_estimate
from .pattern_suites import SUITE_NAMES, pattern_suite
from .patterns import (
    LINE_COUNT,
    REVERSAL_ORBIT_COUNT,
    REVERSAL_ORBITS,
    line_pattern_counts,
    phase_basis,
)
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
# How the line term is computed: from dense per-record line counts, or by summing the weights
# of each record's 36 patterns. Both optimise the same model; only rounding differs.
TRAINING_METHODS = ("dense", "lookup")
# Occurrences per block of the lookup gradient's fixed-order reduction.
LOOKUP_BLOCK = 256


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


def _phase_means(plys: np.ndarray, residuals: np.ndarray,
                 weights: np.ndarray | None = None) -> np.ndarray:
    result = np.zeros(len(PHASE_FEATURE_NAMES), dtype=np.float32)
    for ply in range(result.size):
        selected = plys == ply
        if bool(selected.any()):
            result[ply] = float(residuals[selected].mean() if weights is None else
                                np.average(residuals[selected], weights=weights[selected]))
    return result


@dataclass(frozen=True)
class _TrainColumns:
    """The narrow per-record columns that metrics and phase means read."""

    plys: np.ndarray
    residuals: np.ndarray
    teacher_values: np.ndarray
    two_ply_closure_values: np.ndarray
    flags: np.ndarray

    @property
    def records(self) -> int:
        return int(self.plys.size)


class _LineMoments:
    """Weighted first and second moments of dense line counts, summed piece by piece.

    Counts are small integers, so the float64 sums are exact for unit weights.
    """

    def __init__(self) -> None:
        self.total = 0.0
        self.first = np.zeros(REVERSAL_ORBIT_COUNT, dtype=np.float64)
        self.second = np.zeros(REVERSAL_ORBIT_COUNT, dtype=np.float64)

    def add(self, orbits: np.ndarray, weights: np.ndarray) -> None:
        rows = np.repeat(np.arange(orbits.shape[0], dtype=np.int64), orbits.shape[1])
        cells, counts = np.unique(rows * REVERSAL_ORBIT_COUNT + orbits.ravel(),
                                  return_counts=True)
        weight = weights[cells // REVERSAL_ORBIT_COUNT]
        orbit = cells % REVERSAL_ORBIT_COUNT
        self.first += np.bincount(orbit, weights=weight * counts, minlength=REVERSAL_ORBIT_COUNT)
        self.second += np.bincount(orbit, weights=weight * np.square(counts),
                                   minlength=REVERSAL_ORBIT_COUNT)
        self.total += float(weights.sum())

    def standardization(self) -> tuple[np.ndarray, np.ndarray]:
        mean = self.first / self.total
        variance = np.maximum(self.second / self.total - np.square(mean), 0.0)
        center = mean.astype(np.float32)
        scale = np.sqrt(variance).astype(np.float32)
        scale[scale < 1.0e-6] = 1.0
        return center, scale


@dataclass(frozen=True)
class _PatternLookup:
    """Each record's 36 pattern IDs, and for training data, a plan for summing gradients.

    The plan lists the records of every occurrence grouped by pattern and padded to whole
    blocks; padding points one past the last record, at a zero gradient row.
    """

    records: int
    flat: Any
    plan: Any | None = None
    block_patterns: Any | None = None

    def sums(self, values: Any) -> Any:
        """Return, for each record and column of values, the sum over its 36 patterns."""
        return _lookup_sum_function().apply(values, self)


@functools.cache
def _lookup_sum_function() -> Any:
    import torch

    class LookupSum(torch.autograd.Function):
        # Deterministic scatter backwards sort every occurrence at every step. The plan groups
        # occurrences once, so a step sums fixed blocks and index_adds only the block sums.
        @staticmethod
        def forward(context: Any, values: Any, lookup: _PatternLookup) -> Any:
            context.lookup = lookup
            return values.index_select(0, lookup.flat).view(
                lookup.records, LINE_COUNT, values.shape[1]).sum(dim=1)

        @staticmethod
        def backward(context: Any, gradient: Any) -> tuple[Any, None]:
            lookup = context.lookup
            if lookup.plan is None:
                raise PatternExperimentError("pattern lookup has no gradient plan")
            columns = gradient.shape[1]
            padded = torch.cat((gradient, gradient.new_zeros((1, columns))))
            blocks = padded.index_select(0, lookup.plan).view(-1, LOOKUP_BLOCK, columns).sum(dim=1)
            result = gradient.new_zeros((REVERSAL_ORBIT_COUNT, columns))
            return result.index_add_(0, lookup.block_patterns, blocks), None

    return LookupSum


def _pattern_lookup(orbits: np.ndarray, device: Any, *, plan: bool) -> _PatternLookup:
    import torch

    records = int(orbits.shape[0])
    flat = orbits.reshape(-1)
    lookup = _PatternLookup(records, torch.as_tensor(flat.astype(np.int32), device=device))
    if not plan:
        return lookup
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=REVERSAL_ORBIT_COUNT)
    blocks = -(-counts // LOOKUP_BLOCK)
    starts = (np.cumsum(blocks) - blocks) * LOOKUP_BLOCK
    rank = np.arange(flat.size) - np.repeat(np.cumsum(counts) - counts, counts)
    padded = np.full(int(blocks.sum()) * LOOKUP_BLOCK, records, dtype=np.int32)
    padded[np.repeat(starts, counts) + rank] = (order // LINE_COUNT).astype(np.int32)
    return _PatternLookup(
        records, lookup.flat, plan=torch.as_tensor(padded, device=device),
        block_patterns=torch.as_tensor(
            np.repeat(np.arange(REVERSAL_ORBIT_COUNT), blocks), device=device))


@dataclass(frozen=True)
class _FitData:
    """One split on the training device: dense line counts, or pattern IDs expanded per piece."""

    plys: np.ndarray
    residuals: np.ndarray
    ply_index: Any
    target: Any
    gains: Any
    lines: Any | None = None
    orbits: Any | None = None
    line_center: Any | None = None
    line_scale: Any | None = None
    piece_records: int | None = None
    lookup: _PatternLookup | None = None
    # Normalized to sum to one; None means the plain mean, as for a single dataset.
    weights: Any | None = None
    host_weights: np.ndarray | None = None

    @property
    def records(self) -> int:
        return int(self.plys.size)

    def slices(self) -> list[slice]:
        if self.piece_records is None:
            return [slice(None)]
        return [slice(start, start + self.piece_records)
                for start in range(0, self.records, self.piece_records)]

    def dense_lines(self, piece: slice) -> Any:
        if self.lines is not None:
            return self.lines[piece]
        import torch

        orbits = self.orbits[piece].long()
        dense = torch.zeros((orbits.shape[0], REVERSAL_ORBIT_COUNT), dtype=torch.float32,
                            device=orbits.device)
        dense.scatter_add_(1, orbits, torch.ones(orbits.shape, dtype=torch.float32,
                                                 device=orbits.device))
        return (dense - self.line_center) / self.line_scale


def _gain_design(plys: np.ndarray, gains: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate(
        (phase_features(plys), phase_interactions(gains, phase_interpolation(plys, GAIN_KNOTS))),
        axis=1), dtype=np.float64)


def _fit_gain_control(
    train: NumpyFeatureBatch | _TrainColumns,
    validation: NumpyFeatureBatch,
    train_gain: np.ndarray,
    validation_gain: np.ndarray,
    device: Any,
    ridge_lambdas: Sequence[float],
    *,
    weights: np.ndarray | None = None,
    piece_records: int | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    validation_design = _gain_design(validation.plys, validation_gain)
    residuals = np.asarray(train.residuals, dtype=np.float64)
    if piece_records is None:
        train_design = _gain_design(train.plys, train_gain)
        fit = fit_ridge(
            train_design, residuals, validation_design,
            np.asarray(validation.residuals, dtype=np.float64), ridge_lambdas, device,
            error_type=PatternExperimentError, weights=weights,
        )
        train_prediction = train_design @ fit.weights
    else:
        weight = np.ones(train.records) if weights is None else np.asarray(weights, np.float64)
        pieces = [slice(start, start + piece_records)
                  for start in range(0, train.records, piece_records)]
        gram = right_hand = 0.0
        for piece in pieces:
            design = _gain_design(train.plys[piece], train_gain[piece])
            weighted = design * weight[piece, None]
            gram = gram + design.T @ weighted
            right_hand = right_hand + weighted.T @ residuals[piece]
        total = float(weight.sum())
        fit = fit_ridge_moments(
            np.asarray(gram) / total, np.asarray(right_hand) / total, validation_design,
            np.asarray(validation.residuals, dtype=np.float64), ridge_lambdas, device,
            error_type=PatternExperimentError,
        )
        train_prediction = np.concatenate([
            _gain_design(train.plys[piece], train_gain[piece]) @ fit.weights
            for piece in pieces])
    model = {
        "name": "gain_summary_phase_5_control",
        "parameters": int(fit.weights.size),
        "phase_knots": list(GAIN_KNOTS),
        "selected_ridge_lambda": fit.selected_lambda,
        "ridge_path": list(fit.path),
        "weights": fit.weights.tolist(),
        "train": model_metrics(train, train_prediction, 8,
                               error_type=PatternExperimentError),
        "validation": model_metrics(validation, validation_design @ fit.weights, 8,
                                    error_type=PatternExperimentError),
    }
    return model, fit.weights.astype(np.float32)


def _train_model(
    config: ModelConfig,
    train: _FitData,
    validation: _FitData,
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
        (len(config.line_knots), REVERSAL_ORBIT_COUNT),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    if config.gain_knots is None:
        phase_weights = torch.tensor(
            _phase_means(train.plys, train.residuals, train.host_weights),
            dtype=torch.float32, device=device, requires_grad=True
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

    def line_term(data: _FitData, line_phase: Any, piece: slice) -> Any:
        if data.lookup is None:
            return ((data.dense_lines(piece) @ line_weights.T) * line_phase[piece]).sum(dim=1)
        # Standardized counts times weights: each pattern's weight over its scale, summed over
        # the record's patterns, less a constant per phase component for the centering.
        values = (line_weights / data.line_scale).T.contiguous()
        return ((data.lookup.sums(values) - data.line_center @ values) * line_phase).sum(dim=1)

    def predict(data: _FitData, line_phase: Any, gain_phase: Any | None,
                piece: slice = slice(None)) -> Any:
        result = phase_weights[data.ply_index[piece]] + line_term(data, line_phase, piece)
        if gain_weights is not None:
            result = result + ((data.gains[piece] @ gain_weights.T) * gain_phase[piece]).sum(dim=1)
        return result

    def predict_all(data: _FitData, line_phase: Any, gain_phase: Any | None) -> Any:
        parts = [predict(data, line_phase, gain_phase, piece) for piece in data.slices()]
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def record_loss(prediction: Any, target: Any, reduction: str) -> Any:
        if config.loss == "mse":
            squared = (prediction - target).square()
            return {"mean": squared.mean, "sum": squared.sum}.get(reduction, lambda: squared)()
        if config.loss == "huber":
            return functional.huber_loss(prediction, target, delta=config.huber_delta,
                                         reduction=reduction)
        raise PatternExperimentError(f"unknown pattern loss: {config.loss}")

    def data_loss(prediction: Any, data: _FitData, piece: slice | None = None) -> Any:
        """Return the data term over every record, or one piece's share of it."""
        target = data.target if piece is None else data.target[piece]
        if data.weights is not None:
            weights = data.weights if piece is None else data.weights[piece]
            return (weights * record_loss(prediction, target, "none")).sum()
        if piece is None:
            return record_loss(prediction, target, "mean")
        return record_loss(prediction, target, "sum") / data.records

    def penalty() -> Any:
        total = line_weights.square().sum()
        if gain_weights is not None:
            total = total + gain_weights.square().sum()
        return total

    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    best_key = (math.inf, math.inf, math.inf)
    best_step = 0
    best_state: tuple[Any, Any, Any | None] | None = None
    trace: list[dict[str, float | int]] = []
    for step in range(config.max_steps + 1):
        if step > 0:
            optimizer.zero_grad(set_to_none=True)
            if train.piece_records is None:
                measured_loss = data_loss(predict(train, train_line_phase, train_gain_phase), train)
                loss = measured_loss + config.l2 * penalty()
                loss.backward()
            else:
                # Gradients of the data term accumulate piece by piece into one full-batch step.
                for piece in train.slices():
                    data_loss(predict(train, train_line_phase, train_gain_phase, piece),
                              train, piece).backward()
                (config.l2 * penalty()).backward()
            optimizer.step()

        if step % config.evaluation_interval == 0:
            with torch.no_grad():
                train_checkpoint = predict_all(train, train_line_phase, train_gain_phase)
                validation_checkpoint = predict_all(
                    validation, validation_line_phase, validation_gain_phase)
                training_loss = float(data_loss(train_checkpoint, train).cpu())
                validation_loss = float(data_loss(validation_checkpoint, validation).cpu())
                error = validation_checkpoint - validation.target
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
        train_prediction = predict_all(train, train_line_phase, train_gain_phase).cpu().numpy()
        validation_prediction = predict_all(
            validation, validation_line_phase, validation_gain_phase).cpu().numpy()
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


@dataclass(frozen=True)
class _Prepared:
    feature_metadata: dict[str, Any]
    gain_control: dict[str, Any]
    gain_initial: np.ndarray
    train: _FitData
    validation: _FitData
    train_columns: NumpyFeatureBatch | _TrainColumns


def _feature_metadata(line_center: np.ndarray, line_scale: np.ndarray,
                      gain_standardization: Any) -> dict[str, Any]:
    return {
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


def _normalized(weights: np.ndarray | None) -> np.ndarray | None:
    return None if weights is None else np.asarray(weights, np.float64) / float(np.sum(weights))


def _prepare_in_memory(train: NumpyFeatureBatch, validation: NumpyFeatureBatch,
                       weights: np.ndarray | None, device: Any) -> _Prepared:
    """Expand every record's line counts at once, the original training path."""
    import torch

    raw_train_lines = line_pattern_counts(train.line_patterns)
    raw_validation_lines = line_pattern_counts(validation.line_patterns)
    if weights is None:
        train_line_values, validation_line_values, line_center, line_scale = (
            _standardize_counts(raw_train_lines, raw_validation_lines))
    else:
        moments = _LineMoments()
        moments.add(REVERSAL_ORBITS.raw_to_orbit[train.line_patterns], weights)
        line_center, line_scale = moments.standardization()
        train_line_values = (raw_train_lines - line_center) / line_scale
        validation_line_values = (raw_validation_lines - line_center) / line_scale
    raw_train_gain = gain_summaries(train.own_gains, train.opponent_gains)
    raw_validation_gain = gain_summaries(validation.own_gains, validation.opponent_gains)
    gain_standardization = fit_standardization(raw_train_gain, weights)
    train_gain_values = gain_standardization.transform(raw_train_gain)
    validation_gain_values = gain_standardization.transform(raw_validation_gain)
    gain_control, gain_initial = _fit_gain_control(
        train, validation, train_gain_values, validation_gain_values, device,
        DEFAULT_RIDGE_LAMBDAS, weights=weights,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def data(batch: NumpyFeatureBatch, lines: np.ndarray, gains: np.ndarray,
             weights: np.ndarray | None) -> _FitData:
        return _FitData(
            plys=batch.plys, residuals=batch.residuals,
            ply_index=torch.as_tensor(batch.plys.astype(np.int64), device=device),
            target=torch.as_tensor(batch.residuals, dtype=torch.float32, device=device),
            gains=torch.as_tensor(gains.astype(np.float32), device=device),
            lines=torch.as_tensor(lines, device=device),
            weights=(None if weights is None else
                     torch.as_tensor(weights, dtype=torch.float32, device=device)),
            host_weights=weights,
        )

    return _Prepared(
        feature_metadata=_feature_metadata(line_center, line_scale, gain_standardization),
        gain_control=gain_control, gain_initial=gain_initial,
        train=data(train, train_line_values, train_gain_values, weights),
        validation=data(validation, validation_line_values, validation_gain_values, None),
        train_columns=train,
    )


def _orbit_ids(patterns: np.ndarray) -> np.ndarray:
    return REVERSAL_ORBITS.raw_to_orbit[patterns].astype(np.int16)


def _prepare_lookup(train: NumpyFeatureBatch, validation: NumpyFeatureBatch,
                    weights: np.ndarray | None, device: Any) -> _Prepared:
    """Keep 36 pattern IDs per record and look up their weights; never expand line counts."""
    import torch

    train_orbits = _orbit_ids(train.line_patterns)
    moments = _LineMoments()
    step = 65536
    for start in range(0, train.records, step):
        piece = slice(start, start + step)
        moments.add(train_orbits[piece], np.ones(train_orbits[piece].shape[0])
                    if weights is None else weights[piece])
    line_center, line_scale = moments.standardization()
    raw_train_gain = gain_summaries(train.own_gains, train.opponent_gains)
    raw_validation_gain = gain_summaries(validation.own_gains, validation.opponent_gains)
    gain_standardization = fit_standardization(raw_train_gain, weights)
    train_gain_values = gain_standardization.transform(raw_train_gain)
    validation_gain_values = gain_standardization.transform(raw_validation_gain)
    gain_control, gain_initial = _fit_gain_control(
        train, validation, train_gain_values, validation_gain_values, device,
        DEFAULT_RIDGE_LAMBDAS, weights=weights,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    center = torch.as_tensor(line_center, device=device)
    scale = torch.as_tensor(line_scale, device=device)

    def data(batch: NumpyFeatureBatch, orbits: np.ndarray, gains: np.ndarray,
             weights: np.ndarray | None, *, plan: bool) -> _FitData:
        return _FitData(
            plys=batch.plys, residuals=batch.residuals,
            ply_index=torch.as_tensor(batch.plys.astype(np.int64), device=device),
            target=torch.as_tensor(batch.residuals, dtype=torch.float32, device=device),
            gains=torch.as_tensor(gains.astype(np.float32), device=device),
            line_center=center, line_scale=scale,
            lookup=_pattern_lookup(orbits, device, plan=plan),
            weights=(None if weights is None else
                     torch.as_tensor(weights, dtype=torch.float32, device=device)),
            host_weights=weights,
        )

    return _Prepared(
        feature_metadata=_feature_metadata(line_center, line_scale, gain_standardization),
        gain_control=gain_control, gain_initial=gain_initial,
        train=data(train, train_orbits, train_gain_values, weights, plan=True),
        validation=data(validation, _orbit_ids(validation.line_patterns),
                        validation_gain_values, None, plan=False),
        train_columns=train,
    )


def _prepare_pieces(selection: TrainingSelection, validation: NumpyFeatureBatch,
                    piece_records: int, device: Any) -> _Prepared:
    """Keep 36 pattern IDs per record and expand dense line counts one piece at a time."""
    import torch

    weighted = not selection.uniform
    moments = _LineMoments()
    orbits, gains, columns, weights = [], [], [], []
    for piece, piece_weights in selection.pieces(piece_records):
        piece_orbits = REVERSAL_ORBITS.raw_to_orbit[piece.line_patterns].astype(np.int16)
        moments.add(piece_orbits, piece_weights if weighted else np.ones(piece.records))
        orbits.append(piece_orbits)
        gains.append(gain_summaries(piece.own_gains, piece.opponent_gains))
        columns.append({name: getattr(piece, name) for name in _TrainColumns.__dataclass_fields__})
        weights.append(piece_weights)
    line_center, line_scale = moments.standardization()
    train = _TrainColumns(**{name: np.concatenate([piece[name] for piece in columns])
                             for name in _TrainColumns.__dataclass_fields__})
    del columns
    host_weights = _normalized(np.concatenate(weights)) if weighted else None
    raw_train_gain = np.concatenate(gains)
    del gains
    raw_validation_gain = gain_summaries(validation.own_gains, validation.opponent_gains)
    gain_standardization = fit_standardization(raw_train_gain, host_weights)
    train_gain_values = gain_standardization.transform(raw_train_gain)
    validation_gain_values = gain_standardization.transform(raw_validation_gain)
    gain_control, gain_initial = _fit_gain_control(
        train, validation, train_gain_values, validation_gain_values, device,
        DEFAULT_RIDGE_LAMBDAS, weights=host_weights, piece_records=piece_records,
    )
    center = torch.as_tensor(line_center, device=device)
    scale = torch.as_tensor(line_scale, device=device)

    def data(columns: Any, orbit_ids: np.ndarray, gains: np.ndarray,
             weights: np.ndarray | None) -> _FitData:
        return _FitData(
            plys=columns.plys, residuals=columns.residuals,
            ply_index=torch.as_tensor(columns.plys.astype(np.int64), device=device),
            target=torch.as_tensor(columns.residuals, dtype=torch.float32, device=device),
            gains=torch.as_tensor(gains.astype(np.float32), device=device),
            orbits=torch.as_tensor(orbit_ids, device=device),
            line_center=center, line_scale=scale, piece_records=piece_records,
            weights=(None if weights is None else
                     torch.as_tensor(weights, dtype=torch.float32, device=device)),
            host_weights=weights,
        )

    return _Prepared(
        feature_metadata=_feature_metadata(line_center, line_scale, gain_standardization),
        gain_control=gain_control, gain_initial=gain_initial,
        train=data(train, np.concatenate(orbits), train_gain_values, host_weights),
        validation=data(validation,
                        REVERSAL_ORBITS.raw_to_orbit[validation.line_patterns].astype(np.int16),
                        validation_gain_values, None),
        train_columns=train,
    )


def _check_training_memory(method: str, piece_records: int | None, records: int,
                           validation_records: int, device: Any) -> None:
    def described(method: str, pieces: int | None = None) -> str:
        needed = training_estimate(method, records, validation_records, pieces)
        gpu = f", {needed.device / GIB:.1f} GiB GPU" if device.type == "cuda" else ""
        return f"about {needed.host / GIB:.1f} GiB host{gpu}"

    remedies = [*([f'method = "lookup" ({described("lookup")})'] if method != "lookup" else []),
                *([f"piece_records = 50000 ({described('dense', 50_000)})"]
                  if method == "dense" and piece_records is None else []),
                "max_records", "a smaller window", "more free memory"]
    check_memory(
        f"pattern training ({method}"
        f"{f' in pieces of {piece_records:,}' if piece_records else ''}, {records:,} train and "
        f"{validation_records:,} validation records)",
        training_estimate(method, records, validation_records, piece_records), device=device,
        remedies=f"Ways out: {', '.join(remedies[:-1])}, or {remedies[-1]}.")


def run_pattern_experiment(
    dataset_directory: Path | str,
    output_directory: Path | str,
    *,
    requested_device: str = "cuda",
    seed: int = 20260818,
    configs: Sequence[ModelConfig] | None = None,
    extra_datasets: Sequence[tuple[Path | str, float]] = (),
    max_records: int | None = None,
    piece_records: int | None = None,
    method: str = "dense",
) -> dict[str, Any]:
    """Train configured model candidates using train/validation only.

    Extra datasets contribute weighted train records only; validation, selection, and the
    held-out test split always come from the primary dataset.
    """
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
    _require(max_records is None or max_records > 0, "max_records must be positive")
    _require(piece_records is None or piece_records > 0, "piece_records must be positive")
    _require(method in TRAINING_METHODS,
             f"training method must be one of {', '.join(TRAINING_METHODS)}")
    # Pieces bound the dense line counts, which lookup never builds.
    _require(method == "dense" or piece_records is None,
             "piece_records applies only to the dense training method")

    output = Path(output_directory).resolve()
    with ExitStack() as stack:
        dataset = stack.enter_context(MappedFeatureDataset(dataset_directory, verify_digest=True))
        extras = [(stack.enter_context(MappedFeatureDataset(path, verify_digest=True)), weight)
                  for path, weight in extra_datasets]
        selection = (TrainingSelection(dataset, extras, max_records=max_records)
                     if extras or max_records is not None or piece_records is not None else None)
        _check_training_memory(
            method, piece_records,
            selection.records if selection is not None else int(dataset.indices("train").size),
            int(dataset.indices("validation").size), device)
        reserve_report_directory(output, SCHEMA, error_type=PatternExperimentError)
        validation = dataset.materialize("validation")
        if piece_records is not None:
            assert selection is not None
            _require(validation.records > 0, "train and validation splits must be nonempty")
            prepared = _prepare_pieces(selection, validation, piece_records, device)
        else:
            if selection is None:
                train, weights = dataset.materialize("train"), None
            else:
                train, raw_weights = selection.materialize()
                weights = None if selection.uniform else _normalized(raw_weights)
            _require(train.records > 0 and validation.records > 0,
                     "train and validation splits must be nonempty")
            prepare = _prepare_lookup if method == "lookup" else _prepare_in_memory
            prepared = prepare(train, validation, weights, device)

        models: list[dict[str, Any]] = []
        for config in selected_configs:
            fit = _train_model(config, prepared.train, prepared.validation,
                               prepared.gain_initial, device)
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
                "train": model_metrics(prepared.train_columns, fit.train_prediction, 8,
                                       error_type=PatternExperimentError),
                "validation": model_metrics(validation, fit.validation_prediction, 8,
                                            error_type=PatternExperimentError),
            }
            model["quantization"] = _quantization_report(
                validation, prepared.feature_metadata, model, fit.validation_prediction
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
            "features": prepared.feature_metadata,
            "provenance": {
                "training_source_sha256": training_source_digest(),
                "git": git_provenance(),
                "runtime": runtime_provenance(torch, device),
            },
            "gain_control": prepared.gain_control,
            "models": models,
            "selection": {
                "best_model": best["name"],
                "best_validation_mae": best["validation"]["overall"]["mae"],
                "best_validation_rmse": best["validation"]["overall"]["rmse"],
                "best_vs_gain_control": model_comparison(prepared.gain_control, best),
            },
        }
        if method != "dense":
            report["training"]["method"] = method
        if selection is not None:
            report["training_data"] = {
                "datasets": selection.describe(), "records": selection.records,
                "max_records": max_records, "piece_records": piece_records,
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
    parser.add_argument("--extra-dataset", nargs=2, action="append", default=[],
                        metavar=("PATH", "WEIGHT"),
                        help="also train on this artifact's train records at this loss weight; "
                             "repeat newest first")
    parser.add_argument("--max-records", type=int,
                        help="cap the training records, keeping whole datasets newest first")
    parser.add_argument("--piece-records", type=int,
                        help="expand dense line counts this many records at a time")
    parser.add_argument("--method", choices=TRAINING_METHODS, default="dense",
                        help="compute the line term from dense counts or by pattern lookup")
    arguments = parser.parse_args()
    configs = pattern_suite(arguments.suite)
    try:
        report = run_pattern_experiment(
            arguments.dataset,
            arguments.output_dir,
            requested_device=arguments.device,
            seed=arguments.seed,
            configs=configs,
            extra_datasets=[(Path(path), float(weight))
                            for path, weight in arguments.extra_dataset],
            max_records=arguments.max_records,
            piece_records=arguments.piece_records,
            method=arguments.method,
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
