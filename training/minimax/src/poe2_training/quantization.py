"""Fold and quantize trained pattern residuals for integer engine inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset import NumpyFeatureBatch
from .patterns import REVERSAL_ORBITS, phase_basis
from .summaries import gain_summaries


@dataclass(frozen=True)
class FoldedPatternModel:
    """Training standardization folded into raw feature weights."""

    name: str
    line_knots: tuple[int, ...]
    gain_knots: tuple[int, ...] | None
    intercept: np.ndarray
    line_weights: np.ndarray
    gain_weights: np.ndarray | None


@dataclass(frozen=True)
class QuantizedPatternModel:
    """One common power-of-two fixed-point scale for deterministic inference."""

    name: str
    fractional_bits: int
    line_knots: tuple[int, ...]
    gain_knots: tuple[int, ...] | None
    intercept: np.ndarray
    line_weights: np.ndarray
    gain_weights: np.ndarray | None

    @property
    def scale(self) -> int:
        return 1 << self.fractional_bits


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not bool(np.isfinite(result).all()):
        raise ValueError(f"{name} has invalid dimensions or values")
    return result


def fold_pattern_model(experiment: dict[str, Any], model: dict[str, Any]) -> FoldedPatternModel:
    """Remove training-only centering/scaling without changing predictions."""
    features = experiment.get("features")
    config = model.get("config")
    if not isinstance(features, dict) or not isinstance(config, dict):
        raise ValueError("pattern experiment model metadata is incomplete")
    line_knots = tuple(config.get("line_knots", ()))
    gain_value = config.get("gain_knots")
    gain_knots = tuple(gain_value) if gain_value is not None else None
    if not line_knots:
        raise ValueError("pattern model has no line knots")

    line_center = _array(features.get("line_center"), (1716,), "line center")
    line_scale = _array(features.get("line_scale"), (1716,), "line scale")
    if not bool((line_scale > 0.0).all()):
        raise ValueError("line scale must be positive")
    standardized_line = _array(
        model.get("line_weights"), (len(line_knots), 1716), "line weights"
    )
    phase_weights = _array(model.get("phase_weights"), (50,), "phase weights")
    line_weights = standardized_line / line_scale
    line_offset = -(line_weights @ line_center)
    plys = np.arange(50, dtype=np.uint8)
    intercept = phase_weights + phase_basis(plys, line_knots) @ line_offset

    gain_weights: np.ndarray | None = None
    if gain_knots is not None:
        gain_center = _array(features.get("gain_center"), (19,), "gain center")
        gain_scale = _array(features.get("gain_scale"), (19,), "gain scale")
        if not bool((gain_scale > 0.0).all()):
            raise ValueError("gain scale must be positive")
        standardized_gain = _array(
            model.get("gain_weights"), (len(gain_knots), 19), "gain weights"
        )
        gain_weights = standardized_gain / gain_scale
        gain_offset = -(gain_weights @ gain_center)
        intercept = intercept + phase_basis(plys, gain_knots) @ gain_offset
    elif model.get("gain_weights") is not None:
        raise ValueError("gain-free pattern model contains gain weights")

    return FoldedPatternModel(
        name=str(model.get("name")),
        line_knots=line_knots,
        gain_knots=gain_knots,
        intercept=np.ascontiguousarray(intercept, dtype=np.float64),
        line_weights=np.ascontiguousarray(line_weights, dtype=np.float64),
        gain_weights=(np.ascontiguousarray(gain_weights, dtype=np.float64)
                      if gain_weights is not None else None),
    )


def predict_folded(batch: NumpyFeatureBatch, model: FoldedPatternModel) -> np.ndarray:
    """Evaluate folded floating-point weights using runtime-style line lookups."""
    orbit_ids = REVERSAL_ORBITS.raw_to_orbit[np.asarray(batch.line_patterns, dtype=np.int64)]
    line_sums = np.stack(
        [weights[orbit_ids].sum(axis=1) for weights in model.line_weights], axis=1
    )
    result = model.intercept[np.asarray(batch.plys, dtype=np.int64)]
    result = result + (
        line_sums * phase_basis(batch.plys, model.line_knots)
    ).sum(axis=1)
    if model.gain_weights is not None:
        if model.gain_knots is None:
            raise ValueError("folded gain weights have no phase knots")
        gains = gain_summaries(batch.own_gains, batch.opponent_gains)
        gain_sums = gains @ model.gain_weights.T
        result = result + (
            gain_sums * phase_basis(batch.plys, model.gain_knots)
        ).sum(axis=1)
    return np.ascontiguousarray(result, dtype=np.float64)


def quantize_pattern_model(model: FoldedPatternModel,
                           fractional_bits: int) -> QuantizedPatternModel:
    """Quantize tables to int16 and the per-ply intercept to int32."""
    if fractional_bits < 0 or fractional_bits > 20:
        raise ValueError("fractional bits must be between zero and twenty")
    scale = 1 << fractional_bits

    def quantize(values: np.ndarray, dtype: Any, name: str) -> np.ndarray:
        rounded = np.rint(values * scale)
        limits = np.iinfo(dtype)
        if bool((rounded < limits.min).any()) or bool((rounded > limits.max).any()):
            raise OverflowError(f"{name} does not fit {np.dtype(dtype).name}")
        return np.ascontiguousarray(rounded, dtype=dtype)

    return QuantizedPatternModel(
        name=model.name,
        fractional_bits=fractional_bits,
        line_knots=model.line_knots,
        gain_knots=model.gain_knots,
        intercept=quantize(model.intercept, np.int32, "intercept"),
        line_weights=quantize(model.line_weights, np.int16, "line weights"),
        gain_weights=(quantize(model.gain_weights, np.int16, "gain weights")
                      if model.gain_weights is not None else None),
    )


def _rounded_divide(numerator: np.ndarray, denominator: int) -> np.ndarray:
    if denominator <= 0:
        raise ValueError("integer interpolation denominator must be positive")
    values = np.asarray(numerator, dtype=np.int64)
    magnitude = (np.abs(values) + denominator // 2) // denominator
    return np.where(values < 0, -magnitude, magnitude)


def _interpolate_integer(values: np.ndarray, plys: np.ndarray,
                         knots: tuple[int, ...]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.int64)
    phases = np.asarray(plys, dtype=np.int64)
    if matrix.shape != (phases.size, len(knots)):
        raise ValueError("integer interpolation inputs are incompatible")
    if len(knots) == 1:
        return matrix[:, 0]
    result = np.empty(phases.size, dtype=np.int64)
    for lower, upper in zip(range(len(knots) - 1), range(1, len(knots))):
        left = knots[lower]
        right = knots[upper]
        selected = (phases >= left) & (phases <= right)
        if lower > 0:
            selected &= phases > left
        distance = phases[selected] - left
        numerator = ((right - phases[selected]) * matrix[selected, lower] +
                     distance * matrix[selected, upper])
        result[selected] = _rounded_divide(numerator, right - left)
    return result


def predict_quantized_fixed(batch: NumpyFeatureBatch,
                            model: QuantizedPatternModel) -> np.ndarray:
    """Evaluate the integer tables and retain the fixed-point residual accumulator."""
    plys = np.asarray(batch.plys, dtype=np.int64)
    orbit_ids = REVERSAL_ORBITS.raw_to_orbit[np.asarray(batch.line_patterns, dtype=np.int64)]
    line_sums = np.stack(
        [weights[orbit_ids].sum(axis=1, dtype=np.int64) for weights in model.line_weights],
        axis=1,
    )
    total = model.intercept[plys].astype(np.int64)
    total += _interpolate_integer(line_sums, plys, model.line_knots)
    if model.gain_weights is not None:
        if model.gain_knots is None:
            raise ValueError("quantized gain weights have no phase knots")
        gains = gain_summaries(batch.own_gains, batch.opponent_gains).astype(np.int64)
        gain_sums = gains @ model.gain_weights.astype(np.int64).T
        total += _interpolate_integer(gain_sums, plys, model.gain_knots)
    return np.ascontiguousarray(total, dtype=np.int64)


def predict_quantized(batch: NumpyFeatureBatch,
                      model: QuantizedPatternModel) -> np.ndarray:
    """Evaluate the integer tables and round the residual to whole score units."""
    total = predict_quantized_fixed(batch, model)
    return np.ascontiguousarray(_rounded_divide(total, model.scale), dtype=np.float64)
