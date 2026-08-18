"""Cheap, deterministic features derived from two-ply-closure primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CELL_COUNT = 49
PLY_COUNT = 50
OCCUPIED_GAIN = np.iinfo(np.int16).min
GAIN_THRESHOLDS = (2, 4, 8, 16)
GAIN_SUMMARY_NAMES = (
    *(f"own_gain_rank_{rank}" for rank in range(1, 5)),
    *(f"opponent_gain_rank_{rank}" for rank in range(1, 5)),
    *(f"own_count_ge_{threshold}" for threshold in GAIN_THRESHOLDS),
    *(f"opponent_count_ge_{threshold}" for threshold in GAIN_THRESHOLDS),
    "contested_best_square",
    "opponent_unique_best_square",
    "denied_opponent_best_gain",
)
PHASE_FEATURE_NAMES = tuple(f"ply_{ply:02d}" for ply in range(PLY_COUNT))


@dataclass(frozen=True)
class Standardization:
    """Training-only location and scale for continuous gain summaries."""

    center: np.ndarray
    scale: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray((values - self.center) / self.scale, dtype=np.float64)


def phase_features(plys: np.ndarray) -> np.ndarray:
    """Return a one-hot per-ply calibration matrix."""
    values = np.asarray(plys, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("plys must be a one-dimensional array")
    if values.size and (int(values.min()) < 0 or int(values.max()) >= PLY_COUNT):
        raise ValueError("ply is outside the 0..49 board range")
    result = np.zeros((values.size, PLY_COUNT), dtype=np.float64)
    result[np.arange(values.size), values] = 1.0
    return result


def phase_interpolation(plys: np.ndarray, knots: tuple[int, ...]) -> np.ndarray:
    """Return piecewise-linear interpolation weights over strictly increasing ply knots."""
    values = np.asarray(plys, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("plys must be a one-dimensional array")
    if len(knots) < 2 or knots[0] != 0 or knots[-1] != PLY_COUNT - 1 or any(
            left >= right for left, right in zip(knots, knots[1:])):
        raise ValueError("phase knots must increase strictly from ply 0 through 49")
    if values.size and (int(values.min()) < 0 or int(values.max()) >= PLY_COUNT):
        raise ValueError("ply is outside the 0..49 board range")

    result = np.zeros((values.size, len(knots)), dtype=np.float64)
    for index, ply in enumerate(values):
        upper = int(np.searchsorted(knots, ply, side="right"))
        if upper == 0:
            result[index, 0] = 1.0
        elif upper == len(knots):
            result[index, -1] = 1.0
        else:
            lower = upper - 1
            span = knots[upper] - knots[lower]
            right_weight = (int(ply) - knots[lower]) / span
            result[index, lower] = 1.0 - right_weight
            result[index, upper] = right_weight
    return result


def phase_interactions(values: np.ndarray, interpolation: np.ndarray) -> np.ndarray:
    """Flatten phase-weighted copies of continuous features in knot-major order."""
    features = np.asarray(values, dtype=np.float64)
    phase = np.asarray(interpolation, dtype=np.float64)
    if features.ndim != 2 or phase.ndim != 2 or features.shape[0] != phase.shape[0]:
        raise ValueError("phase interaction inputs must be aligned two-dimensional matrices")
    interacted = phase[:, :, None] * features[:, None, :]
    return np.ascontiguousarray(interacted.reshape(features.shape[0], -1), dtype=np.float64)


def _top_four(values: np.ndarray, legal: np.ndarray) -> np.ndarray:
    masked = np.where(legal, values, OCCUPIED_GAIN)
    ordered = np.sort(masked, axis=1)[:, ::-1]
    top = np.array(ordered[:, :4], dtype=np.float64, copy=True, order="C")
    top[top == OCCUPIED_GAIN] = 0.0
    return top


def gain_summaries(own_gains: np.ndarray, opponent_gains: np.ndarray) -> np.ndarray:
    """Summarize both marginal-gain landscapes without inspecting line patterns."""
    own = np.asarray(own_gains, dtype=np.int16)
    opponent = np.asarray(opponent_gains, dtype=np.int16)
    if own.ndim != 2 or own.shape[1:] != (CELL_COUNT,) or own.shape != opponent.shape:
        raise ValueError("gain arrays must have matching shape (records, 49)")

    legal = own != OCCUPIED_GAIN
    if not np.array_equal(legal, opponent != OCCUPIED_GAIN):
        raise ValueError("own and opponent gain occupancy masks differ")
    if bool((legal.sum(axis=1) == 0).any()):
        raise ValueError("gain summaries require at least one legal square")
    if bool((own[legal] < 1).any()) or bool((opponent[legal] < 1).any()):
        raise ValueError("legal marginal gains must be positive")

    own_top = _top_four(own, legal)
    opponent_top = _top_four(opponent, legal)
    own_best = own_top[:, 0].astype(np.int16, copy=False)
    opponent_best = opponent_top[:, 0].astype(np.int16, copy=False)
    own_best_mask = legal & (own == own_best[:, None])
    opponent_best_mask = legal & (opponent == opponent_best[:, None])
    contested = (own_best_mask & opponent_best_mask).any(axis=1)
    opponent_best_count = opponent_best_mask.sum(axis=1)

    # Two-ply closure chooses the square maximizing own gain minus the opponent's best reply.
    # Only a unique opponent-best square can lower that reply value when occupied.
    opponent_second = opponent_top[:, 1]
    unique_opponent_best = opponent_best_count == 1
    reply = np.broadcast_to(opponent_top[:, :1], own.shape).astype(np.float64, copy=True)
    unique_rows = np.flatnonzero(unique_opponent_best)
    if unique_rows.size:
        unique_cells = np.argmax(opponent_best_mask[unique_rows], axis=1)
        reply[unique_rows, unique_cells] = opponent_second[unique_rows]
    candidates = np.where(legal, own.astype(np.float64) - reply, -np.inf)
    best_candidate = candidates.max(axis=1)
    optimal = candidates == best_candidate[:, None]
    denial = opponent_top[:, :1] - reply
    # Several squares can tie under the closure own-gain-minus-reply objective. A row-major
    # argmax makes the denial feature change when the board is rotated or reflected,
    # which is incompatible with the engine's D4 transposition table. Conservatively
    # take the minimum denial among all closure-optimal squares; this is permutation invariant.
    denied = np.where(optimal, denial, np.inf).min(axis=1)

    columns: list[np.ndarray] = [own_top, opponent_top]
    columns.extend(
        np.count_nonzero(legal & (own >= threshold), axis=1)[:, None]
        for threshold in GAIN_THRESHOLDS
    )
    columns.extend(
        np.count_nonzero(legal & (opponent >= threshold), axis=1)[:, None]
        for threshold in GAIN_THRESHOLDS
    )
    columns.extend((
        contested[:, None],
        unique_opponent_best[:, None],
        denied[:, None],
    ))
    result = np.ascontiguousarray(np.concatenate(columns, axis=1), dtype=np.float64)
    if result.shape[1] != len(GAIN_SUMMARY_NAMES):
        raise AssertionError("gain summary names differ from the encoded columns")
    return result


def fit_standardization(values: np.ndarray) -> Standardization:
    """Fit deterministic population moments using training records only."""
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("standardization requires a nonempty matrix")
    center = matrix.mean(axis=0, dtype=np.float64)
    scale = matrix.std(axis=0, dtype=np.float64)
    scale[scale < 1.0e-12] = 1.0
    return Standardization(
        center=np.ascontiguousarray(center),
        scale=np.ascontiguousarray(scale),
    )
