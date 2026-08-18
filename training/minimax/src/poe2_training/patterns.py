"""Reversal-tied representations of the 3,276 raw scoring-line patterns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LINE_LENGTHS = tuple(range(2, 8))
RAW_PATTERN_COUNT = sum(3 ** length for length in LINE_LENGTHS)
LINE_COUNT = 36


def reverse_ternary(code: int, length: int) -> int:
    """Reverse exactly ``length`` base-three digits, retaining leading zeros."""
    if length not in LINE_LENGTHS or code < 0 or code >= 3 ** length:
        raise ValueError("ternary pattern code is outside its line length")
    result = 0
    for _ in range(length):
        result = result * 3 + code % 3
        code //= 3
    return result


@dataclass(frozen=True)
class ReversalOrbits:
    raw_to_orbit: np.ndarray
    orbit_lengths: np.ndarray
    pattern_offsets: tuple[int, ...]

    @property
    def count(self) -> int:
        return int(self.orbit_lengths.size)


def build_reversal_orbits() -> ReversalOrbits:
    """Map raw length-offset pattern IDs to deterministic reversal orbits."""
    raw_to_orbit = np.empty(RAW_PATTERN_COUNT, dtype=np.int64)
    orbit_lengths: list[int] = []
    offsets: list[int] = []
    raw_offset = 0
    orbit = 0
    for length in LINE_LENGTHS:
        offsets.append(raw_offset)
        representatives: dict[int, int] = {}
        for code in range(3 ** length):
            representative = min(code, reverse_ternary(code, length))
            if representative not in representatives:
                representatives[representative] = orbit
                orbit_lengths.append(length)
                orbit += 1
            raw_to_orbit[raw_offset + code] = representatives[representative]
        raw_offset += 3 ** length
    if raw_offset != RAW_PATTERN_COUNT:
        raise AssertionError("raw pattern offsets have the wrong total")
    return ReversalOrbits(
        raw_to_orbit=raw_to_orbit,
        orbit_lengths=np.asarray(orbit_lengths, dtype=np.uint8),
        pattern_offsets=tuple(offsets),
    )


REVERSAL_ORBITS = build_reversal_orbits()
REVERSAL_ORBIT_COUNT = REVERSAL_ORBITS.count
if REVERSAL_ORBIT_COUNT != 1716:
    raise AssertionError("unexpected number of reversal-tied patterns")


def line_pattern_counts(patterns: np.ndarray) -> np.ndarray:
    """Convert 36 raw pattern IDs per position to dense reversal-orbit counts."""
    values = np.asarray(patterns, dtype=np.int64)
    if values.ndim != 2 or values.shape[1:] != (LINE_COUNT,):
        raise ValueError("line patterns must have shape (records, 36)")
    if values.size and (int(values.min()) < 0 or int(values.max()) >= RAW_PATTERN_COUNT):
        raise ValueError("line pattern ID is outside the raw pattern table")
    orbit_ids = REVERSAL_ORBITS.raw_to_orbit[values]
    result = np.zeros((values.shape[0], REVERSAL_ORBIT_COUNT), dtype=np.float32)
    np.add.at(
        result,
        (np.repeat(np.arange(values.shape[0]), LINE_COUNT), orbit_ids.ravel()),
        1.0,
    )
    return result


def phase_basis(plys: np.ndarray, knots: tuple[int, ...]) -> np.ndarray:
    """Return one global component or piecewise-linear phase weights."""
    if len(knots) == 1:
        if knots != (0,):
            raise ValueError("the global phase basis must use the (0,) marker")
        values = np.asarray(plys)
        if values.ndim != 1:
            raise ValueError("plys must be a one-dimensional array")
        return np.ones((values.size, 1), dtype=np.float32)
    from .summaries import phase_interpolation

    return phase_interpolation(plys, knots).astype(np.float32)
