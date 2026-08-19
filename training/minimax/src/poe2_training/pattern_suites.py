"""Central registry of supported pattern-experiment model suites."""

from __future__ import annotations

from collections.abc import Callable


SUITE_NAMES = ("default", "frozen-pattern-gain", "line-pattern-audit")


def pattern_suite(name: str):
    """Resolve a suite lazily to avoid a module import cycle."""
    from .pattern_experiment import (
        default_suite,
        frozen_pattern_gain_suite,
        line_pattern_audit_suite,
    )
    registry: dict[str, Callable] = {
        "default": default_suite,
        "frozen-pattern-gain": frozen_pattern_gain_suite,
        "line-pattern-audit": line_pattern_audit_suite,
    }
    try:
        return registry[name]()
    except KeyError as error:
        raise ValueError(f"unknown pattern suite: {name}") from error

