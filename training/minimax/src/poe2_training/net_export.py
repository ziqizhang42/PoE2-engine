"""Export a trained pattern-gain model as a net file the engine loads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .net_file import (
    FRACTIONAL_BITS,
    GAIN_KNOTS,
    LINE_KNOTS,
    NetFileError,
    PatternGainParameters,
    decode_net,
    encode_net,
    net_metadata,
)
from .pattern_experiment import FROZEN_PATTERN_GAIN_MODEL_ID, open_pattern_report
from .shared import atomic_write_bytes, sha256_file

QUANTIZATION = "int16-tables-int32-intercept-power-of-two-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NetFileError(message)


def _flatten(rows: Any, label: str) -> tuple[int, ...]:
    _require(isinstance(rows, list) and all(isinstance(row, list) for row in rows),
             f"quantized {label} must be a list of rows")
    return tuple(value for row in rows for value in row)


def selected_parameters(report: dict[str, Any]) -> PatternGainParameters:
    """Return the quantized parameters of the report's selected pattern-gain model."""
    selection = report.get("selection")
    _require(isinstance(selection, dict) and
             selection.get("best_model") == FROZEN_PATTERN_GAIN_MODEL_ID,
             "the selected model is not the pattern-gain-v1 architecture")
    matches = [model for model in report.get("models", [])
               if isinstance(model, dict) and model.get("name") == FROZEN_PATTERN_GAIN_MODEL_ID]
    _require(len(matches) == 1, "the pattern-gain model is missing or duplicated")
    config = matches[0].get("config")
    quantized = matches[0].get("quantization")
    _require(isinstance(config, dict) and isinstance(quantized, dict),
             "the pattern-gain model has no configuration or quantization")
    _require(tuple(config.get("line_knots", ())) == LINE_KNOTS and
             tuple(config.get("gain_knots", ())) == GAIN_KNOTS,
             "the model's phase knots differ from design pattern-gain-v1")
    _require(quantized.get("definition") == QUANTIZATION and
             quantized.get("fractional_bits") == FRACTIONAL_BITS,
             "the model's quantization differs from design pattern-gain-v1")
    intercept = quantized.get("intercept")
    _require(isinstance(intercept, list), "quantized intercepts must be a list")
    return PatternGainParameters(
        line_knots=LINE_KNOTS, gain_knots=GAIN_KNOTS, intercept=tuple(intercept),
        line_weights=_flatten(quantized.get("line_weights"), "line weights"),
        gain_weights=_flatten(quantized.get("gain_weights"), "gain weights"))


def export_net(experiment: Path, *, name: str, teacher: str, created: str,
               parent: str | None = None) -> bytes:
    """Return the net file of a completed training report; the same inputs give the same bytes."""
    report = open_pattern_report(experiment)
    provenance = report.get("input")
    _require(isinstance(provenance, dict) and all(
        isinstance(provenance.get(key), str)
        for key in ("corpus_id", "corpus_digest", "feature_binary_sha256")),
        "the training report's input provenance is incomplete")
    return encode_net(selected_parameters(report), net_metadata(
        name=name, teacher=teacher, parent=parent, created=created,
        corpus=provenance["corpus_id"], corpus_digest=provenance["corpus_digest"],
        feature_binary_sha256=provenance["feature_binary_sha256"],
        training_report_sha256=sha256_file(experiment / "report.json")))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True,
                        help="completed pattern-gain training directory")
    parser.add_argument("--output", type=Path, required=True, help="net file to create")
    parser.add_argument("--name", required=True, help="the net's name, <run>/<generation>")
    parser.add_argument("--teacher", required=True,
                        help="the teacher's net name, or two-ply-closure")
    parser.add_argument("--parent", help="the parent's net name")
    parser.add_argument("--created", required=True, help="UTC date the model was trained")
    arguments = parser.parse_args()
    try:
        data = export_net(arguments.experiment, name=arguments.name, teacher=arguments.teacher,
                          parent=arguments.parent, created=arguments.created)
        net = decode_net(data)
        atomic_write_bytes(arguments.output, data)
    except (NetFileError, OSError, ValueError) as error:
        print(f"net export failed: {error}", file=sys.stderr)
        return 1
    print(f"net_export_complete name={net.name} id={net.net_id} sha256={net.sha256} "
          f"bytes={len(data)} output={arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
