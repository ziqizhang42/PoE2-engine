#!/usr/bin/env python3
"""Export the authenticated frozen pattern/gain model as C++ lookup tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


REPORT_SCHEMA = "poe2-minimax-pattern-experiment"
# Immutable identifier in the authenticated report; new code uses descriptive names.
AUTHENTICATED_MODEL_ID = "frozen_c_line_0_28_49_gain_phase_5_huber8"
LINE_KNOTS = (0, 28, 49)
GAIN_KNOTS = (0, 12, 24, 36, 49)
FRACTIONAL_BITS = 5
SCALE = 1 << FRACTIONAL_BITS
PLY_COUNT = 50
ORBIT_COUNT = 1716
GAIN_COUNT = 19


class ExportError(ValueError):
    """Raised when the frozen model artifact is not exactly the expected model."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ExportError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ExportError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    require(isinstance(value, int) and not isinstance(value, bool),
            f"{name} must be an integer")
    require(minimum <= value <= maximum, f"{name} is outside its C++ storage range")
    return value


def integer_vector(value: Any, name: str, size: int,
                   minimum: int, maximum: int) -> list[int]:
    require(isinstance(value, list) and len(value) == size,
            f"{name} must contain {size} entries")
    return [integer(item, f"{name}[{index}]", minimum, maximum)
            for index, item in enumerate(value)]


def integer_matrix(value: Any, name: str, rows: int, columns: int,
                   minimum: int, maximum: int) -> list[list[int]]:
    require(isinstance(value, list) and len(value) == rows,
            f"{name} must contain {rows} rows")
    return [integer_vector(row, f"{name}[{index}]", columns, minimum, maximum)
            for index, row in enumerate(value)]


def open_report(directory: Path) -> tuple[dict[str, Any], str]:
    root = directory.resolve()
    complete = root / "COMPLETE"
    report_path = root / "report.json"
    require(root.is_dir() and not (root / "INCOMPLETE").exists(),
            f"experiment is incomplete: {root}")
    require(complete.is_file() and report_path.is_file(),
            f"experiment lacks COMPLETE or report.json: {root}")
    try:
        lines = complete.read_text(encoding="utf-8").splitlines()
        report = json.loads(report_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportError(f"could not read frozen model report: {error}") from error
    require(lines[:1] == [REPORT_SCHEMA] and len(lines) == 2,
            "experiment COMPLETE marker is malformed")
    name, separator, expected_digest = lines[1].partition("=")
    require(name == "report_sha256" and bool(separator) and len(expected_digest) == 64,
            "experiment COMPLETE digest is malformed")
    actual_digest = sha256(report_path)
    require(actual_digest == expected_digest,
            "experiment report digest differs from COMPLETE")
    require(isinstance(report, dict) and report.get("schema") == REPORT_SCHEMA and
            report.get("schema_version") == 1,
            "experiment report schema is unsupported")
    return report, actual_digest


def flatten(rows: Iterable[Sequence[int]]) -> list[int]:
    return [value for row in rows for value in row]


def cpp_array(name: str, cpp_type: str, values: Sequence[int],
              *, values_per_line: int = 16) -> str:
    lines = [f"inline constexpr std::array<{cpp_type}, {len(values)}> {name}{{{{"]
    for begin in range(0, len(values), values_per_line):
        chunk = values[begin:begin + values_per_line]
        lines.append("    " + ", ".join(str(value) for value in chunk) + ",")
    lines.append("}};")
    return "\n".join(lines)


def render_header(report: dict[str, Any], report_digest: str) -> str:
    selection = report.get("selection")
    models = report.get("models")
    require(isinstance(selection, dict) and
            selection.get("best_model") == AUTHENTICATED_MODEL_ID,
            "the selected model is not the frozen pattern/gain architecture")
    require(isinstance(models, list), "experiment models must be a list")
    matches = [model for model in models
               if isinstance(model, dict) and model.get("name") == AUTHENTICATED_MODEL_ID]
    require(len(matches) == 1, "the frozen pattern/gain model is missing or duplicated")
    model = matches[0]
    config = model.get("config")
    quantized = model.get("quantization")
    require(isinstance(config, dict) and isinstance(quantized, dict),
            "frozen model configuration or quantization is missing")
    require(tuple(config.get("line_knots", ())) == LINE_KNOTS and
            tuple(config.get("gain_knots", ())) == GAIN_KNOTS,
            "frozen model phase knots differ from the selected architecture")
    require(quantized.get("definition") ==
            "int16-tables-int32-intercept-power-of-two-v1" and
            quantized.get("fractional_bits") == FRACTIONAL_BITS and
            quantized.get("scale") == SCALE,
            "frozen model quantization differs from the selected scale")

    intercept = integer_vector(
        quantized.get("intercept"), "quantization.intercept", PLY_COUNT,
        -(1 << 31), (1 << 31) - 1,
    )
    line_weights = integer_matrix(
        quantized.get("line_weights"), "quantization.line_weights",
        len(LINE_KNOTS), ORBIT_COUNT, -(1 << 15), (1 << 15) - 1,
    )
    gain_weights = integer_matrix(
        quantized.get("gain_weights"), "quantization.gain_weights",
        len(GAIN_KNOTS), GAIN_COUNT, -(1 << 15), (1 << 15) - 1,
    )

    input_metadata = report.get("input")
    require(isinstance(input_metadata, dict), "experiment input metadata is missing")
    corpus_id = input_metadata.get("corpus_id")
    corpus_digest = input_metadata.get("corpus_digest")
    feature_digest = input_metadata.get("feature_binary_sha256")
    require(all(isinstance(value, str) for value in
                (corpus_id, corpus_digest, feature_digest)),
            "experiment input provenance is incomplete")

    pieces = [
        "// Generated by tools/export_minimax_pattern_gain.py. Do not edit by hand.",
        "#ifndef POE2_MINIMAX_FROZEN_PATTERN_GAIN_MODEL_HPP",
        "#define POE2_MINIMAX_FROZEN_PATTERN_GAIN_MODEL_HPP",
        "",
        "#include <array>",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <string_view>",
        "",
        "// clang-format off",
        "namespace poe2::minimax::frozen_pattern_gain_model {",
        "",
        f'inline constexpr std::string_view kArtifactModelId = "{AUTHENTICATED_MODEL_ID}";',
        f'inline constexpr std::string_view kReportSha256 = "{report_digest}";',
        f'inline constexpr std::string_view kCorpusId = "{corpus_id}";',
        f'inline constexpr std::string_view kCorpusDigest = "{corpus_digest}";',
        f'inline constexpr std::string_view kFeatureBinarySha256 = "{feature_digest}";',
        f"inline constexpr int kFractionalBits = {FRACTIONAL_BITS};",
        f"inline constexpr int kScale = {SCALE};",
        f"inline constexpr std::size_t kReversalOrbitCount = {ORBIT_COUNT};",
        f"inline constexpr std::size_t kGainFeatureCount = {GAIN_COUNT};",
        "",
        cpp_array("kLineKnots", "std::uint8_t", LINE_KNOTS),
        "",
        cpp_array("kGainKnots", "std::uint8_t", GAIN_KNOTS),
        "",
        cpp_array("kIntercept", "std::int32_t", intercept, values_per_line=10),
        "",
        cpp_array("kLineWeights", "std::int16_t", flatten(line_weights)),
        "",
        cpp_array("kGainWeights", "std::int16_t", flatten(gain_weights),
                  values_per_line=19),
        "",
        "}  // namespace poe2::minimax::frozen_pattern_gain_model",
        "// clang-format on",
        "",
        "#endif  // POE2_MINIMAX_FROZEN_PATTERN_GAIN_MODEL_HPP",
        "",
    ]
    return "\n".join(pieces)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True,
                        help="completed frozen pattern/gain experiment directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="generated C++ header")
    parser.add_argument("--check", action="store_true",
                        help="verify that --output is already byte-for-byte current")
    arguments = parser.parse_args()

    try:
        report, digest = open_report(arguments.experiment)
        generated = render_header(report, digest).encode()
        output = arguments.output.resolve()
        if arguments.check:
            require(output.is_file() and output.read_bytes() == generated,
                    f"generated model header is stale: {output}")
            print(f"pattern_gain_export_current report_sha256={digest} output={output}")
            return 0

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_bytes(generated)
        os.replace(temporary, output)
        print(
            "pattern_gain_export_complete"
            f" report_sha256={digest}"
            f" line_weights={len(LINE_KNOTS) * ORBIT_COUNT}"
            f" gain_weights={len(GAIN_KNOTS) * GAIN_COUNT}"
            f" output={output}"
        )
    except (ExportError, OSError) as error:
        print(f"pattern/gain export failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
