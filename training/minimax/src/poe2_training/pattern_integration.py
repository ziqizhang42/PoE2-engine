"""Verify frozen pattern/gain C++ inference against Python fixed-point inference."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import MappedFeatureDataset, subset_batch
from .pattern_evaluation import _selected_model
from .pattern_experiment import open_pattern_report
from .quantization import QuantizedPatternModel, predict_quantized_fixed


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _quantized_model(model: dict[str, Any]) -> QuantizedPatternModel:
    config = model.get("config")
    quantization = model.get("quantization")
    _require(isinstance(config, dict) and isinstance(quantization, dict),
             "selected model lacks quantized tables")
    line_knots = tuple(config.get("line_knots", ()))
    gain_value = config.get("gain_knots")
    gain_knots = tuple(gain_value) if gain_value is not None else None
    fractional_bits = quantization.get("fractional_bits")
    _require(isinstance(fractional_bits, int) and not isinstance(fractional_bits, bool),
             "quantized model fractional bits are invalid")
    intercept = np.asarray(quantization.get("intercept"), dtype=np.int64)
    line_weights = np.asarray(quantization.get("line_weights"), dtype=np.int64)
    gain_weights = np.asarray(quantization.get("gain_weights"), dtype=np.int64)
    _require(line_knots == (0, 28, 49) and gain_knots == (0, 12, 24, 36, 49),
             "selected model is not the frozen combined pattern/gain architecture")
    _require(fractional_bits == 5 and quantization.get("scale") == 32,
             "selected model does not use the frozen scale-32 quantization")
    _require(intercept.shape == (50,) and line_weights.shape == (3, 1716) and
             gain_weights.shape == (5, 19),
             "quantized model table dimensions are invalid")
    _require(bool((intercept >= np.iinfo(np.int32).min).all()) and
             bool((intercept <= np.iinfo(np.int32).max).all()) and
             bool((line_weights >= np.iinfo(np.int16).min).all()) and
             bool((line_weights <= np.iinfo(np.int16).max).all()) and
             bool((gain_weights >= np.iinfo(np.int16).min).all()) and
             bool((gain_weights <= np.iinfo(np.int16).max).all()),
             "quantized model table value is outside its declared storage type")
    return QuantizedPatternModel(
        name=str(model.get("name")),
        fractional_bits=fractional_bits,
        line_knots=line_knots,
        gain_knots=gain_knots,
        intercept=np.ascontiguousarray(intercept, dtype=np.int32),
        line_weights=np.ascontiguousarray(line_weights, dtype=np.int16),
        gain_weights=np.ascontiguousarray(gain_weights, dtype=np.int16),
    )


def _run_inference(binary: Path, evaluator: str,
                   player_one: np.ndarray, player_two: np.ndarray,
                   iterations: int = 0) -> str:
    _require(player_one.shape == player_two.shape, "inference bitboards are not aligned")
    lines = "".join(
        f"{int(one):016x} {int(two):016x}\n"
        for one, two in zip(player_one, player_two)
    )
    command = [str(binary), "--evaluator", evaluator]
    if iterations > 0:
        command.extend(("--iterations", str(iterations)))
    completed = subprocess.run(
        command,
        input=lines,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"C++ inference failed with status {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _predictions(binary: Path, evaluator: str,
                 player_one: np.ndarray, player_two: np.ndarray) -> np.ndarray:
    output = _run_inference(binary, evaluator, player_one, player_two)
    try:
        values = np.asarray([int(line) for line in output.splitlines()], dtype=np.int64)
    except ValueError as error:
        raise RuntimeError("C++ inference emitted a malformed fixed score") from error
    _require(values.shape == player_one.shape,
             "C++ inference emitted the wrong number of fixed scores")
    return values


def _transform_bitboard(bits: int, symmetry: int) -> int:
    transformed = 0
    while bits:
        index = (bits & -bits).bit_length() - 1
        bits &= bits - 1
        row, column = divmod(index, 7)
        if symmetry == 1:
            next_row, next_column = column, 6 - row
        elif symmetry == 2:
            next_row, next_column = 6 - row, 6 - column
        elif symmetry == 3:
            next_row, next_column = 6 - column, row
        elif symmetry == 4:
            next_row, next_column = 6 - row, column
        elif symmetry == 5:
            next_row, next_column = row, 6 - column
        elif symmetry == 6:
            next_row, next_column = column, row
        elif symmetry == 7:
            next_row, next_column = 6 - column, 6 - row
        else:
            next_row, next_column = row, column
        transformed |= 1 << (next_row * 7 + next_column)
    return transformed


def run_verification(dataset_path: Path, experiment_path: Path, binary: Path,
                     sample_count: int, symmetry_sample_count: int,
                     verify_digest: bool, benchmark_iterations: int) -> None:
    report = open_pattern_report(experiment_path)
    model = _quantized_model(_selected_model(report))
    _require(binary.is_file(), f"C++ inference binary does not exist: {binary}")

    with MappedFeatureDataset(dataset_path, verify_digest=verify_digest) as dataset:
        metadata = report.get("input")
        _require(isinstance(metadata, dict) and
                 metadata.get("corpus_id") == dataset.artifact.manifest["corpus"]["id"] and
                 metadata.get("feature_binary_sha256") == dataset.artifact.binary_digest and
                 metadata.get("feature_manifest_sha256") == dataset.artifact.manifest_digest,
                 "training report and feature artifact identities differ")
        count = min(sample_count, dataset.artifact.record_count)
        indices = np.unique(np.linspace(
            0, dataset.artifact.record_count - 1, num=count, dtype=np.int64,
        ))
        mask = np.zeros(dataset.artifact.record_count, dtype=np.bool_)
        mask[indices] = True
        batch = subset_batch(dataset.materialize(), mask)
        records = dataset.records[indices]
        player_one = np.ascontiguousarray(records["player_one"], dtype=np.uint64)
        player_two = np.ascontiguousarray(records["player_two"], dtype=np.uint64)

    python_residual = predict_quantized_fixed(batch, model)
    closure_fixed = np.asarray(batch.two_ply_closure_values, dtype=np.int64) * model.scale
    python_pattern_gain = closure_fixed + python_residual
    exact_endgame = (49 - np.asarray(batch.plys, dtype=np.int64)) <= 2
    python_pattern_gain[exact_endgame] = closure_fixed[exact_endgame]

    cpp_closure = _predictions(binary, "two-ply-closure", player_one, player_two)
    if not np.array_equal(cpp_closure, closure_fixed):
        mismatch = int(np.flatnonzero(cpp_closure != closure_fixed)[0])
        raise ValueError(
            "C++ position reconstruction/two-ply closure differs from the feature artifact at "
            f"sample {mismatch}: cpp={cpp_closure[mismatch]} "
            f"python={closure_fixed[mismatch]}"
        )
    cpp_pattern_gain = _predictions(binary, "pattern-gain", player_one, player_two)
    if not np.array_equal(cpp_pattern_gain, python_pattern_gain):
        mismatch = int(np.flatnonzero(cpp_pattern_gain != python_pattern_gain)[0])
        raise ValueError(
            "C++ frozen pattern/gain inference differs from Python at "
            f"sample {mismatch}: cpp={cpp_pattern_gain[mismatch]} "
            f"python={python_pattern_gain[mismatch]}"
        )

    symmetry_count = min(symmetry_sample_count, player_one.size)
    transformed_one: list[int] = []
    transformed_two: list[int] = []
    transformed_expected: list[int] = []
    for index in range(symmetry_count):
        for symmetry in range(1, 8):
            transformed_one.append(_transform_bitboard(int(player_one[index]), symmetry))
            transformed_two.append(_transform_bitboard(int(player_two[index]), symmetry))
            transformed_expected.append(int(python_pattern_gain[index]))
    if transformed_one:
        symmetry_cpp = _predictions(
            binary,
            "pattern-gain",
            np.asarray(transformed_one, dtype=np.uint64),
            np.asarray(transformed_two, dtype=np.uint64),
        )
        symmetry_expected = np.asarray(transformed_expected, dtype=np.int64)
        if not np.array_equal(symmetry_cpp, symmetry_expected):
            mismatch = int(np.flatnonzero(symmetry_cpp != symmetry_expected)[0])
            raise ValueError(
                "C++ frozen pattern/gain inference is not D4 invariant at transformed sample "
                f"{mismatch}: cpp={symmetry_cpp[mismatch]} "
                f"python={symmetry_expected[mismatch]}"
            )

    print(
        "pattern_gain_cpp_python_parity_valid"
        f" base_positions={player_one.size}"
        f" transformed_positions={len(transformed_one)}"
        f" exact_endgames={int(np.count_nonzero(exact_endgame))}"
        f" feature_binary_sha256={report['input'].get('feature_binary_sha256')}"
    )
    if benchmark_iterations > 0:
        for evaluator in ("two-ply-closure", "pattern-gain"):
            print(_run_inference(
                binary, evaluator, player_one, player_two, benchmark_iterations
            ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="authenticated feature artifact used for training")
    parser.add_argument("--experiment", type=Path, required=True,
                        help="authenticated frozen pattern/gain experiment")
    parser.add_argument("--inference-binary", type=Path, required=True,
                        help="release poe2_minimax_infer executable")
    parser.add_argument("--samples", type=int, default=4096,
                        help="deterministically sampled base positions")
    parser.add_argument("--symmetry-samples", type=int, default=512,
                        help="base positions checked under seven nonidentity D4 transforms")
    parser.add_argument("--benchmark-iterations", type=int, default=0,
                        help="also benchmark this many full passes over the base sample")
    parser.add_argument("--skip-digest", action="store_true",
                        help="skip the full feature-binary SHA-256 pass")
    arguments = parser.parse_args()
    if arguments.samples <= 0 or arguments.symmetry_samples < 0 or \
            arguments.benchmark_iterations < 0:
        parser.error("sample counts must be positive/nonnegative")
    try:
        run_verification(
            arguments.dataset,
            arguments.experiment,
            arguments.inference_binary.resolve(),
            arguments.samples,
            arguments.symmetry_samples,
            not arguments.skip_digest,
            arguments.benchmark_iterations,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"pattern/gain integration verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
