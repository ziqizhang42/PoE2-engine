"""Train deterministic two-ply-closure residual baselines without the test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dataset import MappedFeatureDataset, NumpyFeatureBatch, TERMINAL_FLAG
from .summaries import (
    GAIN_SUMMARY_NAMES,
    PHASE_FEATURE_NAMES,
    fit_standardization,
    gain_summaries,
    phase_features,
    phase_interactions,
    phase_interpolation,
)


EXPERIMENT_SCHEMA = "poe2-minimax-baseline-experiment"
EXPERIMENT_SCHEMA_VERSION = 1
COMPLETE_HEADER = "poe2-minimax-baseline-experiment"
DEFAULT_RIDGE_LAMBDAS = (
    1.0e-8,
    1.0e-7,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
    100.0,
)
PHASE_BUCKETS = (
    ("ply_00_06", 0, 6),
    ("ply_07_12", 7, 12),
    ("ply_13_18", 13, 18),
    ("ply_19_24", 19, 24),
    ("ply_25_30", 25, 30),
    ("ply_31_36", 31, 36),
    ("ply_37_42", 37, 42),
    ("ply_43_49", 43, 49),
)
GAIN_PHASE_KNOTS = (
    (0, 49),
    (0, 24, 49),
    (0, 12, 24, 36, 49),
)


class BaselineExperimentError(ValueError):
    """Raised when a baseline experiment cannot be run or authenticated."""


@dataclass(frozen=True)
class RidgeFit:
    weights: np.ndarray
    selected_lambda: float
    path: tuple[dict[str, float], ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineExperimentError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise BaselineExperimentError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def _training_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_digest() -> str:
    root = _training_project_root()
    fixed = (root / ".python-version", root / "pyproject.toml", root / "uv.lock")
    sources = sorted((root / "src").rglob("*.py"))
    digest = hashlib.sha256()
    for path in (*fixed, *sources):
        relative = path.relative_to(root).as_posix().encode()
        contents = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "little"))
        digest.update(contents)
    return digest.hexdigest()


def _repository_root() -> Path | None:
    current = _training_project_root()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _git_provenance() -> dict[str, Any]:
    root = _repository_root()
    if root is None:
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise BaselineExperimentError(f"could not inspect Git provenance: {error}") from error
    _require(len(commit) == 40, "Git did not return a full commit ID")
    return {"commit": commit, "dirty": bool(status)}


def _device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        selected = requested
    if selected == "cuda" and not torch.cuda.is_available():
        raise BaselineExperimentError("CUDA/ROCm was requested but no device is available")
    return torch.device(selected)


def _fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    lambdas: Sequence[float],
    device: Any,
) -> RidgeFit:
    import torch

    if train_x.ndim != 2 or validation_x.ndim != 2 or train_x.shape[1] != validation_x.shape[1]:
        raise BaselineExperimentError("ridge design matrices have incompatible shapes")
    if train_x.shape[0] == 0 or validation_x.shape[0] == 0 or train_x.shape[1] == 0:
        raise BaselineExperimentError("ridge fitting requires nonempty train and validation data")
    if train_y.shape != (train_x.shape[0],) or validation_y.shape != (validation_x.shape[0],):
        raise BaselineExperimentError("ridge targets have incompatible shapes")
    if not lambdas or any(not math.isfinite(value) or value <= 0.0 for value in lambdas):
        raise BaselineExperimentError("ridge lambdas must be finite and positive")

    train_design = torch.as_tensor(train_x, dtype=torch.float64, device=device)
    train_target = torch.as_tensor(train_y, dtype=torch.float64, device=device)
    validation_design = torch.as_tensor(validation_x, dtype=torch.float64, device=device)
    validation_target = torch.as_tensor(validation_y, dtype=torch.float64, device=device)
    count = float(train_x.shape[0])
    gram = (train_design.T @ train_design) / count
    right_hand = (train_design.T @ train_target) / count
    identity = torch.eye(train_x.shape[1], dtype=torch.float64, device=device)

    candidates: list[tuple[tuple[float, float, float], Any, dict[str, float]]] = []
    for ridge_lambda in lambdas:
        weights = torch.linalg.solve(gram + ridge_lambda * identity, right_hand)
        prediction = validation_design @ weights
        error = prediction - validation_target
        mae = float(error.abs().mean().cpu())
        rmse = float(error.square().mean().sqrt().cpu())
        if not math.isfinite(mae) or not math.isfinite(rmse):
            raise BaselineExperimentError("ridge path produced a non-finite validation metric")
        entry = {
            "lambda": float(ridge_lambda),
            "validation_mae": mae,
            "validation_rmse": rmse,
        }
        candidates.append(((mae, rmse, float(ridge_lambda)), weights, entry))

    candidates.sort(key=lambda candidate: candidate[0])
    _, selected_weights, selected_entry = candidates[0]
    ordered_path = tuple(candidate[2] for candidate in sorted(candidates, key=lambda value: value[2]["lambda"]))
    return RidgeFit(
        weights=np.ascontiguousarray(selected_weights.detach().cpu().numpy(), dtype=np.float64),
        selected_lambda=selected_entry["lambda"],
        path=ordered_path,
    )


def _scalar_metrics(teacher: np.ndarray, prediction: np.ndarray, close_margin: int) -> dict[str, Any]:
    target = np.asarray(teacher, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if target.shape != estimate.shape or target.ndim != 1 or target.size == 0:
        raise BaselineExperimentError("metrics require matching nonempty vectors")
    error = estimate - target
    absolute = np.abs(error)
    close = np.abs(target) <= close_margin
    sign_correct = np.sign(estimate) == np.sign(target)
    return {
        "records": int(target.size),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_error": float(error.mean()),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "maximum_absolute_error": float(absolute.max()),
        "sign_accuracy": float(sign_correct.mean()),
        "close_margin": int(close_margin),
        "close_records": int(close.sum()),
        "close_sign_accuracy": float(sign_correct[close].mean()) if bool(close.any()) else None,
    }


def _metrics(batch: NumpyFeatureBatch, residual_prediction: np.ndarray,
             close_margin: int) -> dict[str, Any]:
    prediction = (np.asarray(batch.two_ply_closure_values, dtype=np.float64) +
                  residual_prediction)
    teacher = np.asarray(batch.teacher_values, dtype=np.float64)
    plys = np.asarray(batch.plys)
    terminal = (np.asarray(batch.flags) & TERMINAL_FLAG) != 0

    result: dict[str, Any] = {
        "overall": _scalar_metrics(teacher, prediction, close_margin),
        "by_label": {},
        "by_parity": {},
        "by_phase": {},
    }
    for name, mask in (("exact_terminal", terminal), ("teacher", ~terminal)):
        if bool(mask.any()):
            result["by_label"][name] = _scalar_metrics(
                teacher[mask], prediction[mask], close_margin
            )
    for name, mask in (("even_ply", plys % 2 == 0), ("odd_ply", plys % 2 == 1)):
        if bool(mask.any()):
            result["by_parity"][name] = _scalar_metrics(
                teacher[mask], prediction[mask], close_margin
            )
    for name, first, last in PHASE_BUCKETS:
        mask = (plys >= first) & (plys <= last)
        if bool(mask.any()):
            result["by_phase"][name] = _scalar_metrics(
                teacher[mask], prediction[mask], close_margin
            )
    return result


def _improvement(reference: float, candidate: float) -> float | None:
    if reference == 0.0:
        return None
    return float(100.0 * (reference - candidate) / reference)


def _model_comparison(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_overall = reference["validation"]["overall"]
    candidate_overall = candidate["validation"]["overall"]
    return {
        "validation_mae_reduction_percent": _improvement(
            reference_overall["mae"], candidate_overall["mae"]
        ),
        "validation_rmse_reduction_percent": _improvement(
            reference_overall["rmse"], candidate_overall["rmse"]
        ),
        "validation_sign_accuracy_change": float(
            candidate_overall["sign_accuracy"] - reference_overall["sign_accuracy"]
        ),
    }


def _reserve_output(directory: Path) -> None:
    try:
        directory.parent.mkdir(parents=True, exist_ok=True)
        directory.mkdir()
        (directory / "INCOMPLETE").write_text(
            f"{EXPERIMENT_SCHEMA}\n", encoding="utf-8"
        )
    except OSError as error:
        raise BaselineExperimentError(f"could not reserve output directory {directory}: {error}") from error


def _write_bytes(path: Path, contents: bytes) -> None:
    with path.open("xb") as destination:
        destination.write(contents)
        destination.flush()
        os.fsync(destination.fileno())


def _complete_output(directory: Path, report: dict[str, Any]) -> None:
    report_bytes = (json.dumps(
        report, sort_keys=True, indent=2, allow_nan=False
    ) + "\n").encode()
    report_digest = hashlib.sha256(report_bytes).hexdigest()
    temporary_report = directory / "report.json.tmp"
    final_report = directory / "report.json"
    temporary_complete = directory / "COMPLETE.tmp"
    final_complete = directory / "COMPLETE"
    complete_bytes = (
        f"{COMPLETE_HEADER}\n"
        f"report_sha256={report_digest}\n"
    ).encode()
    try:
        _write_bytes(temporary_report, report_bytes)
        os.replace(temporary_report, final_report)
        _write_bytes(temporary_complete, complete_bytes)
        os.replace(temporary_complete, final_complete)
        (directory / "INCOMPLETE").unlink()
    except OSError as error:
        raise BaselineExperimentError(f"could not complete output directory {directory}: {error}") from error


def open_baseline_report(directory: Path | str) -> dict[str, Any]:
    """Authenticate a completed baseline report."""
    root = Path(directory).resolve()
    _require(root.is_dir(), f"baseline report directory does not exist: {root}")
    _require(not (root / "INCOMPLETE").exists(), "baseline report is incomplete")
    complete_path = root / "COMPLETE"
    report_path = root / "report.json"
    _require(complete_path.is_file() and report_path.is_file(), "baseline report is incomplete")
    try:
        lines = complete_path.read_text(encoding="utf-8").splitlines()
        report = json.loads(report_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BaselineExperimentError(f"could not read baseline report: {error}") from error
    _require(lines[:1] == [COMPLETE_HEADER] and len(lines) == 2,
             "baseline COMPLETE marker is malformed")
    name, separator, digest = lines[1].partition("=")
    _require(name == "report_sha256" and bool(separator) and len(digest) == 64,
             "baseline COMPLETE digest is malformed")
    _require(_sha256(report_path) == digest, "baseline report digest differs from COMPLETE")
    _require(isinstance(report, dict) and report.get("schema") == EXPERIMENT_SCHEMA and
             report.get("schema_version") == EXPERIMENT_SCHEMA_VERSION,
             "baseline report schema is unsupported")
    return report


def _runtime(torch: Any, device: Any) -> dict[str, Any]:
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = platform.processor() or platform.machine()
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device_type": device.type,
        "device_name": device_name,
        "float_dtype": "float64",
    }


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
        _reserve_output(output)
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

        phase_fit = _fit_ridge(
            train_phase, train_target, validation_phase, validation_target,
            ridge_lambdas, device,
        )
        gain_fit = _fit_ridge(
            train_combined, train_target, validation_combined, validation_target,
            ridge_lambdas, device,
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
            fit = _fit_ridge(
                train_design, train_target, validation_design, validation_target,
                ridge_lambdas, device,
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
            "train": _metrics(train, zero_train, close_margin),
            "validation": _metrics(validation, zero_validation, close_margin),
        }
        phase_model = {
            "name": "phase",
            "definition": "one-hot-ply-residual-ridge-v1",
            "parameters": len(PHASE_FEATURE_NAMES),
            "feature_names": list(PHASE_FEATURE_NAMES),
            "weights": phase_fit.weights.tolist(),
            "selected_ridge_lambda": phase_fit.selected_lambda,
            "ridge_path": list(phase_fit.path),
            "train": _metrics(train, phase_train_prediction, close_margin),
            "validation": _metrics(validation, phase_validation_prediction, close_margin),
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
            "train": _metrics(train, gain_train_prediction, close_margin),
            "validation": _metrics(validation, gain_validation_prediction, close_margin),
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
                "train": _metrics(train, train_design @ fit.weights, close_margin),
                "validation": _metrics(
                    validation, validation_design @ fit.weights, close_margin
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
                "training_source_sha256": _source_digest(),
                "git": _git_provenance(),
                "runtime": _runtime(torch, device),
            },
            "models": [closure_model, phase_model, gain_model, *phase_aware_models],
            "comparisons": {
                "phase_vs_two_ply_closure": _model_comparison(closure_model, phase_model),
                "gain_summary_vs_two_ply_closure": _model_comparison(
                    closure_model, gain_model
                ),
                "gain_summary_vs_phase": _model_comparison(phase_model, gain_model),
                **{
                    f"{model['name']}_vs_gain_summary": _model_comparison(gain_model, model)
                    for model in phase_aware_models
                },
            },
        }
        _complete_output(output, report)
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
