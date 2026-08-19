"""Public model fitting and metric helpers shared by training experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .dataset import NumpyFeatureBatch, TERMINAL_FLAG


DEFAULT_RIDGE_LAMBDAS = (
    1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3,
    1.0e-2, 1.0e-1, 1.0, 10.0, 100.0,
)
PHASE_BUCKETS = (
    ("ply_00_06", 0, 6), ("ply_07_12", 7, 12),
    ("ply_13_18", 13, 18), ("ply_19_24", 19, 24),
    ("ply_25_30", 25, 30), ("ply_31_36", 31, 36),
    ("ply_37_42", 37, 42), ("ply_43_49", 43, 49),
)


@dataclass(frozen=True)
class RidgeFit:
    weights: np.ndarray
    selected_lambda: float
    path: tuple[dict[str, float], ...]


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray,
              validation_x: np.ndarray, validation_y: np.ndarray,
              lambdas: Sequence[float], device: Any,
              *, error_type: type[ValueError] = ValueError) -> RidgeFit:
    """Fit and select deterministic float64 ridge regression."""
    import torch

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise error_type(message)

    require(train_x.ndim == 2 and validation_x.ndim == 2 and
            train_x.shape[1] == validation_x.shape[1],
            "ridge design matrices have incompatible shapes")
    require(train_x.shape[0] > 0 and validation_x.shape[0] > 0 and train_x.shape[1] > 0,
            "ridge fitting requires nonempty train and validation data")
    require(train_y.shape == (train_x.shape[0],) and
            validation_y.shape == (validation_x.shape[0],),
            "ridge targets have incompatible shapes")
    require(bool(lambdas) and all(math.isfinite(value) and value > 0.0 for value in lambdas),
            "ridge lambdas must be finite and positive")

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
        error = validation_design @ weights - validation_target
        mae = float(error.abs().mean().cpu())
        rmse = float(error.square().mean().sqrt().cpu())
        require(math.isfinite(mae) and math.isfinite(rmse),
                "ridge path produced a non-finite validation metric")
        entry = {"lambda": float(ridge_lambda), "validation_mae": mae,
                 "validation_rmse": rmse}
        candidates.append(((mae, rmse, float(ridge_lambda)), weights, entry))
    candidates.sort(key=lambda candidate: candidate[0])
    _, selected_weights, selected_entry = candidates[0]
    ordered_path = tuple(candidate[2] for candidate in
                         sorted(candidates, key=lambda value: value[2]["lambda"]))
    return RidgeFit(
        weights=np.ascontiguousarray(selected_weights.detach().cpu().numpy(), dtype=np.float64),
        selected_lambda=selected_entry["lambda"], path=ordered_path,
    )


def scalar_metrics(teacher: np.ndarray, prediction: np.ndarray, close_margin: int,
                   *, error_type: type[ValueError] = ValueError) -> dict[str, Any]:
    target = np.asarray(teacher, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    if target.shape != estimate.shape or target.ndim != 1 or target.size == 0:
        raise error_type("metrics require matching nonempty vectors")
    error = estimate - target
    absolute = np.abs(error)
    close = np.abs(target) <= close_margin
    sign_correct = np.sign(estimate) == np.sign(target)
    return {
        "records": int(target.size), "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mean_error": float(error.mean()),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "maximum_absolute_error": float(absolute.max()),
        "sign_accuracy": float(sign_correct.mean()), "close_margin": int(close_margin),
        "close_records": int(close.sum()),
        "close_sign_accuracy": float(sign_correct[close].mean()) if bool(close.any()) else None,
    }


def model_metrics(batch: NumpyFeatureBatch, residual_prediction: np.ndarray,
                  close_margin: int, *,
                  error_type: type[ValueError] = ValueError) -> dict[str, Any]:
    prediction = np.asarray(batch.two_ply_closure_values, dtype=np.float64) + residual_prediction
    teacher = np.asarray(batch.teacher_values, dtype=np.float64)
    plys = np.asarray(batch.plys)
    terminal = (np.asarray(batch.flags) & TERMINAL_FLAG) != 0
    metric = lambda one, two: scalar_metrics(one, two, close_margin, error_type=error_type)
    result: dict[str, Any] = {
        "overall": metric(teacher, prediction), "by_label": {},
        "by_parity": {}, "by_phase": {},
    }
    for name, mask in (("exact_terminal", terminal), ("teacher", ~terminal)):
        if bool(mask.any()):
            result["by_label"][name] = metric(teacher[mask], prediction[mask])
    for name, mask in (("even_ply", plys % 2 == 0), ("odd_ply", plys % 2 == 1)):
        if bool(mask.any()):
            result["by_parity"][name] = metric(teacher[mask], prediction[mask])
    for name, first, last in PHASE_BUCKETS:
        mask = (plys >= first) & (plys <= last)
        if bool(mask.any()):
            result["by_phase"][name] = metric(teacher[mask], prediction[mask])
    return result


def model_comparison(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_overall = reference["validation"]["overall"]
    candidate_overall = candidate["validation"]["overall"]

    def improvement(old: float, new: float) -> float | None:
        return None if old == 0.0 else float(100.0 * (old - new) / old)

    return {
        "validation_mae_reduction_percent": improvement(
            reference_overall["mae"], candidate_overall["mae"]),
        "validation_rmse_reduction_percent": improvement(
            reference_overall["rmse"], candidate_overall["rmse"]),
        "validation_sign_accuracy_change": float(
            candidate_overall["sign_accuracy"] - reference_overall["sign_accuracy"]),
    }

