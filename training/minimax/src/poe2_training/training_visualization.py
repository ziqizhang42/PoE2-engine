"""Dependency-free SVG dashboards for authenticated training reports."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

from .shared import write_report_attachment


_COLORS = ("#58a6ff", "#f778ba", "#3fb950", "#d29922")


def _text(x: float, y: float, value: object, *, size: int = 14,
          fill: str = "#c9d1d9", anchor: str = "start", weight: int = 400) -> str:
    escaped = html.escape(str(value))
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{escaped}</text>')


def _number(value: float) -> str:
    if value and abs(value) < 0.001:
        return f"{value:.1e}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}"


def _line_panel(x: float, y: float, width: float, height: float, *, title: str,
                x_label: str, y_label: str,
                series: list[tuple[str, str, list[tuple[float, float]]]],
                best_x: float | None = None) -> list[str]:
    result = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" '
        'fill="#161b22" stroke="#30363d"/>',
        _text(x + 22, y + 32, title, size=18, fill="#f0f6fc", weight=600),
    ]
    points = [point for _, _, values in series for point in values]
    if not points:
        result.append(_text(x + width / 2, y + height / 2, "No checkpoint path",
                            fill="#8b949e", anchor="middle"))
        return result
    left, right = x + 72, x + width - 24
    top, bottom = y + 62, y + height - 58
    x_values, y_values = zip(*points, strict=False)
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_max = x_min + 1.0
    padding = max((y_max - y_min) * 0.08, max(abs(y_max), 1.0) * 0.01)
    y_min, y_max = y_min - padding, y_max + padding

    def sx(value: float) -> float:
        return left + (value - x_min) * (right - left) / (x_max - x_min)

    def sy(value: float) -> float:
        return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)

    for index in range(5):
        ratio = index / 4
        gy = top + ratio * (bottom - top)
        value = y_max - ratio * (y_max - y_min)
        result.extend((
            f'<line x1="{left}" y1="{gy:.1f}" x2="{right}" y2="{gy:.1f}" '
            'stroke="#30363d"/>',
            _text(left - 10, gy + 5, _number(value), size=11, fill="#8b949e",
                  anchor="end"),
        ))
    for index in range(5):
        ratio = index / 4
        gx = left + ratio * (right - left)
        value = x_min + ratio * (x_max - x_min)
        result.append(_text(gx, bottom + 22, _number(value), size=11,
                            fill="#8b949e", anchor="middle"))
    if best_x is not None and x_min <= best_x <= x_max:
        bx = sx(best_x)
        result.append(f'<line x1="{bx:.1f}" y1="{top}" x2="{bx:.1f}" y2="{bottom}" '
                      'stroke="#d29922" stroke-dasharray="5 5"/>')
        result.append(_text(bx + 5, top + 14, "selected", size=11, fill="#d29922"))
    for label, color, values in series:
        coordinates = " ".join(f"{sx(px):.1f},{sy(py):.1f}" for px, py in values)
        result.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
                      'stroke-width="2.5" stroke-linejoin="round"/>')
    legend_x = left
    for label, color, _ in series:
        result.append(f'<line x1="{legend_x}" y1="{y + 50}" x2="{legend_x + 20}" '
                      f'y2="{y + 50}" stroke="{color}" stroke-width="3"/>')
        result.append(_text(legend_x + 26, y + 55, label, size=12))
        legend_x += 32 + len(label) * 7
    result.extend((
        _text((left + right) / 2, y + height - 16, x_label, size=12,
              fill="#8b949e", anchor="middle"),
        _text(x + 16, (top + bottom) / 2, y_label, size=12,
              fill="#8b949e", anchor="middle"),
    ))
    return result


def _selected_model(report: dict[str, Any]) -> dict[str, Any]:
    models = report.get("models", [])
    if not isinstance(models, list) or not models:
        raise ValueError("training report has no models to visualize")
    selected_name = report.get("selection", {}).get("best_model")
    matches = [model for model in models if model.get("name") == selected_name]
    return matches[0] if len(matches) == 1 else min(
        models, key=lambda model: model["validation"]["overall"]["mae"])


def _quantized_mae(model: dict[str, Any]) -> float | None:
    quantization = model.get("quantization")
    if not isinstance(quantization, dict):
        return None
    value = quantization.get("validation", {}).get("overall", {}).get("mae")
    return float(value) if isinstance(value, (int, float)) else None


def _curve(selected: dict[str, Any]) -> tuple[
        str, str, str, list[tuple[str, str, list[tuple[float, float]]]]]:
    trace = selected.get("trace", [])
    if trace and all("training_loss" in point and "validation_loss" in point for point in trace):
        return (
            "Selected-model loss", "training step", "data loss",
            [
                ("training", _COLORS[0], [
                    (float(point["step"]), float(point["training_loss"])) for point in trace]),
                ("validation", _COLORS[1], [
                    (float(point["step"]), float(point["validation_loss"])) for point in trace]),
            ],
        )
    ridge = selected.get("ridge_path", [])
    if ridge:
        return (
            "Selected-model ridge path", "log10(lambda)", "validation error",
            [
                ("MAE", _COLORS[2], [
                    (math.log10(float(point["lambda"])), float(point["validation_mae"]))
                    for point in ridge]),
                ("RMSE", _COLORS[3], [
                    (math.log10(float(point["lambda"])), float(point["validation_rmse"]))
                    for point in ridge]),
            ],
        )
    return "Checkpoint validation", "training step", "error", [
        ("MAE", _COLORS[2], [
            (float(point["step"]), float(point["validation_mae"])) for point in trace]),
        ("RMSE", _COLORS[3], [
            (float(point["step"]), float(point["validation_rmse"])) for point in trace]),
    ]


def render_training_metrics(report: dict[str, Any]) -> bytes:
    """Render one deterministic dashboard from a completed in-memory report."""
    models = report["models"]
    selected = _selected_model(report)
    metrics = selected["validation"]["overall"]
    width = 1280
    comparison_height = max(390, 82 + 34 * len(models))
    height = max(650, 210 + comparison_height)
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" role="img" '
        f'viewBox="0 0 {width} {height}" '
        'style="font-family:ui-sans-serif,system-ui,sans-serif">',
        '<title>Training metrics dashboard</title>',
        '<desc>Selected-model loss or regularization path and validation model comparison.</desc>',
        f'<rect width="{width}" height="{height}" fill="#0d1117"/>',
        _text(48, 52, "Training metrics", size=30, fill="#f0f6fc", weight=700),
        _text(48, 82, f"selected · {selected['name']}", size=15, fill="#8b949e"),
        _text(48, 122, f"MAE  {_number(float(metrics['mae']))}", size=19,
              fill=_COLORS[2], weight=600),
        _text(230, 122, f"RMSE  {_number(float(metrics['rmse']))}", size=19,
              fill=_COLORS[3], weight=600),
        _text(430, 122, f"sign  {100 * float(metrics['sign_accuracy']):.1f}%", size=19,
              fill=_COLORS[0], weight=600),
    ]
    if (quantized_mae := _quantized_mae(selected)) is not None:
        elements.append(_text(615, 122, f"quantized MAE  {_number(quantized_mae)}", size=19,
                              fill=_COLORS[1], weight=600))
    title, x_label, y_label, series = _curve(selected)
    selected_x = (float(selected["best_step"]) if "best_step" in selected else
                  math.log10(float(selected["selected_ridge_lambda"]))
                  if "selected_ridge_lambda" in selected else None)
    elements.extend(_line_panel(
        40, 150, 760, comparison_height, title=title, x_label=x_label, y_label=y_label,
        series=series, best_x=selected_x,
    ))

    x, y, panel_width = 830, 150, 410
    has_quantized = any(_quantized_mae(model) is not None for model in models)
    subtitle = "lower is better · pink tick is quantized" if has_quantized else "lower is better"
    elements.extend((
        f'<rect x="{x}" y="{y}" width="{panel_width}" height="{comparison_height}" '
        'rx="12" fill="#161b22" stroke="#30363d"/>',
        _text(x + 22, y + 32, "Validation MAE by model", size=18,
              fill="#f0f6fc", weight=600),
        _text(x + 22, y + 53, subtitle, size=12, fill="#8b949e"),
    ))
    values = [float(model["validation"]["overall"]["mae"]) for model in models]
    values.extend(value for model in models if (value := _quantized_mae(model)) is not None)
    maximum = max(max(values), 1.0e-12)
    for index, model in enumerate(models):
        row_y = y + 82 + index * 34
        value = float(model["validation"]["overall"]["mae"])
        label = str(model["name"])
        short = label if len(label) <= 31 else label[:28] + "…"
        color = _COLORS[2] if model is selected else _COLORS[0]
        elements.extend((
            f'<g><title>{html.escape(label)}</title>',
            _text(x + 18, row_y + 14, short, size=11,
                  fill="#f0f6fc" if model is selected else "#c9d1d9"),
            f'<rect x="{x + 212}" y="{row_y + 2}" width="150" height="15" rx="4" '
            'fill="#21262d"/>',
            f'<rect x="{x + 212}" y="{row_y + 2}" width="{150 * value / maximum:.1f}" '
            f'height="15" rx="4" fill="{color}"/>',
            _text(x + 372, row_y + 14, f"{value:.3f}", size=11, anchor="end"),
        ))
        if (quantized := _quantized_mae(model)) is not None:
            marker_x = x + 212 + 150 * quantized / maximum
            elements.append(
                f'<line x1="{marker_x:.1f}" y1="{row_y}" x2="{marker_x:.1f}" '
                f'y2="{row_y + 19}" stroke="{_COLORS[1]}" stroke-width="2"/>')
        elements.append("</g>")
    elements.extend((
        _text(48, height - 22, "Generated deterministically from authenticated report data",
              size=11, fill="#6e7681"),
        "</svg>",
    ))
    return ("\n".join(elements) + "\n").encode()


def write_training_metrics(
    output_directory: Path,
    report: dict[str, Any],
    *,
    error_type: type[ValueError],
) -> dict[str, Any]:
    try:
        contents = render_training_metrics(report)
        return write_report_attachment(
            output_directory, "training-metrics.svg", contents,
            media_type="image/svg+xml", error_type=error_type,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise error_type(f"could not generate training visualization: {error}") from error
