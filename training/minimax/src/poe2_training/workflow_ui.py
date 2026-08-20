"""Terminal-aware color, timing, and progress for training workflows."""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from collections.abc import Mapping
from typing import Any, TextIO

from tqdm import tqdm

from .workflow_config import WorkflowConfig, stage_fingerprints
from .workflow_state import RunState


_ANSI = {
    "bold": "1", "dim": "2", "red": "31", "green": "32",
    "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36",
}
_FIELD = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
_NINJA_PROGRESS = re.compile(r"^\[(\d+)/(\d+)\]")


def is_interactive(stream: TextIO = sys.stdout) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)()) and os.environ.get("TERM") != "dumb"


def paint(text: str, *styles: str, stream: TextIO = sys.stdout) -> str:
    if not is_interactive(stream) or "NO_COLOR" in os.environ:
        return text
    return f"\033[{';'.join(_ANSI[item] for item in styles)}m{text}\033[0m"


def format_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 10:
        return f"{seconds:.1f}s"
    rounded = int(seconds + 0.5)
    if rounded < 60:
        return f"{rounded}s"
    minutes, remaining = divmod(rounded, 60)
    if minutes < 60:
        return f"{minutes}m {remaining:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def status_symbol(status: str, stream: TextIO = sys.stdout) -> str:
    if status == "complete":
        return paint("✓", "green", stream=stream)
    if status in {"blocked", "failed", "incomplete", "unexpected"}:
        return paint("!", "red", stream=stream)
    if status == "running":
        return paint("●", "cyan", stream=stream)
    return paint("○", "dim", stream=stream)


def human_bar(completed: int, total: int, stream: TextIO = sys.stdout,
              width: int = 24) -> str:
    ratio = completed / total if total else 1.0
    filled = min(width, max(0, round(width * ratio)))
    return paint("█" * filled + "░" * (width - filled), "cyan", stream=stream)


def iteration_pipeline(snapshot: Mapping[str, Any], stream: TextIO = sys.stdout) -> str:
    parts = [
        f"{index} {status_symbol(str(value['status']), stream)} {name}"
        for index, (name, value) in enumerate(snapshot["iterations"].items(), 1)
    ]
    return paint("iterations", "bold", stream=stream) + "  " + "  ──  ".join(parts)


def aggregate_status(statuses: list[str]) -> str:
    """Reduce child statuses with failure and active work taking precedence."""
    if "failed" in statuses:
        return "failed"
    if "running" in statuses:
        return "running"
    return "complete" if statuses and all(value == "complete" for value in statuses) else "pending"


def recorded_seconds(state: RunState) -> float:
    total = 0.0
    for event in state.events:
        if event["type"] == "stage_completed" and isinstance(event.get("timing"), dict):
            total += float(event["timing"].get("creation_seconds", 0.0))
            total += float(event["timing"].get("authentication_seconds", 0.0))
        elif event["type"] in {"stage_failed", "stage_abandoned"}:
            total += float(event.get("wall_seconds", 0.0))
    return total


def progress_detail(operation: str, line: str) -> str | None:
    """Reduce verbose child progress to the current tqdm postfix."""
    values = dict(_FIELD.findall(line))
    for completed_name, total_name in (
        ("completed_trajectories", "total_trajectories"),
        ("completed_games", "total_games"),
        ("completed", "total"),
    ):
        try:
            completed, total = int(values[completed_name]), int(values[total_name])
        except (KeyError, ValueError):
            continue
        label = operation.replace("-", " ")
        if match := re.fullmatch(r"create shard (\d+)", label):
            label = f"shard {int(match.group(1)) + 1}"
        percent = 100.0 * completed / total if total else 0.0
        return f"{label} {completed:,}/{total:,} ({percent:.0f}%)"
    if match := _NINJA_PROGRESS.match(line):
        return f"{operation.replace('-', ' ')} {match[1]}/{match[2]}"
    if line.startswith("pattern_model_complete ") and "name" in values:
        return f"trained {values['name']}"
    return None


def pretty_stage(stage: str) -> str:
    kind, name, step = stage.split(":", 2)
    return f"{kind} {name} · {step.replace('-', ' ')}"


class WorkflowProgress:
    """A single stage bar; full child output remains in per-command logs."""

    def __init__(self, config: WorkflowConfig, state: RunState,
                 stream: TextIO = sys.stdout) -> None:
        self.config, self.stream = config, stream
        self.interactive = is_interactive(stream)
        self._lock = threading.RLock()
        self._started = time.perf_counter()
        self._stage_started: dict[str, float] = {}
        planned = set(stage_fingerprints(config))
        stage_state = state.stage_state()
        self._completed = {
            stage for stage, value in stage_state.items()
            if stage in planned and value["status"] == "complete"
        }
        iteration_status = {}
        for item in config.iterations:
            stages = [stage for stage in planned
                      if stage.startswith(f"iteration:{item.name}:")]
            statuses = [stage_state.get(stage, {}).get("status", "pending")
                        for stage in stages]
            iteration_status[item.name] = {"status": aggregate_status(statuses)}
        print(iteration_pipeline({"iterations": iteration_status}, stream), file=stream)
        self._bar: Any | None = None
        if self.interactive:
            self._bar = tqdm(
                total=len(planned), initial=len(self._completed), desc=config.name,
                unit="stage", dynamic_ncols=True, colour=(
                    "cyan" if "NO_COLOR" not in os.environ else None),
                file=stream, leave=True,
                bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}] {postfix}",
            )
        else:
            print(f"run {config.name}: {len(self._completed)}/{len(planned)} stages complete",
                  file=stream)

    def _render(self, stage: str, detail: str) -> None:
        if self._bar is None:
            return
        self._bar.set_description_str(f"{self.config.name} · {pretty_stage(stage)}", refresh=False)
        self._bar.set_postfix_str(detail, refresh=True)

    def _write(self, message: str) -> None:
        if self._bar is not None:
            tqdm.write(message, file=self.stream)
        else:
            print(message, file=self.stream, flush=True)

    def stage_started(self, stage: str, action: str) -> None:
        with self._lock:
            self._stage_started[stage] = time.perf_counter()
            self._render(stage, action.replace("-", " "))
            if self._bar is None:
                self._write(f"→ {pretty_stage(stage)}  {action.replace('-', ' ')}")

    def command_started(self, stage: str, operation: str) -> None:
        with self._lock:
            self._render(stage, operation.replace("-", " "))

    def command_output(self, stage: str, operation: str, line: str, _elapsed: float) -> None:
        if detail := progress_detail(operation, line):
            with self._lock:
                self._render(stage, detail)

    def stage_completed(self, stage: str, *, created: bool) -> None:
        with self._lock:
            elapsed = time.perf_counter() - self._stage_started.pop(stage, self._started)
            if stage not in self._completed:
                self._completed.add(stage)
                if self._bar is not None:
                    self._bar.update(1)
            reuse = "authenticated · " if not created else ""
            self._write(f"{paint('✓', 'green', stream=self.stream)} {pretty_stage(stage)}  "
                        f"{reuse}{format_duration(elapsed)}")

    def stage_failed(self, stage: str, error: BaseException) -> None:
        with self._lock:
            elapsed = time.perf_counter() - self._stage_started.pop(stage, self._started)
            self._write(f"{paint('✗', 'red', stream=self.stream)} {pretty_stage(stage)}  "
                        f"failed after {format_duration(elapsed)}: {error}")

    def iteration_completed(self, name: str) -> None:
        index = next(index for index, item in enumerate(self.config.iterations, 1)
                     if item.name == name)
        self._write(f"{paint('✓', 'green', stream=self.stream)} iteration "
                    f"{index}/{len(self.config.iterations)}  {name}")

    def close(self, *, success: bool) -> None:
        if self._bar is None:
            return
        self._bar.set_description_str(
            f"{self.config.name} · {'complete' if success else 'stopped'}", refresh=False)
        self._bar.set_postfix_str(format_duration(time.perf_counter() - self._started))
        self._bar.close()
