"""Append-only local state and atomic summaries for training workflows."""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .shared import atomic_write_bytes, canonical_json, sha256_file
from .workflow_config import WorkflowConfig, stage_fingerprints


EVENT_SCHEMA = "poe2-training-event-v1"
SUMMARY_SCHEMA = "poe2-training-summary"
ARTIFACT_START = "artifact_started"
# Events that settle a started artifact; one started later with none of these is unfinished.
ARTIFACT_OUTCOMES = frozenset({
    "artifact_completed", "artifact_failed", "artifact_abandoned",
    "interrupted_artifact_removed", "crash_artifact_removed", "crash_artifact_kept",
})


class StateError(ValueError):
    """Raised when workflow state is malformed or incompatible with a config."""


def now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class RunState:
    """Read, append, and reduce one run's operational state."""

    def __init__(self, config: WorkflowConfig) -> None:
        self.config = config
        self.root = config.output_directory
        self.state_directory = self.root / "state"
        self.events_path = self.state_directory / "events.jsonl"
        self.summary_path = self.root / "summary.json"
        self.snapshots_directory = self.state_directory / "configs"
        self._lock = threading.Lock()
        # A crash can cut the final record short; it is ignored here and dropped under the lock.
        self.torn_tail = b""
        self.events = self._read_events()

    def _read_events(self) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        if not self.events_path.is_file():
            raise StateError(f"workflow event path is not a file: {self.events_path}")
        try:
            contents = self.events_path.read_bytes()
        except OSError as error:
            raise StateError(f"could not read {self.events_path}: {error}") from error
        lines = contents.split(b"\n")
        partial = lines.pop()
        result: list[dict[str, Any]] = []
        for index, line in enumerate(lines, 1):
            try:
                event = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                if index == len(lines) and not partial:
                    partial = line + b"\n"
                    break
                raise StateError(
                    f"malformed event at {self.events_path}:{index}; restore the append-only log"
                ) from error
            if (not isinstance(event, dict) or event.get("schema") != EVENT_SCHEMA or
                    event.get("sequence") != index or not isinstance(event.get("type"), str)):
                raise StateError(f"invalid event at {self.events_path}:{index}")
            result.append(event)
        self.torn_tail = partial
        return result

    def repair_torn_tail(self) -> int:
        """Drop an incomplete final record left by a crash; call only under the run lock."""
        dropped = self.torn_tail
        if not dropped:
            return 0
        try:
            with self.events_path.open("r+b") as events:
                events.truncate(self.events_path.stat().st_size - len(dropped))
                events.flush()
                os.fsync(events.fileno())
        except OSError as error:
            raise StateError(f"could not repair {self.events_path}: {error}") from error
        self.torn_tail = b""
        self.append("event_log_repaired", dropped_bytes=len(dropped),
                    dropped_text=dropped[:200].decode("utf-8", "replace"))
        return len(dropped)

    def append(self, event_type: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            if self.torn_tail:
                raise StateError(f"repair the torn final record of {self.events_path} first")
            self.state_directory.mkdir(parents=True, exist_ok=True)
            event = {
                "schema": EVENT_SCHEMA, "sequence": len(self.events) + 1,
                "time_utc": now_utc(), "type": event_type, **fields,
            }
            line = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            try:
                with self.events_path.open("a", encoding="utf-8") as destination:
                    destination.write(line)
                    destination.flush()
                    os.fsync(destination.fileno())
            except OSError as error:
                raise StateError(f"could not append {self.events_path}: {error}") from error
            self.events.append(event)
            return event

    def unfinished_artifacts(self) -> list[str]:
        """Return run-relative artifacts started with no later outcome: work a stop cut short."""
        unfinished: dict[str, None] = {}
        for event in self.events:
            artifact = event.get("artifact")
            if not isinstance(artifact, str):
                continue
            if event["type"] == ARTIFACT_START:
                unfinished[artifact] = None
            elif event["type"] in ARTIFACT_OUTCOMES:
                unfinished.pop(artifact, None)
        return list(unfinished)

    def started_fingerprints(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for event in self.events:
            if event["type"] == "stage_started":
                stage = event.get("stage")
                value = event.get("config_fingerprint")
                if isinstance(stage, str) and isinstance(value, str):
                    result.setdefault(stage, value)
        return result

    def started_input_fingerprint(self, stage: str) -> str | None:
        """Return the immutable input fingerprint from a stage's first attempt."""
        for event in self.events:
            if event["type"] == "stage_started" and event.get("stage") == stage:
                value = event.get("input_fingerprint")
                if isinstance(value, str):
                    return value
        return None

    def validate_evolution(self) -> None:
        current = stage_fingerprints(self.config)
        for stage, locked in self.started_fingerprints().items():
            if stage not in current:
                raise StateError(
                    f"configuration removes started stage {stage}; restore it in {self.config.path}")
            if current[stage] != locked:
                raise StateError(
                    f"configuration changes started stage {stage}; use its accepted snapshot under "
                    f"{self.snapshots_directory}")

    def accept_config(self) -> tuple[int, Path, bool]:
        """Snapshot a compatible config revision, returning revision/path/new."""
        self.validate_evolution()
        accepted = [event for event in self.events if event["type"] == "config_accepted"]
        for event in accepted:
            if event.get("digest") == self.config.digest:
                snapshot = self.root / str(event["snapshot"])
                if not snapshot.is_file() or sha256_file(snapshot) != self.config.digest:
                    raise StateError(f"accepted config snapshot is missing or changed: {snapshot}")
                return int(event["revision"]), snapshot, False

        self.snapshots_directory.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.snapshots_directory.glob("*.toml"))
        revision = len(existing) + 1
        snapshot = self.snapshots_directory / f"{revision:04d}-{self.config.digest}.toml"
        if snapshot.exists():
            if snapshot.read_bytes() != self.config.contents:
                raise StateError(f"config snapshot collision: {snapshot}")
        else:
            try:
                atomic_write_bytes(snapshot, self.config.contents)
            except OSError as error:
                raise StateError(f"could not snapshot config at {snapshot}: {error}") from error
        relative = snapshot.relative_to(self.root).as_posix()
        self.append("config_accepted", revision=revision, digest=self.config.digest,
                    snapshot=relative, config_path=str(self.config.path))
        return revision, snapshot, True

    def stage_state(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for event in self.events:
            stage = event.get("stage")
            if not isinstance(stage, str):
                continue
            record = result.setdefault(stage, {
                "status": "pending", "attempts": 0, "failures": 0,
                "statistics": None, "last_event": None,
            })
            event_type = event["type"]
            if event_type == "stage_started":
                record["status"] = "running"
                record["attempts"] += 1
                record["config_fingerprint"] = event.get("config_fingerprint")
                record["input_fingerprint"] = event.get("input_fingerprint")
                record["git_commit"] = event.get("git_commit")
            elif event_type in {"stage_failed", "stage_abandoned"}:
                record["status"] = "failed"
                record["failures"] += 1
                record["error"] = event.get("error")
            elif event_type == "stage_completed":
                record["status"] = "complete"
                record["statistics"] = event.get("statistics")
                record["timing"] = event.get("timing")
                record.pop("error", None)
            elif event_type == "stage_authenticated":
                record["last_authentication"] = event.get("timing")
            record["last_event"] = event["sequence"]
        return result

    def pin_for_dataset(self, name: str) -> dict[str, Any] | None:
        matches = [event for event in self.events
                   if event["type"] == "import_pinned" and event.get("dataset") == name]
        return matches[-1].get("digests") if matches else None

    def recover_running_stages(self) -> None:
        for stage, record in self.stage_state().items():
            if record["status"] == "running":
                self.append(
                    "stage_abandoned", stage=stage,
                    error="previous orchestrator ended before recording stage completion")

    def attempt_count(self, stage: str) -> int:
        return int(self.stage_state().get(stage, {}).get("attempts", 0))

    def summary(self, live: dict[str, Any] | None = None) -> dict[str, Any]:
        accepted = [event for event in self.events if event["type"] == "config_accepted"]
        revisions = [{key: event.get(key) for key in ("revision", "digest", "snapshot", "time_utc")}
                     for event in accepted]
        attempts = sum(1 for event in self.events if event["type"] == "stage_started")
        failures = sum(1 for event in self.events
                       if event["type"] in {"stage_failed", "stage_abandoned"})
        return {
            "schema": SUMMARY_SCHEMA, "schema_version": 1,
            "run": self.config.name, "output_directory": str(self.root),
            "config_digest": self.config.digest, "config_revisions": revisions,
            "event_count": len(self.events), "attempts": attempts, "failures": failures,
            "stages": self.stage_state(), "live": live,
            "updated_at_utc": now_utc(),
        }

    def write_summary(self, live: dict[str, Any] | None = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            atomic_write_bytes(self.summary_path, canonical_json(self.summary(live)), replace=True)
        except OSError as error:
            raise StateError(f"could not update {self.summary_path}: {error}") from error
