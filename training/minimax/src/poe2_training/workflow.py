"""Create-only orchestration for reproducible PoE2 training runs."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from .shared import (
    atomic_write_bytes,
    canonical_json,
    complete_json_report,
    reserve_report_directory,
    sha256_file,
)
from .workflow_adapters import (
    CANDIDATE_SCHEMA,
    GATE_SCHEMA,
    aggregate_label_stats,
    authenticate_candidate,
    authenticate_evaluation,
    authenticate_features,
    authenticate_gate,
    authenticate_label_shard,
    authenticate_source,
    authenticate_training,
    candidate_build_id,
    data_binary,
    engine_gate_command,
    evaluation_command,
    export_command,
    feature_command,
    feature_audit_command,
    gate_build_directory,
    label_command,
    label_audit_command,
    label_preflight_command,
    label_shard_path,
    parity_command,
    resolve_eval_binary,
    search_benchmark_command,
    source_command,
    source_audit_command,
    training_command,
)
from .workflow_config import (
    ImportedDatasetConfig,
    IterationConfig,
    ManagedDatasetConfig,
    WorkflowConfig,
    fingerprint,
    stage_fingerprints,
)
from .workflow_state import RunState, now_utc


class WorkflowError(ValueError):
    """Raised when an orchestration operation cannot safely proceed."""


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    exit_code: int
    output: str
    log_path: Path
    metrics_path: Path
    metrics: dict[str, Any]


class CommandExecutionError(WorkflowError):
    def __init__(self, result: CommandResult) -> None:
        self.result = result
        super().__init__(
            f"command exited {result.exit_code}; inspect {result.log_path}: "
            f"{shlex.join(result.command)}")


class CommandRunner:
    """Run logged subprocess groups and terminate all of them on interruption."""

    def __init__(self, repository: Path) -> None:
        self.repository = repository
        self._active: dict[int, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()
        self._print_lock = threading.Lock()

    @staticmethod
    def _parse_time(path: Path) -> dict[str, float | int | None]:
        result: dict[str, float | int | None] = {
            "cpu_user_seconds": None, "cpu_system_seconds": None, "peak_rss_kb": None,
        }
        if not path.is_file():
            return result
        try:
            fields = dict(line.split("=", 1) for line in
                          path.read_text(encoding="utf-8").splitlines() if "=" in line)
            result["cpu_user_seconds"] = float(fields["user_seconds"])
            result["cpu_system_seconds"] = float(fields["system_seconds"])
            result["peak_rss_kb"] = int(fields["peak_rss_kb"])
        except (OSError, KeyError, ValueError):
            pass
        return result

    def run(self, command: Sequence[str], *, log_path: Path, metrics_path: Path,
            accepted_exit_codes: set[int] | None = None) -> CommandResult:
        accepted = accepted_exit_codes or {0}
        log_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists() or metrics_path.exists():
            raise WorkflowError(
                f"command log or metrics path already exists; preserve it and use a new attempt: "
                f"{log_path}")
        raw_time = metrics_path.with_suffix(metrics_path.suffix + ".time")
        wrapped = list(command)
        if Path("/usr/bin/time").is_file():
            wrapped = [
                "/usr/bin/time", "-f",
                "user_seconds=%U\nsystem_seconds=%S\npeak_rss_kb=%M",
                "-o", str(raw_time), "--", *wrapped,
            ]
        started = time.perf_counter()
        started_utc = now_utc()
        output_parts: list[str] = []
        with log_path.open("x", encoding="utf-8") as log:
            log.write(f"started_at_utc={started_utc}\ncommand={shlex.join(command)}\n")
            log.flush()
            try:
                process = subprocess.Popen(
                    wrapped, cwd=self.repository, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True,
                )
            except OSError as error:
                raise WorkflowError(f"could not start {shlex.join(command)}: {error}") from error
            with self._lock:
                self._active[process.pid] = process
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    output_parts.append(line)
                    log.write(line)
                    log.flush()
                    with self._print_lock:
                        print(line, end="", flush=True)
                process.stdout.close()
                exit_code = process.wait()
            except BaseException:
                self._terminate(process)
                raise
            finally:
                if process.stdout is not None and not process.stdout.closed:
                    process.stdout.close()
                with self._lock:
                    self._active.pop(process.pid, None)
        wall = time.perf_counter() - started
        metrics: dict[str, Any] = {
            "started_at_utc": started_utc, "finished_at_utc": now_utc(),
            "wall_seconds": wall, "exit_code": exit_code,
            **self._parse_time(raw_time),
        }
        try:
            atomic_write_bytes(metrics_path, canonical_json(metrics))
            if raw_time.exists():
                raw_time.unlink()
        except OSError as error:
            raise WorkflowError(f"could not record command metrics at {metrics_path}: {error}") from error
        result = CommandResult(tuple(command), exit_code, "".join(output_parts),
                               log_path, metrics_path, metrics)
        if exit_code not in accepted:
            raise CommandExecutionError(result)
        return result

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 5.0
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()

    def terminate_all(self) -> None:
        with self._lock:
            processes = list(self._active.values())
        for process in processes:
            self._terminate(process)


def resolved_label_concurrency(config: WorkflowConfig,
                               dataset: ManagedDatasetConfig | None = None) -> int:
    """Resolve label processes from logical CPUs and per-shard workers."""
    if config.label_concurrency is not None:
        return config.label_concurrency
    workers = dataset.labels.workers if dataset is not None else max(
        (item.labels.workers for item in config.datasets
         if isinstance(item, ManagedDatasetConfig)), default=1)
    return max(1, ((os.cpu_count() or 1) - 1) // workers)


def _git_status(config: WorkflowConfig) -> tuple[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.repository,
            check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=config.repository, check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise WorkflowError(f"could not inspect Git worktree: {error}") from error
    if len(commit) != 40:
        raise WorkflowError("Git did not return a full commit ID")
    return commit, status


def require_clean_committed(config: WorkflowConfig) -> str:
    commit, status = _git_status(config)
    if status:
        details = "\n".join(line for line in status.splitlines()[:20])
        raise WorkflowError(
            "artifact creation requires a clean committed worktree; commit or stash these paths:\n"
            f"{details}")
    try:
        config_relative = config.path.relative_to(config.repository)
    except ValueError as error:
        raise WorkflowError(
            f"artifact creation requires a repository-owned config: {config.path}") from error
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", config_relative.as_posix()],
        cwd=config.repository, capture_output=True, text=True, check=False)
    if tracked.returncode != 0:
        raise WorkflowError(
            f"artifact creation requires a committed config file: {config.path}")
    try:
        output_relative = config.output_directory.relative_to(config.repository)
    except ValueError:
        output_relative = None
    if output_relative is not None:
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", output_relative.as_posix()],
            cwd=config.repository, check=False)
        if ignored.returncode == 1:
            raise WorkflowError(
                "a run root inside the repository must be ignored so artifacts do not dirty "
                f"later stages: {config.output_directory}")
        if ignored.returncode not in {0, 1}:
            raise WorkflowError(
                f"could not verify that the run root is ignored: {config.output_directory}")
    return commit


@contextmanager
def _interruptions_as_keyboard() -> Iterator[None]:
    """Translate process-termination signals so subprocess groups can be reaped."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    handled = (signal.SIGTERM, signal.SIGHUP)
    previous = {number: signal.getsignal(number) for number in handled}

    def interrupt(number: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {number}")

    try:
        for number in handled:
            signal.signal(number, interrupt)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


@contextmanager
def run_lock(config: WorkflowConfig) -> Iterator[None]:
    config.output_directory.mkdir(parents=True, exist_ok=True)
    path = config.output_directory / ".orchestrator.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WorkflowError(f"another orchestrator holds the run lock: {path}") from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _path_state(path: Path) -> str:
    if not path.exists():
        return "missing"
    if not path.is_dir():
        return "unexpected"
    if (path / "INCOMPLETE").exists():
        return "incomplete"
    if (path / "COMPLETE").is_file():
        return "complete"
    return "unexpected"


def _blocked_path(path: Path, kind: str) -> WorkflowError:
    return WorkflowError(
        f"{kind} path exists but is not a complete authenticatable artifact; "
        f"move it aside or remove it explicitly, then retry: {path}")


def workflow_snapshot(config: WorkflowConfig) -> dict[str, Any]:
    """Return a read-only marker-level snapshot of all configured work."""
    datasets: dict[str, Any] = {}
    for dataset in config.datasets:
        if isinstance(dataset, ImportedDatasetConfig):
            datasets[dataset.name] = {
                "kind": "imported", "features": _path_state(dataset.feature_directory),
                "path": str(dataset.feature_directory),
            }
            continue
        shard_states = [_path_state(label_shard_path(dataset, index))
                        for index in range(dataset.source.shards)]
        datasets[dataset.name] = {
            "kind": "managed", "source": _path_state(dataset.source_directory),
            "labels": {
                "status": ("complete" if shard_states and
                           all(item == "complete" for item in shard_states) else
                           "partial" if any(item == "complete" for item in shard_states) else
                           "missing" if all(item == "missing" for item in shard_states) else "blocked"),
                "complete_shards": sum(item == "complete" for item in shard_states),
                "total_shards": len(shard_states), "shards": shard_states,
            },
            "features": _path_state(dataset.feature_directory),
        }
    iterations: dict[str, Any] = {}
    for iteration in config.iterations:
        value: dict[str, Any] = {
            "dataset": iteration.dataset, "depends_on": list(iteration.depends_on),
            "training": _path_state(iteration.training_directory),
        }
        if iteration.sealed_evaluation is not None:
            value["sealed_evaluation"] = _path_state(iteration.evaluation_directory)
        if iteration.candidate_validation is not None:
            value["candidate_validation"] = _path_state(iteration.candidate_directory)
        if iteration.engine_gate is not None:
            value["engine_gate"] = _path_state(iteration.gate_directory)
        stages = [item for key, item in value.items()
                  if key not in {"dataset", "depends_on"}]
        value["status"] = ("complete" if stages and all(item == "complete" for item in stages)
                           else "blocked" if any(item in {"incomplete", "unexpected"} for item in stages)
                           else "pending")
        iterations[iteration.name] = value
    all_dataset_complete = all(
        value["features"] == "complete" for value in datasets.values())
    all_iterations_complete = all(value["status"] == "complete" for value in iterations.values())
    return {
        "schema": "poe2-training-status", "schema_version": 1,
        "run": config.name, "config": str(config.path), "config_digest": config.digest,
        "output_directory": str(config.output_directory), "build_preset": config.build_preset,
        "logical_cpus": os.cpu_count() or 1,
        "resolved_label_concurrency": resolved_label_concurrency(config),
        "datasets": datasets, "iterations": iterations,
        "status": "complete" if all_dataset_complete and all_iterations_complete else "pending",
    }


def print_status(config: WorkflowConfig, *, json_output: bool = False) -> None:
    snapshot = workflow_snapshot(config)
    state = RunState(config)
    snapshot["state"] = state.summary(snapshot).get("stages") if state.events else {}
    if json_output:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
        return
    print(f"run {config.name}: {snapshot['status']} ({config.output_directory})")
    for name, dataset in snapshot["datasets"].items():
        if dataset["kind"] == "managed":
            labels = dataset["labels"]
            print(f"  dataset {name}: source={dataset['source']} "
                  f"labels={labels['complete_shards']}/{labels['total_shards']} "
                  f"features={dataset['features']}")
        else:
            print(f"  dataset {name}: imported features={dataset['features']}")
    for name, iteration in snapshot["iterations"].items():
        stages = " ".join(f"{key}={value}" for key, value in iteration.items()
                          if key not in {"dataset", "depends_on", "status"})
        print(f"  iteration {name}: {iteration['status']} {stages}")


def _format_command(command: Sequence[str]) -> str:
    return shlex.join([str(item) for item in command])


def print_plan(config: WorkflowConfig) -> None:
    """Print a read-only execution plan with derived paths and representative commands."""
    print(f"run {config.name}")
    print(f"  output: {config.output_directory}")
    print(f"  preset: {config.build_preset}")
    print(f"  logical CPUs: {os.cpu_count() or 1}")
    print(f"  default label concurrency: {resolved_label_concurrency(config)}")
    for dataset in config.datasets:
        print(f"dataset {dataset.name} ({dataset.kind})")
        if isinstance(dataset, ImportedDatasetConfig):
            print(f"  authenticate: {dataset.feature_directory}")
            continue
        print(f"  source: {dataset.source_directory}")
        print(f"    {_format_command(source_command(config, dataset))}")
        print(f"  labels: {dataset.labels_directory} "
              f"({dataset.source.shards} shards, concurrency "
              f"{resolved_label_concurrency(config, dataset)})")
        print(f"    {_format_command(label_command(config, dataset, 0))}")
        print(f"  features: {dataset.feature_directory}")
        print(f"    {_format_command(feature_command(config, dataset))}")
    for iteration in config.iterations:
        dataset = config.dataset(iteration.dataset)
        print(f"iteration {iteration.name}: dataset={iteration.dataset} "
              f"depends_on={','.join(iteration.depends_on) or '-'}")
        print(f"  training: {_format_command(training_command(sys.executable, dataset, iteration))}")
        if iteration.sealed_evaluation is not None:
            print(f"  sealed evaluation: "
                  f"{_format_command(evaluation_command(sys.executable, config, iteration))}")
        if iteration.candidate_validation is not None:
            print(f"  candidate validation: {iteration.candidate_directory} "
                  f"promotable={str(iteration.candidate_validation.promotable).lower()}")
        if iteration.engine_gate is not None:
            print(f"  engine gate: {iteration.gate_directory} base={iteration.engine_gate.base}")


def _direct_audit(command: Sequence[str], repository: Path) -> str:
    completed = subprocess.run(command, cwd=repository, text=True,
                               capture_output=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise WorkflowError(
            f"authentication command failed ({_format_command(command)}): {detail}")
    return completed.stdout


def validate_existing(config: WorkflowConfig) -> dict[str, Any]:
    """Read-only validation of config evolution and every artifact currently present."""
    state = RunState(config)
    state.validate_evolution()
    authenticated: dict[str, Any] = {}
    for dataset in config.datasets:
        feature_state = _path_state(dataset.feature_directory)
        if isinstance(dataset, ImportedDatasetConfig):
            if feature_state != "complete":
                raise _blocked_path(dataset.feature_directory, "imported feature")
            stats = authenticate_features(dataset)
            pin = state.pin_for_dataset(dataset.name)
            if pin is not None and pin != {
                    "binary_sha256": stats["binary_sha256"],
                    "manifest_sha256": stats["manifest_sha256"]}:
                raise WorkflowError(f"imported dataset changed after pinning: {dataset.name}")
            authenticated[f"dataset:{dataset.name}:features"] = stats
            continue
        source_state = _path_state(dataset.source_directory)
        if source_state in {"incomplete", "unexpected"}:
            raise _blocked_path(dataset.source_directory, "position source")
        if source_state == "complete":
            _direct_audit(
                source_audit_command(sys.executable, config, dataset), config.repository)
            authenticated[f"dataset:{dataset.name}:source"] = authenticate_source(dataset)
        complete_shards = 0
        for index in range(dataset.source.shards):
            shard_path = label_shard_path(dataset, index)
            shard_state = _path_state(shard_path)
            if shard_state in {"incomplete", "unexpected"}:
                raise _blocked_path(shard_path, "label shard")
            if shard_state == "complete":
                _direct_audit(
                    label_audit_command(sys.executable, config, dataset, index),
                    config.repository)
                authenticate_label_shard(dataset, index)
                complete_shards += 1
        if complete_shards not in {0, dataset.source.shards}:
            authenticated[f"dataset:{dataset.name}:labels"] = {
                "complete_shards": complete_shards, "total_shards": dataset.source.shards}
        elif complete_shards == dataset.source.shards:
            output = _direct_audit(
                label_preflight_command(sys.executable, config, dataset), config.repository)
            authenticated[f"dataset:{dataset.name}:labels"] = json.loads(output)
        if feature_state in {"incomplete", "unexpected"}:
            raise _blocked_path(dataset.feature_directory, "feature artifact")
        if feature_state == "complete":
            _direct_audit(
                feature_audit_command(sys.executable, config, dataset), config.repository)
            authenticated[f"dataset:{dataset.name}:features"] = authenticate_features(dataset)
    for iteration in config.iterations:
        stages = [
            ("training", iteration.training_directory,
             lambda: authenticate_training(config.dataset(iteration.dataset), iteration)),
        ]
        if iteration.sealed_evaluation is not None:
            stages.append(("sealed-evaluation", iteration.evaluation_directory,
                           lambda item=iteration: authenticate_evaluation(config, item)))
        if iteration.candidate_validation is not None:
            stages.append(("candidate-validation", iteration.candidate_directory,
                           lambda item=iteration: authenticate_candidate(config, item)))
        if iteration.engine_gate is not None:
            stages.append(("engine-gate", iteration.gate_directory,
                           lambda item=iteration: authenticate_gate(config, item)))
        for name, path, authenticate in stages:
            path_state = _path_state(path)
            if path_state in {"incomplete", "unexpected"}:
                raise _blocked_path(path, name)
            if path_state == "complete":
                authenticated[f"iteration:{iteration.name}:{name}"] = authenticate()
    return authenticated


def _parse_key_values(line: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for item in line.split()[1:]:
        name, separator, value = item.partition("=")
        if not separator:
            continue
        try:
            fields[name] = int(value)
            continue
        except ValueError:
            pass
        try:
            fields[name] = float(value)
        except ValueError:
            fields[name] = value
    return fields


class WorkflowRunner:
    def __init__(self, config: WorkflowConfig, state: RunState, commit: str) -> None:
        self.config = config
        self.state = state
        self.commit = commit
        self.commands = CommandRunner(config.repository)
        self._data_ready = False

    def _paths(self, stage: str, operation: str, attempt: int) -> tuple[Path, Path]:
        safe = stage.replace(":", "-")
        directory = self.config.output_directory / "logs" / safe
        stem = f"attempt-{attempt:03d}-{operation}"
        return directory / f"{stem}.log", directory / f"{stem}.resource.json"

    def _command(self, stage: str, operation: str, command: Sequence[str],
                 *, attempt: int | None = None,
                 accepted_exit_codes: set[int] | None = None) -> CommandResult:
        number = attempt or max(1, self.state.attempt_count(stage))
        log, metrics = self._paths(stage, operation, number)
        return self.commands.run(command, log_path=log, metrics_path=metrics,
                                 accepted_exit_codes=accepted_exit_codes)

    def _ensure_data_binary(self, stage: str, attempt: int) -> None:
        if self._data_ready and data_binary(self.config).is_file():
            return
        build = self.config.output_directory / "build" / self.config.build_preset
        self._command(stage, "configure", [
            "cmake", "--preset", self.config.build_preset, "-B", str(build)],
            attempt=attempt)
        self._command(stage, "build-data", [
            "cmake", "--build", str(build),
            "--target", "poe2_minimax_data"], attempt=attempt)
        if not data_binary(self.config).is_file():
            raise WorkflowError(f"data binary was not built: {data_binary(self.config)}")
        self._data_ready = True

    def _begin(self, stage: str, action: str, input_value: Any) -> int:
        input_fingerprint = fingerprint(input_value)
        locked_input = self.state.started_input_fingerprint(stage)
        if locked_input is not None and locked_input != input_fingerprint:
            raise WorkflowError(
                f"inputs to started stage {stage} changed; restore the authenticated "
                "upstream artifacts recorded by its first attempt")
        attempt = self.state.attempt_count(stage) + 1
        label_concurrency = None
        if stage.startswith("dataset:") and stage.endswith(":labels"):
            dataset_name = stage.split(":", 2)[1]
            dataset = self.config.dataset(dataset_name)
            if isinstance(dataset, ManagedDatasetConfig):
                label_concurrency = resolved_label_concurrency(self.config, dataset)
        self.state.append(
            "stage_started", stage=stage, attempt=attempt, action=action,
            config_fingerprint=stage_fingerprints(self.config)[stage],
            input_fingerprint=input_fingerprint, git_commit=self.commit,
            resolved_label_concurrency=label_concurrency,
        )
        self.state.write_summary(workflow_snapshot(self.config))
        return attempt

    def _complete(self, stage: str, stats: dict[str, Any], *, creation: float,
                  authentication: float, import_seconds: float = 0.0) -> None:
        self.state.append(
            "stage_completed", stage=stage, statistics=stats,
            timing={"creation_seconds": creation,
                    "authentication_seconds": authentication,
                    "import_seconds": import_seconds},
        )
        self.state.write_summary(workflow_snapshot(self.config))

    def _fail(self, stage: str, error: BaseException, started: float) -> None:
        self.state.append("stage_failed", stage=stage,
                          wall_seconds=time.perf_counter() - started, error=str(error))
        self.state.write_summary(workflow_snapshot(self.config))

    def _run_simple(self, stage: str, path: Path, kind: str, input_value: Any,
                    create: Callable[[int], None],
                    authenticate: Callable[[int], dict[str, Any]]) -> dict[str, Any]:
        path_state = _path_state(path)
        if path_state in {"incomplete", "unexpected"}:
            raise _blocked_path(path, kind)
        action = "create" if path_state == "missing" else "authenticate"
        attempt = self._begin(stage, action, input_value)
        stage_started = time.perf_counter()
        creation = 0.0
        try:
            if path_state == "missing":
                started = time.perf_counter()
                create(attempt)
                creation = time.perf_counter() - started
            auth_started = time.perf_counter()
            stats = authenticate(attempt)
            authentication = time.perf_counter() - auth_started
            self._complete(stage, stats, creation=creation, authentication=authentication)
            return stats
        except BaseException as error:
            self._fail(stage, error, stage_started)
            raise

    def prepare_dataset(self, name: str) -> dict[str, Any]:
        dataset = self.config.dataset(name)
        if isinstance(dataset, ImportedDatasetConfig):
            return self._prepare_import(dataset)
        source_stats = self._prepare_source(dataset)
        label_stats = self._prepare_labels(dataset, source_stats)
        return self._prepare_features(dataset, source_stats, label_stats)

    def _prepare_import(self, dataset: ImportedDatasetConfig) -> dict[str, Any]:
        stage = f"dataset:{dataset.name}:features"
        if _path_state(dataset.feature_directory) != "complete":
            raise _blocked_path(dataset.feature_directory, "imported feature")
        attempt = self._begin(stage, "import", {"path": str(dataset.feature_directory)})
        started = time.perf_counter()
        try:
            stats = authenticate_features(dataset)
            pin = {"binary_sha256": stats["binary_sha256"],
                   "manifest_sha256": stats["manifest_sha256"]}
            existing = self.state.pin_for_dataset(dataset.name)
            if existing is not None and existing != pin:
                raise WorkflowError(
                    f"imported dataset {dataset.name} changed after first use: "
                    f"{dataset.feature_directory}")
            if existing is None:
                self.state.append("import_pinned", dataset=dataset.name, digests=pin,
                                  path=str(dataset.feature_directory))
            elapsed = time.perf_counter() - started
            self._complete(stage, stats, creation=0.0, authentication=elapsed,
                           import_seconds=elapsed)
            return stats
        except BaseException as error:
            self._fail(stage, error, started)
            raise

    def _prepare_source(self, dataset: ManagedDatasetConfig) -> dict[str, Any]:
        stage = f"dataset:{dataset.name}:source"

        def create(attempt: int) -> None:
            self._ensure_data_binary(stage, attempt)
            self._command(stage, "create", source_command(self.config, dataset), attempt=attempt)

        def authenticate(attempt: int) -> dict[str, Any]:
            self._command(
                stage, "audit",
                source_audit_command(sys.executable, self.config, dataset), attempt=attempt)
            return authenticate_source(dataset)

        return self._run_simple(stage, dataset.source_directory, "position source",
                                asdict(dataset.source), create, authenticate)

    def _prepare_labels(self, dataset: ManagedDatasetConfig,
                        source_stats: dict[str, Any]) -> dict[str, Any]:
        stage = f"dataset:{dataset.name}:labels"
        root = dataset.labels_directory
        if root.exists():
            if not root.is_dir():
                raise _blocked_path(root, "label corpus")
            unexpected = [item for item in root.iterdir() if item.name != "shards"]
            if unexpected:
                raise WorkflowError(
                    f"label corpus has unexpected output; move it aside explicitly: {unexpected[0]}")
        attempt = self._begin(stage, "resume-or-create", source_stats)
        stage_started = time.perf_counter()
        creation = 0.0
        authentication = 0.0
        try:
            root.mkdir(parents=True, exist_ok=True)
            (root / "shards").mkdir(exist_ok=True)
            complete: dict[int, dict[str, Any]] = {}
            missing: list[int] = []
            for index in range(dataset.source.shards):
                shard = label_shard_path(dataset, index)
                shard_state = _path_state(shard)
                if shard_state in {"incomplete", "unexpected"}:
                    raise _blocked_path(shard, "label shard")
                if shard_state == "missing":
                    missing.append(index)
                    continue
                auth_started = time.perf_counter()
                self._command(
                    stage, f"audit-shard-{index:03d}",
                    label_audit_command(sys.executable, self.config, dataset, index),
                    attempt=attempt)
                complete[index] = authenticate_label_shard(dataset, index)
                authentication += time.perf_counter() - auth_started

            if missing:
                self._ensure_data_binary(stage, attempt)
                create_started = time.perf_counter()

                def run_shard(index: int) -> tuple[int, dict[str, Any], float]:
                    command_result = self._command(
                        stage, f"create-shard-{index:03d}",
                        label_command(self.config, dataset, index), attempt=attempt)
                    auth_started = time.perf_counter()
                    self._command(
                        stage, f"audit-new-shard-{index:03d}",
                        label_audit_command(sys.executable, self.config, dataset, index),
                        attempt=attempt)
                    stats = authenticate_label_shard(dataset, index)
                    stats["command_metrics"] = command_result.metrics
                    return index, stats, time.perf_counter() - auth_started

                executor = ThreadPoolExecutor(
                    max_workers=resolved_label_concurrency(self.config, dataset),
                    thread_name_prefix=f"labels-{dataset.name}")
                futures: list[Future[tuple[int, dict[str, Any], float]]] = []
                try:
                    futures = [executor.submit(run_shard, index) for index in missing]
                    for future in as_completed(futures):
                        index, stats, auth_elapsed = future.result()
                        complete[index] = stats
                        authentication += auth_elapsed
                        self.state.append(
                            "label_shard_completed", stage=stage, shard=index,
                            complete_shards=len(complete), total_shards=dataset.source.shards,
                            statistics=stats)
                        self.state.write_summary(workflow_snapshot(self.config))
                except BaseException:
                    for future in futures:
                        future.cancel()
                    self.commands.terminate_all()
                    raise
                finally:
                    executor.shutdown(wait=True, cancel_futures=True)
                creation = time.perf_counter() - create_started

            ordered = [complete[index] for index in range(dataset.source.shards)]
            auth_started = time.perf_counter()
            preflight = self._command(
                stage, "preflight",
                label_preflight_command(sys.executable, self.config, dataset), attempt=attempt)
            authentication += time.perf_counter() - auth_started
            try:
                preflight_stats = json.loads(preflight.output)
            except json.JSONDecodeError as error:
                raise WorkflowError(f"corpus preflight did not emit JSON: {preflight.log_path}") from error
            stats = {**aggregate_label_stats(ordered), "preflight": preflight_stats,
                     "resolved_concurrency": resolved_label_concurrency(self.config, dataset)}
            self._complete(stage, stats, creation=creation, authentication=authentication)
            return stats
        except BaseException as error:
            self._fail(stage, error, stage_started)
            raise

    def _prepare_features(self, dataset: ManagedDatasetConfig,
                          source_stats: dict[str, Any],
                          label_stats: dict[str, Any]) -> dict[str, Any]:
        stage = f"dataset:{dataset.name}:features"

        def create(attempt: int) -> None:
            self._ensure_data_binary(stage, attempt)
            self._command(stage, "create", feature_command(self.config, dataset), attempt=attempt)

        def authenticate(attempt: int) -> dict[str, Any]:
            self._command(
                stage, "audit",
                feature_audit_command(sys.executable, self.config, dataset), attempt=attempt)
            return authenticate_features(dataset)

        return self._run_simple(
            stage, dataset.feature_directory, "feature artifact",
            {"source": source_stats, "labels": label_stats}, create, authenticate)

    def run_iteration(self, iteration: IterationConfig) -> None:
        dependency_stats = {}
        for name in iteration.depends_on:
            dependency = self.config.iteration(name)
            if not self._iteration_artifacts_complete(dependency):
                raise WorkflowError(
                    f"iteration {iteration.name} requires completed dependency {name}; "
                    "rerun with --with-dependencies or run that iteration first")
            dependency_stats[name] = self._authenticate_iteration_outputs(dependency)
        dataset_stats = self.prepare_dataset(iteration.dataset)
        training_stats = self._run_training(iteration, dataset_stats, dependency_stats)
        evaluation_stats = None
        if iteration.sealed_evaluation is not None:
            for name in iteration.sealed_evaluation.exclude_datasets:
                self.prepare_dataset(name)
            evaluation_stats = self._run_evaluation(iteration, training_stats)
        candidate_stats = None
        if iteration.candidate_validation is not None:
            candidate_stats = self._run_candidate(
                iteration, dataset_stats, training_stats, evaluation_stats)
        if iteration.engine_gate is not None:
            self._run_gate(iteration, candidate_stats)

    def _authenticate_iteration_outputs(self,
                                        iteration: IterationConfig) -> dict[str, Any]:
        """Authenticate every configured output before a dependency is reused."""
        training = authenticate_training(
            self.config.dataset(iteration.dataset), iteration)
        if iteration.sealed_evaluation is not None:
            authenticate_evaluation(self.config, iteration)
        if iteration.candidate_validation is not None:
            authenticate_candidate(self.config, iteration)
        if iteration.engine_gate is not None:
            authenticate_gate(self.config, iteration)
        return training

    def _run_training(self, iteration: IterationConfig, dataset_stats: dict[str, Any],
                      dependencies: dict[str, Any]) -> dict[str, Any]:
        stage = f"iteration:{iteration.name}:training"
        dataset = self.config.dataset(iteration.dataset)

        def create(attempt: int) -> None:
            self._command(stage, "create", training_command(
                sys.executable, dataset, iteration), attempt=attempt)

        def authenticate(_attempt: int) -> dict[str, Any]:
            return authenticate_training(dataset, iteration)

        return self._run_simple(
            stage, iteration.training_directory, "training report",
            {"dataset": dataset_stats, "dependencies": dependencies}, create, authenticate)

    def _run_evaluation(self, iteration: IterationConfig,
                        training_stats: dict[str, Any]) -> dict[str, Any]:
        stage = f"iteration:{iteration.name}:sealed-evaluation"

        def create(attempt: int) -> None:
            self._command(stage, "create", evaluation_command(
                sys.executable, self.config, iteration), attempt=attempt)

        def authenticate(_attempt: int) -> dict[str, Any]:
            return authenticate_evaluation(self.config, iteration)

        return self._run_simple(
            stage, iteration.evaluation_directory, "sealed evaluation",
            training_stats, create, authenticate)

    def _run_candidate(self, iteration: IterationConfig, dataset_stats: dict[str, Any],
                       training_stats: dict[str, Any],
                       evaluation_stats: dict[str, Any] | None) -> dict[str, Any]:
        stage = f"iteration:{iteration.name}:candidate-validation"
        candidate = iteration.candidate_validation
        assert candidate is not None
        if not candidate.opening_book.is_file():
            raise WorkflowError(
                f"candidate opening book is missing: {candidate.opening_book}")
        dataset = self.config.dataset(iteration.dataset)
        root = iteration.candidate_directory

        def create(attempt: int) -> None:
            reserve_report_directory(root, CANDIDATE_SCHEMA)
            command_results: list[CommandResult] = []
            export = self._command(stage, "export", export_command(
                sys.executable, self.config, iteration), attempt=attempt)
            command_results.append(export)
            header = root / "frozen_pattern_gain_model.hpp"
            build = root / "build"
            configure = self._command(stage, "configure", [
                "cmake", "--preset", self.config.build_preset, "-B", str(build),
                f"-DPOE2_MINIMAX_MODEL_HEADER={header}",
            ], attempt=attempt)
            command_results.append(configure)
            compile_result = self._command(stage, "build", [
                "cmake", "--build", str(build), "--target",
                "poe2_minimax_infer", "poe2_minimax", "poe2_runner",
            ], attempt=attempt)
            command_results.append(compile_result)
            inference = build / "engines" / "minimax" / "poe2_minimax_infer"
            engine = build / "engines" / "minimax" / "poe2_minimax"
            parity = self._command(stage, "parity", parity_command(
                sys.executable, dataset, iteration, inference), attempt=attempt)
            command_results.append(parity)
            search = self._command(stage, "search-benchmark", search_benchmark_command(
                sys.executable, self.config, iteration, engine), attempt=attempt)
            command_results.append(search)
            parity_lines = [line for line in parity.output.splitlines()
                            if line.startswith("pattern_gain_cpp_python_parity_valid")]
            if len(parity_lines) != 1:
                raise WorkflowError(f"parity command emitted no unique result: {parity.log_path}")
            evaluator_benchmarks = {
                fields.get("evaluator", f"result-{index}"): fields
                for index, line in enumerate(parity.output.splitlines())
                if line.startswith("benchmark ")
                for fields in [_parse_key_values(line)]
            }
            search_benchmarks = {
                fields.get("evaluator", f"result-{index}"): fields
                for index, line in enumerate(search.output.splitlines())
                if line.startswith("search_benchmark ")
                for fields in [_parse_key_values(line)]
            }
            comparison = next((
                _parse_key_values(line) for line in search.output.splitlines()
                if line.startswith("search_benchmark_comparison")), {})
            artifacts = {
                "model_header": header, "inference_binary": inference, "engine_binary": engine,
                "runner_binary": build / "runner" / "poe2_runner",
            }
            for name, path in artifacts.items():
                if not path.is_file():
                    raise WorkflowError(f"candidate build did not produce {name}: {path}")
            report = {
                "schema": CANDIDATE_SCHEMA, "schema_version": 1,
                "created_at_utc": now_utc(), "promotable": candidate.promotable,
                "git_commit": self.commit,
                "settings": {
                    "promotable": candidate.promotable, "samples": candidate.samples,
                    "symmetry_samples": candidate.symmetry_samples,
                    "benchmark_iterations": candidate.benchmark_iterations,
                    "opening_book": str(candidate.opening_book),
                    "search_positions": candidate.search_positions,
                    "search_movetime_ms": candidate.search_movetime_ms,
                },
                "opening_book_sha256": sha256_file(candidate.opening_book),
                "training_report_sha256": training_stats["report_sha256"],
                "sealed_evaluation_report_sha256": (
                    evaluation_stats["report_sha256"] if evaluation_stats else None),
                "feature_binary_sha256": dataset_stats["binary_sha256"],
                "model": {
                    "name": training_stats["selected_model"],
                    "float": {
                        "validation": training_stats["selected_validation"],
                        "sealed_test": (evaluation_stats.get("metrics")
                                        if evaluation_stats else None),
                    },
                    "quantized": {
                        "validation": training_stats["selected_quantized_validation"],
                        "sealed_test": (evaluation_stats.get("quantized_metrics")
                                        if evaluation_stats else None),
                    },
                },
                "parity": {"passed": True, **_parse_key_values(parity_lines[0])},
                "evaluator_benchmark": evaluator_benchmarks,
                "search_benchmark": {"evaluators": search_benchmarks,
                                     "comparison": comparison},
                "artifacts": {
                    name: {"path": path.relative_to(root).as_posix(),
                           "sha256": sha256_file(path)} for name, path in artifacts.items()
                },
                "commands": [{
                    "command": list(item.command),
                    "log": item.log_path.relative_to(self.config.output_directory).as_posix(),
                    "log_sha256": sha256_file(item.log_path), "resources": item.metrics,
                } for item in command_results],
            }
            complete_json_report(root, CANDIDATE_SCHEMA, report)

        def authenticate(_attempt: int) -> dict[str, Any]:
            return authenticate_candidate(self.config, iteration)

        return self._run_simple(
            stage, root, "candidate", {
                "dataset": dataset_stats,
                "training": training_stats,
                "evaluation": evaluation_stats,
                "opening_book_sha256": sha256_file(candidate.opening_book),
            }, create, authenticate)

    def _run_gate(self, iteration: IterationConfig,
                  candidate_stats: dict[str, Any] | None) -> dict[str, Any]:
        stage = f"iteration:{iteration.name}:engine-gate"
        gate = iteration.engine_gate
        assert gate is not None
        if not gate.opening_book.is_file():
            raise WorkflowError(f"engine-gate opening book is missing: {gate.opening_book}")
        root = iteration.gate_directory
        base_engine_binary = resolve_eval_binary(
            self.config, gate.base, gate.base_engine)
        if candidate_stats is None:
            raise WorkflowError("engine gate requires authenticated candidate statistics")
        header_metadata = candidate_stats.get("artifacts", {}).get("model_header")
        if not isinstance(header_metadata, dict) or not isinstance(
                header_metadata.get("sha256"), str):
            raise WorkflowError("candidate report has no authenticated model header")
        build_id = candidate_build_id(candidate_stats)
        build = gate_build_directory(self.config, iteration, candidate_stats)
        tracked_header = (self.config.repository / "engines" / "minimax" / "src" /
                          "frozen_pattern_gain_model.hpp")
        if sha256_file(tracked_header) != header_metadata["sha256"]:
            raise WorkflowError(
                "engine gate requires the candidate header to be installed and committed; "
                f"run handoff --apply for {iteration.name}: {tracked_header}")

        def create(attempt: int) -> None:
            reserve_report_directory(root, GATE_SCHEMA)
            self._command(stage, "configure", [
                "cmake", "--preset", self.config.build_preset, "-B", str(build)],
                attempt=attempt)
            self._command(stage, "build", ["cmake", "--build", str(build)], attempt=attempt)
            self._command(stage, "test", [
                "ctest", "--test-dir", str(build), "--output-on-failure"], attempt=attempt)
            if sha256_file(tracked_header) != header_metadata["sha256"]:
                raise WorkflowError(
                    f"tracked model header changed while building the engine gate: {tracked_header}")
            runner = build / "runner" / "poe2_runner"
            command = engine_gate_command(self.config, iteration, build, runner)
            result = self._command(stage, "evaluate", command, attempt=attempt)
            matches = re.findall(r"^eval_run run_dir=(.+?) ledger=disabled$",
                                 result.output, re.MULTILINE)
            if len(matches) != 1:
                raise WorkflowError(f"engine evaluator emitted no unique run path: {result.log_path}")
            run_path = Path(matches[0]).resolve()
            if root not in run_path.parents:
                raise WorkflowError(f"engine evaluator escaped gate root: {run_path}")
            manifest = json.loads((run_path / "manifest.json").read_bytes())
            if manifest.get("valid") is not True:
                raise WorkflowError(f"engine gate is invalid: {run_path}")
            build_artifacts = {
                "new_engine_binary": build / "engines" / gate.new_engine,
                "runner_binary": runner,
            }
            for name, path in build_artifacts.items():
                if not path.is_file():
                    raise WorkflowError(f"engine gate did not produce {name}: {path}")
            report = {
                "schema": GATE_SCHEMA, "schema_version": 1,
                "created_at_utc": now_utc(), "git_commit": self.commit,
                "settings": {
                    **asdict(gate), "opening_book": str(gate.opening_book),
                },
                "opening_book_sha256": sha256_file(gate.opening_book),
                "run_id": manifest.get("run_id"),
                "run_directory": run_path.relative_to(root).as_posix(),
                "candidate_report_sha256": candidate_stats["report_sha256"],
                "candidate_build_id": build_id,
                "build_directory": build.relative_to(root).as_posix(),
                "tracked_model_header_sha256": header_metadata["sha256"],
                "base_engine_artifact": {
                    "path": str(base_engine_binary),
                    "sha256": sha256_file(base_engine_binary),
                },
                "build_artifacts": {
                    name: {"path": path.relative_to(root).as_posix(),
                           "sha256": sha256_file(path)}
                    for name, path in build_artifacts.items()
                },
                "artifacts": {name: sha256_file(run_path / name) for name in
                              ("manifest.json", "summary.json", "games.csv", "ledger-row.csv")},
            }
            complete_json_report(root, GATE_SCHEMA, report)

        def authenticate(_attempt: int) -> dict[str, Any]:
            return authenticate_gate(self.config, iteration)

        return self._run_simple(stage, root, "engine gate", {
            "candidate": candidate_stats,
            "opening_book_sha256": sha256_file(gate.opening_book),
            "base_engine_sha256": sha256_file(base_engine_binary),
            "tracked_model_header_sha256": sha256_file(tracked_header),
        }, create, authenticate)

    @staticmethod
    def _iteration_artifacts_complete(iteration: IterationConfig) -> bool:
        paths = [iteration.training_directory]
        if iteration.sealed_evaluation is not None:
            paths.append(iteration.evaluation_directory)
        if iteration.candidate_validation is not None:
            paths.append(iteration.candidate_directory)
        if iteration.engine_gate is not None:
            paths.append(iteration.gate_directory)
        return all(_path_state(path) == "complete" for path in paths)


def _dependency_order(config: WorkflowConfig, iteration: IterationConfig) -> list[IterationConfig]:
    result: list[IterationConfig] = []
    added: set[str] = set()

    def add(item: IterationConfig) -> None:
        for name in item.depends_on:
            add(config.iteration(name))
        if item.name not in added:
            result.append(item)
            added.add(item.name)

    add(iteration)
    return result


def run_workflow(config: WorkflowConfig, *, iteration_name: str | None = None,
                 with_dependencies: bool = False) -> None:
    """Execute a full run or one iteration under its create-only run lock."""
    if iteration_name is None:
        selected = list(config.iterations)
    else:
        requested = config.iteration(iteration_name)
        selected = (_dependency_order(config, requested)
                    if with_dependencies else [requested])
    commit = require_clean_committed(config)
    with run_lock(config):
        state = RunState(config)
        state.validate_evolution()
        state.recover_running_stages()
        state.accept_config()
        state.append("run_started", git_commit=commit, config_digest=config.digest,
                     command=("iteration" if iteration_name else "run"),
                     iteration=iteration_name)
        state.write_summary(workflow_snapshot(config))
        runner = WorkflowRunner(config, state, commit)
        try:
            with _interruptions_as_keyboard():
                for iteration in selected:
                    runner.run_iteration(iteration)
                    state.append("iteration_completed", iteration=iteration.name)
                    state.write_summary(workflow_snapshot(config))
                state.append(
                    "run_completed", command=("iteration" if iteration_name else "run"),
                    iteration=iteration_name)
                state.write_summary(workflow_snapshot(config))
        except KeyboardInterrupt as error:
            runner.commands.terminate_all()
            state.append("run_interrupted", error="keyboard interrupt")
            state.write_summary(workflow_snapshot(config))
            raise WorkflowError("workflow interrupted; complete shards were preserved") from error
        except BaseException as error:
            runner.commands.terminate_all()
            state.append("run_failed", error=str(error))
            state.write_summary(workflow_snapshot(config))
            raise


def handoff_candidate(config: WorkflowConfig, iteration_name: str, *, apply: bool = False) -> dict[str, Any]:
    """Preview or deliberately install one authenticated generated header."""
    if apply:
        with run_lock(config):
            return _handoff_candidate(config, iteration_name, apply=True)
    return _handoff_candidate(config, iteration_name, apply=False)


def _handoff_candidate(config: WorkflowConfig, iteration_name: str, *,
                       apply: bool) -> dict[str, Any]:
    iteration = config.iteration(iteration_name)
    candidate = authenticate_candidate(config, iteration)
    header_metadata = candidate["artifacts"].get("model_header")
    if not isinstance(header_metadata, dict):
        raise WorkflowError("candidate report has no model header")
    source = iteration.candidate_directory / header_metadata["path"]
    target = config.repository / "engines" / "minimax" / "src" / "frozen_pattern_gain_model.hpp"
    tracked_digest = sha256_file(target)
    preview = {
        "iteration": iteration_name, "promotable": candidate["promotable"],
        "candidate_header": str(source), "candidate_sha256": header_metadata["sha256"],
        "tracked_header": str(target), "tracked_sha256": tracked_digest,
        "different": tracked_digest != header_metadata["sha256"], "applied": False,
    }
    if not apply:
        return preview
    if not candidate["promotable"]:
        raise WorkflowError(
            f"candidate {iteration_name} is explicitly non-promotable")
    require_clean_committed(config)
    if preview["different"]:
        try:
            contents = source.read_bytes()
            if hashlib.sha256(contents).hexdigest() != header_metadata["sha256"]:
                raise WorkflowError(f"candidate header changed during handoff: {source}")
            atomic_write_bytes(target, contents, replace=True)
        except OSError as error:
            raise WorkflowError(f"could not install candidate header at {target}: {error}") from error
        preview["applied"] = True
    return preview


def _read_ledger_row(path: Path) -> tuple[list[str], list[str]]:
    try:
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.reader(source))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise WorkflowError(f"could not read saved ledger row {path}: {error}") from error
    if len(rows) != 2 or not rows[0] or len(rows[0]) != len(rows[1]):
        raise WorkflowError(f"saved engine-gate ledger row is malformed: {path}")
    return rows[0], rows[1]


def _ledger_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _ledger_number_matches(value: str | None, expected: Any) -> bool:
    if expected is None:
        return value == ""
    if (not isinstance(expected, (int, float)) or isinstance(expected, bool) or
            value is None or value == ""):
        return False
    try:
        actual = float(value)
    except ValueError:
        return False
    return math.isfinite(actual) and actual == float(expected)


def _ledger_lock_path(path: Path) -> Path:
    return Path(f"{path}.lock")


@contextmanager
def _ledger_lock(path: Path) -> Iterator[None]:
    lock_path = _ledger_lock_path(path)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    except OSError as error:
        raise WorkflowError(f"could not open ledger lock {lock_path}: {error}") from error
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as error:
            raise WorkflowError(f"could not lock ledger {path}: {error}") from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def promote_gate(config: WorkflowConfig, iteration_name: str, *, apply: bool = False) -> dict[str, Any]:
    """Preview or append one authenticated local engine-gate row to the master ledger."""
    if apply:
        with run_lock(config):
            return _promote_gate(config, iteration_name, apply=True)
    return _promote_gate(config, iteration_name, apply=False)


def _promote_gate(config: WorkflowConfig, iteration_name: str, *,
                  apply: bool) -> dict[str, Any]:
    iteration = config.iteration(iteration_name)
    gate = authenticate_gate(config, iteration)
    saved_path = Path(gate["ledger_row"])
    header, row = _read_ledger_row(saved_path)
    try:
        run_index = header.index("run_id")
    except ValueError as error:
        raise WorkflowError("saved ledger schema has no run_id") from error
    run_id = row[run_index]
    if run_id != gate["run_id"]:
        raise WorkflowError("saved ledger row run ID differs from authenticated gate")
    saved = dict(zip(header, row, strict=True))
    manifest = gate.get("manifest")
    summary = gate.get("summary")
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        raise WorkflowError("authenticated gate lacks manifest or summary statistics")
    gate_settings = iteration.engine_gate
    assert gate_settings is not None
    expected_saved = {
        "kind": "training-gate",
        "new_id": gate["candidate_build_id"],
        "new_engine": gate_settings.new_engine,
        "new_engine_args": gate_settings.new_engine_args,
        "base_engine": gate_settings.base_engine,
        "base_engine_args": gate_settings.base_engine_args,
        "games_requested": str(gate_settings.games),
        "opening_book": str(gate_settings.opening_book),
        "go_movetime_ms": str(gate_settings.go_movetime_ms),
        "timeout_ms": str(gate_settings.timeout_ms),
        "run_dir": gate["run_directory"],
        "valid": "true",
        "workers_requested": str(gate_settings.workers),
    }
    manifest_fields = {
        "created_at_utc": "created_at_utc",
        "base_id": "base_id",
        "games_played": "games_played",
        "opening_book_digest": "opening_book_digest",
        "opening_count": "opening_count",
        "go_depth": "go_depth",
        "go_movetime_ms": "go_movetime_ms",
        "go_nodes": "go_nodes",
        "analysis_version": "analysis_version",
        "statistical_unit": "statistical_unit",
        "confidence_method": "confidence_method",
        "sequential_test_method": "sequential_test_method",
        "invalid_reason": "invalid_reason",
        "opening_seed": "opening_seed",
        "opening_seed_source": "opening_seed_source",
        "opening_sampling": "opening_sampling",
        "unique_openings": "unique_openings",
        "sequential_bound_unit": "sequential_bound_unit",
        "sequential_decision": "sequential_decision",
        "workers_used": "workers_used",
        "games_discarded": "games_discarded",
    }
    summary_fields = {
        "engine_one_wins": "engine_one_wins",
        "engine_two_wins": "engine_two_wins",
        "no_winner": "no_winner",
        "statistical_samples": "statistical_samples",
        "statistical_games": "statistical_games",
    }
    missing_manifest = [name for name in manifest_fields.values() if name not in manifest]
    missing_summary = [name for name in summary_fields.values() if name not in summary]
    if missing_manifest or missing_summary:
        missing = sorted(set(missing_manifest + missing_summary))
        raise WorkflowError(
            "authenticated gate lacks ledger provenance fields: " + ", ".join(missing))
    expected_saved.update({
        ledger_name: _ledger_scalar(manifest[manifest_name])
        for ledger_name, manifest_name in manifest_fields.items()
    })
    expected_saved.update({
        ledger_name: _ledger_scalar(summary[summary_name])
        for ledger_name, summary_name in summary_fields.items()
    })
    expected_saved["sequential_model"] = _ledger_scalar(
        manifest["sequential_test_method"])
    expected_saved["analysis_note"] = "paired normalized-Elo GSPRT"
    score_counts = summary.get("statistical_score_rate_counts")
    if not isinstance(score_counts, dict):
        raise WorkflowError("authenticated gate lacks statistical score counts")
    for ledger_name, score_name in {
            "pair_score_0": "0", "pair_score_0_5": "0.25", "pair_score_1": "0.5",
            "pair_score_1_5": "0.75", "pair_score_2": "1"}.items():
        if score_name not in score_counts:
            raise WorkflowError(
                f"authenticated gate lacks statistical score count {score_name}")
        expected_saved[ledger_name] = _ledger_scalar(score_counts[score_name])
    mismatches = [name for name, value in expected_saved.items()
                  if saved.get(name) != value]
    for ledger_name, manifest_name in {
            "sequential_null": "sequential_null",
            "sequential_alt": "sequential_alt",
            "sequential_llr": "sequential_llr",
            "sequential_lower_bound": "sequential_lower_bound",
            "sequential_upper_bound": "sequential_upper_bound",
            "normalized_elo": "normalized_elo",
            "betting_log_evidence_above_even": "betting_log_evidence_above_even",
            "betting_log_evidence_below_even": "betting_log_evidence_below_even",
    }.items():
        if manifest_name not in manifest or not _ledger_number_matches(
                saved.get(ledger_name), manifest[manifest_name]):
            mismatches.append(ledger_name)
    if mismatches:
        raise WorkflowError(
            "saved ledger row differs from the authenticated gate in: " +
            ", ".join(sorted(set(mismatches))))
    master = config.repository / "eval" / "results.csv"

    def inspect_master() -> bool:
        try:
            with master.open(newline="", encoding="utf-8") as source:
                existing = list(csv.reader(source))
        except (OSError, UnicodeDecodeError, csv.Error) as error:
            raise WorkflowError(f"could not read master ledger {master}: {error}") from error
        if not existing or existing[0] != header:
            raise WorkflowError(f"master and saved ledger schemas differ: {master}")
        if any(len(item) != len(header) for item in existing[1:]):
            raise WorkflowError(f"master ledger contains a malformed row: {master}")
        return any(item[run_index] == run_id for item in existing[1:])

    preview = {"iteration": iteration_name, "run_id": run_id,
               "saved_row": str(saved_path), "ledger": str(master),
               "duplicate": False, "applied": False}
    if not apply:
        preview["duplicate"] = inspect_master()
        if preview["duplicate"]:
            raise WorkflowError(f"master ledger already contains run ID {run_id}")
        return preview
    require_clean_committed(config)
    try:
        with _ledger_lock(master):
            require_clean_committed(config)
            preview["duplicate"] = inspect_master()
            if preview["duplicate"]:
                raise WorkflowError(f"master ledger already contains run ID {run_id}")
            with master.open("a", newline="", encoding="utf-8") as destination:
                writer = csv.writer(destination, lineterminator="\n")
                writer.writerow(row)
                destination.flush()
                os.fsync(destination.fileno())
    except (OSError, csv.Error) as error:
        raise WorkflowError(f"could not append master ledger {master}: {error}") from error
    preview["applied"] = True
    return preview
