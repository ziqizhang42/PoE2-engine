"""Create-only orchestration for reproducible PoE2 training runs."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Sequence

from .shared import (
    atomic_write_bytes,
    canonical_json,
    complete_json_report,
    reserve_report_directory,
    sha256_file,
    sync_path,
    sync_tree,
)
from .workflow_adapters import (
    CANDIDATE_SCHEMA,
    COMPARISON_KIND,
    COMPARISON_SCHEMA,
    GATE_KIND,
    GATE_SCHEMA,
    GATE_SCHEMA_VERSION,
    SOURCE_CHUNK_TRAJECTORIES,
    aggregate_label_stats,
    authenticate_candidate,
    authenticate_comparison,
    authenticate_evaluation,
    authenticate_features,
    authenticate_gate,
    authenticate_label_shard,
    authenticate_source,
    authenticate_training,
    candidate_build_id,
    candidate_training_datasets,
    names_path,
    catch2_cache_directory,
    catch2_candidates,
    catch2_pin,
    configured_elsewhere,
    comparison_command,
    comparison_side,
    data_binary,
    data_build_directory,
    engine_gate_command,
    evaluation_command,
    export_command,
    feature_command,
    feature_audit_command,
    gate_base_binary,
    gate_base_build_directory,
    gate_build_directory,
    label_command,
    label_audit_command,
    label_preflight_command,
    label_shard_path,
    parity_command,
    search_benchmark_command,
    source_assemble_command,
    source_chunk_command,
    source_chunk_commit,
    source_chunk_path,
    source_chunk_ranges,
    source_chunks_directory,
    source_audit_command,
    training_command,
    verified_checkout,
)
from .workflow_config import (
    CANDIDATE_ENGINE,
    MODEL_ARCHIVE_DIRECTORY,
    ComparisonConfig,
    ImportedDatasetConfig,
    IterationConfig,
    ManagedDatasetConfig,
    MatchConfig,
    ModelReference,
    WorkflowConfig,
    comparison_stage,
    extra_training_datasets,
    fingerprint,
    match_settings,
    recorded_gate_settings,
    rendered_paths,
    stage_fingerprints,
    trained_models_used,
    with_trained_model,
)
from .workflow_state import RunState, now_utc
from .workflow_ui import (
    WorkflowProgress,
    aggregate_status,
    format_duration,
    human_bar,
    iteration_pipeline,
    paint,
    recorded_seconds,
    status_symbol,
)

MODEL_ARCHIVE_SCHEMA = "poe2-archived-model"
MODEL_INDEX_PATH = MODEL_ARCHIVE_DIRECTORY / "index.csv"
MODEL_INDEX_HEADER = (
    "recorded_utc", "run", "iteration", "model", "base", "base_model", "decision", "games",
    "score_pct", "sealed_mae", "gate_run_id",
)
COMPARISON_INDEX_PATH = MODEL_ARCHIVE_DIRECTORY / "comparisons.csv"
COMPARISON_INDEX_HEADER = (
    "recorded_utc", "run", "comparison", "new", "new_model", "base", "base_model", "decision",
    "games", "score_pct", "run_id",
)
ITERATION_NON_STAGE_FIELDS = frozenset({"dataset", "depends_on", "status", "blocked_by"})


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
        # Commands inherit the run lock, so a command that outlives a killed orchestrator keeps
        # the next one from treating its unfinished artifact as a crash leftover.
        self.lock_descriptors: tuple[int, ...] = ()
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
            accepted_exit_codes: set[int] | None = None,
            output_callback: Callable[[str, float], None] | None = None) -> CommandResult:
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
                    pass_fds=self.lock_descriptors,
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
                        if output_callback is None:
                            print(line, end="", flush=True)
                        else:
                            output_callback(line, time.perf_counter() - started)
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
def run_lock(config: WorkflowConfig) -> Iterator[int]:
    """Hold the run lock, yielding its descriptor; commands that inherit it hold it too."""
    config.output_directory.mkdir(parents=True, exist_ok=True)
    path = config.output_directory / ".orchestrator.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise WorkflowError(
                f"another orchestrator, or a command a stopped one started, holds the run "
                f"lock; `fuser -v {path}` lists them") from error
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
        yield descriptor
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


def _artifact_record(config: WorkflowConfig, path: Path) -> dict[str, str]:
    """Return the event fields naming an artifact, by its path inside the run folder."""
    if config.output_directory not in path.parents:
        raise WorkflowError(f"refusing to track an artifact outside the run: {path}")
    return {"path": str(path), "artifact": path.relative_to(config.output_directory).as_posix()}


def _remove_artifact(config: WorkflowConfig, path: Path) -> None:
    if config.output_directory not in path.parents:
        raise WorkflowError(f"refusing to discard a path outside the run: {path}")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _recorded_artifact(config: WorkflowConfig, artifact: str) -> Path | None:
    relative = PurePosixPath(artifact)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        return None
    return config.output_directory.joinpath(*relative.parts)


def recover_after_crash(config: WorkflowConfig, state: RunState) -> list[Path]:
    """Repair a torn event log and remove artifacts a stopped invocation left unfinished.

    Call under the run lock. Only paths the log records starting, with no later outcome, are
    touched; complete ones are kept for authentication, which reports any damage.
    """
    dropped = state.repair_torn_tail()
    if dropped:
        print(f"dropped an incomplete final record ({dropped} bytes) from {state.events_path}",
              file=sys.stderr)
    removed: list[Path] = []
    for artifact in state.unfinished_artifacts():
        path = _recorded_artifact(config, artifact)
        if path is None:
            state.append("crash_artifact_kept", artifact=artifact,
                         reason="recorded path is not inside the run folder")
            print(f"kept {artifact}: its recorded path is not inside the run folder",
                  file=sys.stderr)
            continue
        fields = _artifact_record(config, path)
        path_state = _path_state(path)
        if path_state == "complete":
            state.append("artifact_completed", **fields, recovered=True)
        elif path_state == "missing" and not path.is_symlink():
            state.append("artifact_abandoned", **fields)
        else:
            _remove_artifact(config, path)
            state.append("crash_artifact_removed", **fields)
            print(f"removed unfinished {path} left by a stopped run", file=sys.stderr)
            removed.append(path)
    return removed


@contextmanager
def _recovered_run_lock(config: WorkflowConfig) -> Iterator[tuple[int, RunState]]:
    with run_lock(config) as descriptor:
        state = RunState(config)
        recover_after_crash(config, state)
        yield descriptor, state


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
        stages = [item for key, item in value.items() if key not in ITERATION_NON_STAGE_FIELDS]
        value["status"] = ("complete" if stages and all(item == "complete" for item in stages)
                           else "blocked" if any(item in {"incomplete", "unexpected"} for item in stages)
                           else "pending")
        iterations[iteration.name] = value
    blocked = blocked_iterations(config)
    for name, item in blocked.items():
        iterations[name].update(status="blocked", blocked_by=item.reason)
    comparisons: dict[str, Any] = {}
    blocked_matches = blocked_comparisons(config, blocked)
    for comparison in config.comparisons:
        match = _path_state(comparison.root)
        value = {"new": comparison.new, "base": comparison.base, "match": match,
                 "status": "complete" if match == "complete" else
                 "blocked" if match in {"incomplete", "unexpected"} else "pending"}
        if comparison.name in blocked_matches:
            value.update(status="blocked", blocked_by=blocked_matches[comparison.name].reason)
        comparisons[comparison.name] = value
    all_dataset_complete = all(
        value["features"] == "complete" for value in datasets.values())
    work = [*iterations.items(), *comparisons.items()]
    all_work_complete = all(value["status"] == "complete" for _, value in work)
    stopped = bool(blocked or blocked_matches) and all(
        value["status"] == "complete" or "blocked_by" in value for _, value in work)
    return {
        "schema": "poe2-training-status", "schema_version": 1,
        "run": config.name, "config": str(config.path), "config_digest": config.digest,
        "output_directory": str(config.output_directory), "build_preset": config.build_preset,
        "logical_cpus": os.cpu_count() or 1,
        "resolved_label_concurrency": resolved_label_concurrency(config),
        "datasets": datasets, "iterations": iterations, "comparisons": comparisons,
        "status": ("complete" if all_dataset_complete and all_work_complete else
                   "stopped" if stopped else "pending"),
    }


def print_status(config: WorkflowConfig, *, json_output: bool = False) -> None:
    snapshot = workflow_snapshot(config)
    state = RunState(config)
    snapshot["state"] = state.summary(snapshot).get("stages") if state.events else {}
    if json_output:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
        return
    operational = state.stage_state()
    for name, iteration in snapshot["iterations"].items():
        statuses = [value["status"] for stage, value in operational.items()
                    if stage.startswith(f"iteration:{name}:")]
        operational_status = aggregate_status(statuses)
        if operational_status in {"failed", "running"}:
            iteration["status"] = operational_status
    stage_statuses: list[str] = []
    for dataset in snapshot["datasets"].values():
        if dataset["kind"] == "managed":
            stage_statuses.extend((dataset["source"], dataset["labels"]["status"]))
        stage_statuses.append(dataset["features"])
    for iteration in snapshot["iterations"].values():
        stage_statuses.extend(value for key, value in iteration.items()
                              if key not in ITERATION_NON_STAGE_FIELDS)
    stage_statuses.extend(value["match"] for value in snapshot["comparisons"].values())
    completed, total = stage_statuses.count("complete"), len(stage_statuses)
    percent = 100.0 * completed / total if total else 100.0
    iteration_statuses = [value["status"] for value in snapshot["iterations"].values()]
    overall = aggregate_status(iteration_statuses)
    if overall == "pending":
        overall = str(snapshot["status"])
    style = {"complete": "green", "failed": "red", "running": "cyan"}.get(
        overall, "yellow")
    print(f"{paint('run', 'bold')} {config.name}  {paint(overall.upper(), 'bold', style)}")
    print(f"{human_bar(completed, total)}  {completed}/{total} stages ({percent:.0f}%)  ·  "
          f"recorded {format_duration(recorded_seconds(state))}")
    print(iteration_pipeline(snapshot))
    for name, dataset in snapshot["datasets"].items():
        if dataset["kind"] == "managed":
            labels = dataset["labels"]
            print(f"  {status_symbol(str(dataset['features']))} dataset {name}: "
                  f"source={dataset['source']} labels={labels['complete_shards']}/"
                  f"{labels['total_shards']} features={dataset['features']}")
        else:
            print(f"  {status_symbol(str(dataset['features']))} dataset {name}: "
                  f"imported features={dataset['features']}")
    for index, (name, iteration) in enumerate(snapshot["iterations"].items(), 1):
        stages = " ".join(f"{key}={value}" for key, value in iteration.items()
                          if key not in ITERATION_NON_STAGE_FIELDS)
        decision = _recorded_gate_decision(config.iteration(name))
        if decision is not None:
            stages += f" gate_decision={decision}"
        print(f"  {status_symbol(str(iteration['status']))} iteration {index}. {name}: {stages}")
        if "blocked_by" in iteration:
            print(f"      not started: {iteration['blocked_by']}")
    for name, comparison in snapshot["comparisons"].items():
        decision = _recorded_decision(config.comparison(name).root)
        print(f"  {status_symbol(str(comparison['status']))} comparison {name}: "
              f"new={comparison['new']} base={comparison['base']} match={comparison['match']}"
              f"{f' decision={decision}' if decision is not None else ''}")
        if "blocked_by" in comparison:
            print(f"      not started: {comparison['blocked_by']}")


def _recorded_decision(root: Path) -> str | None:
    """Return a completed match's recorded decision for display, without authenticating it."""
    if _path_state(root) != "complete":
        return None
    try:
        report = json.loads((root / "report.json").read_bytes())
        manifest = json.loads((root / report["run_directory"] / "manifest.json").read_bytes())
    except (OSError, ValueError, KeyError, TypeError):
        return None
    decision = manifest.get("sequential_decision") if isinstance(manifest, dict) else None
    return decision if isinstance(decision, str) else None


def _recorded_gate_decision(iteration: IterationConfig) -> str | None:
    if iteration.engine_gate is None:
        return None
    return _recorded_decision(iteration.gate_directory)


@dataclass(frozen=True)
class Blocked:
    """An iteration or comparison that cannot start because a model it needs was not trained."""

    name: str
    # The iteration whose engine gate did not accept its candidate, and that gate's decision.
    loser: str
    decision: str
    kind: str = "iteration"

    @property
    def reason(self) -> str:
        return f"{self.loser} did not win its engine gate (decision {self.decision})"

    @property
    def label(self) -> str:
        return self.name if self.kind == "iteration" else f"{self.kind} {self.name}"


def blocked_comparisons(config: WorkflowConfig,
                        iterations: dict[str, Blocked] | None = None) -> dict[str, Blocked]:
    """Return comparisons with a side whose iteration is never started.

    A side whose own gate did not win still plays: comparing a loser is a fair question.
    """
    iterations = blocked_iterations(config) if iterations is None else iterations
    blocked: dict[str, Blocked] = {}
    for comparison in config.comparisons:
        for side in (comparison.new, comparison.base):
            iteration = config.model(side).iteration
            cause = None if iteration is None else iterations.get(iteration)
            if cause is not None:
                blocked[comparison.name] = replace(
                    cause, name=comparison.name, kind="comparison")
                break
    return blocked


def blocked_work(config: WorkflowConfig) -> list[Blocked]:
    iterations = blocked_iterations(config)
    return [*iterations.values(), *blocked_comparisons(config, iterations).values()]


def blocked_iterations(config: WorkflowConfig) -> dict[str, Blocked]:
    """Return iterations that need a trained model whose gate decided anything but accept_alt.

    This reads recorded decisions without authenticating them; resolving a model to use it
    authenticates its gate.
    """
    blocked: dict[str, Blocked] = {}
    for iteration in config.iterations:
        producers = [model.iteration for model in trained_models_used(config, iteration)]
        for name in (*iteration.depends_on, *producers):
            cause = blocked.get(name)
            if cause is None and name in producers:
                decision = _recorded_gate_decision(config.iteration(name))
                if decision is not None and decision != "accept_alt":
                    cause = Blocked(iteration.name, name, decision)
            if cause is not None:
                blocked[iteration.name] = replace(cause, name=iteration.name)
                break
    return blocked


def describe_not_started(blocked: Sequence[Blocked]) -> list[str]:
    """Return one line per losing iteration naming the work it kept from starting."""
    groups: dict[str, list[Blocked]] = {}
    for item in blocked:
        groups.setdefault(item.loser, []).append(item)
    return [f"{items[0].reason}; not started: {', '.join(item.label for item in items)}"
            for items in groups.values()]


def _candidate_model(model: ModelReference, iteration: IterationConfig,
                     candidate: dict[str, Any]) -> ModelReference:
    header = candidate["artifacts"]["model_header"]
    return replace(model, model_id=candidate_build_id(candidate),
                   header=iteration.candidate_directory / header["path"])


def _accepted_model(config: WorkflowConfig, model: ModelReference) -> ModelReference:
    """Return a trained model filled in from its iteration's authenticated, accepted candidate."""
    assert model.iteration is not None
    iteration = config.iteration(model.iteration)
    candidate = authenticate_candidate(config, iteration)
    gate = authenticate_gate(config, iteration, candidate=candidate)
    if gate["sequential_decision"] != "accept_alt":
        raise WorkflowError(
            f"model {model.name} is not usable: iteration {model.iteration}'s authenticated "
            f"engine gate decided {gate['sequential_decision']}, not accept_alt")
    return _candidate_model(model, iteration, candidate)


def _comparison_ready(config: WorkflowConfig, comparison: ComparisonConfig) -> bool:
    """Return whether both sides exist: archived, or trained with a complete candidate."""
    return all(
        model.iteration is None or
        _path_state(config.iteration(model.iteration).candidate_directory) == "complete"
        for model in (config.model(comparison.new), config.model(comparison.base)))


def _comparison_model(config: WorkflowConfig, name: str) -> ModelReference:
    """Return an intact comparison side; a trained one needs only its authenticated candidate.

    Pass a config whose accepted models are resolved, so the candidate's teacher is known.
    """
    model = config.model(name)
    if model.iteration is None:
        return _model_by_id(config, model.model_id, "comparison model")
    iteration = config.iteration(model.iteration)
    return _candidate_model(model, iteration, authenticate_candidate(config, iteration))


def _comparison_models(config: WorkflowConfig,
                       comparison: ComparisonConfig) -> tuple[ModelReference, ModelReference]:
    new, base = (_comparison_model(config, name) for name in (comparison.new, comparison.base))
    if new.model_id == base.model_id:
        raise WorkflowError(
            f"comparison {comparison.name} would play {new.model_id} against itself")
    return new, base


def resolve_trained_models(config: WorkflowConfig) -> WorkflowConfig:
    """Fill in every model this recipe trains whose engine gate accepted it.

    Models are resolved in iteration order, so each one's own teacher and gate base are filled
    in before its candidate and gate are authenticated.
    """
    order = {iteration.name: index for index, iteration in enumerate(config.iterations)}
    trained = sorted((model for model in config.models if model.iteration is not None),
                     key=lambda model: order[model.iteration])
    for model in trained:
        if _recorded_gate_decision(config.iteration(model.iteration)) == "accept_alt":
            config = with_trained_model(config, _accepted_model(config, model))
    return config


def _format_command(command: Sequence[str]) -> str:
    return shlex.join([str(item) for item in command])


def print_plan(config: WorkflowConfig) -> None:
    """Print a read-only execution plan with derived paths and representative commands."""
    print(paint(f"run {config.name}", "bold", "cyan"))
    print(f"  output: {config.output_directory}")
    print(f"  preset: {config.build_preset}")
    print(f"  logical CPUs: {os.cpu_count() or 1}")
    print(f"  default label concurrency: {resolved_label_concurrency(config)}")
    for dataset in config.datasets:
        print(paint(f"dataset {dataset.name} ({dataset.kind})", "bold", "blue"))
        if isinstance(dataset, ImportedDatasetConfig):
            print(f"  authenticate: {dataset.feature_directory}")
            continue
        if dataset.teacher is not None:
            print(f"  teacher: {_describe_model(config, dataset.teacher)}")
        if dataset.positions is not None:
            print(f"  positions: relabeled from dataset {dataset.positions}'s source "
                  f"{dataset.source_directory}")
        else:
            chunks = source_chunk_ranges(dataset)
            print(f"  source: {dataset.source_directory} ({len(chunks)} chunks of up to "
                  f"{SOURCE_CHUNK_TRAJECTORIES} trajectories)")
            print(f"    {_format_command(source_chunk_command(config, dataset, *chunks[0]))}")
            print(f"    {_format_command(source_assemble_command(config, dataset))}")
        print(f"  labels: {dataset.labels_directory} "
              f"({dataset.source.shards} shards, concurrency "
              f"{resolved_label_concurrency(config, dataset)})")
        print(f"    {_format_command(label_command(config, dataset, 0))}")
        print(f"  features: {dataset.feature_directory}")
        print(f"    {_format_command(feature_command(config, dataset))}")
    blocked = blocked_iterations(config)
    for iteration in config.iterations:
        dataset = config.dataset(iteration.dataset)
        print(paint(f"iteration {iteration.name}: dataset={iteration.dataset} "
                    f"depends_on={','.join(iteration.depends_on) or '-'}", "bold", "magenta"))
        if iteration.name in blocked:
            print(f"  not started: {blocked[iteration.name].reason}")
        print(f"  training: {_format_command(training_command(
            sys.executable, dataset, iteration, extra_training_datasets(config, iteration)))}")
        if iteration.sealed_evaluation is not None:
            print(f"  sealed evaluation: "
                  f"{_format_command(evaluation_command(sys.executable, config, iteration))}")
        if iteration.candidate_validation is not None:
            print(f"  candidate validation: {iteration.candidate_directory} "
                  f"promotable={str(iteration.candidate_validation.promotable).lower()}")
        if iteration.engine_gate is not None:
            gate = iteration.engine_gate
            base = (gate.base if gate.base is not None else
                    f"{_describe_model(config, _configured_base_model(config, iteration))}, "
                    "built privately")
            print(f"  engine gate: {iteration.gate_directory} base={base}")
    blocked_matches = blocked_comparisons(config, blocked)
    for comparison in config.comparisons:
        print(paint(f"comparison {comparison.name}: {comparison.root}", "bold", "magenta"))
        if comparison.name in blocked_matches:
            print(f"  not started: {blocked_matches[comparison.name].reason}")
        for side, name in (("new", comparison.new), ("base", comparison.base)):
            print(f"  {side}: {_describe_model(config, config.model(name), accepted=False)}, "
                  "built privately")


def _describe_model(config: WorkflowConfig, model: ModelReference, *,
                    accepted: bool = True) -> str:
    if model.iteration is None:
        return f"{config.name}/{model.name} ({model.model_id})"
    return (f"{config.name}/{model.name} (the candidate of iteration {model.iteration}"
            f"{', once its gate accepts it' if accepted else ''})")


def _configured_base_model(config: WorkflowConfig, iteration: IterationConfig) -> ModelReference:
    """Return the [models] entry a gate's base names, filled in or not."""
    gate = iteration.engine_gate
    assert gate is not None and gate.base is None
    if gate.base_iteration is not None:
        return config.trained_model(gate.base_iteration)
    return _model_by_id(config, gate.base_model, "engine-gate base model")


def _direct_audit(command: Sequence[str], repository: Path) -> str:
    completed = subprocess.run(command, cwd=repository, text=True,
                               capture_output=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise WorkflowError(
            f"authentication command failed ({_format_command(command)}): {detail}")
    return completed.stdout


def _model_by_id(config: WorkflowConfig, model_id: str, label: str) -> ModelReference:
    """Return the intact archived model that the recipe's [models] table names by this ID."""
    for model in config.models:
        if model.model_id == model_id:
            if not model.header.is_file() or sha256_file(model.header) != model.sha256:
                raise WorkflowError(f"{label} {model.name} is missing or changed: {model.header}")
            return model
    raise WorkflowError(f"{label} is not named in config.models: {model_id}")


def _expected_model_id(model: ModelReference, user: str) -> str:
    """Return a model's ID, or explain that its iteration's gate has not accepted it yet."""
    if model.model_id is None:
        raise WorkflowError(
            f"{user} needs model {model.name}, which iteration {model.iteration}'s engine gate "
            f"has not accepted; run {model.iteration} first or rerun with --with-dependencies")
    return model.model_id


def _gate_base_model(config: WorkflowConfig, iteration: IterationConfig) -> ModelReference | None:
    """Return the intact model a gate's base is built from, or None for an existing build."""
    gate = iteration.engine_gate
    if gate is None or gate.base is not None:
        return None
    model = _configured_base_model(config, iteration)
    _expected_model_id(model, f"engine gate for {iteration.name}")
    return model


def check_model_lineage(config: WorkflowConfig, model: ModelReference | None = None) -> None:
    """Require each archived [models] entry named after an iteration to be its candidate.

    An entry is compared once its iteration's candidate exists; before then nothing disagrees.
    An entry for a model this recipe trains is that candidate by construction.
    """
    iterations = {iteration.name: iteration for iteration in config.iterations}
    for entry in config.models if model is None else (model,):
        iteration = iterations.get(entry.name)
        if (entry.iteration is not None or iteration is None or
                _path_state(iteration.candidate_directory) != "complete"):
            continue
        produced = candidate_build_id(authenticate_candidate(config, iteration))
        if produced != entry.model_id:
            raise WorkflowError(
                f"config.models.{entry.name} is {entry.model_id}, but iteration "
                f"{entry.name} produced {produced}")


def _pending_source_chunks(dataset: ManagedDatasetConfig,
                           leftovers: frozenset[Path] = frozenset()) -> list[tuple[int, int]]:
    """Return chunk ranges still to generate; reject unfinished or unexpected chunk paths.

    Crash leftovers count as not yet generated, since the next run removes them first.
    """
    expected = {source_chunk_path(dataset, *item): item for item in source_chunk_ranges(dataset)}
    directory = source_chunks_directory(dataset)
    if directory.exists():
        unexpected = [item for item in directory.iterdir() if item not in expected]
        if unexpected:
            raise WorkflowError(
                f"source chunks have unexpected output; move it aside explicitly: "
                f"{unexpected[0]}")
    pending: list[tuple[int, int]] = []
    for path, item in expected.items():
        state = "missing" if path in leftovers else _path_state(path)
        if state in {"incomplete", "unexpected"}:
            raise _blocked_path(path, "source chunk")
        if state == "missing":
            pending.append(item)
    return pending


def _waiting_models(config: WorkflowConfig, iteration: IterationConfig) -> list[ModelReference]:
    return [model for model in trained_models_used(config, iteration) if model.model_id is None]


def crash_leftovers(config: WorkflowConfig, state: RunState | None = None) -> list[Path]:
    """Return unfinished artifacts a stopped invocation recorded; the next run removes them."""
    state = RunState(config) if state is None else state
    paths = [_recorded_artifact(config, artifact) for artifact in state.unfinished_artifacts()]
    return [path for path in paths if path is not None and _path_state(path) != "complete" and
            (path.exists() or path.is_symlink())]


def _require_not_started(path: Path, label: str, models: Sequence[ModelReference]) -> None:
    """Reject artifacts of work that needs a trained model that is not accepted."""
    if path.exists():
        names = ", ".join(model.name for model in models)
        raise WorkflowError(
            f"{label} has artifacts, but it needs {names}, whose engine gate has not accepted "
            f"it; move them aside explicitly: {path}")


def validate_existing(config: WorkflowConfig) -> dict[str, Any]:
    """Read-only validation of config evolution and every artifact currently present.

    Crash leftovers count as missing: the next mutating command removes them before it starts.
    """
    state = RunState(config)
    state.validate_evolution()
    leftovers = frozenset(crash_leftovers(config, state))

    def state_of(path: Path) -> str:
        return "missing" if path in leftovers else _path_state(path)

    check_model_lineage(config)
    config = resolve_trained_models(config)
    authenticated: dict[str, Any] = {}
    for dataset in config.datasets:
        if (isinstance(dataset, ManagedDatasetConfig) and dataset.teacher is not None and
                dataset.teacher.model_id is None):
            _require_not_started(dataset.root, f"dataset {dataset.name}", [dataset.teacher])
            continue
        feature_state = state_of(dataset.feature_directory)
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
        source_state = state_of(dataset.source_directory)
        if dataset.positions is None:
            if source_state in {"incomplete", "unexpected"}:
                raise _blocked_path(dataset.source_directory, "position source")
            if source_state == "missing":
                _pending_source_chunks(dataset, leftovers)
            if source_state == "complete":
                _direct_audit(
                    source_audit_command(sys.executable, config, dataset), config.repository)
                authenticated[f"dataset:{dataset.name}:source"] = authenticate_source(dataset)
        complete_shards = 0
        for index in range(dataset.source.shards):
            shard_path = label_shard_path(dataset, index)
            shard_state = state_of(shard_path)
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
    blocked_matches = blocked_comparisons(config)
    for comparison in config.comparisons:
        path_state = state_of(comparison.root)
        if comparison.name in blocked_matches or not _comparison_ready(config, comparison):
            if path_state != "missing":
                raise WorkflowError(
                    f"comparison {comparison.name} has artifacts, but one of its models was "
                    f"never trained; move them aside explicitly: {comparison.root}")
            continue
        if path_state in {"incomplete", "unexpected"}:
            raise _blocked_path(comparison.root, "comparison")
        if path_state == "complete":
            authenticated[comparison_stage(comparison)] = authenticate_comparison(
                config, comparison, *_comparison_models(config, comparison))
    for iteration in config.iterations:
        if waiting := _waiting_models(config, iteration):
            _require_not_started(iteration.root, f"iteration {iteration.name}", waiting)
            continue
        stages = [
            ("training", iteration.training_directory,
             lambda: authenticate_training(config.dataset(iteration.dataset), iteration,
                                           extra_training_datasets(config, iteration))),
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
            path_state = state_of(path)
            if path_state in {"incomplete", "unexpected"}:
                raise _blocked_path(path, name)
            if path_state == "complete":
                authenticated[f"iteration:{iteration.name}:{name}"] = authenticate()
    return authenticated


def _evaluator_run(config: WorkflowConfig, result: CommandResult, root: Path,
                   label: str) -> Path:
    """Return the one valid evaluator run a match command reported inside its root."""
    matches = re.findall(r"^eval_run run_dir=(.+?) ledger=disabled$", result.output, re.MULTILINE)
    if len(matches) != 1:
        raise WorkflowError(f"engine evaluator emitted no unique run path: {result.log_path}")
    run_path = (config.repository / matches[0]).resolve()
    if root not in run_path.parents:
        raise WorkflowError(f"engine evaluator escaped {label} root: {run_path}")
    manifest = json.loads((run_path / "manifest.json").read_bytes())
    if manifest.get("valid") is not True:
        raise WorkflowError(f"{label} is invalid: {run_path}")
    return run_path


def _run_artifacts(run_path: Path) -> dict[str, str]:
    return {name: sha256_file(run_path / name) for name in
            ("manifest.json", "summary.json", "games.csv", "ledger-row.csv")}


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
    def __init__(self, config: WorkflowConfig, state: RunState, commit: str,
                 progress: WorkflowProgress | None = None) -> None:
        self.config = config
        self.state = state
        self.commit = commit
        self.commands = CommandRunner(config.repository)
        self.progress = progress
        # Commit each run-local data binary was built or authenticated at, keyed by binary path.
        self._data_commits: dict[Path, str] = {}
        # Artifact paths this runner began creating; an interruption discards unfinished ones.
        self._started_artifacts: set[Path] = set()
        self._artifact_lock = threading.Lock()

    def _start_artifact(self, path: Path) -> None:
        """Record durably that this invocation is about to create path."""
        self.state.append("artifact_started", **_artifact_record(self.config, path))
        with self._artifact_lock:
            self._started_artifacts.add(path)

    def _finish_artifact(self, path: Path) -> None:
        with self._artifact_lock:
            self._started_artifacts.discard(path)
        self.state.append("artifact_completed", **_artifact_record(self.config, path))

    def _take_unfinished_artifacts(self) -> list[Path]:
        with self._artifact_lock:
            started = sorted(self._started_artifacts)
            self._started_artifacts.clear()
        return started

    def discard_interrupted_artifacts(self) -> list[Path]:
        """Remove artifacts this runner started but did not finish; call once its commands stop."""
        removed: list[Path] = []
        for path in self._take_unfinished_artifacts():
            fields = _artifact_record(self.config, path)
            path_state = _path_state(path)
            if path_state == "complete":
                self.state.append("artifact_completed", **fields)
            elif path_state == "missing" and not path.is_symlink():
                self.state.append("artifact_abandoned", **fields)
            else:
                _remove_artifact(self.config, path)
                self.state.append("interrupted_artifact_removed", **fields)
                removed.append(path)
        return removed

    def keep_failed_artifacts(self) -> list[Path]:
        """Record unfinished artifacts as failed, keeping them for inspection; return them."""
        failed = self._take_unfinished_artifacts()
        for path in failed:
            self.state.append("artifact_failed", **_artifact_record(self.config, path))
        return failed

    def _catch2_arguments(self) -> list[str]:
        """Point a private build at a verified Catch2 checkout on disk, copied into the run.

        Without one the value is empty, which makes CMake fetch normally even in a build tree
        configured with a checkout before.
        """
        pin = catch2_pin(self.config.repository)
        if pin is None:
            return []
        cache = catch2_cache_directory(self.config, pin)
        if not cache.exists() and not cache.is_symlink():
            source = next((path for path in catch2_candidates(self.config)
                           if verified_checkout(path, pin)), None)
            if source is not None:
                self._copy_checkout(source, cache)
        checkout = str(cache) if verified_checkout(cache, pin) else ""
        return [f"-DFETCHCONTENT_SOURCE_DIR_{pin.name.upper()}={checkout}"]

    def _copy_checkout(self, source: Path, destination: Path) -> None:
        self._start_artifact(destination)
        try:
            shutil.copytree(source, destination, symlinks=True)
            sync_tree(destination)
            sync_path(destination.parent)
        except OSError:
            with self._artifact_lock:
                self._started_artifacts.discard(destination)
            shutil.rmtree(destination, ignore_errors=True)
            self.state.append("artifact_abandoned",
                              **_artifact_record(self.config, destination))
            return
        self._finish_artifact(destination)

    def _configure(self, stage: str, operation: str, build: Path, arguments: Sequence[str],
                   attempt: int) -> CommandResult:
        moved_from = configured_elsewhere(build, self.config.repository)
        if moved_from is not None:
            # Like a half-built tree, a moved one is rebuilt from scratch; it is only ever
            # configured again when its outputs are about to be rebuilt anyway.
            _remove_artifact(self.config, build)
            self.state.append("moved_build_removed", **_artifact_record(self.config, build),
                              configured_at=moved_from)
            self._notice(f"removed {build}, configured at {moved_from}, to rebuild it here")
        return self._command(stage, operation, [
            "cmake", "--preset", self.config.build_preset, "-B", str(build), *arguments,
            *self._catch2_arguments()], attempt=attempt)

    def _notice(self, message: str) -> None:
        if self.progress is None:
            print(message, file=sys.stderr)
        else:
            self.progress.note(message)

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
        if self.progress is not None:
            self.progress.command_started(stage, operation)
        return self.commands.run(
            command, log_path=log, metrics_path=metrics,
            accepted_exit_codes=accepted_exit_codes,
            output_callback=(
                (lambda line, elapsed: self.progress.command_output(
                    stage, operation, line, elapsed))
                if self.progress is not None else None),
        )

    def _ensure_data_binary(self, stage: str, attempt: int, dataset: ManagedDatasetConfig, *,
                            required_commit: str | None = None) -> None:
        binary = data_binary(self.config, dataset)
        teacher = dataset.teacher
        if required_commit is not None and re.fullmatch(
                r"[0-9a-f]{40}", required_commit) is None:
            raise WorkflowError(
                f"label shards have invalid build commit {required_commit!r}")
        target_commit = required_commit or self.commit
        ready_commit = self._data_commits.get(binary)
        if ready_commit is not None and binary.is_file():
            if ready_commit == target_commit:
                return
            if required_commit is not None:
                raise WorkflowError(
                    f"run-local data binary is from commit {ready_commit}, but the "
                    f"completed label shards require {required_commit}")

        def binary_contains(*values: str) -> bool:
            try:
                contents = binary.read_bytes()
            except OSError as error:
                raise WorkflowError(
                    f"could not authenticate data binary {binary}: {error}") from error
            return all(value.encode() in contents for value in values)

        identity = (teacher.sha256,) if teacher is not None else ()
        # A partially completed label corpus must continue with the exact
        # run-local binary that created its earlier shards. Reconfiguring this
        # tree at a later source commit would silently embed a different build
        # identity, which the native feature reader correctly rejects.
        if required_commit is not None and binary.is_file():
            if binary_contains(required_commit, *identity):
                self._data_commits[binary] = required_commit
                return
        if required_commit is not None and required_commit != self.commit:
            raise WorkflowError(
                f"cannot resume label artifacts created by commit {required_commit}: restore "
                f"their run-local data binary at {binary}; the current commit is {self.commit}")

        build = data_build_directory(self.config, dataset)
        model_arguments: list[str] = []
        if teacher is not None:
            if not teacher.header.is_file() or sha256_file(teacher.header) != teacher.sha256:
                raise WorkflowError(
                    f"teacher model {teacher.name} is missing or changed: {teacher.header}")
            model_arguments.append(f"-DPOE2_MINIMAX_MODEL_HEADER={teacher.header}")
        # A build stopped midway may leave truncated outputs with fresh timestamps, so an
        # unfinished build tree is discarded whole and rebuilt rather than resumed.
        self._start_artifact(build)
        self._configure(stage, "configure", build, model_arguments, attempt)
        self._command(stage, "build-data", [
            "cmake", "--build", str(build),
            "--target", "poe2_minimax_data"], attempt=attempt)
        if not binary.is_file():
            raise WorkflowError(f"data binary was not built: {binary}")
        if not binary_contains(self.commit, *identity):
            raise WorkflowError(
                f"data binary does not contain the configured build commit "
                f"{self.commit}{' and teacher model' if teacher is not None else ''}: {binary}")
        self._finish_artifact(build)
        self._data_commits[binary] = self.commit

    def _begin(self, stage: str, action: str, input_value: Any) -> int:
        input_fingerprint = fingerprint(rendered_paths(input_value, self.config.recorded_path))
        locked_input = self.state.started_input_fingerprint(stage)
        # Inputs locked before paths were recorded relative to the repository named them
        # absolutely, which still matches where the repository was then.
        if locked_input not in {None, input_fingerprint,
                                fingerprint(rendered_paths(input_value, str))}:
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
        if self.progress is not None:
            self.progress.stage_started(stage, action)
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
        if self.progress is not None:
            self.progress.stage_completed(stage, created=creation > 0.0)

    def _fail(self, stage: str, error: BaseException, started: float) -> None:
        message = str(error)
        if isinstance(error, KeyboardInterrupt):
            message = f"interrupted{': ' + message if message else ''}"
        self.state.append("stage_failed", stage=stage,
                          wall_seconds=time.perf_counter() - started, error=message)
        self.state.write_summary(workflow_snapshot(self.config))
        if self.progress is not None:
            self.progress.stage_failed(stage, error)

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
                self._start_artifact(path)
                create(attempt)
                self._finish_artifact(path)
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
        if dataset.teacher is not None:
            # The teacher's iteration may have finished earlier in this same invocation.
            check_model_lineage(self.config, dataset.teacher)
        source = dataset if dataset.positions is None else self.config.dataset(dataset.positions)
        source_stats = self._prepare_source(source)
        label_stats = self._prepare_labels(dataset, source_stats)
        return self._prepare_features(dataset, source_stats, label_stats)

    def _prepare_import(self, dataset: ImportedDatasetConfig) -> dict[str, Any]:
        stage = f"dataset:{dataset.name}:features"
        if _path_state(dataset.feature_directory) != "complete":
            raise _blocked_path(dataset.feature_directory, "imported feature")
        attempt = self._begin(stage, "import", {"path": dataset.feature_directory})
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
            pending = _pending_source_chunks(dataset)
            commits = {source_chunk_commit(source_chunk_path(dataset, *item))
                       for item in source_chunk_ranges(dataset) if item not in pending}
            if len(commits) > 1:
                raise WorkflowError(
                    "completed source chunks have inconsistent build commits: " +
                    ", ".join(sorted(commits)))
            # Like label shards, earlier chunks can only be assembled by the build that made them.
            self._ensure_data_binary(
                stage, attempt, dataset, required_commit=next(iter(commits), None))
            for begin, end in pending:
                chunk = source_chunk_path(dataset, begin, end)
                self._start_artifact(chunk)
                self._command(stage, f"create-chunk-{begin:09d}",
                              source_chunk_command(self.config, dataset, begin, end),
                              attempt=attempt)
                self._finish_artifact(chunk)
            self._command(stage, "assemble", source_assemble_command(self.config, dataset),
                          attempt=attempt)

        def authenticate(attempt: int) -> dict[str, Any]:
            self._command(
                stage, "audit",
                source_audit_command(sys.executable, self.config, dataset), attempt=attempt)
            return authenticate_source(dataset)

        source_input = asdict(dataset.source)
        if dataset.teacher is not None and dataset.teacher.iteration is not None:
            # The stage fingerprint names the teacher's iteration; lock the model it became.
            source_input["teacher"] = dataset.teacher.model_id
        return self._run_simple(stage, dataset.source_directory, "position source",
                                source_input, create, authenticate)

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
        label_input = source_stats
        if dataset.positions is not None:
            # The source statistics name the positions' teacher; lock the one relabeling them.
            label_input = {"source": source_stats, "teacher": (
                None if dataset.teacher is None else dataset.teacher.model_id)}
        attempt = self._begin(stage, "resume-or-create", label_input)
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
                build_commits = {str(item["git_commit"]) for item in complete.values()}
                if len(build_commits) > 1:
                    raise WorkflowError(
                        "completed label shards have inconsistent build commits: " +
                        ", ".join(sorted(build_commits)))
                required_commit = next(iter(build_commits), None)
                self._ensure_data_binary(
                    stage, attempt, dataset, required_commit=required_commit)
                create_started = time.perf_counter()

                def run_shard(index: int) -> tuple[int, dict[str, Any], float]:
                    command_result = self._create_label_shard(stage, dataset, index, attempt)
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

    def _create_label_shard(self, stage: str, dataset: ManagedDatasetConfig, index: int,
                            attempt: int) -> CommandResult:
        shard = label_shard_path(dataset, index)
        self._start_artifact(shard)
        result = self._command(
            stage, f"create-shard-{index:03d}",
            label_command(self.config, dataset, index), attempt=attempt)
        self._finish_artifact(shard)
        return result

    def _prepare_features(self, dataset: ManagedDatasetConfig,
                          source_stats: dict[str, Any],
                          label_stats: dict[str, Any]) -> dict[str, Any]:
        stage = f"dataset:{dataset.name}:features"

        def create(attempt: int) -> None:
            label_commit = authenticate_label_shard(dataset, 0)["git_commit"]
            self._ensure_data_binary(
                stage, attempt, dataset, required_commit=str(label_commit))
            self._command(
                stage, "create", feature_command(self.config, dataset), attempt=attempt)

        def authenticate(attempt: int) -> dict[str, Any]:
            self._command(
                stage, "audit",
                feature_audit_command(sys.executable, self.config, dataset), attempt=attempt)
            return authenticate_features(dataset)

        return self._run_simple(
            stage, dataset.feature_directory, "feature artifact",
            {"source": source_stats, "labels": label_stats}, create, authenticate)

    def run_iteration(self, iteration: IterationConfig) -> None:
        for model in trained_models_used(self.config, iteration):
            _expected_model_id(model, f"iteration {iteration.name}")
        dependency_stats = {}
        for name in iteration.depends_on:
            dependency = self.config.iteration(name)
            if not self._iteration_artifacts_complete(dependency):
                raise WorkflowError(
                    f"iteration {iteration.name} requires completed dependency {name}; "
                    "rerun with --with-dependencies or run that iteration first")
            dependency_stats[name] = self._authenticate_iteration_outputs(dependency)
        dataset_stats = self.prepare_dataset(iteration.dataset)
        extra_stats = [self.prepare_dataset(dataset.name)
                       for dataset in extra_training_datasets(self.config, iteration)]
        training_stats = self._run_training(
            iteration, dataset_stats, dependency_stats, extra_stats)
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
            self.config.dataset(iteration.dataset), iteration,
            extra_training_datasets(self.config, iteration))
        if iteration.sealed_evaluation is not None:
            authenticate_evaluation(self.config, iteration)
        if iteration.candidate_validation is not None:
            authenticate_candidate(self.config, iteration)
        if iteration.engine_gate is not None:
            authenticate_gate(self.config, iteration)
        return training

    def _run_training(self, iteration: IterationConfig, dataset_stats: dict[str, Any],
                      dependencies: dict[str, Any],
                      extra_stats: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
        stage = f"iteration:{iteration.name}:training"
        dataset = self.config.dataset(iteration.dataset)
        extras = extra_training_datasets(self.config, iteration)

        def create(attempt: int) -> None:
            self._command(stage, "create", training_command(
                sys.executable, dataset, iteration, extras), attempt=attempt)

        def authenticate(_attempt: int) -> dict[str, Any]:
            return authenticate_training(dataset, iteration, extras)

        inputs: dict[str, Any] = {"dataset": dataset_stats, "dependencies": dependencies}
        if extra_stats:
            inputs["extra_datasets"] = list(extra_stats)
        return self._run_simple(stage, iteration.training_directory, "training report",
                                inputs, create, authenticate)

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
            configure = self._configure(
                stage, "configure", build, [f"-DPOE2_MINIMAX_MODEL_HEADER={header}"], attempt)
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
                    "opening_book": self.config.recorded_path(candidate.opening_book),
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
            training_datasets = candidate_training_datasets(iteration)
            if training_datasets is not None:
                report["training_datasets"] = training_datasets
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
        base_model = _gate_base_model(self.config, iteration)
        if base_model is not None:
            check_model_lineage(self.config, base_model)
        base_engine_binary = gate_base_binary(self.config, iteration)
        if candidate_stats is None:
            raise WorkflowError("engine gate requires authenticated candidate statistics")
        header_metadata = candidate_stats.get("artifacts", {}).get("model_header")
        if not isinstance(header_metadata, dict) or not isinstance(
                header_metadata.get("sha256"), str) or not isinstance(
                header_metadata.get("path"), str):
            raise WorkflowError("candidate report has no authenticated model header")
        build_id = candidate_build_id(candidate_stats)
        if base_model is not None and base_model.model_id == build_id:
            raise WorkflowError(
                f"engine gate for {iteration.name} would play its candidate against itself")
        build = gate_build_directory(self.config, iteration, candidate_stats)
        candidate_header = iteration.candidate_directory / header_metadata["path"]

        def require_candidate_header() -> None:
            if (not candidate_header.is_file() or
                    sha256_file(candidate_header) != header_metadata["sha256"]):
                raise WorkflowError(
                    f"candidate model header is missing or changed: {candidate_header}")

        require_candidate_header()

        def create(attempt: int) -> None:
            reserve_report_directory(root, GATE_SCHEMA)
            self._configure(stage, "configure", build,
                            [f"-DPOE2_MINIMAX_MODEL_HEADER={candidate_header}"], attempt)
            self._command(stage, "build", ["cmake", "--build", str(build)], attempt=attempt)
            self._command(stage, "test", [
                "ctest", "--test-dir", str(build), "--output-on-failure"], attempt=attempt)
            require_candidate_header()
            if base_model is not None:
                # Both sides build from this commit, so the gate measures only the weights.
                self._build_model_engine(
                    stage, attempt, base_model, gate_base_build_directory(self.config, iteration),
                    "base", engine=gate.base_engine)
            runner = build / "runner" / "poe2_runner"
            command = engine_gate_command(self.config, iteration, build, runner)
            result = self._command(stage, "evaluate", command, attempt=attempt)
            run_path = _evaluator_run(self.config, result, root, "engine gate")
            manifest = json.loads((run_path / "manifest.json").read_bytes())
            build_artifacts = {
                "new_engine_binary": build / "engines" / gate.new_engine,
                "runner_binary": runner,
            }
            for name, path in build_artifacts.items():
                if not path.is_file():
                    raise WorkflowError(f"engine gate did not produce {name}: {path}")
            report = {
                "schema": GATE_SCHEMA, "schema_version": GATE_SCHEMA_VERSION,
                "created_at_utc": now_utc(), "git_commit": self.commit,
                "settings": {
                    **recorded_gate_settings(gate),
                    "opening_book": self.config.recorded_path(gate.opening_book),
                },
                "opening_book_sha256": sha256_file(gate.opening_book),
                "run_id": manifest.get("run_id"),
                "run_directory": run_path.relative_to(root).as_posix(),
                "candidate_report_sha256": candidate_stats["report_sha256"],
                "candidate_build_id": build_id,
                "build_directory": build.relative_to(root).as_posix(),
                "model_header_sha256": header_metadata["sha256"],
                "base_engine_artifact": {
                    "path": self.config.recorded_path(base_engine_binary),
                    "sha256": sha256_file(base_engine_binary),
                },
                "build_artifacts": {
                    name: {"path": path.relative_to(root).as_posix(),
                           "sha256": sha256_file(path)}
                    for name, path in build_artifacts.items()
                },
                "artifacts": _run_artifacts(run_path),
            }
            complete_json_report(root, GATE_SCHEMA, report)

        def authenticate(_attempt: int) -> dict[str, Any]:
            return authenticate_gate(self.config, iteration)

        return self._run_simple(stage, root, "engine gate", {
            "candidate": candidate_stats,
            "opening_book_sha256": sha256_file(gate.opening_book),
            **({"base_engine_sha256": sha256_file(base_engine_binary)} if base_model is None
               else {"base_model": base_model.model_id}),
        }, create, authenticate)

    def _build_model_engine(self, stage: str, attempt: int, model: ModelReference, build: Path,
                            role: str, *, engine: str = CANDIDATE_ENGINE,
                            runner: bool = False) -> Path:
        """Build one model's engine privately at this commit, optionally with the match runner."""
        if model.header is None or not model.header.is_file() or (
                sha256_file(model.header) != model.sha256):
            raise WorkflowError(f"model {model.name} is missing or changed: {model.header}")
        self._configure(stage, f"configure-{role}", build,
                        [f"-DPOE2_MINIMAX_MODEL_HEADER={model.header}"], attempt)
        targets = ["poe2_minimax", *(["poe2_runner"] if runner else [])]
        self._command(stage, f"build-{role}", ["cmake", "--build", str(build), "--target",
                                                *targets], attempt=attempt)
        binary = build / "engines" / engine
        if not binary.is_file():
            raise WorkflowError(f"{stage} did not build model {model.name}: {binary}")
        return binary

    def run_comparison(self, comparison: ComparisonConfig) -> dict[str, Any]:
        """Play a comparison between both models, each built privately at this commit."""
        stage = comparison_stage(comparison)
        if not comparison.opening_book.is_file():
            raise WorkflowError(
                f"comparison opening book is missing: {comparison.opening_book}")
        new, base = _comparison_models(self.config, comparison)
        root = comparison.root

        def create(attempt: int) -> None:
            reserve_report_directory(root, COMPARISON_SCHEMA)
            sides = {"new": comparison_side(self.config, comparison, new),
                     "base": comparison_side(self.config, comparison, base)}
            binaries = {
                "new_engine_binary": self._build_model_engine(
                    stage, attempt, new, root / sides["new"]["build_directory"], "new",
                    runner=True),
                "base_engine_binary": self._build_model_engine(
                    stage, attempt, base, root / sides["base"]["build_directory"], "base"),
            }
            binaries["runner_binary"] = (
                root / sides["new"]["build_directory"] / "runner" / "poe2_runner")
            result = self._command(stage, "evaluate", comparison_command(
                self.config, comparison, new.model_id, base.model_id),
                attempt=attempt)
            run_path = _evaluator_run(self.config, result, root, "comparison")
            report = {
                "schema": COMPARISON_SCHEMA, "schema_version": 1,
                "created_at_utc": now_utc(), "git_commit": self.commit,
                "settings": {**match_settings(comparison),
                             "opening_book": self.config.recorded_path(comparison.opening_book)},
                "opening_book_sha256": sha256_file(comparison.opening_book),
                "run_id": json.loads((run_path / "manifest.json").read_bytes()).get("run_id"),
                "run_directory": run_path.relative_to(root).as_posix(),
                **sides,
                "build_artifacts": {
                    name: {"path": path.relative_to(root).as_posix(),
                           "sha256": sha256_file(path)}
                    for name, path in binaries.items()
                },
                "artifacts": _run_artifacts(run_path),
            }
            complete_json_report(root, COMPARISON_SCHEMA, report)

        def authenticate(_attempt: int) -> dict[str, Any]:
            return authenticate_comparison(self.config, comparison, new, base)

        return self._run_simple(stage, root, "comparison", {
            "new": new.model_id, "base": base.model_id,
            "opening_book_sha256": sha256_file(comparison.opening_book),
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
        producers = [model.iteration for model in trained_models_used(config, item)]
        for name in (*item.depends_on, *producers):
            add(config.iteration(name))
        if item.name not in added:
            result.append(item)
            added.add(item.name)

    add(iteration)
    return result


def _comparison_order(config: WorkflowConfig,
                      comparison: ComparisonConfig) -> list[IterationConfig]:
    """Return the iterations, with their dependencies, that train a comparison's sides."""
    needed: set[str] = set()
    for name in (comparison.new, comparison.base):
        iteration = config.model(name).iteration
        if iteration is not None:
            needed.update(item.name for item in
                          _dependency_order(config, config.iteration(iteration)))
    return [item for item in config.iterations if item.name in needed]


def _require_unblocked(config: WorkflowConfig, name: str, kind: str) -> None:
    blocked = (blocked_iterations(config) if kind == "iteration" else
               blocked_comparisons(config)).get(name)
    if blocked is not None:
        raise WorkflowError(f"{kind} {name} is not started: {blocked.reason}")


def _selected_work(config: WorkflowConfig, iteration_name: str | None,
                   comparison_name: str | None, with_dependencies: bool,
                   ) -> tuple[list[IterationConfig], list[ComparisonConfig]]:
    if comparison_name is not None:
        comparison = config.comparison(comparison_name)
        _require_unblocked(config, comparison_name, "comparison")
        if with_dependencies:
            return _comparison_order(config, comparison), [comparison]
        if not _comparison_ready(config, comparison):
            raise WorkflowError(
                f"comparison {comparison_name} needs the candidates of the iterations that train "
                "its models; run them first or rerun with --with-dependencies")
        return [], [comparison]
    if iteration_name is not None:
        requested = config.iteration(iteration_name)
        _require_unblocked(config, iteration_name, "iteration")
        return (_dependency_order(config, requested) if with_dependencies else [requested]), []
    return list(config.iterations), list(config.comparisons)


def run_workflow(config: WorkflowConfig, *, iteration_name: str | None = None,
                 comparison_name: str | None = None,
                 with_dependencies: bool = False) -> list[Blocked]:
    """Execute a full run, one iteration, or one comparison under its create-only run lock.

    Comparisons play one at a time, in recipe order, as soon as both models exist. Work that
    needs a model whose gate did not accept it is skipped and returned; requested work that is
    blocked is an error instead.
    """
    selected, comparisons = _selected_work(
        config, iteration_name, comparison_name, with_dependencies)
    command = ("comparison" if comparison_name else "iteration" if iteration_name else "run")
    commit = require_clean_committed(config)
    with _recovered_run_lock(config) as (lock_descriptor, state):
        state.validate_evolution()
        check_model_lineage(config)
        state.recover_running_stages()
        state.accept_config()
        state.append("run_started", git_commit=commit, config_digest=config.digest,
                     command=command, iteration=iteration_name, comparison=comparison_name)
        state.write_summary(workflow_snapshot(config))
        progress = WorkflowProgress(config, state)
        runner = WorkflowRunner(config, state, commit, progress)
        runner.commands.lock_descriptors = (lock_descriptor,)
        not_started: list[Blocked] = []
        pending = list(comparisons)
        success = False

        def play_ready_comparisons() -> None:
            for comparison in list(pending):
                if _comparison_ready(config, comparison):
                    pending.remove(comparison)
                    runner.config = resolve_trained_models(config)
                    runner.run_comparison(comparison)
                    state.append("comparison_completed", comparison=comparison.name)
                    state.write_summary(workflow_snapshot(config))
                    progress.comparison_completed(comparison.name)

        try:
            with _interruptions_as_keyboard():
                play_ready_comparisons()
                for iteration in selected:
                    if iteration.name == iteration_name:
                        _require_unblocked(config, iteration_name, "iteration")
                    blocked = blocked_iterations(config).get(iteration.name)
                    if blocked is not None:
                        not_started.append(blocked)
                        state.append("iteration_not_started", iteration=iteration.name,
                                     reason=blocked.reason)
                        progress.not_started(blocked.kind, blocked.name, blocked.reason)
                        continue
                    # A model this recipe trains may have been accepted earlier in this run.
                    runner.config = resolve_trained_models(config)
                    runner.run_iteration(iteration)
                    state.append("iteration_completed", iteration=iteration.name)
                    state.write_summary(workflow_snapshot(config))
                    progress.iteration_completed(iteration.name)
                    play_ready_comparisons()
                for comparison in pending:
                    if comparison.name == comparison_name:
                        _require_unblocked(config, comparison_name, "comparison")
                    blocked = blocked_comparisons(config).get(comparison.name)
                    if blocked is None:
                        raise WorkflowError(
                            f"comparison {comparison.name} has a model that was not trained")
                    not_started.append(blocked)
                    state.append("comparison_not_started", comparison=comparison.name,
                                 reason=blocked.reason)
                    progress.not_started(blocked.kind, blocked.name, blocked.reason)
                state.append("run_completed", command=command, iteration=iteration_name,
                             comparison=comparison_name,
                             not_started=[item.label for item in not_started])
                state.write_summary(workflow_snapshot(config))
                success = True
        except KeyboardInterrupt as error:
            runner.commands.terminate_all()
            removed = runner.discard_interrupted_artifacts()
            state.append("run_interrupted", error="keyboard interrupt",
                         removed=[str(path) for path in removed])
            state.write_summary(workflow_snapshot(config))
            discarded = "".join(f"\n  removed unfinished {path}" for path in removed)
            raise WorkflowError(
                "workflow interrupted; completed artifacts were preserved and the same command "
                f"resumes{discarded}") from error
        except BaseException as error:
            runner.commands.terminate_all()
            failed = runner.keep_failed_artifacts()
            state.append("run_failed", error=str(error), kept=[str(path) for path in failed])
            state.write_summary(workflow_snapshot(config))
            raise
        finally:
            progress.close(success=success)
    return not_started


def handoff_candidate(config: WorkflowConfig, iteration_name: str, *, apply: bool = False) -> dict[str, Any]:
    """Preview or deliberately install one authenticated generated header."""
    if apply:
        with _recovered_run_lock(config):
            return _handoff_candidate(config, iteration_name, apply=True)
    return _handoff_candidate(config, iteration_name, apply=False)


def _handoff_candidate(config: WorkflowConfig, iteration_name: str, *,
                       apply: bool) -> dict[str, Any]:
    config = resolve_trained_models(config)
    iteration = config.iteration(iteration_name)
    candidate = authenticate_candidate(config, iteration)
    header_metadata = candidate["artifacts"].get("model_header")
    if not isinstance(header_metadata, dict):
        raise WorkflowError("candidate report has no model header")
    source = iteration.candidate_directory / header_metadata["path"]
    target = config.repository / "engines" / "minimax" / "src" / "frozen_pattern_gain_model.hpp"
    tracked_digest = sha256_file(target)
    archive = _archive_entry(config, iteration, candidate)
    gate = (authenticate_gate(config, iteration, candidate=candidate)
            if iteration.engine_gate is not None and
            _path_state(iteration.gate_directory) == "complete" else None)
    preview = {
        "iteration": iteration_name, "promotable": candidate["promotable"],
        "candidate_header": str(source), "candidate_sha256": header_metadata["sha256"],
        "tracked_header": str(target), "tracked_sha256": tracked_digest,
        "gate_decision": gate["sequential_decision"] if gate is not None else None,
        "archived_header": str(archive.header_path),
        "archived": _archived_model_state(archive),
        "different": tracked_digest != header_metadata["sha256"], "applied": False,
    }
    if not apply:
        return preview
    if not candidate["promotable"]:
        raise WorkflowError(
            f"candidate {iteration_name} is explicitly non-promotable")
    if iteration.engine_gate is None:
        raise WorkflowError(
            f"candidate {iteration_name} has no engine gate; only gate winners can be installed")
    if gate is None:
        raise WorkflowError(f"engine gate for {iteration_name} is not complete")
    if gate["sequential_decision"] != "accept_alt":
        raise WorkflowError(
            f"candidate {iteration_name} did not win its engine gate "
            f"(decision={gate['sequential_decision']}); only gate winners can be installed")
    if not preview["archived"] or not _master_ledger_has_run(config, gate["run_id"]):
        raise WorkflowError(
            f"record the gate first: run promote-gate --apply for {iteration_name}, then "
            "commit its ledger row and archived model")
    require_clean_committed(config)
    if preview["different"]:
        try:
            contents = archive.header_path.read_bytes()
            if hashlib.sha256(contents).hexdigest() != header_metadata["sha256"]:
                raise WorkflowError(
                    f"archived header changed during handoff: {archive.header_path}")
            atomic_write_bytes(target, contents, replace=True)
        except OSError as error:
            raise WorkflowError(f"could not install candidate header at {target}: {error}") from error
        preview["applied"] = True
    return preview


@dataclass(frozen=True)
class ArchiveEntry:
    """One gated model's content-addressed files under the repository's models/ folder."""

    header_source: Path
    header_sha256: str
    header_path: Path
    metrics_path: Path
    metrics: bytes


def _repository_relative(config: WorkflowConfig, path: Path) -> str:
    try:
        return path.relative_to(config.repository).as_posix()
    except ValueError:
        return str(path)


def _archive_entry(config: WorkflowConfig, iteration: IterationConfig,
                   candidate: dict[str, Any]) -> ArchiveEntry:
    header_metadata = candidate["artifacts"]["model_header"]
    model_id = candidate_build_id(candidate)
    directory = config.repository / MODEL_ARCHIVE_DIRECTORY
    model = candidate["model"]
    # Only fields fixed by the authenticated candidate belong here, so a retried promotion
    # regenerates byte-identical metrics and the archive stays idempotent.
    record = {
        "schema": MODEL_ARCHIVE_SCHEMA, "schema_version": 1,
        "model_id": model_id, "model_header_sha256": header_metadata["sha256"],
        "architecture": model["name"],
        "recipe": {"path": _repository_relative(config, config.path),
                   "run": config.name, "iteration": iteration.name},
        "git_commit": candidate["git_commit"],
        "validated_at_utc": candidate["created_at_utc"],
        "corpus_id": candidate["corpus_id"],
        "teacher": ({"evaluator": "two-ply-closure"} if candidate["teacher"] is None else
                    {"evaluator": "pattern-gain", "model_id": candidate["teacher"]}),
        "provenance": {
            "feature_binary_sha256": candidate["feature_binary_sha256"],
            "training_report_sha256": candidate["training_report_sha256"],
            "sealed_evaluation_report_sha256": candidate["sealed_evaluation_report_sha256"],
            "candidate_report_sha256": candidate["report_sha256"],
        },
        "metrics": {"float": model["float"], "quantized": model["quantized"]},
        **({"training_datasets": candidate["training_datasets"]}
           if "training_datasets" in candidate else {}),
        "parity": candidate["parity"],
        "benchmarks": {"evaluator": candidate["evaluator_benchmark"],
                       "search": candidate["search_benchmark"]},
    }
    return ArchiveEntry(
        header_source=iteration.candidate_directory / header_metadata["path"],
        header_sha256=header_metadata["sha256"],
        header_path=directory / f"{model_id}.hpp",
        metrics_path=directory / f"{model_id}.json",
        metrics=(json.dumps(record, indent=2, sort_keys=True) + "\n").encode(),
    )


def _archived_model_state(entry: ArchiveEntry) -> bool:
    """Return whether both archive files exist; reject files that disagree with the candidate."""
    if entry.header_path.exists() and sha256_file(entry.header_path) != entry.header_sha256:
        raise WorkflowError(
            f"archived model header differs from the candidate: {entry.header_path}")
    if entry.metrics_path.exists() and entry.metrics_path.read_bytes() != entry.metrics:
        raise WorkflowError(
            f"archived model metrics differ from the candidate: {entry.metrics_path}")
    return entry.header_path.is_file() and entry.metrics_path.is_file()


def _write_archived_model(entry: ArchiveEntry) -> None:
    if _archived_model_state(entry):
        return
    try:
        contents = entry.header_source.read_bytes()
        if hashlib.sha256(contents).hexdigest() != entry.header_sha256:
            raise WorkflowError(
                f"candidate header changed before archiving: {entry.header_source}")
        for path, data in ((entry.header_path, contents), (entry.metrics_path, entry.metrics)):
            if not path.exists():
                atomic_write_bytes(path, data)
    except OSError as error:
        raise WorkflowError(f"could not archive model {entry.header_path.stem}: {error}") from error


def _games_and_score(saved: dict[str, str]) -> tuple[int, str]:
    """Return games played and the new side's score, counting a no-winner game as half."""
    games = int(saved["games_played"])
    if games <= 0:
        raise WorkflowError("authenticated match played no games")
    score = 100.0 * (int(saved["engine_one_wins"]) + 0.5 * int(saved["no_winner"])) / games
    return games, f"{score:.1f}"


def _model_index_row(config: WorkflowConfig, iteration: IterationConfig,
                     candidate: dict[str, Any], gate: dict[str, Any],
                     saved: dict[str, str]) -> list[str]:
    """Return one readable index row built only from authenticated gate and candidate fields."""
    games, score = _games_and_score(saved)
    sealed = candidate["model"]["quantized"].get("sealed_test")
    sealed_mae = f"{sealed['mae']:.3f}" if isinstance(sealed, dict) else ""
    model_digest = candidate["artifacts"]["model_header"]["sha256"]
    base, base_model = saved["base_id"], ""
    model = _gate_base_model(config, iteration)
    if model is not None:
        # People read the run-scoped generation name; the digest keeps the row exact.
        base, base_model = f"{config.name}/{model.name}", model.sha256[:12]
    return [
        saved["created_at_utc"], config.name, iteration.name, model_digest[:12],
        base, base_model, gate["sequential_decision"], str(games), score,
        sealed_mae, gate["run_id"],
    ]


def _comparison_index_row(config: WorkflowConfig, comparison: ComparisonConfig,
                          new: ModelReference, base: ModelReference, match: dict[str, Any],
                          saved: dict[str, str]) -> list[str]:
    """Return one readable comparison row built only from authenticated fields."""
    games, score = _games_and_score(saved)
    return [
        saved["created_at_utc"], config.name, comparison.name,
        f"{config.name}/{new.name}", new.sha256[:12], f"{config.name}/{base.name}",
        base.sha256[:12], match["sequential_decision"], str(games), score, match["run_id"],
    ]


def _csv_has_row(path: Path, header: Sequence[str], key: str, row: list[str]) -> bool:
    """Return whether a tracked index already records this row's key; reject conflicts."""
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.reader(source))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise WorkflowError(f"could not read index {path}: {error}") from error
    if not rows or tuple(rows[0]) != tuple(header):
        raise WorkflowError(f"index has an unexpected header: {path}")
    key_index = list(header).index(key)
    matches = [item for item in rows[1:] if len(item) > key_index and item[key_index] == row[key_index]]
    if any(item != row for item in matches):
        raise WorkflowError(
            f"index already records {row[key_index]} with different values: {path}")
    return bool(matches)


def _append_csv_row(path: Path, header: Sequence[str], key: str, row: list[str]) -> None:
    if _csv_has_row(path, header, key, row):
        return
    try:
        if not path.exists():
            contents = io.StringIO()
            csv.writer(contents, lineterminator="\n").writerows([header, row])
            atomic_write_bytes(path, contents.getvalue().encode("utf-8"))
            return
        with path.open("a", newline="", encoding="utf-8") as destination:
            csv.writer(destination, lineterminator="\n").writerow(row)
            destination.flush()
            os.fsync(destination.fileno())
    except (OSError, csv.Error) as error:
        raise WorkflowError(f"could not append index {path}: {error}") from error


def _master_ledger_path(config: WorkflowConfig) -> Path:
    return config.repository / "eval" / "results.csv"


def _read_master_ledger(master: Path) -> list[list[str]]:
    try:
        with master.open(newline="", encoding="utf-8") as source:
            return list(csv.reader(source))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise WorkflowError(f"could not read master ledger {master}: {error}") from error


def _master_ledger_has_run(config: WorkflowConfig, run_id: str) -> bool:
    rows = _read_master_ledger(_master_ledger_path(config))
    if not rows or "run_id" not in rows[0]:
        raise WorkflowError("master ledger has no run_id column")
    index = rows[0].index("run_id")
    return any(len(row) > index and row[index] == run_id for row in rows[1:])


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


@dataclass(frozen=True)
class SavedLedgerRow:
    """The ledger row a match saved beside its run, checked against its manifest and summary."""

    path: Path
    header: list[str]
    row: list[str]

    @property
    def run_index(self) -> int:
        return self.header.index("run_id")

    @property
    def run_id(self) -> str:
        return self.row[self.run_index]

    @property
    def values(self) -> dict[str, str]:
        return dict(zip(self.header, self.row, strict=True))


def _verified_ledger_row(config: WorkflowConfig, match: dict[str, Any], settings: MatchConfig,
                         *, kind: str, new_id: str, label: str) -> SavedLedgerRow:
    """Return a match's saved ledger row after checking every field it shares with the run.

    The row names its paths relative to the repository, as the tracked ledger does, even if the
    match recorded them absolutely.
    """
    path = Path(match["ledger_row"])
    header, row = _read_ledger_row(path)
    if "run_id" not in header:
        raise WorkflowError("saved ledger schema has no run_id")
    saved = SavedLedgerRow(path, header, row)
    if saved.run_id != match["run_id"]:
        raise WorkflowError(f"saved ledger row run ID differs from authenticated {label}")
    values = saved.values
    manifest = match.get("manifest")
    summary = match.get("summary")
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        raise WorkflowError(f"authenticated {label} lacks manifest or summary statistics")
    expected_saved = {
        "kind": kind,
        "new_id": new_id,
        "new_engine": settings.new_engine,
        "new_engine_args": settings.new_engine_args,
        "base_engine": settings.base_engine,
        "base_engine_args": settings.base_engine_args,
        "games_requested": str(settings.games),
        "go_movetime_ms": str(settings.go_movetime_ms),
        "timeout_ms": str(settings.timeout_ms),
        "valid": "true",
        "workers_requested": str(settings.workers),
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
            f"authenticated {label} lacks ledger provenance fields: " + ", ".join(missing))
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
        raise WorkflowError(f"authenticated {label} lacks statistical score counts")
    for ledger_name, score_name in {
            "pair_score_0": "0", "pair_score_0_5": "0.25", "pair_score_1": "0.5",
            "pair_score_1_5": "0.75", "pair_score_2": "1"}.items():
        if score_name not in score_counts:
            raise WorkflowError(
                f"authenticated {label} lacks statistical score count {score_name}")
        expected_saved[ledger_name] = _ledger_scalar(score_counts[score_name])
    mismatches = [name for name, value in expected_saved.items()
                  if values.get(name) != value]
    paths = {"opening_book": settings.opening_book, "run_dir": Path(match["run_directory"])}
    mismatches.extend(name for name, path in paths.items()
                      if not names_path(config, values.get(name), path))
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
                values.get(ledger_name), manifest[manifest_name]):
            mismatches.append(ledger_name)
    if mismatches:
        raise WorkflowError(
            f"saved ledger row differs from the authenticated {label} in: " +
            ", ".join(sorted(set(mismatches))))
    portable = {name: config.recorded_path(path) for name, path in paths.items()}
    return replace(saved, row=[portable.get(name, value) for name, value in zip(header, row)])


def _inspect_master(master: Path, saved: SavedLedgerRow) -> bool:
    """Return whether the master ledger already has this run; reject a schema mismatch."""
    existing = _read_master_ledger(master)
    if not existing or existing[0] != saved.header:
        raise WorkflowError(f"master and saved ledger schemas differ: {master}")
    if any(len(item) != len(saved.header) for item in existing[1:]):
        raise WorkflowError(f"master ledger contains a malformed row: {master}")
    return any(item[saved.run_index] == saved.run_id for item in existing[1:])


def _preview_ledger_row(config: WorkflowConfig, saved: SavedLedgerRow) -> None:
    if _inspect_master(_master_ledger_path(config), saved):
        raise WorkflowError(f"master ledger already contains run ID {saved.run_id}")


def _append_ledger_row(config: WorkflowConfig, saved: SavedLedgerRow,
                       record_first: Callable[[], None]) -> None:
    """Append a verified row under the ledger lock, after the records that describe it."""
    master = _master_ledger_path(config)
    require_clean_committed(config)
    try:
        with _ledger_lock(master):
            require_clean_committed(config)
            _preview_ledger_row(config, saved)
            # Records are written before the row so an interrupted apply retries cleanly: they
            # are idempotent for the same run, while the ledger rejects duplicates.
            record_first()
            with master.open("a", newline="", encoding="utf-8") as destination:
                writer = csv.writer(destination, lineterminator="\n")
                writer.writerow(saved.row)
                destination.flush()
                os.fsync(destination.fileno())
    except (OSError, csv.Error) as error:
        raise WorkflowError(f"could not append master ledger {master}: {error}") from error


def _is_archived(config: WorkflowConfig, model: ModelReference) -> bool:
    directory = config.repository / MODEL_ARCHIVE_DIRECTORY
    header = directory / f"{model.model_id}.hpp"
    return (header.is_file() and (directory / f"{model.model_id}.json").is_file() and
            sha256_file(header) == model.sha256)


def promote_gate(config: WorkflowConfig, iteration_name: str, *, apply: bool = False) -> dict[str, Any]:
    """Preview or append one authenticated local engine-gate row to the master ledger."""
    if apply:
        with _recovered_run_lock(config):
            return _promote_gate(config, iteration_name, apply=True)
    return _promote_gate(config, iteration_name, apply=False)


def _require_archived_inputs(config: WorkflowConfig, iteration: IterationConfig) -> None:
    """Require every trained model a gate's iteration used to be archived before it is."""
    for model in trained_models_used(config, iteration):
        _expected_model_id(model, f"iteration {iteration.name}")
        if not _is_archived(config, model):
            raise WorkflowError(
                f"promote {model.iteration} first: {iteration.name} uses its model "
                f"{model.name}, which is not archived under {MODEL_ARCHIVE_DIRECTORY}/; run "
                f"promote-gate --apply {model.iteration} and commit the result")


def _promote_gate(config: WorkflowConfig, iteration_name: str, *,
                  apply: bool) -> dict[str, Any]:
    config = resolve_trained_models(config)
    iteration = config.iteration(iteration_name)
    _require_archived_inputs(config, iteration)
    candidate = authenticate_candidate(config, iteration)
    gate = authenticate_gate(config, iteration, candidate=candidate)
    assert iteration.engine_gate is not None
    saved = _verified_ledger_row(config, gate, iteration.engine_gate, kind=GATE_KIND,
                                 new_id=gate["candidate_build_id"], label="gate")
    archive = _archive_entry(config, iteration, candidate)
    model_index = config.repository / MODEL_INDEX_PATH
    index_row = _model_index_row(config, iteration, candidate, gate, saved.values)
    preview = {"iteration": iteration_name, "run_id": saved.run_id,
               "model_id": gate["candidate_build_id"],
               "sequential_decision": gate["sequential_decision"],
               "saved_row": str(saved.path), "ledger": str(_master_ledger_path(config)),
               "archived_header": str(archive.header_path),
               "archived_metrics": str(archive.metrics_path),
               "archived": _archived_model_state(archive),
               "model_index": str(model_index),
               "index_row": dict(zip(MODEL_INDEX_HEADER, index_row, strict=True)),
               "indexed": _csv_has_row(model_index, MODEL_INDEX_HEADER, "gate_run_id",
                                       index_row),
               "duplicate": False, "applied": False}
    if not apply:
        _preview_ledger_row(config, saved)
        return preview

    def record() -> None:
        _write_archived_model(archive)
        preview["archived"] = True
        _append_csv_row(model_index, MODEL_INDEX_HEADER, "gate_run_id", index_row)
        preview["indexed"] = True

    _append_ledger_row(config, saved, record)
    preview["applied"] = True
    return preview


def record_comparison(config: WorkflowConfig, name: str, *,
                      apply: bool = False) -> dict[str, Any]:
    """Preview or append one finished comparison to the master ledger and comparison index."""
    if apply:
        with _recovered_run_lock(config):
            return _record_comparison(config, name, apply=True)
    return _record_comparison(config, name, apply=False)


def _record_comparison(config: WorkflowConfig, name: str, *, apply: bool) -> dict[str, Any]:
    config = resolve_trained_models(config)
    comparison = config.comparison(name)
    _require_unblocked(config, name, "comparison")
    if _path_state(comparison.root) != "complete":
        raise WorkflowError(f"comparison {name} has not finished; play it first")
    new, base = _comparison_models(config, comparison)
    for model in (new, base):
        if not _is_archived(config, model):
            raise WorkflowError(
                f"promote {model.iteration} first: comparison {name} plays its model "
                f"{model.name}, which is not archived under {MODEL_ARCHIVE_DIRECTORY}/; run "
                f"promote-gate --apply {model.iteration} and commit the result")
    match = authenticate_comparison(config, comparison, new, base)
    saved = _verified_ledger_row(config, match, comparison, kind=COMPARISON_KIND,
                                 new_id=new.model_id, label="comparison")
    index = config.repository / COMPARISON_INDEX_PATH
    row = _comparison_index_row(config, comparison, new, base, match, saved.values)
    preview = {"comparison": name, "run_id": saved.run_id,
               "new_model": new.model_id, "base_model": base.model_id,
               "sequential_decision": match["sequential_decision"],
               "saved_row": str(saved.path), "ledger": str(_master_ledger_path(config)),
               "comparison_index": str(index),
               "index_row": dict(zip(COMPARISON_INDEX_HEADER, row, strict=True)),
               "indexed": _csv_has_row(index, COMPARISON_INDEX_HEADER, "run_id", row),
               "applied": False}
    if not apply:
        _preview_ledger_row(config, saved)
        return preview

    def record() -> None:
        _append_csv_row(index, COMPARISON_INDEX_HEADER, "run_id", row)
        preview["indexed"] = True

    _append_ledger_row(config, saved, record)
    preview["applied"] = True
    return preview
