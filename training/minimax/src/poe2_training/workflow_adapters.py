"""Typed adapters between workflow stages and the authoritative lower-level tools."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .artifact import FeatureArtifact, open_feature_artifact
from .baseline import open_baseline_report
from .net_file import FILE_SUFFIX, NetFileError, read_net
from .pattern_evaluation import open_pattern_evaluation
from .pattern_experiment import (
    FROZEN_PATTERN_GAIN_FRACTIONAL_BITS,
    open_pattern_report,
)
from .pattern_suites import pattern_suite
from .shared import open_json_report, sha256_file
from .workflow_config import (
    BaselineTrainingConfig,
    ComparisonConfig,
    DatasetConfig,
    IterationConfig,
    ManagedDatasetConfig,
    MatchConfig,
    ModelReference,
    PatternTrainingConfig,
    WorkflowConfig,
    match_settings,
    recorded_gate_settings,
)


CANDIDATE_SCHEMA = "poe2-training-candidate"
# Version 2 candidates are net files checked with the run's shared build; version 1 candidates
# were C++ headers built privately.
CANDIDATE_SCHEMA_VERSION = 2
GATE_SCHEMA = "poe2-training-engine-gate"
# Version 3 gates play net files on one engine build. Version 2 gates built the candidate's
# C++ header privately.
GATE_SCHEMA_VERSION = 3
HEADER_GATE_SCHEMA_VERSION = 2
COMPARISON_SCHEMA = "poe2-training-comparison"
# Version 2 comparisons play both nets on one engine build.
COMPARISON_SCHEMA_VERSION = 2
# Ledger kinds; only gate rows decide champions and chains.
GATE_KIND = "training-gate"
COMPARISON_KIND = "training-comparison"


class AdapterError(ValueError):
    """Raised when a lower-level stage artifact fails orchestration checks."""


@dataclass(frozen=True)
class StageAdapter:
    """Registry metadata for one typed orchestration stage."""

    name: str
    output_kind: str
    create_only: bool
    command_builders: tuple[Callable[..., list[str]], ...]
    audit_builders: tuple[Callable[..., list[str]], ...]
    authenticator: Callable[..., dict[str, Any]]


def names_path(config: WorkflowConfig, recorded: Any, path: Path) -> bool:
    """Return whether a record names path: relative to the repository, or absolutely as before.

    Records written before paths were relative name them absolutely, which matches only while
    the repository stays where it was.
    """
    return isinstance(recorded, str) and recorded in {config.recorded_path(path), str(path)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def _require_report_layout(directory: Path, label: str,
                           attachments: set[str] | None = None) -> None:
    try:
        names = {path.name for path in directory.iterdir()}
    except OSError as error:
        raise AdapterError(f"could not inspect {directory}: {error}") from error
    expected = {"COMPLETE", "report.json"} | (attachments or set())
    _require(names == expected,
             f"{label} has unexpected output files: {directory}")


def data_build_directory(config: WorkflowConfig) -> Path:
    """Return the run's build tree. Teachers are nets the data tool loads, so one serves all."""
    return config.output_directory / "build" / config.build_preset


def configured_elsewhere(build: Path, repository: Path) -> str | None:
    """Return where a CMake build tree was configured, if not at build from repository.

    CMake refuses to reconfigure a tree whose cache names another build or source folder.
    """
    try:
        lines = (build / "CMakeCache.txt").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as error:
        raise AdapterError(f"could not read the CMake cache of {build}: {error}") from error
    cache = dict(line.split("=", 1) for line in lines if "=" in line and ":" in line)
    for key, expected in (("CMAKE_CACHEFILE_DIR:INTERNAL", build),
                          ("CMAKE_HOME_DIRECTORY:INTERNAL", repository)):
        recorded = cache.get(key)
        if recorded is not None and Path(recorded).resolve() != expected.resolve():
            return recorded
    return None


def data_binary(config: WorkflowConfig) -> Path:
    return data_build_directory(config) / "runner" / "poe2_minimax_data"


@dataclass(frozen=True)
class PinnedCheckout:
    name: str
    repository: str
    tag: str


def catch2_pin(repository: Path) -> PinnedCheckout | None:
    """Return the Catch2 checkout the build fetches, as pinned in tests/CMakeLists.txt."""
    try:
        text = (repository / "tests" / "CMakeLists.txt").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"FetchContent_Declare\(\s*(Catch2)\s+GIT_REPOSITORY\s+(\S+)\s+"
                      r"GIT_TAG\s+([\w.-]+)\s*\)", text)
    return None if match is None else PinnedCheckout(*match.groups())


def catch2_cache_directory(config: WorkflowConfig, pin: PinnedCheckout) -> Path:
    return config.output_directory / "build" / f"{pin.name.lower()}-{pin.tag}"


def catch2_candidates(config: WorkflowConfig) -> list[Path]:
    """Return Catch2 checkouts already on disk, the run's private builds before the repository's."""
    builds = [data_build_directory(config)]
    for iteration in config.iterations:
        builds.append(iteration.candidate_directory / "build")
        builds.extend(sorted(iteration.gate_directory.glob("builds/*/*")))
    for comparison in config.comparisons:
        builds.extend(sorted(comparison.root.glob("builds/*/*")))
    builds.extend(sorted((config.repository / "build").glob("*")))
    return list(dict.fromkeys(build / "_deps" / "catch2-src" for build in builds))


def verified_checkout(path: Path, pin: PinnedCheckout) -> bool:
    """Return whether path is a clean Git checkout of the pinned tag from the pinned remote."""
    if path.is_symlink() or not (path / ".git").is_dir():
        return False
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith("GIT_")}
    environment.update(GIT_DIR=str(path / ".git"), GIT_WORK_TREE=str(path),
                       GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                       GIT_OPTIONAL_LOCKS="0")

    def git(*arguments: str) -> str | None:
        try:
            result = subprocess.run(["git", *arguments], cwd=path, env=environment,
                                    capture_output=True, text=True, check=False)
        except OSError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    head = git("rev-parse", "--verify", "HEAD^{commit}")
    return (head is not None and
            head == git("rev-parse", "--verify", f"refs/tags/{pin.tag}^{{commit}}") and
            git("config", "--get", "remote.origin.url") == pin.repository and
            git("status", "--porcelain", "--untracked-files=all") == "")


def _evaluator_arguments(dataset: ManagedDatasetConfig) -> list[str]:
    teacher = dataset.teacher
    if teacher is None:
        return []
    if teacher.net is None and teacher.iteration is not None:
        net = f"<net of {teacher.name} once iteration {teacher.iteration}'s gate accepts it>"
    else:
        net = str(model_net(teacher))
    return ["--evaluator", "pattern-gain", "--net", net]


def candidate_build_id(candidate: dict[str, Any]) -> str:
    """Return the model ID of one authenticated candidate: its net's, or its header's before nets."""
    artifacts = candidate.get("artifacts", {})
    identity = artifacts.get("net") if "net" in artifacts else artifacts.get("model_header")
    _require(isinstance(identity, dict) and
             isinstance(identity.get("sha256"), str) and
             re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is not None,
             "candidate report has no valid net or model-header digest")
    return f"candidate-sha256-{identity['sha256']}"


def candidate_net_name(config: WorkflowConfig, iteration: IterationConfig) -> str:
    """Return the name of the net an iteration trains: <run>/<iteration>."""
    return f"{config.name}/{iteration.name}"


def model_net_name(config: WorkflowConfig, model: ModelReference) -> str:
    """Return a model's net name: its file's, or <run>/<iteration> for one this recipe trains."""
    if model.net_name is not None:
        return model.net_name
    _require(model.iteration is not None, f"model {model.name} has no net")
    return f"{config.name}/{model.iteration}"


def model_net(model: ModelReference) -> Path:
    """Return the intact net file of a model, or explain why it cannot be loaded."""
    _require(model.net is not None and model.net_sha256 is not None,
             f"model {model.name} has no net file; archive it with promote-gate --apply first")
    _require(model.net.is_file() and sha256_file(model.net) == model.net_sha256,
             f"net of model {model.name} is missing or changed: {model.net}")
    return model.net


def private_build_directory(config: WorkflowConfig, root: Path, model_id: str) -> Path:
    """Return a model's private build under a match root, a layout the build-ID reader names."""
    return root / "builds" / model_id / config.build_preset


def gate_build_directory(config: WorkflowConfig, iteration: IterationConfig,
                         candidate: dict[str, Any]) -> Path:
    return private_build_directory(
        config, iteration.gate_directory, candidate_build_id(candidate))


def gate_base_model_id(config: WorkflowConfig, iteration: IterationConfig) -> str | None:
    """Return the model ID a gate's base is built from, or None for an existing build base."""
    gate = iteration.engine_gate
    _require(gate is not None, "iteration has no engine gate")
    if gate.base_iteration is None:
        return gate.base_model
    model = config.trained_model(gate.base_iteration)
    _require(model.model_id is not None,
             f"engine-gate base {model.name} has no accepted candidate from iteration "
             f"{gate.base_iteration}")
    return model.model_id


def gate_base_build_directory(config: WorkflowConfig, iteration: IterationConfig) -> Path:
    """Return the private build of a gate's base model, in the layout the build-ID reader names."""
    model_id = gate_base_model_id(config, iteration)
    _require(model_id is not None, "gate has no base model")
    return private_build_directory(config, iteration.gate_directory, model_id)


def gate_base_binary(config: WorkflowConfig, iteration: IterationConfig) -> Path:
    """Return the base engine a gate plays; a base model's build exists only once the gate ran."""
    gate = iteration.engine_gate
    _require(gate is not None, "iteration has no engine gate")
    if gate.base is None:
        return (gate_base_build_directory(config, iteration) / "engines" /
                gate.base_engine).resolve()
    return resolve_eval_binary(config, gate.base, gate.base_engine)


def resolve_eval_binary(config: WorkflowConfig, value: str, engine: str) -> Path:
    """Resolve an evaluator input exactly like the native evaluation CLI."""
    raw = Path(value)
    input_path = raw if raw.is_absolute() else config.repository / raw
    if input_path.is_file():
        return input_path.resolve()
    if input_path.is_dir():
        for candidate in (
                input_path / "engines" / engine,
                input_path / config.build_preset / "engines" / engine):
            if candidate.is_file():
                return candidate.resolve()
    by_commit = (config.repository / "build" / "by-commit" / raw /
                 config.build_preset / "engines" / engine)
    if by_commit.is_file():
        return by_commit.resolve()
    raise AdapterError(
        f"could not resolve engine {engine!r} from evaluator input {value!r}")


# Position sources are generated in chunks of this many trajectories so a pause loses at most one
# chunk. Assembly is byte-identical for any chunk size, so this is not part of any fingerprint.
SOURCE_CHUNK_TRAJECTORIES = 2000


def source_chunk_ranges(dataset: ManagedDatasetConfig) -> list[tuple[int, int]]:
    total = dataset.source.trajectories
    return [(begin, min(begin + SOURCE_CHUNK_TRAJECTORIES, total))
            for begin in range(0, total, SOURCE_CHUNK_TRAJECTORIES)]


def source_chunks_directory(dataset: ManagedDatasetConfig) -> Path:
    return dataset.root / "source-chunks"


def source_chunk_path(dataset: ManagedDatasetConfig, begin: int, end: int) -> Path:
    return source_chunks_directory(dataset) / f"chunk-{begin:09d}-{end:09d}"


def source_chunk_commit(path: Path) -> str:
    """Return the build commit recorded by one completed source chunk."""
    try:
        with (path / "chunk.jsonl").open(encoding="utf-8") as source:
            header = json.loads(source.readline())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterError(f"could not read source chunk {path}: {error}") from error
    commit = header.get("git_commit") if isinstance(header, dict) else None
    _require(header.get("type") == "poe2-position-source-chunk" and isinstance(commit, str) and
             re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
             f"source chunk has invalid build provenance: {path}")
    return commit


def source_chunk_command(config: WorkflowConfig, dataset: ManagedDatasetConfig,
                         begin: int, end: int) -> list[str]:
    return [
        str(data_binary(config)), "source-chunk",
        "--output-dir", str(source_chunk_path(dataset, begin, end)),
        "--trajectory-begin", str(begin), "--trajectory-end", str(end),
        *_source_arguments(dataset),
    ]


def source_assemble_command(config: WorkflowConfig, dataset: ManagedDatasetConfig) -> list[str]:
    return [
        str(data_binary(config)), "source-assemble",
        "--output-dir", str(dataset.source_directory),
        "--chunks", str(source_chunks_directory(dataset)),
        *_source_arguments(dataset),
    ]


def _source_arguments(dataset: ManagedDatasetConfig) -> list[str]:
    source = dataset.source
    return [
        "--corpus-id", source.corpus_id, "--seed", str(source.seed),
        "--trajectories", str(source.trajectories),
        "--samples-per-trajectory", str(source.samples_per_trajectory),
        "--shards", str(source.shards), "--workers", str(source.workers),
        "--search-nodes", str(source.search_nodes),
        "--search-hash-mb", str(source.search_hash_mb),
        "--noise-percent", str(source.noise_percent),
        "--random-weight", str(source.random_weight),
        "--greedy-weight", str(source.greedy_weight),
        "--opponent-weight", str(source.opponent_weight),
        "--search-weight", str(source.search_weight),
        "--progress-every", str(source.progress_every),
        *_evaluator_arguments(dataset),
    ]


def source_audit_command(python: str, config: WorkflowConfig,
                         dataset: ManagedDatasetConfig) -> list[str]:
    return [
        python, str(config.repository / "tools" / "inspect_position_source.py"),
        str(dataset.source_directory),
    ]


def label_shard_path(dataset: ManagedDatasetConfig, index: int) -> Path:
    return dataset.labels_directory / "shards" / f"shard-{index:03d}"


def source_shard_path(dataset: ManagedDatasetConfig, index: int) -> Path:
    manifest = json.loads((dataset.source_directory / "manifest.json").read_bytes())
    shards = manifest.get("shards")
    _require(isinstance(shards, list) and len(shards) == dataset.source.shards,
             "source manifest has the wrong shard list")
    entry = shards[index]
    _require(isinstance(entry, dict) and entry.get("index") == index and
             isinstance(entry.get("name"), str),
             f"source manifest shard {index} is malformed")
    return dataset.source_directory / "shards" / entry["name"]


def label_command(config: WorkflowConfig, dataset: ManagedDatasetConfig,
                  index: int) -> list[str]:
    labels = dataset.labels
    command = [
        str(data_binary(config)), "labels", "--input", str(dataset.source_directory),
        "--source-shard", str(index), "--output-dir", str(label_shard_path(dataset, index)),
        "--mode", labels.mode, "--nodes", str(labels.nodes),
        "--hash-mb", str(labels.hash_mb), "--workers", str(labels.workers),
        "--progress-every", str(labels.progress_every),
        *_evaluator_arguments(dataset),
    ]
    if labels.require_all:
        command.append("--require-all")
    return command


def label_audit_command(python: str, config: WorkflowConfig,
                        dataset: ManagedDatasetConfig, index: int) -> list[str]:
    return [
        python, str(config.repository / "tools" / "inspect_minimax_labels.py"),
        str(label_shard_path(dataset, index)), "--source",
        str(source_shard_path(dataset, index)),
    ]


def label_preflight_command(python: str, config: WorkflowConfig,
                            dataset: ManagedDatasetConfig) -> list[str]:
    return [
        python, str(config.repository / "tools" / "preflight_minimax_corpus.py"),
        str(dataset.labels_directory), "--source", str(dataset.source_directory), "--json",
    ]


def feature_command(config: WorkflowConfig, dataset: ManagedDatasetConfig) -> list[str]:
    return [
        str(data_binary(config)), "features",
        "--source", str(dataset.source_directory),
        "--labels", str(dataset.labels_directory), "--output-dir",
        str(dataset.feature_directory),
    ]


def feature_audit_command(python: str, config: WorkflowConfig,
                          dataset: ManagedDatasetConfig) -> list[str]:
    return [
        python, str(config.repository / "tools" / "inspect_minimax_features.py"),
        str(dataset.feature_directory), "--labels", str(dataset.labels_directory),
        "--gain-samples", str(dataset.features.gain_samples),
    ]


def training_command(python: str, dataset: DatasetConfig, iteration: IterationConfig,
                     extras: Sequence[DatasetConfig] = ()) -> list[str]:
    feature_directory = dataset.feature_directory
    training = iteration.training
    if isinstance(training, BaselineTrainingConfig):
        return [
            python, "-m", "poe2_training.baseline", "--dataset", str(feature_directory),
            "--output-dir", str(iteration.training_directory), "--device", training.device,
            "--ridge-lambdas", ",".join(str(value) for value in training.ridge_lambdas),
            "--close-margin", str(training.close_margin), "--seed", str(training.seed),
        ]
    command = [
        python, "-m", "poe2_training.pattern_experiment", "--dataset",
        str(feature_directory), "--output-dir", str(iteration.training_directory),
        "--device", training.device, "--seed", str(training.seed), "--suite", training.suite,
    ]
    for extra, item in zip(extras, training.extra_datasets, strict=True):
        command.extend(("--extra-dataset", str(extra.feature_directory), repr(item.weight)))
    if training.max_records is not None:
        command.extend(("--max-records", str(training.max_records)))
    if training.piece_records is not None:
        command.extend(("--piece-records", str(training.piece_records)))
    if training.method != "dense":
        command.extend(("--method", training.method))
    return command


def evaluation_command(python: str, config: WorkflowConfig,
                       iteration: IterationConfig) -> list[str]:
    _require(iteration.sealed_evaluation is not None,
             "iteration has no sealed evaluation")
    dataset = config.dataset(iteration.dataset)
    command = [
        python, "-m", "poe2_training.pattern_evaluation", "--dataset",
        str(dataset.feature_directory), "--experiment", str(iteration.training_directory),
        "--output-dir", str(iteration.evaluation_directory),
    ]
    for name in iteration.sealed_evaluation.exclude_datasets:
        command.extend(("--exclude-dataset", str(config.dataset(name).feature_directory)))
    return command


CANDIDATE_NET = f"net{FILE_SUFFIX}"


def candidate_net_path(iteration: IterationConfig) -> Path:
    return iteration.candidate_directory / CANDIDATE_NET


def export_command(python: str, config: WorkflowConfig, iteration: IterationConfig, *,
                   teacher: str, created: str, parent: str | None) -> list[str]:
    command = [
        python, "-m", "poe2_training.net_export", "--experiment",
        str(iteration.training_directory), "--output", str(candidate_net_path(iteration)),
        "--name", candidate_net_name(config, iteration), "--teacher", teacher,
        "--created", created,
    ]
    if parent is not None:
        command.extend(("--parent", parent))
    return command


def parity_command(python: str, dataset: DatasetConfig, iteration: IterationConfig,
                   binary: Path) -> list[str]:
    candidate = iteration.candidate_validation
    _require(candidate is not None, "iteration has no candidate validation")
    return [
        python, "-m", "poe2_training.pattern_integration", "--dataset",
        str(dataset.feature_directory), "--experiment", str(iteration.training_directory),
        "--inference-binary", str(binary), "--net", str(candidate_net_path(iteration)),
        "--samples", str(candidate.samples),
        "--symmetry-samples", str(candidate.symmetry_samples),
        "--benchmark-iterations", str(candidate.benchmark_iterations),
    ]


def search_benchmark_command(python: str, config: WorkflowConfig,
                             iteration: IterationConfig, engine: Path) -> list[str]:
    candidate = iteration.candidate_validation
    _require(candidate is not None, "iteration has no candidate validation")
    return [
        python, str(config.repository / "tools" / "benchmark_minimax_search.py"),
        "--engine", str(engine), "--net", str(candidate_net_path(iteration)),
        "--opening-book", str(candidate.opening_book),
        "--positions", str(candidate.search_positions),
        "--movetime-ms", str(candidate.search_movetime_ms), "--seed", str(iteration.training.seed),
    ]


def match_command(config: WorkflowConfig, settings: MatchConfig, *, runner: Path,
                  new_build: Path, base: str, kind: str, root: Path,
                  new_net: Path | None = None, base_net: Path | None = None) -> list[str]:
    """Return the evaluator command for one match, recorded under root/runs without a ledger.

    The evaluator runs in the repository and records paths as given, so they are given
    relative to it and its manifest and ledger row stay valid if the repository moves.
    """
    command = [
        str(runner), "eval", "--new-build", config.recorded_path(new_build), "--base", base,
        "--new-engine", settings.new_engine, "--base-engine", settings.base_engine,
        "--preset", config.build_preset, "--kind", kind,
        "--run-root", config.recorded_path(root / "runs"), "--no-ledger",
        "--opening-book", config.recorded_path(settings.opening_book), "--shuffle-openings",
        "--games", str(settings.games), "--workers", str(settings.workers),
        "--timeout-ms", str(settings.timeout_ms),
        "--go-movetime-ms", str(settings.go_movetime_ms),
        "--sequential-null", str(settings.sequential_null),
        "--sequential-alt", str(settings.sequential_alt),
        "--sequential-alpha", str(settings.sequential_alpha),
        "--sequential-beta", str(settings.sequential_beta),
    ]
    if settings.new_engine_args:
        command.extend(("--new-engine-args", settings.new_engine_args))
    if settings.base_engine_args:
        command.extend(("--base-engine-args", settings.base_engine_args))
    for flag, net in (("--new-net", new_net), ("--base-net", base_net)):
        if net is not None:
            command.extend((flag, config.recorded_path(net)))
    if settings.opening_seed is not None:
        command.extend(("--opening-seed", str(settings.opening_seed)))
    if settings.sequential_stop:
        command.append("--sequential-stop")
    return command


def engine_gate_command(config: WorkflowConfig, iteration: IterationConfig, build: Path,
                        runner: Path, *, new_net: Path,
                        base_model: ModelReference | None) -> list[str]:
    """Return the gate's match: the candidate's net against its base, on one engine build.

    A base model is its net on the same build; a configured base build plays its own net.
    """
    gate = iteration.engine_gate
    _require(gate is not None, "iteration has no engine gate")
    if base_model is None:
        return match_command(config, gate, runner=runner, new_build=build, base=gate.base or "",
                             kind=GATE_KIND, root=iteration.gate_directory, new_net=new_net)
    return match_command(config, gate, runner=runner, new_build=build,
                         base=config.recorded_path(build), kind=GATE_KIND,
                         root=iteration.gate_directory, new_net=new_net,
                         base_net=model_net(base_model))


def comparison_command(config: WorkflowConfig, comparison: ComparisonConfig, build: Path,
                       new: ModelReference, base: ModelReference) -> list[str]:
    return match_command(
        config, comparison, runner=build / "runner" / "poe2_runner", new_build=build,
        base=config.recorded_path(build), kind=COMPARISON_KIND, root=comparison.root,
        new_net=model_net(new), base_net=model_net(base))


def _manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterError(f"could not read {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def recorded_teacher(search: dict[str, Any], label: str) -> str | None:
    """Return the net a manifest's search used, or None for two-ply closure.

    Data made with a net file records its file hash as net_sha256, named here
    net-sha256-<hash>. Data made before nets were files records the model header compiled into
    the data tool, which is the model's ID.
    """
    evaluator = search.get("evaluator", "two-ply-closure")
    header = search.get("model_header_sha256")
    net = search.get("net_sha256")
    if evaluator in {"two-ply-closure", "b"}:
        _require(header is None and net is None,
                 f"{label} records a model without a learned evaluator")
        return None
    digest = net if header is None else header
    _require(evaluator == "pattern-gain" and (header is None) != (net is None) and
             isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
             f"{label} search evaluator is unsupported")
    return f"candidate-sha256-{digest}" if net is None else f"net-sha256-{digest}"


def _teacher_matches(recorded: str | None, dataset: ManagedDatasetConfig) -> bool:
    teacher = dataset.teacher
    if teacher is None or recorded is None:
        return teacher is None and recorded is None
    return recorded in {_expected_teacher(dataset), f"net-sha256-{teacher.net_sha256}"}


def _expected_teacher(dataset: ManagedDatasetConfig) -> str | None:
    teacher = dataset.teacher
    if teacher is None:
        return None
    _require(teacher.model_id is not None,
             f"dataset {dataset.name} has teacher {teacher.name}, which iteration "
             f"{teacher.iteration}'s engine gate has not accepted")
    return teacher.model_id


def authenticate_source(dataset: ManagedDatasetConfig) -> dict[str, Any]:
    manifest_path = dataset.source_directory / "manifest.json"
    manifest = _manifest(manifest_path)
    generator = manifest.get("generator")
    results = manifest.get("results")
    corpus = manifest.get("corpus")
    _require(manifest.get("schema") == "poe2-position-source" and
             isinstance(generator, dict) and isinstance(results, dict) and
             isinstance(corpus, dict), "position source manifest schema is unsupported")
    expected = dataset.source
    _require(corpus.get("id") == expected.corpus_id and
             generator.get("seed") == f"{expected.seed:016x}" and
             generator.get("trajectory_count") == expected.trajectories and
             generator.get("samples_per_trajectory") == expected.samples_per_trajectory and
             generator.get("shard_count") == expected.shards and
             generator.get("workers_requested") == expected.workers and
             generator.get("search_nodes") == expected.search_nodes and
             generator.get("search_hash_bytes") == expected.search_hash_mb * 1024 * 1024 and
             generator.get("noise_percent") == expected.noise_percent,
             "position source manifest differs from configured source settings")
    _require(generator.get("policy_weights") == {
        "random": expected.random_weight, "immediate_gain": expected.greedy_weight,
        "opponent_aware": expected.opponent_weight, "noisy_search": expected.search_weight,
    }, "position source policy mixture differs from configuration")
    _require(_teacher_matches(recorded_teacher(generator, "position source"), dataset),
             "position source teacher differs from configuration")
    build = manifest.get("build")
    _require(isinstance(build, dict) and build.get("git_dirty") is False,
             "position source was produced from a dirty build")
    stats = {
        "manifest_sha256": sha256_file(manifest_path), "corpus_id": corpus.get("id"),
        "records": results.get("records"), "trajectories": expected.trajectories,
        "shards": expected.shards, "duplicates": results.get("duplicate_positions"),
        "split_counts": results.get("split_counts"), "policy_counts": results.get("policy_counts"),
        "workers_requested": generator.get("workers_requested"),
        "workers_used": generator.get("workers_used"),
    }
    # Stage statistics feed later stages' input fingerprints, so teacher-less runs keep their shape.
    if dataset.teacher is not None:
        stats["teacher"] = _expected_teacher(dataset)
    return stats


def authenticate_label_shard(dataset: ManagedDatasetConfig, index: int) -> dict[str, Any]:
    path = label_shard_path(dataset, index)
    manifest_path = path / "manifest.json"
    manifest = _manifest(manifest_path)
    corpus = manifest.get("corpus")
    search = manifest.get("search")
    results = manifest.get("results")
    build = manifest.get("build")
    _require(manifest.get("schema") == "poe2-minimax-labels" and
             isinstance(corpus, dict) and isinstance(search, dict) and
             isinstance(results, dict) and isinstance(build, dict),
             f"label shard {index} manifest schema is unsupported")
    expected = dataset.labels
    _require(corpus.get("id") == dataset.source.corpus_id and
             corpus.get("shard_index") == index and corpus.get("shard_count") == dataset.source.shards,
             f"label shard {index} corpus provenance differs")
    _require(search.get("mode") == expected.mode and search.get("node_limit") == expected.nodes and
             search.get("hash_bytes_requested") == expected.hash_mb * 1024 * 1024 and
             search.get("workers_requested") == expected.workers and
             search.get("require_all") == expected.require_all,
             f"label shard {index} search settings differ from configuration")
    # Schema-v2 manifests written before the evaluator rename used the alias
    # "b".  The authoritative producer now writes "two-ply-closure", and the
    # native feature reader intentionally accepts both spellings.
    _require(search.get("target_selection") == "deepest_terminal_parity" and
             search.get("symmetry") is True and
             search.get("two_ply_closure") is True,
             f"label shard {index} search semantics are unsupported")
    _require(_teacher_matches(recorded_teacher(search, f"label shard {index}"), dataset),
             f"label shard {index} teacher differs from configuration")
    _require(build.get("git_dirty") is False and
             isinstance(build.get("git_commit"), str) and
             re.fullmatch(r"[0-9a-f]{40}", build["git_commit"]) is not None,
             f"label shard {index} has invalid build provenance")
    return {
        "index": index, "manifest_sha256": sha256_file(manifest_path),
        "binary_sha256": str(manifest.get("binary_digest", "")).removeprefix("sha256:"),
        "records": results.get("records"), "terminal_records": results.get("terminal_records"),
        "parity_backoffs": results.get("parity_backoffs"),
        "previous_records": results.get("previous_records"), "unsolved": results.get("unsolved"),
        "workers_requested": search.get("workers_requested"),
        "workers_used": search.get("workers_used"),
        "git_commit": build["git_commit"],
        **({} if dataset.teacher is None else {"teacher": _expected_teacher(dataset)}),
    }


def aggregate_label_stats(shards: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "shards": len(shards),
        "records": sum(int(item.get("records") or 0) for item in shards),
        "terminal_records": sum(int(item.get("terminal_records") or 0) for item in shards),
        "parity_backoffs": sum(int(item.get("parity_backoffs") or 0) for item in shards),
        "previous_records": sum(int(item.get("previous_records") or 0) for item in shards),
        "unsolved": sum(int(item.get("unsolved") or 0) for item in shards),
        "manifest_set_sha256": hashlib.sha256("".join(
            item["manifest_sha256"] for item in shards).encode()).hexdigest(),
    }


def authenticate_features(dataset: DatasetConfig, *, require_clean: bool = True) -> dict[str, Any]:
    artifact: FeatureArtifact = open_feature_artifact(
        dataset.feature_directory, verify_digest=True, require_clean=require_clean)
    label_search = artifact.manifest.get("inputs", {}).get("label_search")
    _require(isinstance(label_search, dict), "feature manifest has no label search record")
    teacher = recorded_teacher(label_search, "feature artifact")
    if isinstance(dataset, ManagedDatasetConfig):
        _require(_teacher_matches(teacher, dataset),
                 "feature artifact teacher differs from configuration")
        teacher = _expected_teacher(dataset)
    return {
        "binary_sha256": artifact.binary_digest,
        "manifest_sha256": artifact.manifest_digest,
        "records": artifact.record_count, "source_records": artifact.source_record_count,
        "duplicates_removed": artifact.duplicates_removed, "shards": artifact.shard_count,
        "split_counts": {"train": artifact.split_counts[0],
                         "validation": artifact.split_counts[1], "test": artifact.split_counts[2]},
        "corpus_id": artifact.manifest["corpus"]["id"],
        **({} if teacher is None else {"teacher": teacher}),
    }


def _authenticate_training_data(report: dict[str, Any], training: PatternTrainingConfig,
                                features: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check that a report trained on exactly the configured datasets, weights, and options."""
    data = report.get("training_data")
    _require(isinstance(data, dict) and data.get("max_records") == training.max_records and
             data.get("piece_records") == training.piece_records and
             isinstance(data.get("datasets"), list),
             "pattern report training data options differ from configuration")
    weights = [1.0, *(item.weight for item in training.extra_datasets)]
    datasets = data["datasets"]
    _require(len(datasets) == len(features) and all(
        isinstance(entry, dict) and
        entry.get("feature_binary_sha256") == feature["binary_sha256"] and
        entry.get("feature_manifest_sha256") == feature["manifest_sha256"] and
        entry.get("weight") == weight
        for entry, feature, weight in zip(datasets, features, weights)),
        "pattern report trained on different datasets or weights than configured")
    return datasets


def authenticate_training(dataset: DatasetConfig, iteration: IterationConfig,
                          extras: Sequence[DatasetConfig] = ()) -> dict[str, Any]:
    feature = authenticate_features(dataset)
    training = iteration.training
    if isinstance(training, BaselineTrainingConfig):
        report = open_baseline_report(iteration.training_directory)
        selected = min(report["models"],
                       key=lambda model: model["validation"]["overall"]["mae"])
        _require(report.get("training", {}).get("seed") == training.seed and
                 report.get("training", {}).get("close_margin") == training.close_margin and
                 tuple(report.get("training", {}).get("ridge_lambdas", ())) ==
                 training.ridge_lambdas,
                 "baseline report settings differ from configuration")
    else:
        report = open_pattern_report(iteration.training_directory)
        # Reports have passed through JSON, so tuple-valued model knots are arrays.
        expected_configs = json.loads(json.dumps(
            [asdict(item) for item in pattern_suite(training.suite)]))
        _require(report.get("training", {}).get("seed") == training.seed and
                 report.get("training", {}).get("configs") == expected_configs and
                 report["training"].get("method", "dense") == training.method,
                 "pattern report settings differ from configuration")
        selected_name = report.get("selection", {}).get("best_model")
        matches = [item for item in report.get("models", []) if item.get("name") == selected_name]
        _require(len(matches) == 1, "pattern report selected model is missing or duplicated")
        selected = matches[0]
    attachments = {
        metadata["path"] for metadata in report.get("attachments", {}).values()
    }
    _require_report_layout(iteration.training_directory, "training report", attachments)
    metadata = report.get("input", {})
    _require(metadata.get("feature_binary_sha256") == feature["binary_sha256"] and
             metadata.get("feature_manifest_sha256") == feature["manifest_sha256"],
             "training report input differs from the configured feature artifact")
    _require(report.get("provenance", {}).get("git", {}).get("dirty") is False,
             "training report was produced from a dirty worktree")
    runtime_device = report.get("provenance", {}).get("runtime", {}).get("device_type")
    _require(training.device == "auto" or runtime_device == training.device,
             "training report device differs from configuration")
    quantization = selected.get("quantization")
    quantized_validation = (
        quantization.get("validation", {}).get("overall")
        if isinstance(quantization, dict) else None)
    if iteration.candidate_validation is not None:
        _require(isinstance(quantized_validation, dict),
                 "candidate training report has no quantized validation metrics")
        _require(quantization.get("fractional_bits") ==
                 FROZEN_PATTERN_GAIN_FRACTIONAL_BITS and
                 quantization.get("scale") ==
                 1 << FROZEN_PATTERN_GAIN_FRACTIONAL_BITS,
                 "candidate training report does not use deployment scale-32 quantization")
    result = {
        "report_sha256": sha256_file(iteration.training_directory / "report.json"),
        "kind": training.type, "selected_model": selected.get("name"),
        "selected_validation": selected.get("validation", {}).get("overall"),
        "selected_quantized_validation": quantized_validation,
        "input": metadata, "attachments": report.get("attachments", {}),
    }
    if isinstance(training, PatternTrainingConfig) and training.uses_training_data_options:
        result["training_datasets"] = _authenticate_training_data(
            report, training, [feature, *(authenticate_features(item) for item in extras)])
    else:
        _require("training_data" not in report,
                 "pattern report used training data options that are not configured")
    return result


def authenticate_evaluation(config: WorkflowConfig,
                            iteration: IterationConfig) -> dict[str, Any]:
    _require(iteration.sealed_evaluation is not None, "iteration has no sealed evaluation")
    _require_report_layout(iteration.evaluation_directory, "sealed evaluation")
    report = open_pattern_evaluation(iteration.evaluation_directory)
    training_digest = sha256_file(iteration.training_directory / "report.json")
    _require(report.get("input", {}).get("experiment_report_sha256") == training_digest,
             "sealed evaluation references a different training report")
    target = authenticate_features(config.dataset(iteration.dataset))
    input_metadata = report.get("input", {})
    _require(input_metadata.get("feature_binary_sha256") == target["binary_sha256"] and
             input_metadata.get("feature_manifest_sha256") == target["manifest_sha256"],
             "sealed evaluation references a different target feature artifact")
    expected = [authenticate_features(config.dataset(name))
                for name in iteration.sealed_evaluation.exclude_datasets]
    actual = input_metadata.get("excluded_datasets", [])
    _require([(item.get("feature_binary_sha256"), item.get("feature_manifest_sha256"))
              for item in actual] ==
             [(item["binary_sha256"], item["manifest_sha256"]) for item in expected],
             "sealed evaluation exclusions differ from configuration")
    evaluation = report.get("evaluation", {})
    _require(report.get("provenance", {}).get("git", {}).get("dirty") is False,
             "sealed evaluation was produced from a dirty worktree")
    return {
        "report_sha256": sha256_file(iteration.evaluation_directory / "report.json"),
        "test_records": report.get("input", {}).get("test_records"),
        "overlap_excluded": report.get("input", {}).get(
            "development_overlap_records_excluded"),
        "model": evaluation.get("model"),
        "metrics": evaluation.get("model_metrics", {}).get("overall"),
        "quantized_metrics": (evaluation.get("quantized_model_metrics") or {}).get("overall"),
    }


def _authenticate_candidate_header(iteration: IterationConfig,
                                   report: dict[str, Any]) -> dict[str, Any]:
    """Authenticate the private build of a candidate validated before nets were files."""
    artifacts = report.get("artifacts")
    _require(isinstance(artifacts, dict), "candidate artifact map is missing")
    expected_artifacts = {
        "model_header": "frozen_pattern_gain_model.hpp",
        "inference_binary": "build/engines/minimax/poe2_minimax_infer",
        "engine_binary": "build/engines/minimax/poe2_minimax",
        "runner_binary": "build/runner/poe2_runner",
    }
    _require(set(artifacts) == set(expected_artifacts),
             "candidate artifact map has missing or unexpected entries")
    for name, metadata in artifacts.items():
        _require(isinstance(metadata, dict) and isinstance(metadata.get("path"), str) and
                 isinstance(metadata.get("sha256"), str),
                 f"candidate artifact {name} metadata is malformed")
        _require(metadata["path"] == expected_artifacts[name],
                 f"candidate artifact {name} has an unexpected path")
        path = (iteration.candidate_directory / metadata["path"]).resolve()
        _require(iteration.candidate_directory in path.parents and path.is_file(),
                 f"candidate artifact is missing or escapes its root: {name}")
        _require(sha256_file(path) == metadata["sha256"],
                 f"candidate artifact digest differs: {name}")
    return artifacts


def _authenticate_candidate_net(config: WorkflowConfig, iteration: IterationConfig,
                                report: dict[str, Any],
                                feature: dict[str, Any]) -> dict[str, Any]:
    """Authenticate a candidate's net file against its report and training provenance.

    The tools that checked it ran from the run's shared build, which later commits rebuild, so
    the report records their digests as provenance rather than as artifacts.
    """
    artifacts = report.get("artifacts")
    _require(isinstance(artifacts, dict) and set(artifacts) == {"net"},
             "candidate artifact map must hold exactly its net")
    metadata = artifacts["net"]
    _require(isinstance(metadata, dict) and metadata.get("path") == CANDIDATE_NET and
             isinstance(metadata.get("sha256"), str),
             "candidate net metadata is malformed")
    try:
        net = read_net(candidate_net_path(iteration))
    except NetFileError as error:
        raise AdapterError(f"candidate net is unusable: {error}") from error
    _require(net.sha256 == metadata["sha256"], "candidate net digest differs from its report")
    _require(metadata == {"path": CANDIDATE_NET, "sha256": net.sha256, "id": net.net_id,
                          "name": net.name, "design": net.design},
             "candidate net metadata differs from the net file")
    _require(net.name == candidate_net_name(config, iteration) and
             "header_sha256" not in net.metadata,
             f"candidate net is not named {candidate_net_name(config, iteration)}")
    _require(net.metadata.get("training_report_sha256") == report.get("training_report_sha256") and
             net.metadata.get("feature_binary_sha256") == feature["binary_sha256"],
             "candidate net was exported from a different training report or corpus")
    tools = report.get("tools")
    _require(isinstance(tools, dict) and set(tools) == {"inference_binary", "engine_binary"} and
             all(isinstance(item, dict) and isinstance(item.get("path"), str) and
                 isinstance(item.get("sha256"), str) for item in tools.values()),
             "candidate tool provenance is malformed")
    return artifacts


def authenticate_candidate(config: WorkflowConfig,
                           iteration: IterationConfig) -> dict[str, Any]:
    """Authenticate a candidate: a net file (schema 2), or a C++ header from before nets (1)."""
    candidate = iteration.candidate_validation
    _require(candidate is not None, "iteration has no candidate validation")
    report, report_digest = open_json_report(
        iteration.candidate_directory, schema=CANDIDATE_SCHEMA,
        schema_version=(1, CANDIDATE_SCHEMA_VERSION), error_type=AdapterError)
    with_net = report.get("schema_version") == CANDIDATE_SCHEMA_VERSION
    names = {path.name for path in iteration.candidate_directory.iterdir()}
    expected_names = ({"COMPLETE", "report.json", CANDIDATE_NET} if with_net else
                      {"COMPLETE", "report.json", "frozen_pattern_gain_model.hpp", "build"})
    _require(names == expected_names,
             f"candidate has unexpected output files: {iteration.candidate_directory}")
    _require(report.get("promotable") == candidate.promotable,
             "candidate promotable setting differs from configuration")
    expected_settings = {
        "promotable": candidate.promotable, "samples": candidate.samples,
        "symmetry_samples": candidate.symmetry_samples,
        "benchmark_iterations": candidate.benchmark_iterations,
        "search_positions": candidate.search_positions,
        "search_movetime_ms": candidate.search_movetime_ms,
    }
    settings = report.get("settings")
    _require(isinstance(settings, dict) and
             {key: value for key, value in settings.items() if key != "opening_book"} ==
             expected_settings and
             names_path(config, settings.get("opening_book"), candidate.opening_book),
             "candidate validation settings differ from configuration")
    _require(candidate.opening_book.is_file(),
             f"candidate opening book is missing: {candidate.opening_book}")
    _require(report.get("opening_book_sha256") == sha256_file(candidate.opening_book),
             "candidate opening book changed after validation")
    _require(report.get("training_report_sha256") ==
             sha256_file(iteration.training_directory / "report.json"),
             "candidate references a different training report")
    expected_evaluation_digest = (
        sha256_file(iteration.evaluation_directory / "report.json")
        if iteration.sealed_evaluation is not None else None)
    _require(report.get("sealed_evaluation_report_sha256") == expected_evaluation_digest,
             "candidate references a different sealed evaluation")
    feature = authenticate_features(config.dataset(iteration.dataset))
    _require(report.get("feature_binary_sha256") == feature["binary_sha256"],
             "candidate references a different feature artifact")
    _require(isinstance(report.get("git_commit"), str) and
             re.fullmatch(r"[0-9a-f]{40}", report["git_commit"]) is not None,
             "candidate Git commit is malformed")
    model = report.get("model")
    float_metrics = model.get("float") if isinstance(model, dict) else None
    quantized_metrics = model.get("quantized") if isinstance(model, dict) else None
    _require(isinstance(model, dict) and isinstance(model.get("name"), str) and
             bool(model["name"]) and isinstance(float_metrics, dict) and
             isinstance(float_metrics.get("validation"), dict) and
             isinstance(quantized_metrics, dict) and
             isinstance(quantized_metrics.get("validation"), dict),
             "candidate selected-model metadata is malformed")
    if iteration.sealed_evaluation is not None:
        _require(isinstance(float_metrics.get("sealed_test"), dict) and
                 isinstance(quantized_metrics.get("sealed_test"), dict),
                 "candidate float or quantized sealed-test metrics are missing")
    if with_net:
        artifacts = _authenticate_candidate_net(config, iteration, report, feature)
    else:
        artifacts = _authenticate_candidate_header(iteration, report)
    parity = report.get("parity")
    _require(isinstance(parity, dict) and parity.get("passed") is True,
             "candidate did not pass exact Python/C++ parity")
    _require(isinstance(parity.get("base_positions"), int) and
             parity["base_positions"] > 0 and
             isinstance(parity.get("transformed_positions"), int) and
             parity["transformed_positions"] > 0 and
             parity.get("feature_binary_sha256") == feature["binary_sha256"],
             "candidate parity statistics are malformed")
    evaluator_benchmark = report.get("evaluator_benchmark")
    _require(isinstance(evaluator_benchmark, dict) and
             set(evaluator_benchmark) == {"two-ply-closure", "pattern-gain"},
             "candidate evaluator benchmarks are incomplete")
    search_benchmark = report.get("search_benchmark")
    _require(isinstance(search_benchmark, dict) and
             isinstance(search_benchmark.get("evaluators"), dict) and
             set(search_benchmark["evaluators"]) == {"two-ply-closure", "pattern-gain"} and
             isinstance(search_benchmark.get("comparison"), dict) and
             bool(search_benchmark["comparison"]),
             "candidate search benchmarks are incomplete")
    commands = report.get("commands")
    _require(isinstance(commands, list) and bool(commands),
             "candidate command provenance is missing")
    for index, command in enumerate(commands):
        _require(isinstance(command, dict) and
                 isinstance(command.get("command"), list) and
                 all(isinstance(item, str) for item in command["command"]) and
                 isinstance(command.get("log"), str) and
                 isinstance(command.get("log_sha256"), str) and
                 isinstance(command.get("resources"), dict),
                 f"candidate command provenance {index} is malformed")
        log = (config.output_directory / command["log"]).resolve()
        _require(config.output_directory in log.parents and log.is_file() and
                 sha256_file(log) == command["log_sha256"],
                 f"candidate command log {index} is missing, unsafe, or changed")
    training_datasets = candidate_training_datasets(iteration)
    _require(report.get("training_datasets") == training_datasets,
             "candidate lists different training datasets than its training report")
    return {
        "report_sha256": report_digest, "promotable": report["promotable"],
        "git_commit": report["git_commit"], "created_at_utc": report.get("created_at_utc"),
        "corpus_id": feature["corpus_id"],
        "teacher": feature.get("teacher"),
        "feature_binary_sha256": feature["binary_sha256"],
        "training_report_sha256": report["training_report_sha256"],
        "sealed_evaluation_report_sha256": report["sealed_evaluation_report_sha256"],
        "model": report.get("model"), "parity": report.get("parity"),
        "evaluator_benchmark": report.get("evaluator_benchmark"),
        "search_benchmark": report.get("search_benchmark"),
        "artifacts": artifacts,
        **({} if training_datasets is None else {"training_datasets": training_datasets}),
    }


def candidate_training_datasets(iteration: IterationConfig) -> list[dict[str, Any]] | None:
    """Return the datasets a candidate's model learned from, when there was more than one."""
    if not isinstance(iteration.training, PatternTrainingConfig):
        return None
    data = open_pattern_report(iteration.training_directory).get("training_data")
    datasets = data.get("datasets") if isinstance(data, dict) else None
    return datasets if isinstance(datasets, list) and len(datasets) > 1 else None


def _authenticate_build_artifacts(root: Path, report: dict[str, Any],
                                  expected: dict[str, str], label: str) -> dict[str, Path]:
    """Authenticate binaries a match report records by path relative to its root."""
    build_artifacts = report.get("build_artifacts")
    _require(isinstance(build_artifacts, dict) and set(build_artifacts) == set(expected),
             f"{label} build artifact map is incomplete")
    paths: dict[str, Path] = {}
    for name, relative in expected.items():
        metadata = build_artifacts[name]
        _require(isinstance(metadata, dict) and metadata.get("path") == relative and
                 isinstance(metadata.get("sha256"), str),
                 f"{label} build artifact {name} metadata is malformed")
        path = (root / relative).resolve()
        _require(root in path.parents and path.is_file() and
                 sha256_file(path) == metadata["sha256"],
                 f"{label} build artifact {name} is missing, unsafe, or changed")
        paths[name] = path
    return paths


def _authenticate_match_run(config: WorkflowConfig, root: Path, report: dict[str, Any],
                            settings: MatchConfig, recorded: dict[str, Any], *, label: str,
                            kind: str, new_id: str, base_id: str | None,
                            new_engine_path: Path, base_engine_path: Path,
                            nets: dict[str, Path | None] | None = None) -> dict[str, Any]:
    """Authenticate a match report's settings and its single evaluator run against both sides.

    nets, for matches between net files, maps each side to the file it loaded, or to None for a
    base build that plays its own net.
    """
    recorded_settings = report.get("settings")
    _require(isinstance(recorded_settings, dict) and
             {key: value for key, value in recorded_settings.items() if key != "opening_book"} ==
             {key: value for key, value in recorded.items() if key != "opening_book"} and
             names_path(config, recorded_settings.get("opening_book"), settings.opening_book),
             f"{label} settings differ from configuration")
    _require(settings.opening_book.is_file(),
             f"{label} opening book is missing: {settings.opening_book}")
    _require(report.get("opening_book_sha256") == sha256_file(settings.opening_book),
             f"{label} opening book changed after evaluation")
    _require(isinstance(report.get("git_commit"), str) and
             re.fullmatch(r"[0-9a-f]{40}", report["git_commit"]) is not None,
             f"{label} Git commit is malformed")
    run_relative = report.get("run_directory")
    _require(isinstance(run_relative, str) and bool(run_relative),
             f"{label} run directory metadata is malformed")
    run_path = (root / run_relative).resolve()
    _require(root in run_path.parents and run_path.is_dir(),
             f"{label} run directory is missing or unsafe")
    try:
        run_entries = list((root / "runs").iterdir())
    except OSError as error:
        raise AdapterError(f"could not inspect {label} runs: {error}") from error
    _require(len(run_entries) == 1 and run_entries[0].resolve() == run_path,
             f"{label} has missing or unexpected evaluator runs")
    run_artifacts = report.get("artifacts")
    _require(isinstance(run_artifacts, dict), f"{label} run artifact map is missing")
    for name in ("manifest.json", "summary.json", "games.csv", "ledger-row.csv"):
        expected = run_artifacts.get(name)
        _require(isinstance(expected, str) and sha256_file(run_path / name) == expected,
                 f"{label} {name} is missing or changed")
    manifest = _manifest(run_path / "manifest.json")
    summary = _manifest(run_path / "summary.json")
    if base_id is not None:
        _require(manifest.get("base_id") == base_id,
                 f"{label} base build is not the configured base model")
    run_id = report.get("run_id")
    _require(isinstance(run_id, str) and bool(run_id) and
             manifest.get("run_id") == run_id and manifest.get("valid") is True,
             f"{label} manifest is invalid")
    manifest_base_path = manifest.get("base_engine_path")
    _require(isinstance(manifest_base_path, str) and bool(manifest_base_path),
             f"{label} manifest base engine path is malformed")
    normalized_manifest_base = Path(manifest_base_path)
    if not normalized_manifest_base.is_absolute():
        normalized_manifest_base = config.repository / normalized_manifest_base
    _require(
        manifest.get("kind") == kind and
        manifest.get("new_id") == new_id and
        names_path(config, manifest.get("new_engine_path"), new_engine_path) and
        manifest.get("new_engine") == settings.new_engine and
        manifest.get("new_engine_args") == settings.new_engine_args and
        manifest.get("base_engine") == settings.base_engine and
        manifest.get("base_engine_args") == settings.base_engine_args and
        normalized_manifest_base.resolve() == base_engine_path and
        names_path(config, manifest.get("opening_book"), settings.opening_book) and
        manifest.get("games") == settings.games and
        manifest.get("workers_requested") == settings.workers and
        manifest.get("timeout_ms") == settings.timeout_ms and
        manifest.get("go_movetime_ms") == settings.go_movetime_ms and
        manifest.get("sequential_stop") == settings.sequential_stop and
        manifest.get("sequential_null") == settings.sequential_null and
        manifest.get("sequential_alt") == settings.sequential_alt and
        manifest.get("sequential_alpha") == settings.sequential_alpha and
        manifest.get("sequential_beta") == settings.sequential_beta,
        f"{label} manifest settings differ from configuration")
    if settings.opening_seed is not None:
        _require(manifest.get("opening_seed") == settings.opening_seed,
                 f"{label} opening seed differs from configuration")
    for side, net in (nets or {}).items():
        recorded = manifest.get(f"{side}_net")
        _require(recorded is None if net is None else names_path(config, recorded, net),
                 f"{label} {side} side loaded a different net file")
        if net is not None:
            _require(manifest.get(f"{side}_net_sha256") == sha256_file(net),
                     f"{label} {side} side reported a different net")
    decision = manifest.get("sequential_decision")
    _require(isinstance(decision, str) and bool(decision),
             f"{label} manifest has no sequential decision")
    _require(summary.get("valid") is True, f"{label} summary is invalid")
    return {
        "run_id": run_id, "sequential_decision": decision, "run_directory": str(run_path),
        "manifest": manifest, "summary": summary,
        "ledger_row": str(run_path / "ledger-row.csv"),
    }


def authenticate_gate(config: WorkflowConfig, iteration: IterationConfig, *,
                      candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    """Authenticate a completed gate; pass an already authenticated candidate to reuse it."""
    _require(iteration.engine_gate is not None, "iteration has no engine gate")
    if candidate is None:
        candidate = authenticate_candidate(config, iteration)
    report, digest = open_json_report(
        iteration.gate_directory, schema=GATE_SCHEMA,
        schema_version=(HEADER_GATE_SCHEMA_VERSION, GATE_SCHEMA_VERSION), error_type=AdapterError)
    if report["schema_version"] == HEADER_GATE_SCHEMA_VERSION:
        return _authenticate_header_gate(config, iteration, candidate, report, digest)
    return _authenticate_net_gate(config, iteration, candidate, report, digest)


def engine_build_id(count: int, commit: str) -> str:
    """Return the ID a build of commit has under build/by-commit: count-shortsha."""
    return f"{count:06d}-{commit[:12]}"


def _authenticate_shared_build(root: Path, report: dict[str, Any], preset: str,
                               label: str) -> tuple[str, Path]:
    """Return the engine build ID and directory a match built once from its commit."""
    build_id = report.get("engine_build_id")
    commit = report.get("git_commit")
    _require(isinstance(build_id, str) and isinstance(commit, str) and
             re.fullmatch(r"[0-9]{6}-[0-9a-f]{12}", build_id) is not None and
             build_id.endswith(commit[:12]),
             f"{label} engine build ID is malformed")
    build = root / "builds" / build_id / preset
    _require(report.get("build_directory") == build.relative_to(root).as_posix(),
             f"{label} has the wrong build layout")
    builds = {path.name for path in (root / "builds").iterdir()}
    _require(builds == {build_id}, f"{label} has unexpected builds: {root / 'builds'}")
    return build_id, build


def _authenticate_net_gate(config: WorkflowConfig, iteration: IterationConfig,
                           candidate: dict[str, Any], report: dict[str, Any],
                           digest: str) -> dict[str, Any]:
    """Authenticate a gate between net files: one engine build, the candidate's net on one side."""
    gate = iteration.engine_gate
    assert gate is not None
    names = {path.name for path in iteration.gate_directory.iterdir()}
    _require(names == {"COMPLETE", "report.json", "builds", "runs"},
             f"engine gate has unexpected output files: {iteration.gate_directory}")
    _require(report.get("candidate_report_sha256") == candidate["report_sha256"],
             "engine gate references a different candidate report")
    model_id = candidate_build_id(candidate)
    _require(report.get("candidate_build_id") == model_id and "net" in candidate["artifacts"],
             "engine gate played a different candidate")
    build_id, build = _authenticate_shared_build(
        iteration.gate_directory, report, config.build_preset, "engine gate")
    build_relative = build.relative_to(iteration.gate_directory)
    build_paths = _authenticate_build_artifacts(iteration.gate_directory, report, {
        "new_engine_binary": (build_relative / "engines" / gate.new_engine).as_posix(),
        "runner_binary": (build_relative / "runner" / "poe2_runner").as_posix(),
    }, "engine-gate")
    new_net = candidate_net_path(iteration)
    base_model_id = gate_base_model_id(config, iteration)
    if base_model_id is None:
        base_engine_path = resolve_eval_binary(config, gate.base or "", gate.base_engine)
        base_net: Path | None = None
    else:
        base_engine_path = (build / "engines" / gate.base_engine).resolve()
        base_net = model_net(_gate_base_reference(config, iteration))
    base_metadata = report.get("base_engine_artifact")
    _require(isinstance(base_metadata, dict) and
             names_path(config, base_metadata.get("path"), base_engine_path) and
             isinstance(base_metadata.get("sha256"), str) and
             sha256_file(base_engine_path) == base_metadata["sha256"],
             "engine-gate base engine artifact is missing or changed")
    expected_nets = {"new_net": (new_net, model_id), "base_net": (base_net, base_model_id)}
    for key, (net, net_model_id) in expected_nets.items():
        recorded = report.get(key)
        _require(recorded is None if net is None else
                 isinstance(recorded, dict) and set(recorded) == {"path", "sha256", "model_id"} and
                 names_path(config, recorded["path"], net) and
                 recorded["sha256"] == sha256_file(net) and recorded["model_id"] == net_model_id,
                 "engine gate played different nets than its candidate and base")
    match = _authenticate_match_run(
        config, iteration.gate_directory, report, gate, recorded_gate_settings(gate),
        label="engine-gate", kind=GATE_KIND, new_id=build_id,
        base_id=build_id if base_model_id is not None else None,
        new_engine_path=build_paths["new_engine_binary"], base_engine_path=base_engine_path,
        nets={"new": new_net, "base": base_net})
    return {"report_sha256": digest, "candidate_build_id": model_id, "new_id": build_id,
            "engine_build_id": build_id, "new_net_sha256": sha256_file(new_net),
            "base_net_sha256": None if base_net is None else sha256_file(base_net), **match}


def _gate_base_reference(config: WorkflowConfig, iteration: IterationConfig) -> ModelReference:
    """Return the resolved model a gate's base names."""
    gate = iteration.engine_gate
    assert gate is not None
    if gate.base_iteration is not None:
        return config.trained_model(gate.base_iteration)
    for model in config.models:
        if model.model_id == gate.base_model:
            return model
    raise AdapterError(f"engine-gate base model is not named in config.models: {gate.base_model}")


def _authenticate_header_gate(config: WorkflowConfig, iteration: IterationConfig,
                              candidate: dict[str, Any], report: dict[str, Any],
                              digest: str) -> dict[str, Any]:
    """Authenticate a gate played before nets were files: one private build per model."""
    build_id = candidate_build_id(candidate)
    build = gate_build_directory(config, iteration, candidate)
    names = {path.name for path in iteration.gate_directory.iterdir()}
    _require(names == {"COMPLETE", "report.json", "builds", "runs"},
             f"engine gate has unexpected output files: {iteration.gate_directory}")
    gate = iteration.engine_gate
    _require(report.get("candidate_report_sha256") == candidate["report_sha256"],
             "engine gate references a different candidate report")
    _require(report.get("candidate_build_id") == build_id and
             report.get("build_directory") ==
             build.relative_to(iteration.gate_directory).as_posix(),
             "engine gate has the wrong candidate build identity or layout")
    candidate_header = candidate["artifacts"]["model_header"]
    _require(report.get("model_header_sha256") == candidate_header["sha256"],
             "engine gate was not built from the candidate model header")
    build_relative = build.relative_to(iteration.gate_directory)
    build_paths = _authenticate_build_artifacts(iteration.gate_directory, report, {
        "new_engine_binary": (build_relative / "engines" / gate.new_engine).as_posix(),
        "runner_binary": (build_relative / "runner" / "poe2_runner").as_posix(),
    }, "engine-gate")
    base_engine_path = gate_base_binary(config, iteration)
    base_model_id = gate_base_model_id(config, iteration)
    if base_model_id is not None:
        _require(iteration.gate_directory in base_engine_path.parents,
                 "engine-gate base model build escapes the gate directory")
    base_metadata = report.get("base_engine_artifact")
    _require(isinstance(base_metadata, dict) and
             names_path(config, base_metadata.get("path"), base_engine_path) and
             isinstance(base_metadata.get("sha256"), str) and
             sha256_file(base_engine_path) == base_metadata["sha256"],
             "engine-gate base engine artifact is missing or changed")
    match = _authenticate_match_run(
        config, iteration.gate_directory, report, gate, recorded_gate_settings(gate),
        label="engine-gate", kind=GATE_KIND, new_id=build_id, base_id=base_model_id,
        new_engine_path=build_paths["new_engine_binary"], base_engine_path=base_engine_path)
    return {"report_sha256": digest, "candidate_build_id": build_id, "new_id": build_id, **match}


def comparison_side(config: WorkflowConfig, model: ModelReference) -> dict[str, str]:
    """Return how a comparison report records one side: its name, model, and net."""
    _require(model.model_id is not None, f"comparison model {model.name} is not available")
    net = model_net(model)
    return {"name": model.name, "model_id": model.model_id, "net": config.recorded_path(net),
            "net_sha256": sha256_file(net)}


def authenticate_comparison(config: WorkflowConfig, comparison: ComparisonConfig,
                            new: ModelReference, base: ModelReference) -> dict[str, Any]:
    """Authenticate a completed comparison between the nets of the two models it resolved to."""
    root = comparison.root
    report, digest = open_json_report(root, schema=COMPARISON_SCHEMA,
                                      schema_version=COMPARISON_SCHEMA_VERSION,
                                      error_type=AdapterError)
    names = {path.name for path in root.iterdir()}
    _require(names == {"COMPLETE", "report.json", "builds", "runs"},
             f"comparison has unexpected output files: {root}")
    sides = {"new": comparison_side(config, new), "base": comparison_side(config, base)}
    for side, model in (("new", new), ("base", base)):
        recorded = report.get(side)
        expected = sides[side]
        _require(isinstance(recorded, dict) and set(recorded) == set(expected) and
                 all(recorded[key] == expected[key] for key in expected if key != "net") and
                 names_path(config, recorded["net"], model_net(model)),
                 f"comparison {comparison.name} was played between different models")
    build_id, build = _authenticate_shared_build(root, report, config.build_preset, "comparison")
    build_relative = build.relative_to(root).as_posix()
    build_paths = _authenticate_build_artifacts(root, report, {
        "new_engine_binary": f"{build_relative}/engines/{comparison.new_engine}",
        "base_engine_binary": f"{build_relative}/engines/{comparison.base_engine}",
        "runner_binary": f"{build_relative}/runner/poe2_runner",
    }, "comparison")
    match = _authenticate_match_run(
        config, root, report, comparison, match_settings(comparison), label="comparison",
        kind=COMPARISON_KIND, new_id=build_id, base_id=build_id,
        new_engine_path=build_paths["new_engine_binary"],
        base_engine_path=build_paths["base_engine_binary"],
        nets={"new": model_net(new), "base": model_net(base)})
    return {"report_sha256": digest, "new_id": build_id, "base_id": build_id,
            "engine_build_id": build_id, "new_model_id": new.model_id,
            "base_model_id": base.model_id, "new_net_sha256": sides["new"]["net_sha256"],
            "base_net_sha256": sides["base"]["net_sha256"], **match}


STAGE_ADAPTERS: dict[str, StageAdapter] = {
    "source": StageAdapter(
        "source", "position source", True, (source_chunk_command, source_assemble_command),
        (source_audit_command,), authenticate_source),
    "labels": StageAdapter(
        "labels", "label corpus", True, (label_command,),
        (label_audit_command, label_preflight_command), aggregate_label_stats),
    "features": StageAdapter(
        "features", "feature artifact", True, (feature_command,),
        (feature_audit_command,), authenticate_features),
    "imported-features": StageAdapter(
        "imported-features", "imported feature artifact", False, (), (),
        authenticate_features),
    "training-baseline": StageAdapter(
        "training-baseline", "baseline report", True, (training_command,), (),
        authenticate_training),
    "training-pattern": StageAdapter(
        "training-pattern", "pattern report", True, (training_command,), (),
        authenticate_training),
    "sealed-evaluation": StageAdapter(
        "sealed-evaluation", "sealed evaluation", True, (evaluation_command,), (),
        authenticate_evaluation),
    "candidate-validation": StageAdapter(
        "candidate-validation", "candidate report", True,
        (export_command, parity_command, search_benchmark_command), (),
        authenticate_candidate),
    "engine-gate": StageAdapter(
        "engine-gate", "engine gate", True, (engine_gate_command,), (),
        authenticate_gate),
    "comparison": StageAdapter(
        "comparison", "comparison", True, (comparison_command,), (),
        authenticate_comparison),
}
