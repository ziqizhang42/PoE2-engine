"""Typed adapters between workflow stages and the authoritative lower-level tools."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .artifact import FeatureArtifact, open_feature_artifact
from .baseline import open_baseline_report
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
    WorkflowConfig,
    match_settings,
    recorded_gate_settings,
)


CANDIDATE_SCHEMA = "poe2-training-candidate"
GATE_SCHEMA = "poe2-training-engine-gate"
# Version 2 gates build the candidate header privately instead of the tracked engine header.
GATE_SCHEMA_VERSION = 2
COMPARISON_SCHEMA = "poe2-training-comparison"
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


def data_build_directory(config: WorkflowConfig,
                         dataset: ManagedDatasetConfig | None = None) -> Path:
    """Return the run-local build tree, private to the dataset's teacher model when it has one.

    A teacher this recipe trains is named by its iteration, so the plan knows the path before
    the model exists; the binary still embeds and is checked against the model digest.
    """
    root = config.output_directory / "build"
    if dataset is not None and dataset.teacher is not None:
        teacher = dataset.teacher
        name = (teacher.sha256[:12] if teacher.iteration is None else
                f"iteration-{teacher.iteration}")
        root = root / f"teacher-{name}"
    return root / config.build_preset


def data_binary(config: WorkflowConfig, dataset: ManagedDatasetConfig | None = None) -> Path:
    return data_build_directory(config, dataset) / "runner" / "poe2_minimax_data"


def _evaluator_arguments(dataset: ManagedDatasetConfig) -> list[str]:
    return [] if dataset.teacher is None else ["--evaluator", "pattern-gain"]


def candidate_build_id(candidate: dict[str, Any]) -> str:
    """Return the native evaluator identity for one authenticated candidate model."""
    header = candidate.get("artifacts", {}).get("model_header")
    _require(isinstance(header, dict) and
             isinstance(header.get("sha256"), str) and
             re.fullmatch(r"[0-9a-f]{64}", header["sha256"]) is not None,
             "candidate report has no valid model-header digest")
    return f"candidate-sha256-{header['sha256']}"


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


def gate_base_input(config: WorkflowConfig, iteration: IterationConfig) -> str:
    """Return the evaluator --base argument: an existing build or the base model's private build."""
    gate = iteration.engine_gate
    _require(gate is not None, "iteration has no engine gate")
    if gate.base is None:
        return str(gate_base_build_directory(config, iteration))
    return gate.base


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
        str(data_binary(config, dataset)), "source-chunk",
        "--output-dir", str(source_chunk_path(dataset, begin, end)),
        "--trajectory-begin", str(begin), "--trajectory-end", str(end),
        *_source_arguments(dataset),
    ]


def source_assemble_command(config: WorkflowConfig, dataset: ManagedDatasetConfig) -> list[str]:
    return [
        str(data_binary(config, dataset)), "source-assemble",
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
        str(data_binary(config, dataset)), "labels", "--input", str(dataset.source_directory),
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
        str(data_binary(config, dataset)), "features",
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


def training_command(python: str, dataset: DatasetConfig,
                     iteration: IterationConfig) -> list[str]:
    feature_directory = dataset.feature_directory
    training = iteration.training
    if isinstance(training, BaselineTrainingConfig):
        return [
            python, "-m", "poe2_training.baseline", "--dataset", str(feature_directory),
            "--output-dir", str(iteration.training_directory), "--device", training.device,
            "--ridge-lambdas", ",".join(str(value) for value in training.ridge_lambdas),
            "--close-margin", str(training.close_margin), "--seed", str(training.seed),
        ]
    return [
        python, "-m", "poe2_training.pattern_experiment", "--dataset",
        str(feature_directory), "--output-dir", str(iteration.training_directory),
        "--device", training.device, "--seed", str(training.seed), "--suite", training.suite,
    ]


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


def export_command(python: str, config: WorkflowConfig,
                   iteration: IterationConfig) -> list[str]:
    return [
        python, str(config.repository / "tools" / "export_minimax_pattern_gain.py"),
        "--experiment", str(iteration.training_directory), "--output",
        str(iteration.candidate_directory / "frozen_pattern_gain_model.hpp"),
    ]


def parity_command(python: str, dataset: DatasetConfig, iteration: IterationConfig,
                   binary: Path) -> list[str]:
    candidate = iteration.candidate_validation
    _require(candidate is not None, "iteration has no candidate validation")
    return [
        python, "-m", "poe2_training.pattern_integration", "--dataset",
        str(dataset.feature_directory), "--experiment", str(iteration.training_directory),
        "--inference-binary", str(binary), "--samples", str(candidate.samples),
        "--symmetry-samples", str(candidate.symmetry_samples),
        "--benchmark-iterations", str(candidate.benchmark_iterations),
    ]


def search_benchmark_command(python: str, config: WorkflowConfig,
                             iteration: IterationConfig, engine: Path) -> list[str]:
    candidate = iteration.candidate_validation
    _require(candidate is not None, "iteration has no candidate validation")
    return [
        python, str(config.repository / "tools" / "benchmark_minimax_search.py"),
        "--engine", str(engine), "--opening-book", str(candidate.opening_book),
        "--positions", str(candidate.search_positions),
        "--movetime-ms", str(candidate.search_movetime_ms), "--seed", str(iteration.training.seed),
    ]


def match_command(config: WorkflowConfig, settings: MatchConfig, *, runner: Path,
                  new_build: Path, base: str, kind: str, root: Path) -> list[str]:
    """Return the evaluator command for one match, recorded under root/runs without a ledger."""
    command = [
        str(runner), "eval", "--new-build", str(new_build), "--base", base,
        "--new-engine", settings.new_engine, "--base-engine", settings.base_engine,
        "--preset", config.build_preset, "--kind", kind,
        "--run-root", str(root / "runs"), "--no-ledger",
        "--opening-book", str(settings.opening_book), "--shuffle-openings",
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
    if settings.opening_seed is not None:
        command.extend(("--opening-seed", str(settings.opening_seed)))
    if settings.sequential_stop:
        command.append("--sequential-stop")
    return command


def engine_gate_command(config: WorkflowConfig, iteration: IterationConfig,
                        build: Path, runner: Path) -> list[str]:
    gate = iteration.engine_gate
    _require(gate is not None, "iteration has no engine gate")
    return match_command(config, gate, runner=runner, new_build=build,
                         base=gate_base_input(config, iteration), kind=GATE_KIND,
                         root=iteration.gate_directory)


def comparison_command(config: WorkflowConfig, comparison: ComparisonConfig,
                       new_id: str, base_id: str) -> list[str]:
    new_build = private_build_directory(config, comparison.root, new_id)
    return match_command(
        config, comparison, runner=new_build / "runner" / "poe2_runner", new_build=new_build,
        base=str(private_build_directory(config, comparison.root, base_id)),
        kind=COMPARISON_KIND, root=comparison.root)


def _manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterError(f"could not read {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def recorded_teacher(search: dict[str, Any], label: str) -> str | None:
    """Return the archived model ID a manifest's search used, or None for two-ply closure."""
    evaluator = search.get("evaluator", "two-ply-closure")
    model = search.get("model_header_sha256")
    if evaluator in {"two-ply-closure", "b"}:
        _require(model is None, f"{label} records a model without a learned evaluator")
        return None
    _require(evaluator == "pattern-gain" and isinstance(model, str) and
             re.fullmatch(r"[0-9a-f]{64}", model) is not None,
             f"{label} search evaluator is unsupported")
    return f"candidate-sha256-{model}"


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
    _require(recorded_teacher(generator, "position source") == _expected_teacher(dataset),
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
    _require(recorded_teacher(search, f"label shard {index}") == _expected_teacher(dataset),
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
        _require(teacher == _expected_teacher(dataset),
                 "feature artifact teacher differs from configuration")
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


def authenticate_training(dataset: DatasetConfig,
                          iteration: IterationConfig) -> dict[str, Any]:
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
                 report.get("training", {}).get("configs") == expected_configs,
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
    return {
        "report_sha256": sha256_file(iteration.training_directory / "report.json"),
        "kind": training.type, "selected_model": selected.get("name"),
        "selected_validation": selected.get("validation", {}).get("overall"),
        "selected_quantized_validation": quantized_validation,
        "input": metadata, "attachments": report.get("attachments", {}),
    }


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


def authenticate_candidate(config: WorkflowConfig,
                           iteration: IterationConfig) -> dict[str, Any]:
    candidate = iteration.candidate_validation
    _require(candidate is not None, "iteration has no candidate validation")
    report, report_digest = open_json_report(
        iteration.candidate_directory, schema=CANDIDATE_SCHEMA,
        error_type=AdapterError)
    names = {path.name for path in iteration.candidate_directory.iterdir()}
    _require(names == {"COMPLETE", "report.json", "frozen_pattern_gain_model.hpp", "build"},
             f"candidate has unexpected output files: {iteration.candidate_directory}")
    _require(report.get("promotable") == candidate.promotable,
             "candidate promotable setting differs from configuration")
    expected_settings = {
        "promotable": candidate.promotable, "samples": candidate.samples,
        "symmetry_samples": candidate.symmetry_samples,
        "benchmark_iterations": candidate.benchmark_iterations,
        "opening_book": str(candidate.opening_book),
        "search_positions": candidate.search_positions,
        "search_movetime_ms": candidate.search_movetime_ms,
    }
    _require(report.get("settings") == expected_settings,
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
    }


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
                            new_engine_path: Path, base_engine_path: Path) -> dict[str, Any]:
    """Authenticate a match report's settings and its single evaluator run against both sides."""
    _require(report.get("settings") == {**recorded, "opening_book": str(settings.opening_book)},
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
        manifest.get("new_engine_path") == str(new_engine_path) and
        manifest.get("new_engine") == settings.new_engine and
        manifest.get("new_engine_args") == settings.new_engine_args and
        manifest.get("base_engine") == settings.base_engine and
        manifest.get("base_engine_args") == settings.base_engine_args and
        normalized_manifest_base.resolve() == base_engine_path and
        manifest.get("opening_book") == str(settings.opening_book) and
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
    build_id = candidate_build_id(candidate)
    build = gate_build_directory(config, iteration, candidate)
    report, digest = open_json_report(
        iteration.gate_directory, schema=GATE_SCHEMA,
        schema_version=GATE_SCHEMA_VERSION, error_type=AdapterError)
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
             base_metadata.get("path") == str(base_engine_path) and
             isinstance(base_metadata.get("sha256"), str) and
             sha256_file(base_engine_path) == base_metadata["sha256"],
             "engine-gate base engine artifact is missing or changed")
    match = _authenticate_match_run(
        config, iteration.gate_directory, report, gate, recorded_gate_settings(gate),
        label="engine-gate", kind=GATE_KIND, new_id=build_id, base_id=base_model_id,
        new_engine_path=build_paths["new_engine_binary"], base_engine_path=base_engine_path)
    return {"report_sha256": digest, "candidate_build_id": build_id, **match}


def comparison_side(config: WorkflowConfig, comparison: ComparisonConfig,
                    model: ModelReference) -> dict[str, str]:
    """Return how a comparison report records one side: its name, model, and private build."""
    _require(model.model_id is not None, f"comparison model {model.name} is not available")
    return {"name": model.name, "model_id": model.model_id,
            "build_directory": private_build_directory(config, comparison.root, model.model_id)
            .relative_to(comparison.root).as_posix()}


def authenticate_comparison(config: WorkflowConfig, comparison: ComparisonConfig,
                            new: ModelReference, base: ModelReference) -> dict[str, Any]:
    """Authenticate a completed comparison between the two models it resolved to."""
    root = comparison.root
    report, digest = open_json_report(root, schema=COMPARISON_SCHEMA, error_type=AdapterError)
    names = {path.name for path in root.iterdir()}
    _require(names == {"COMPLETE", "report.json", "builds", "runs"},
             f"comparison has unexpected output files: {root}")
    sides = {"new": comparison_side(config, comparison, new),
             "base": comparison_side(config, comparison, base)}
    _require(report.get("new") == sides["new"] and report.get("base") == sides["base"],
             f"comparison {comparison.name} was played between different models")
    builds = {path.name for path in (root / "builds").iterdir()}
    _require(builds == {new.model_id, base.model_id},
             f"comparison has unexpected private builds: {root / 'builds'}")
    build_paths = _authenticate_build_artifacts(root, report, {
        "new_engine_binary": f"{sides['new']['build_directory']}/engines/{comparison.new_engine}",
        "base_engine_binary":
            f"{sides['base']['build_directory']}/engines/{comparison.base_engine}",
        "runner_binary": f"{sides['new']['build_directory']}/runner/poe2_runner",
    }, "comparison")
    match = _authenticate_match_run(
        config, root, report, comparison, match_settings(comparison), label="comparison",
        kind=COMPARISON_KIND, new_id=new.model_id, base_id=base.model_id,
        new_engine_path=build_paths["new_engine_binary"],
        base_engine_path=build_paths["base_engine_binary"])
    return {"report_sha256": digest, "new_id": new.model_id, "base_id": base.model_id, **match}


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
