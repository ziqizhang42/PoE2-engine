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
from .pattern_experiment import open_pattern_report
from .pattern_suites import pattern_suite
from .shared import open_json_report, sha256_file
from .workflow_config import (
    BaselineTrainingConfig,
    DatasetConfig,
    IterationConfig,
    ManagedDatasetConfig,
    WorkflowConfig,
)


CANDIDATE_SCHEMA = "poe2-training-candidate"
GATE_SCHEMA = "poe2-training-engine-gate"


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


def _require_report_layout(directory: Path, label: str) -> None:
    try:
        names = {path.name for path in directory.iterdir()}
    except OSError as error:
        raise AdapterError(f"could not inspect {directory}: {error}") from error
    _require(names == {"COMPLETE", "report.json"},
             f"{label} has unexpected output files: {directory}")


def data_binary(config: WorkflowConfig) -> Path:
    return (config.output_directory / "build" / config.build_preset / "runner" /
            "poe2_minimax_data")


def candidate_build_id(candidate: dict[str, Any]) -> str:
    """Return the native evaluator identity for one authenticated candidate model."""
    header = candidate.get("artifacts", {}).get("model_header")
    _require(isinstance(header, dict) and
             isinstance(header.get("sha256"), str) and
             re.fullmatch(r"[0-9a-f]{64}", header["sha256"]) is not None,
             "candidate report has no valid model-header digest")
    return f"candidate-sha256-{header['sha256']}"


def gate_build_directory(config: WorkflowConfig, iteration: IterationConfig,
                         candidate: dict[str, Any]) -> Path:
    """Choose a candidate-specific layout understood by the native build-ID reader."""
    return (iteration.gate_directory / "builds" / candidate_build_id(candidate) /
            config.build_preset)


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


def source_command(config: WorkflowConfig, dataset: ManagedDatasetConfig) -> list[str]:
    source = dataset.source
    return [
        str(data_binary(config)), "source", "--output-dir", str(dataset.source_directory),
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
        str(data_binary(config)), "features", "--source", str(dataset.source_directory),
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


def engine_gate_command(config: WorkflowConfig, iteration: IterationConfig,
                        build: Path, runner: Path) -> list[str]:
    gate = iteration.engine_gate
    _require(gate is not None, "iteration has no engine gate")
    command = [
        str(runner), "eval", "--new-build", str(build), "--base", gate.base,
        "--new-engine", gate.new_engine, "--base-engine", gate.base_engine,
        "--preset", config.build_preset, "--kind", "training-gate",
        "--run-root", str(iteration.gate_directory / "runs"), "--no-ledger",
        "--opening-book", str(gate.opening_book), "--shuffle-openings",
        "--games", str(gate.games), "--workers", str(gate.workers),
        "--timeout-ms", str(gate.timeout_ms),
        "--go-movetime-ms", str(gate.go_movetime_ms),
        "--sequential-null", str(gate.sequential_null),
        "--sequential-alt", str(gate.sequential_alt),
        "--sequential-alpha", str(gate.sequential_alpha),
        "--sequential-beta", str(gate.sequential_beta),
    ]
    if gate.new_engine_args:
        command.extend(("--new-engine-args", gate.new_engine_args))
    if gate.base_engine_args:
        command.extend(("--base-engine-args", gate.base_engine_args))
    if gate.opening_seed is not None:
        command.extend(("--opening-seed", str(gate.opening_seed)))
    if gate.sequential_stop:
        command.append("--sequential-stop")
    if gate.require_accept_alt:
        command.append("--require-accept-alt")
    return command


def _manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterError(f"could not read {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


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
    build = manifest.get("build")
    _require(isinstance(build, dict) and build.get("git_dirty") is False,
             "position source was produced from a dirty build")
    return {
        "manifest_sha256": sha256_file(manifest_path), "corpus_id": corpus.get("id"),
        "records": results.get("records"), "trajectories": expected.trajectories,
        "shards": expected.shards, "duplicates": results.get("duplicate_positions"),
        "split_counts": results.get("split_counts"), "policy_counts": results.get("policy_counts"),
        "workers_requested": generator.get("workers_requested"),
        "workers_used": generator.get("workers_used"),
    }


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
             search.get("evaluator") in {"two-ply-closure", "b"} and
             search.get("symmetry") is True and
             search.get("two_ply_closure") is True,
             f"label shard {index} search semantics are unsupported")
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
    return {
        "binary_sha256": artifact.binary_digest,
        "manifest_sha256": artifact.manifest_digest,
        "records": artifact.record_count, "source_records": artifact.source_record_count,
        "duplicates_removed": artifact.duplicates_removed, "shards": artifact.shard_count,
        "split_counts": {"train": artifact.split_counts[0],
                         "validation": artifact.split_counts[1], "test": artifact.split_counts[2]},
        "corpus_id": artifact.manifest["corpus"]["id"],
    }


def authenticate_training(dataset: DatasetConfig,
                          iteration: IterationConfig) -> dict[str, Any]:
    _require_report_layout(iteration.training_directory, "training report")
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
    return {
        "report_sha256": sha256_file(iteration.training_directory / "report.json"),
        "kind": training.type, "selected_model": selected.get("name"),
        "selected_validation": selected.get("validation", {}).get("overall"),
        "selected_quantized_validation": quantized_validation,
        "input": metadata,
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
        "model": report.get("model"), "parity": report.get("parity"),
        "evaluator_benchmark": report.get("evaluator_benchmark"),
        "search_benchmark": report.get("search_benchmark"),
        "artifacts": artifacts,
    }


def authenticate_gate(config: WorkflowConfig,
                      iteration: IterationConfig) -> dict[str, Any]:
    _require(iteration.engine_gate is not None, "iteration has no engine gate")
    candidate = authenticate_candidate(config, iteration)
    build_id = candidate_build_id(candidate)
    build = gate_build_directory(config, iteration, candidate)
    report, digest = open_json_report(
        iteration.gate_directory, schema=GATE_SCHEMA, error_type=AdapterError)
    names = {path.name for path in iteration.gate_directory.iterdir()}
    _require(names == {"COMPLETE", "report.json", "builds", "runs"},
             f"engine gate has unexpected output files: {iteration.gate_directory}")
    gate = iteration.engine_gate
    expected_settings = asdict(gate)
    expected_settings["opening_book"] = str(gate.opening_book)
    _require(report.get("settings") == expected_settings,
             "engine-gate settings differ from configuration")
    _require(gate.opening_book.is_file(),
             f"engine-gate opening book is missing: {gate.opening_book}")
    _require(report.get("opening_book_sha256") == sha256_file(gate.opening_book),
             "engine-gate opening book changed after evaluation")
    _require(report.get("candidate_report_sha256") == candidate["report_sha256"],
             "engine gate references a different candidate report")
    _require(report.get("candidate_build_id") == build_id and
             report.get("build_directory") ==
             build.relative_to(iteration.gate_directory).as_posix(),
             "engine gate has the wrong candidate build identity or layout")
    candidate_header = candidate["artifacts"]["model_header"]
    _require(report.get("tracked_model_header_sha256") == candidate_header["sha256"],
             "engine gate was not built from the candidate model header")
    _require(isinstance(report.get("git_commit"), str) and
             re.fullmatch(r"[0-9a-f]{40}", report["git_commit"]) is not None,
             "engine-gate Git commit is malformed")
    run_relative = report.get("run_directory")
    _require(isinstance(run_relative, str) and bool(run_relative),
             "engine-gate run directory metadata is malformed")
    run_path = (iteration.gate_directory / run_relative).resolve()
    _require(iteration.gate_directory in run_path.parents and run_path.is_dir(),
             "engine-gate run directory is missing or unsafe")
    try:
        run_entries = list((iteration.gate_directory / "runs").iterdir())
    except OSError as error:
        raise AdapterError(f"could not inspect engine-gate runs: {error}") from error
    _require(len(run_entries) == 1 and run_entries[0].resolve() == run_path,
             "engine gate has missing or unexpected evaluator runs")
    run_artifacts = report.get("artifacts")
    _require(isinstance(run_artifacts, dict),
             "engine-gate run artifact map is missing")
    for name in ("manifest.json", "summary.json", "games.csv", "ledger-row.csv"):
        expected = run_artifacts.get(name)
        _require(isinstance(expected, str) and sha256_file(run_path / name) == expected,
                 f"engine-gate {name} is missing or changed")
    build_artifacts = report.get("build_artifacts")
    build_relative = build.relative_to(iteration.gate_directory)
    expected_build_artifacts = {
        "new_engine_binary": (build_relative / "engines" / gate.new_engine).as_posix(),
        "runner_binary": (build_relative / "runner" / "poe2_runner").as_posix(),
    }
    _require(isinstance(build_artifacts, dict) and
             set(build_artifacts) == set(expected_build_artifacts),
             "engine-gate build artifact map is incomplete")
    authenticated_build_paths: dict[str, Path] = {}
    for name, relative in expected_build_artifacts.items():
        metadata = build_artifacts[name]
        _require(isinstance(metadata, dict) and metadata.get("path") == relative and
                 isinstance(metadata.get("sha256"), str),
                 f"engine-gate build artifact {name} metadata is malformed")
        path = (iteration.gate_directory / relative).resolve()
        _require(iteration.gate_directory in path.parents and path.is_file() and
                 sha256_file(path) == metadata["sha256"],
                 f"engine-gate build artifact {name} is missing, unsafe, or changed")
        authenticated_build_paths[name] = path
    manifest = _manifest(run_path / "manifest.json")
    summary = _manifest(run_path / "summary.json")
    base_engine_path = resolve_eval_binary(config, gate.base, gate.base_engine)
    base_metadata = report.get("base_engine_artifact")
    _require(isinstance(base_metadata, dict) and
             base_metadata.get("path") == str(base_engine_path) and
             isinstance(base_metadata.get("sha256"), str) and
             sha256_file(base_engine_path) == base_metadata["sha256"],
             "engine-gate base engine artifact is missing or changed")
    run_id = report.get("run_id")
    _require(isinstance(run_id, str) and bool(run_id) and
             manifest.get("run_id") == run_id and manifest.get("valid") is True,
             "engine-gate manifest is invalid")
    manifest_base_path = manifest.get("base_engine_path")
    _require(isinstance(manifest_base_path, str) and bool(manifest_base_path),
             "engine-gate manifest base engine path is malformed")
    normalized_manifest_base = Path(manifest_base_path)
    if not normalized_manifest_base.is_absolute():
        normalized_manifest_base = config.repository / normalized_manifest_base
    normalized_manifest_base = normalized_manifest_base.resolve()
    _require(
        manifest.get("kind") == "training-gate" and
        manifest.get("new_id") == build_id and
        manifest.get("new_engine_path") ==
        str(authenticated_build_paths["new_engine_binary"]) and
        manifest.get("new_engine") == gate.new_engine and
        manifest.get("new_engine_args") == gate.new_engine_args and
        manifest.get("base_engine") == gate.base_engine and
        manifest.get("base_engine_args") == gate.base_engine_args and
        normalized_manifest_base == base_engine_path and
        manifest.get("opening_book") == str(gate.opening_book) and
        manifest.get("games") == gate.games and
        manifest.get("workers_requested") == gate.workers and
        manifest.get("timeout_ms") == gate.timeout_ms and
        manifest.get("go_movetime_ms") == gate.go_movetime_ms and
        manifest.get("sequential_stop") == gate.sequential_stop and
        manifest.get("sequential_null") == gate.sequential_null and
        manifest.get("sequential_alt") == gate.sequential_alt and
        manifest.get("sequential_alpha") == gate.sequential_alpha and
        manifest.get("sequential_beta") == gate.sequential_beta,
        "engine-gate manifest settings differ from configuration")
    if gate.opening_seed is not None:
        _require(manifest.get("opening_seed") == gate.opening_seed,
                 "engine-gate opening seed differs from configuration")
    if gate.require_accept_alt:
        _require(manifest.get("sequential_decision") == "accept_alt",
                 "engine gate did not accept the configured alternative")
    _require(summary.get("valid") is True,
             "engine-gate summary is invalid")
    return {
        "report_sha256": digest, "run_id": report.get("run_id"),
        "candidate_build_id": build_id,
        "run_directory": str(run_path), "manifest": manifest, "summary": summary,
        "ledger_row": str(run_path / "ledger-row.csv"),
    }


STAGE_ADAPTERS: dict[str, StageAdapter] = {
    "source": StageAdapter(
        "source", "position source", True, (source_command,),
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
}
