"""Strict typed TOML configuration for consolidated training runs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .model_metrics import DEFAULT_RIDGE_LAMBDAS
from .pattern_suites import SUITE_NAMES
from .shared import find_repository_root


SCHEMA_VERSION = 1
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CANDIDATE_ENGINE = "minimax/poe2_minimax"
CANDIDATE_ENGINE_ARGS = "--evaluator pattern-gain"


class WorkflowConfigError(ValueError):
    """Raised when a workflow TOML document is invalid or unsafe."""


@dataclass(frozen=True)
class SourceConfig:
    corpus_id: str
    seed: int
    trajectories: int
    samples_per_trajectory: int = 8
    shards: int = 1
    workers: int = 1
    search_nodes: int = 10_000
    search_hash_mb: int = 8
    noise_percent: int = 15
    random_weight: int = 30
    greedy_weight: int = 20
    opponent_weight: int = 20
    search_weight: int = 30
    progress_every: int = 100


@dataclass(frozen=True)
class LabelConfig:
    mode: Literal["exact", "teacher"]
    nodes: int
    hash_mb: int = 16
    workers: int = 6
    progress_every: int = 100
    require_all: bool = True


@dataclass(frozen=True)
class FeatureConfig:
    gain_samples: int = 256


@dataclass(frozen=True)
class ManagedDatasetConfig:
    name: str
    kind: Literal["managed"]
    root: Path
    source: SourceConfig
    labels: LabelConfig
    features: FeatureConfig

    @property
    def source_directory(self) -> Path:
        return self.root / "source"

    @property
    def labels_directory(self) -> Path:
        return self.root / "labels"

    @property
    def feature_directory(self) -> Path:
        return self.root / "features"


@dataclass(frozen=True)
class ImportedDatasetConfig:
    name: str
    kind: Literal["imported"]
    feature_directory: Path


DatasetConfig = ManagedDatasetConfig | ImportedDatasetConfig


@dataclass(frozen=True)
class BaselineTrainingConfig:
    type: Literal["baseline"]
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = 20260818
    close_margin: int = 8
    ridge_lambdas: tuple[float, ...] = DEFAULT_RIDGE_LAMBDAS


@dataclass(frozen=True)
class PatternTrainingConfig:
    type: Literal["pattern"]
    device: Literal["cpu", "cuda"] = "cuda"
    seed: int = 20260818
    suite: str = "default"


TrainingConfig = BaselineTrainingConfig | PatternTrainingConfig


@dataclass(frozen=True)
class SealedEvaluationConfig:
    exclude_datasets: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateValidationConfig:
    promotable: bool = False
    samples: int = 4096
    symmetry_samples: int = 512
    benchmark_iterations: int = 100
    opening_book: Path = Path("eval/openings/development.txt")
    search_positions: int = 32
    search_movetime_ms: int = 100


@dataclass(frozen=True)
class EngineGateConfig:
    base: str
    new_engine: str = CANDIDATE_ENGINE
    base_engine: str = "minimax/poe2_minimax"
    new_engine_args: str = CANDIDATE_ENGINE_ARGS
    base_engine_args: str = "--evaluator pattern-gain"
    opening_book: Path = Path("eval/openings/holdout.txt")
    games: int = 2000
    workers: int = 1
    timeout_ms: int = 1000
    go_movetime_ms: int = 100
    opening_seed: int | None = None
    sequential_stop: bool = True
    sequential_null: float = 0.0
    sequential_alt: float = 20.0
    sequential_alpha: float = 0.05
    sequential_beta: float = 0.05
    require_accept_alt: bool = True


@dataclass(frozen=True)
class IterationConfig:
    name: str
    dataset: str
    depends_on: tuple[str, ...]
    root: Path
    training: TrainingConfig
    sealed_evaluation: SealedEvaluationConfig | None
    candidate_validation: CandidateValidationConfig | None
    engine_gate: EngineGateConfig | None

    @property
    def training_directory(self) -> Path:
        return self.root / "training"

    @property
    def evaluation_directory(self) -> Path:
        return self.root / "sealed-evaluation"

    @property
    def candidate_directory(self) -> Path:
        return self.root / "candidate"

    @property
    def gate_directory(self) -> Path:
        return self.root / "engine-gate"


@dataclass(frozen=True)
class WorkflowConfig:
    path: Path
    contents: bytes
    digest: str
    repository: Path
    name: str
    output_directory: Path
    build_preset: str
    label_concurrency: int | None
    datasets: tuple[DatasetConfig, ...]
    iterations: tuple[IterationConfig, ...]

    def dataset(self, name: str) -> DatasetConfig:
        for dataset in self.datasets:
            if dataset.name == name:
                return dataset
        raise WorkflowConfigError(f"unknown dataset: {name}")

    def iteration(self, name: str) -> IterationConfig:
        for iteration in self.iterations:
            if iteration.name == name:
                return iteration
        raise WorkflowConfigError(f"unknown iteration: {name}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WorkflowConfigError(message)


def _table(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be a table")
    return value


def _unknown(table: dict[str, Any], allowed: set[str], field: str) -> None:
    unexpected = sorted(set(table) - allowed)
    _require(not unexpected, f"{field} has unknown field(s): {', '.join(unexpected)}")


_MISSING = object()


def _value(table: dict[str, Any], name: str, field: str, default: Any = _MISSING) -> Any:
    if name in table:
        return table[name]
    if default is not _MISSING:
        return default
    raise WorkflowConfigError(f"{field}.{name} is required")


def _string(table: dict[str, Any], name: str, field: str, default: Any = _MISSING,
            *, nonempty: bool = True) -> str:
    value = _value(table, name, field, default)
    _require(isinstance(value, str) and (bool(value) or not nonempty),
             f"{field}.{name} must be {'a nonempty ' if nonempty else 'a '}string")
    return value


def _name(value: str, field: str) -> str:
    _require(bool(NAME_PATTERN.fullmatch(value)),
             f"{field} must match {NAME_PATTERN.pattern}")
    _require(value not in {".", ".."}, f"{field} is unsafe")
    return value


def _binary_name(value: str, field: str) -> str:
    path = Path(value)
    _require(not path.is_absolute() and bool(path.parts) and
             all(part not in {"", ".", ".."} for part in path.parts),
             f"{field} must be a safe relative binary path")
    return value


def _integer(table: dict[str, Any], name: str, field: str, default: Any = _MISSING,
             *, minimum: int = 0, maximum: int | None = None) -> int:
    value = _value(table, name, field, default)
    _require(isinstance(value, int) and not isinstance(value, bool),
             f"{field}.{name} must be an integer")
    _require(value >= minimum and (maximum is None or value <= maximum),
             f"{field}.{name} is outside {minimum}..{maximum if maximum is not None else 'max'}")
    return value


def _number(table: dict[str, Any], name: str, field: str, default: Any = _MISSING) -> float:
    value = _value(table, name, field, default)
    _require(isinstance(value, (int, float)) and not isinstance(value, bool) and
             math.isfinite(float(value)), f"{field}.{name} must be a finite number")
    return float(value)


def _boolean(table: dict[str, Any], name: str, field: str,
             default: Any = _MISSING) -> bool:
    value = _value(table, name, field, default)
    _require(isinstance(value, bool), f"{field}.{name} must be a boolean")
    return value


def _string_list(table: dict[str, Any], name: str, field: str,
                 default: Any = _MISSING) -> tuple[str, ...]:
    value = _value(table, name, field, default)
    _require(isinstance(value, list) and all(isinstance(item, str) and item for item in value),
             f"{field}.{name} must be an array of nonempty strings")
    result = tuple(value)
    _require(len(set(result)) == len(result), f"{field}.{name} contains duplicates")
    return result


def _repository_path(repository: Path, value: str, field: str) -> Path:
    raw = Path(value)
    resolved = (raw if raw.is_absolute() else repository / raw).resolve()
    _require(resolved != repository, f"{field} must not be the repository root")
    _require(resolved != repository / ".git" and repository / ".git" not in resolved.parents,
             f"{field} must not be inside .git")
    return resolved


def _parse_source(value: Any, field: str) -> SourceConfig:
    table = _table(value, field)
    _unknown(table, {"corpus_id", "seed", "trajectories", "samples_per_trajectory",
                     "shards", "workers", "search_nodes", "search_hash_mb",
                     "noise_percent", "random_weight", "greedy_weight",
                     "opponent_weight", "search_weight", "progress_every"}, field)
    result = SourceConfig(
        corpus_id=_string(table, "corpus_id", field),
        seed=_integer(table, "seed", field),
        trajectories=_integer(table, "trajectories", field, minimum=1),
        samples_per_trajectory=_integer(table, "samples_per_trajectory", field, 8,
                                        minimum=1, maximum=8),
        shards=_integer(table, "shards", field, 1, minimum=1),
        workers=_integer(table, "workers", field, 1, minimum=1),
        search_nodes=_integer(table, "search_nodes", field, 10_000, minimum=1),
        search_hash_mb=_integer(table, "search_hash_mb", field, 8, minimum=1),
        noise_percent=_integer(table, "noise_percent", field, 15, maximum=100),
        random_weight=_integer(table, "random_weight", field, 30),
        greedy_weight=_integer(table, "greedy_weight", field, 20),
        opponent_weight=_integer(table, "opponent_weight", field, 20),
        search_weight=_integer(table, "search_weight", field, 30),
        progress_every=_integer(table, "progress_every", field, 100, minimum=1),
    )
    _require(sum((result.random_weight, result.greedy_weight, result.opponent_weight,
                  result.search_weight)) > 0, f"{field} requires a positive policy weight")
    _require(result.shards <= result.trajectories,
             f"{field}.shards must not exceed trajectories")
    return result


def _parse_labels(value: Any, field: str) -> LabelConfig:
    table = _table(value, field)
    _unknown(table, {"mode", "nodes", "hash_mb", "workers", "progress_every",
                     "require_all"}, field)
    mode = _string(table, "mode", field, "teacher")
    _require(mode in {"exact", "teacher"}, f"{field}.mode must be exact or teacher")
    return LabelConfig(
        mode=mode, nodes=_integer(table, "nodes", field, minimum=1),
        hash_mb=_integer(table, "hash_mb", field, 16, minimum=1),
        workers=_integer(table, "workers", field, 6, minimum=1),
        progress_every=_integer(table, "progress_every", field, 100, minimum=1),
        require_all=_boolean(table, "require_all", field, True),
    )


def _parse_features(value: Any, field: str) -> FeatureConfig:
    table = _table(value, field)
    _unknown(table, {"gain_samples"}, field)
    return FeatureConfig(gain_samples=_integer(table, "gain_samples", field, 256))


def _parse_training(value: Any, field: str) -> TrainingConfig:
    table = _table(value, field)
    kind = _string(table, "type", field)
    if kind == "baseline":
        _unknown(table, {"type", "device", "seed", "close_margin", "ridge_lambdas"}, field)
        device = _string(table, "device", field, "auto")
        _require(device in {"auto", "cpu", "cuda"},
                 f"{field}.device must be auto, cpu, or cuda")
        raw_lambdas = table.get("ridge_lambdas", list(DEFAULT_RIDGE_LAMBDAS))
        _require(isinstance(raw_lambdas, list) and raw_lambdas and
                 all(isinstance(item, (int, float)) and not isinstance(item, bool) and
                     math.isfinite(float(item)) and float(item) > 0.0 for item in raw_lambdas),
                 f"{field}.ridge_lambdas must be positive finite numbers")
        return BaselineTrainingConfig(
            type="baseline", device=device,
            seed=_integer(table, "seed", field, 20260818),
            close_margin=_integer(table, "close_margin", field, 8),
            ridge_lambdas=tuple(float(item) for item in raw_lambdas),
        )
    if kind == "pattern":
        _unknown(table, {"type", "device", "seed", "suite"}, field)
        device = _string(table, "device", field, "cuda")
        suite = _string(table, "suite", field, "default")
        _require(device in {"cpu", "cuda"}, f"{field}.device must be cpu or cuda")
        _require(suite in SUITE_NAMES,
                 f"{field}.suite must be one of {', '.join(SUITE_NAMES)}")
        return PatternTrainingConfig(
            type="pattern", device=device,
            seed=_integer(table, "seed", field, 20260818), suite=suite,
        )
    raise WorkflowConfigError(f"{field}.type must be baseline or pattern")


def _parse_sealed(value: Any, field: str) -> SealedEvaluationConfig:
    table = _table(value, field)
    _unknown(table, {"exclude_datasets"}, field)
    return SealedEvaluationConfig(
        exclude_datasets=_string_list(table, "exclude_datasets", field, []))


def _parse_candidate(repository: Path, value: Any, field: str) -> CandidateValidationConfig:
    table = _table(value, field)
    _unknown(table, {"promotable", "samples", "symmetry_samples", "benchmark_iterations",
                     "opening_book", "search_positions", "search_movetime_ms"}, field)
    opening = _repository_path(
        repository, _string(table, "opening_book", field, "eval/openings/development.txt"),
        f"{field}.opening_book")
    _require(opening.is_file(), f"{field}.opening_book is not a file: {opening}")
    return CandidateValidationConfig(
        promotable=_boolean(table, "promotable", field, False),
        samples=_integer(table, "samples", field, 4096, minimum=1),
        symmetry_samples=_integer(table, "symmetry_samples", field, 512, minimum=1),
        benchmark_iterations=_integer(
            table, "benchmark_iterations", field, 100, minimum=1),
        opening_book=opening,
        search_positions=_integer(table, "search_positions", field, 32, minimum=1),
        search_movetime_ms=_integer(table, "search_movetime_ms", field, 100, minimum=1),
    )


def _parse_gate(repository: Path, value: Any, field: str) -> EngineGateConfig:
    table = _table(value, field)
    allowed = {"base", "new_engine", "base_engine", "new_engine_args", "base_engine_args",
               "opening_book", "games", "workers", "timeout_ms", "go_movetime_ms",
               "opening_seed", "sequential_stop", "sequential_null", "sequential_alt",
               "sequential_alpha", "sequential_beta", "require_accept_alt"}
    _unknown(table, allowed, field)
    seed = table.get("opening_seed")
    if seed is not None:
        _require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
                 f"{field}.opening_seed must be a nonnegative integer")
    alpha = _number(table, "sequential_alpha", field, 0.05)
    beta = _number(table, "sequential_beta", field, 0.05)
    _require(0.0 < alpha < 1.0 and 0.0 < beta < 1.0,
             f"{field} sequential probabilities must be between zero and one")
    sequential_null = _number(table, "sequential_null", field, 0.0)
    sequential_alt = _number(table, "sequential_alt", field, 20.0)
    _require(sequential_alt > sequential_null,
             f"{field}.sequential_alt must be greater than sequential_null")
    opening_book = _repository_path(
        repository, _string(table, "opening_book", field, "eval/openings/holdout.txt"),
        f"{field}.opening_book")
    _require(opening_book.is_file(),
             f"{field}.opening_book is not a file: {opening_book}")
    new_engine = _binary_name(
        _string(table, "new_engine", field, CANDIDATE_ENGINE),
        f"{field}.new_engine")
    new_engine_args = _string(
        table, "new_engine_args", field, CANDIDATE_ENGINE_ARGS, nonempty=False)
    _require(new_engine == CANDIDATE_ENGINE,
             f"{field}.new_engine must be {CANDIDATE_ENGINE!r} so the gate runs "
             "the candidate model")
    _require(new_engine_args == CANDIDATE_ENGINE_ARGS,
             f"{field}.new_engine_args must be {CANDIDATE_ENGINE_ARGS!r} so the gate "
             "runs the candidate evaluator")
    return EngineGateConfig(
        base=_string(table, "base", field),
        new_engine=new_engine,
        base_engine=_binary_name(
            _string(table, "base_engine", field, "minimax/poe2_minimax"),
            f"{field}.base_engine"),
        new_engine_args=new_engine_args,
        base_engine_args=_string(table, "base_engine_args", field,
                                 "--evaluator pattern-gain", nonempty=False),
        opening_book=opening_book,
        games=_integer(table, "games", field, 2000, minimum=2),
        workers=_integer(table, "workers", field, 1, minimum=1),
        timeout_ms=_integer(table, "timeout_ms", field, 1000, minimum=1),
        go_movetime_ms=_integer(table, "go_movetime_ms", field, 100, minimum=1),
        opening_seed=seed,
        sequential_stop=_boolean(table, "sequential_stop", field, True),
        sequential_null=sequential_null,
        sequential_alt=sequential_alt,
        sequential_alpha=alpha, sequential_beta=beta,
        require_accept_alt=_boolean(table, "require_accept_alt", field, True),
    )


def load_workflow_config(path: Path | str) -> WorkflowConfig:
    """Load one strict workflow definition without creating or modifying run state."""
    config_path = Path(path).resolve()
    try:
        contents = config_path.read_bytes()
        document = tomllib.loads(contents.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise WorkflowConfigError(f"could not read workflow config {config_path}: {error}") from error
    repository = find_repository_root(config_path.parent)
    _require(repository is not None, f"config is not inside a Git repository: {config_path}")
    _unknown(document, {"schema_version", "run", "datasets", "iterations"}, "config")
    _require(document.get("schema_version") == SCHEMA_VERSION,
             f"config.schema_version must be {SCHEMA_VERSION}")

    run = _table(document.get("run"), "config.run")
    _unknown(run, {"name", "output_dir", "build_preset", "label_concurrency"}, "config.run")
    run_name = _name(_string(run, "name", "config.run"), "config.run.name")
    output = _repository_path(
        repository, _string(run, "output_dir", "config.run",
                            f"build/training-runs/{run_name}"), "config.run.output_dir")
    _require(output != config_path and output not in config_path.parents,
             "config.run.output_dir must not contain the workflow config")
    preset = _name(_string(run, "build_preset", "config.run", "release"),
                   "config.run.build_preset")
    try:
        presets_document = json.loads((repository / "CMakePresets.json").read_bytes())
        configure_presets = {
            item.get("name") for item in presets_document.get("configurePresets", [])
            if isinstance(item, dict)}
        build_presets = {
            item.get("name") for item in presets_document.get("buildPresets", [])
            if isinstance(item, dict)}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError) as error:
        raise WorkflowConfigError(
            f"could not read repository CMake presets: {error}") from error
    _require(preset in configure_presets and preset in build_presets,
             f"config.run.build_preset is not a repository configure/build preset: {preset}")
    label_concurrency = (None if "label_concurrency" not in run else
                         _integer(run, "label_concurrency", "config.run", minimum=1))

    dataset_values = _table(document.get("datasets"), "config.datasets")
    _require(bool(dataset_values), "config.datasets must contain at least one named dataset")
    datasets: list[DatasetConfig] = []
    for raw_name, raw_value in dataset_values.items():
        dataset_name = _name(raw_name, f"config.datasets.{raw_name}")
        field = f"config.datasets.{dataset_name}"
        table = _table(raw_value, field)
        kind = _string(table, "kind", field)
        if kind == "managed":
            _unknown(table, {"kind", "source", "labels", "features"}, field)
            datasets.append(ManagedDatasetConfig(
                name=dataset_name, kind="managed", root=output / "datasets" / dataset_name,
                source=_parse_source(_value(table, "source", field), f"{field}.source"),
                labels=_parse_labels(_value(table, "labels", field), f"{field}.labels"),
                features=_parse_features(table.get("features", {}), f"{field}.features"),
            ))
        elif kind == "imported":
            _unknown(table, {"kind", "path"}, field)
            feature_path = _repository_path(
                repository, _string(table, "path", field), f"{field}.path")
            _require(output not in feature_path.parents and feature_path not in output.parents and
                     feature_path != output,
                     f"{field}.path must not overlap the managed run directory")
            datasets.append(ImportedDatasetConfig(
                name=dataset_name, kind="imported", feature_directory=feature_path))
        else:
            raise WorkflowConfigError(f"{field}.kind must be managed or imported")

    raw_iterations = document.get("iterations")
    _require(isinstance(raw_iterations, list) and raw_iterations and
             all(isinstance(item, dict) for item in raw_iterations),
             "config.iterations must be a nonempty array of tables")
    dataset_names = {dataset.name for dataset in datasets}
    iterations: list[IterationConfig] = []
    seen: set[str] = set()
    for index, table in enumerate(raw_iterations):
        field = f"config.iterations[{index}]"
        _unknown(table, {"name", "dataset", "depends_on", "training", "sealed_evaluation",
                         "candidate_validation", "engine_gate"}, field)
        name = _name(_string(table, "name", field), f"{field}.name")
        _require(name not in seen, f"iteration name is repeated: {name}")
        dataset_name = _string(table, "dataset", field)
        _require(dataset_name in dataset_names,
                 f"{field}.dataset names unknown dataset: {dataset_name}")
        dependencies = _string_list(table, "depends_on", field, [])
        _require(all(item in seen for item in dependencies),
                 f"{field}.depends_on may only name earlier iterations")
        training = _parse_training(_value(table, "training", field), f"{field}.training")
        sealed = (_parse_sealed(table["sealed_evaluation"], f"{field}.sealed_evaluation")
                  if "sealed_evaluation" in table else None)
        candidate = (_parse_candidate(repository, table["candidate_validation"],
                                      f"{field}.candidate_validation")
                     if "candidate_validation" in table else None)
        gate = (_parse_gate(repository, table["engine_gate"], f"{field}.engine_gate")
                if "engine_gate" in table else None)
        if sealed is not None:
            _require(isinstance(training, PatternTrainingConfig),
                     f"{field}.sealed_evaluation requires pattern training")
            _require(all(item in dataset_names for item in sealed.exclude_datasets),
                     f"{field}.sealed_evaluation names an unknown exclusion dataset")
            _require(dataset_name not in sealed.exclude_datasets,
                     f"{field}.sealed_evaluation cannot exclude its own dataset")
        if candidate is not None:
            _require(isinstance(training, PatternTrainingConfig) and
                     training.suite == "frozen-pattern-gain",
                     f"{field}.candidate_validation requires frozen-pattern-gain training")
            _require(output != candidate.opening_book and
                     output not in candidate.opening_book.parents,
                     f"{field}.candidate_validation.opening_book must be outside the run root")
        if gate is not None:
            _require(candidate is not None,
                     f"{field}.engine_gate requires candidate_validation")
            _require(candidate.promotable,
                     f"{field}.engine_gate requires a promotable candidate")
            _require(gate.games % 2 == 0, f"{field}.engine_gate.games must be even")
            _require(output != gate.opening_book and output not in gate.opening_book.parents,
                     f"{field}.engine_gate.opening_book must be outside the run root")
        iterations.append(IterationConfig(
            name=name, dataset=dataset_name, depends_on=dependencies,
            root=output / "iterations" / name, training=training,
            sealed_evaluation=sealed, candidate_validation=candidate, engine_gate=gate,
        ))
        seen.add(name)

    referenced_datasets = {iteration.dataset for iteration in iterations}
    referenced_datasets.update(
        name for iteration in iterations
        if iteration.sealed_evaluation is not None
        for name in iteration.sealed_evaluation.exclude_datasets)
    unreferenced = sorted(dataset_names - referenced_datasets)
    _require(not unreferenced,
             "config.datasets contains unreferenced dataset(s): " + ", ".join(unreferenced))

    return WorkflowConfig(
        path=config_path, contents=contents, digest=hashlib.sha256(contents).hexdigest(),
        repository=repository, name=run_name, output_directory=output,
        build_preset=preset, label_concurrency=label_concurrency,
        datasets=tuple(datasets), iterations=tuple(iterations),
    )


def _normalized(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _normalized(asdict(value))
    if isinstance(value, dict):
        return {str(key): _normalized(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_normalized(item) for item in value]
    return value


def fingerprint(value: Any) -> str:
    encoded = json.dumps(_normalized(value), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def stage_definitions(config: WorkflowConfig) -> dict[str, dict[str, Any]]:
    """Return normalized settings and upstream config fingerprints for every stage."""
    result: dict[str, dict[str, Any]] = {}
    for dataset in config.datasets:
        if isinstance(dataset, ManagedDatasetConfig):
            source_id = f"dataset:{dataset.name}:source"
            result[source_id] = {"adapter": "source", "preset": config.build_preset,
                                 "settings": dataset.source}
            labels_id = f"dataset:{dataset.name}:labels"
            result[labels_id] = {
                "adapter": "labels", "preset": config.build_preset,
                "source": fingerprint(result[source_id]), "settings": dataset.labels,
                "label_concurrency": config.label_concurrency,
            }
            features_id = f"dataset:{dataset.name}:features"
            result[features_id] = {
                "adapter": "features", "preset": config.build_preset,
                "source": fingerprint(result[source_id]), "labels": fingerprint(result[labels_id]),
                "settings": dataset.features,
            }
        else:
            result[f"dataset:{dataset.name}:features"] = {
                "adapter": "imported-features", "path": dataset.feature_directory,
            }
    for iteration in config.iterations:
        training_id = f"iteration:{iteration.name}:training"
        dataset_id = f"dataset:{iteration.dataset}:features"
        result[training_id] = {
            "adapter": f"training-{iteration.training.type}",
            "dataset": fingerprint(result[dataset_id]), "settings": iteration.training,
            "dependencies": [fingerprint(result[f"iteration:{name}:training"])
                             for name in iteration.depends_on],
        }
        if iteration.sealed_evaluation is not None:
            stage_id = f"iteration:{iteration.name}:sealed-evaluation"
            result[stage_id] = {
                "adapter": "sealed-evaluation", "training": fingerprint(result[training_id]),
                "dataset": fingerprint(result[dataset_id]),
                "settings": iteration.sealed_evaluation,
                "exclusions": [fingerprint(result[f"dataset:{name}:features"])
                               for name in iteration.sealed_evaluation.exclude_datasets],
            }
        if iteration.candidate_validation is not None:
            stage_id = f"iteration:{iteration.name}:candidate-validation"
            result[stage_id] = {
                "adapter": "candidate-validation", "preset": config.build_preset,
                "training": fingerprint(result[training_id]),
                "dataset": fingerprint(result[dataset_id]),
                "sealed_evaluation": (
                    fingerprint(result[f"iteration:{iteration.name}:sealed-evaluation"])
                    if iteration.sealed_evaluation is not None else None),
                "settings": iteration.candidate_validation,
            }
        if iteration.engine_gate is not None:
            stage_id = f"iteration:{iteration.name}:engine-gate"
            result[stage_id] = {
                "adapter": "engine-gate", "preset": config.build_preset,
                "candidate": fingerprint(result[
                    f"iteration:{iteration.name}:candidate-validation"]),
                "settings": iteration.engine_gate,
            }
    return result


def stage_fingerprints(config: WorkflowConfig) -> dict[str, str]:
    return {stage: fingerprint(definition)
            for stage, definition in stage_definitions(config).items()}
