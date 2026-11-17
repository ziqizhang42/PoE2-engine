"""Strict typed TOML configuration for consolidated training runs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Literal

from .model_metrics import DEFAULT_RIDGE_LAMBDAS
from .net_file import FILE_SUFFIX, NetFileError, read_net
from .pattern_suites import SUITE_NAMES
from .shared import find_repository_root, sha256_file


SCHEMA_VERSION = 1
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CANDIDATE_ENGINE = "minimax/poe2_minimax"
CANDIDATE_ENGINE_ARGS = "--evaluator pattern-gain"
MODEL_ARCHIVE_DIRECTORY = Path("models")
MODEL_ID_PATTERN = re.compile(r"^candidate-sha256-([0-9a-f]{64})$")


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
class ModelReference:
    """A recipe-local name for a model archived under models/ or trained by this recipe."""

    name: str
    model_id: str | None
    # The C++ header of a model trained before nets were files; it no longer drives any build.
    header: Path | None
    # Set for the model an iteration of this recipe trains. Its ID and net are those of the
    # iteration's authenticated candidate, filled in by the workflow once its gate accepts it.
    iteration: str | None = None
    # The net file engines and data tools load for this model, its SHA-256, and its name.
    net: Path | None = None
    net_sha256: str | None = None
    net_name: str | None = None

    @property
    def sha256(self) -> str:
        if self.model_id is None:
            raise WorkflowConfigError(
                f"model {self.name} is not available until iteration {self.iteration}'s "
                "engine gate accepts it")
        return self.model_id.removeprefix("candidate-sha256-")


@dataclass(frozen=True)
class ManagedDatasetConfig:
    name: str
    kind: Literal["managed"]
    root: Path
    source: SourceConfig
    labels: LabelConfig
    features: FeatureConfig
    # The model that drives noisy-search sampling and labels; None means two-ply closure.
    teacher: ModelReference | None = None
    # Set for a dataset that relabels another dataset's positions: it has no source stage and
    # keeps that source's settings and splits.
    positions: str | None = None
    positions_root: Path | None = None

    @property
    def taught_in_run(self) -> bool:
        return self.teacher is not None and self.teacher.iteration is not None

    @property
    def source_directory(self) -> Path:
        return (self.positions_root or self.root) / "source"

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
class TrainingDataset:
    """Another dataset whose train records a pattern iteration also learns from."""

    dataset: str
    weight: float = 1.0


@dataclass(frozen=True)
class PatternTrainingConfig:
    type: Literal["pattern"]
    device: Literal["cpu", "cuda"] = "cuda"
    seed: int = 20260818
    suite: str = "default"
    # Newest first: the order decides which copy of a repeated position is kept.
    extra_datasets: tuple[TrainingDataset, ...] = ()
    max_records: int | None = None
    piece_records: int | None = None
    # "lookup" sums the weights of each record's patterns instead of expanding dense counts.
    method: str = "dense"

    @property
    def uses_training_data_options(self) -> bool:
        return bool(self.extra_datasets) or self.max_records is not None or (
            self.piece_records is not None)


TRAINING_DATA_OPTIONS = ("extra_datasets", "max_records", "piece_records")


def recorded_training_settings(training: TrainingConfig) -> dict[str, Any]:
    """Return training settings as fingerprinted, leaving out unused options and the default method.

    Stages recorded before these options existed therefore keep their fingerprints.
    """
    settings = asdict(training)
    for name in TRAINING_DATA_OPTIONS:
        if name in settings and not settings[name]:
            del settings[name]
    if settings.get("method") == "dense":
        del settings["method"]
    return settings


TrainingConfig = BaselineTrainingConfig | PatternTrainingConfig


def extra_training_datasets(config: WorkflowConfig,
                            iteration: IterationConfig) -> tuple[DatasetConfig, ...]:
    training = iteration.training
    if not isinstance(training, PatternTrainingConfig):
        return ()
    return tuple(config.dataset(item.dataset) for item in training.extra_datasets)


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
class MatchConfig:
    """Settings of one timed match between two engines: a gate or a comparison."""

    new_engine: str = CANDIDATE_ENGINE
    base_engine: str = CANDIDATE_ENGINE
    new_engine_args: str = CANDIDATE_ENGINE_ARGS
    base_engine_args: str = CANDIDATE_ENGINE_ARGS
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


MATCH_FIELDS = frozenset(item.name for item in fields(MatchConfig))


def match_settings(settings: MatchConfig) -> dict[str, Any]:
    return {name: value for name, value in asdict(settings).items() if name in MATCH_FIELDS}


@dataclass(frozen=True, kw_only=True)
class EngineGateConfig(MatchConfig):
    # Exactly one of base (an existing build), base_model (an archived model ID), and
    # base_iteration (an earlier iteration whose accepted candidate is the base) is set.
    base: str | None
    base_model: str | None = None
    base_iteration: str | None = None


@dataclass(frozen=True, kw_only=True)
class ComparisonConfig(MatchConfig):
    """A match between two [models] entries, apart from any iteration's gate."""

    name: str
    new: str
    base: str
    root: Path


def recorded_gate_settings(gate: EngineGateConfig) -> dict[str, Any]:
    """Return a gate's settings as its report records them and its fingerprint covers them.

    base_iteration appears only in gates that use it, so gates from before it existed keep
    their recorded settings and fingerprints.
    """
    settings = asdict(gate)
    if gate.base_iteration is None:
        del settings["base_iteration"]
    return settings


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
    models: tuple[ModelReference, ...] = ()
    comparisons: tuple[ComparisonConfig, ...] = ()

    def recorded_path(self, path: Path) -> str:
        """Return how fingerprints and records name a path: relative to the repository if inside."""
        try:
            return path.relative_to(self.repository).as_posix()
        except ValueError:
            return str(path)

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

    def comparison(self, name: str) -> ComparisonConfig:
        for comparison in self.comparisons:
            if comparison.name == name:
                return comparison
        raise WorkflowConfigError(f"unknown comparison: {name}")

    def model(self, name: str) -> ModelReference:
        for model in self.models:
            if model.name == name:
                return model
        raise WorkflowConfigError(f"unknown model: {name}")

    def trained_model(self, iteration: str) -> ModelReference:
        """Return the [models] entry for the model that one iteration of this recipe trains."""
        for model in self.models:
            if model.iteration == iteration:
                return model
        raise WorkflowConfigError(f"no config.models entry names iteration {iteration}'s model")


def datasets_used(config: WorkflowConfig, iteration: IterationConfig) -> tuple[str, ...]:
    """Return every dataset an iteration reads, each after the dataset its positions come from.

    This is what ordering, blocking, and stage definitions follow.
    """
    names = [iteration.dataset]
    names.extend(dataset.name for dataset in extra_training_datasets(config, iteration))
    if iteration.sealed_evaluation is not None:
        names.extend(iteration.sealed_evaluation.exclude_datasets)
    used: list[str] = []
    for name in names:
        dataset = config.dataset(name)
        positions = dataset.positions if isinstance(dataset, ManagedDatasetConfig) else None
        for item in (positions, name):
            if item is not None and item not in used:
                used.append(item)
    return tuple(used)


def trained_models_used(config: WorkflowConfig,
                        iteration: IterationConfig) -> tuple[ModelReference, ...]:
    """Return the models this recipe trains that an iteration's datasets or gate base use."""
    used: dict[str, ModelReference] = {}
    for name in datasets_used(config, iteration):
        dataset = config.dataset(name)
        if isinstance(dataset, ManagedDatasetConfig) and dataset.taught_in_run:
            used.setdefault(dataset.teacher.name, dataset.teacher)
    gate = iteration.engine_gate
    if gate is not None and gate.base_iteration is not None:
        model = config.trained_model(gate.base_iteration)
        used.setdefault(model.name, model)
    return tuple(used.values())


def with_trained_model(config: WorkflowConfig, model: ModelReference) -> WorkflowConfig:
    """Return the config with one trained model's ID and header filled in wherever it is named."""
    models = tuple(model if item.iteration == model.iteration else item for item in config.models)
    datasets = tuple(
        replace(item, teacher=model)
        if isinstance(item, ManagedDatasetConfig) and item.teacher is not None and
        item.teacher.iteration == model.iteration else item
        for item in config.datasets)
    return replace(config, models=models, datasets=datasets)


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


def _parse_models(repository: Path, value: Any) -> tuple[ModelReference, ...]:
    table = _table(value, "config.models")
    models: list[ModelReference] = []
    for raw_name, model_id in table.items():
        name = _name(raw_name, f"config.models.{raw_name}")
        field = f"config.models.{name}"
        if isinstance(model_id, dict):
            _unknown(model_id, {"iteration"}, field)
            iteration = _name(_string(model_id, "iteration", field), f"{field}.iteration")
            models.append(ModelReference(name=name, model_id=None, header=None,
                                         iteration=iteration))
            continue
        match = MODEL_ID_PATTERN.fullmatch(model_id) if isinstance(model_id, str) else None
        _require(match is not None, f"{field} must be an archived model ID: candidate-sha256-<digest>")
        directory = repository / MODEL_ARCHIVE_DIRECTORY
        net_path = directory / f"{model_id}{FILE_SUFFIX}"
        _require(net_path.is_file() and (directory / f"{model_id}.json").is_file(),
                 f"{field} is not archived under {MODEL_ARCHIVE_DIRECTORY}/: {model_id}")
        try:
            net = read_net(net_path)
        except NetFileError as error:
            raise WorkflowConfigError(f"{field} archived net is unusable: {error}") from error
        _require(net.identity_sha256 == match.group(1),
                 f"{field} archived net does not match its model ID: {net_path}")
        header: Path | None = directory / f"{model_id}.hpp"
        if not header.is_file():
            header = None
        else:
            _require(sha256_file(header) == match.group(1),
                     f"{field} archived header does not match its digest: {header}")
        models.append(ModelReference(name=name, model_id=model_id, header=header, net=net_path,
                                     net_sha256=net.sha256, net_name=net.name))
    return tuple(models)


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


def _parse_training_datasets(value: Any, field: str) -> tuple[TrainingDataset, ...]:
    _require(isinstance(value, list), f"{field} must be an array of tables")
    result: list[TrainingDataset] = []
    for index, item in enumerate(value):
        table = _table(item, f"{field}[{index}]")
        _unknown(table, {"dataset", "weight"}, f"{field}[{index}]")
        weight = _number(table, "weight", f"{field}[{index}]", 1.0)
        _require(weight > 0.0, f"{field}[{index}].weight must be positive")
        result.append(TrainingDataset(
            dataset=_string(table, "dataset", f"{field}[{index}]"), weight=weight))
    names = [item.dataset for item in result]
    _require(len(set(names)) == len(names), f"{field} names a dataset twice")
    return tuple(result)


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
        _unknown(table, {"type", "device", "seed", "suite", "method", *TRAINING_DATA_OPTIONS},
                 field)
        device = _string(table, "device", field, "cuda")
        suite = _string(table, "suite", field, "default")
        method = _string(table, "method", field, "dense")
        _require(device in {"cpu", "cuda"}, f"{field}.device must be cpu or cuda")
        _require(suite in SUITE_NAMES,
                 f"{field}.suite must be one of {', '.join(SUITE_NAMES)}")
        _require(method in {"dense", "lookup"}, f"{field}.method must be dense or lookup")
        _require(method == "dense" or "piece_records" not in table,
                 f"{field}.piece_records expands dense line counts, which lookup never builds")
        return PatternTrainingConfig(
            type="pattern", device=device,
            seed=_integer(table, "seed", field, 20260818), suite=suite,
            extra_datasets=_parse_training_datasets(table.get("extra_datasets", []),
                                                    f"{field}.extra_datasets"),
            max_records=(_integer(table, "max_records", field, minimum=1)
                         if "max_records" in table else None),
            piece_records=(_integer(table, "piece_records", field, minimum=1)
                           if "piece_records" in table else None),
            method=method,
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


def _parse_match(repository: Path, output: Path, table: dict[str, Any],
                 field: str) -> dict[str, Any]:
    """Return validated MatchConfig settings; the new side always runs the pattern-gain engine."""
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
    _require(output != opening_book and output not in opening_book.parents,
             f"{field}.opening_book must be outside the run root")
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
    games = _integer(table, "games", field, 2000, minimum=2)
    _require(games % 2 == 0, f"{field}.games must be even")
    return {
        "new_engine": new_engine,
        "base_engine": _binary_name(
            _string(table, "base_engine", field, CANDIDATE_ENGINE), f"{field}.base_engine"),
        "new_engine_args": new_engine_args,
        "base_engine_args": _string(table, "base_engine_args", field, CANDIDATE_ENGINE_ARGS,
                                    nonempty=False),
        "opening_book": opening_book,
        "games": games,
        "workers": _integer(table, "workers", field, 1, minimum=1),
        "timeout_ms": _integer(table, "timeout_ms", field, 1000, minimum=1),
        "go_movetime_ms": _integer(table, "go_movetime_ms", field, 100, minimum=1),
        "opening_seed": seed,
        "sequential_stop": _boolean(table, "sequential_stop", field, True),
        "sequential_null": sequential_null,
        "sequential_alt": sequential_alt,
        "sequential_alpha": alpha, "sequential_beta": beta,
    }


def _parse_gate(repository: Path, output: Path, value: Any, field: str,
                models: dict[str, ModelReference]) -> EngineGateConfig:
    table = _table(value, field)
    _unknown(table, {"base", "base_model", *MATCH_FIELDS}, field)
    _require(("base" in table) != ("base_model" in table),
             f"{field} must set exactly one of base and base_model")
    base_model = base_iteration = None
    if "base_model" in table:
        model_name = _string(table, "base_model", field)
        _require(model_name in models,
                 f"{field}.base_model must name an entry in config.models: {model_name}")
        base_model = models[model_name].model_id
        base_iteration = models[model_name].iteration
    return EngineGateConfig(
        base=_string(table, "base", field) if "base" in table else None,
        base_model=base_model, base_iteration=base_iteration,
        **_parse_match(repository, output, table, field),
    )


def _parse_comparison(repository: Path, output: Path, value: Any, field: str,
                      models: dict[str, ModelReference]) -> ComparisonConfig:
    table = _table(value, field)
    _unknown(table, {"name", "new", "base", *MATCH_FIELDS}, field)
    new, base = (_string(table, side, field) for side in ("new", "base"))
    for side, model in (("new", new), ("base", base)):
        _require(model in models, f"{field}.{side} must name an entry in config.models: {model}")
    _require(new != base and (models[new].model_id is None or
                              models[new].model_id != models[base].model_id),
             f"{field} compares {new} with the same model")
    for name, fixed in (("new_engine", CANDIDATE_ENGINE), ("base_engine", CANDIDATE_ENGINE),
                        ("new_engine_args", CANDIDATE_ENGINE_ARGS),
                        ("base_engine_args", CANDIDATE_ENGINE_ARGS)):
        _require(table.get(name, fixed) == fixed,
                 f"{field}.{name} must be {fixed!r}; a comparison plays both models with the "
                 "pattern-gain engine")
    name = _name(_string(table, "name", field, f"{new}-vs-{base}"), f"{field}.name")
    return ComparisonConfig(name=name, new=new, base=base, root=output / "comparisons" / name,
                            **_parse_match(repository, output, table, field))


LONG_FORM_KEYS = {"models", "datasets", "iterations"}
COMPACT_KEYS = {"defaults", "generations"}
GENERATION_SECTIONS = ("source", "labels", "features", "training", "sealed_evaluation",
                       "candidate_validation", "engine_gate")
# Settings each trained generation derives from its own entry, so shared defaults cannot set
# them. A generation may override its corpus ID and seeds, but its gate base is its parent.
DERIVED_SETTINGS = {"source": {"corpus_id", "seed"}, "training": {"seed"},
                    "engine_gate": {"base", "base_model"}}


def _generation_sections(defaults: dict[str, Any], generation: dict[str, Any],
                         field: str) -> dict[str, dict[str, Any]]:
    """Merge a generation's own tables key by key over the shared defaults."""
    sections: dict[str, dict[str, Any]] = {}
    for section in GENERATION_SECTIONS:
        if section not in defaults and section not in generation:
            continue
        own = _table(generation.get(section, {}), f"{field}.{section}")
        if section == "engine_gate":
            base = sorted(DERIVED_SETTINGS[section] & set(own))
            _require(not base, f"{field}.engine_gate cannot set {', '.join(base)}; "
                     "a generation's gate base is its parent")
        sections[section] = {**defaults.get(section, {}), **own}
    for section in ("candidate_validation", "engine_gate"):
        _require(section in sections,
                 f"{field} needs [defaults.{section}] or its own [generations.{section}]")
    return sections


# Compact training settings that choose a generation's data; they expand into extra_datasets.
GENERATION_DATA_SETTINGS = {"window", "age_weight", "refresh", "extra"}


def _imported_datasets(document: dict[str, Any]) -> dict[str, Any]:
    """Return a compact recipe's datasets, which may only be imported ones."""
    datasets = _table(document.get("datasets", {}), "config.datasets")
    for name, table in datasets.items():
        _require(_table(table, f"config.datasets.{name}").get("kind") == "imported",
                 f"config.datasets.{name} must be imported; generations make managed datasets")
    return dict(datasets)


def _generation_data(name: str, parent: str, training: dict[str, Any],
                     dataset: dict[str, Any], lineage: dict[str, str | None],
                     available: set[str], field: str) -> tuple[list[Any], dict[str, Any]]:
    """Expand window, age_weight, refresh, and extra into extra datasets, newest first.

    The window holds the datasets of up to window - 1 trained ancestors; a pinned ancestor
    ends it. The refresh newest of those are replaced by copies of their positions labeled by
    this generation's parent, the teacher of its own dataset.
    """
    window = _integer(training, "window", f"{field}.training", 1, minimum=1)
    refresh = _integer(training, "refresh", f"{field}.training", 0, maximum=window - 1)
    age_weight = _number(training, "age_weight", f"{field}.training", 1.0)
    _require(age_weight > 0.0, f"{field}.training.age_weight must be positive")
    extra = training.get("extra", [])
    _require(isinstance(extra, list), f"{field}.training.extra must be an array of tables")
    ancestors: list[str] = []
    ancestor = parent
    while len(ancestors) < window - 1 and lineage.get(ancestor) is not None:
        ancestors.append(ancestor)
        ancestor = lineage[ancestor]
    extras: list[Any] = []
    relabeled: dict[str, Any] = {}
    for age, ancestor in enumerate(ancestors, 1):
        weight = age_weight ** age
        if age <= refresh:
            copy = f"{name}-refresh-{ancestor}"
            relabeled[copy] = {"kind": "managed", "teacher": parent, "positions": ancestor,
                               **{key: dataset[key] for key in ("labels", "features")
                                  if key in dataset}}
            extras.append({"dataset": copy, "weight": weight})
        else:
            extras.append({"dataset": ancestor, "weight": weight})
    for index, item in enumerate(extra):
        entry = _table(item, f"{field}.training.extra[{index}]")
        _require(entry.get("dataset") in available,
                 f"{field}.training.extra[{index}].dataset must name a trained generation or "
                 f"an imported dataset: {entry.get('dataset')}")
        extras.append(entry)
    return extras, relabeled


def expand_generations(document: dict[str, Any]) -> dict[str, Any]:
    """Return the long-form document that a compact [[generations]] recipe stands for."""
    mixed = sorted((LONG_FORM_KEYS - {"datasets"}) & set(document))
    _require(not mixed,
             "a recipe is either long-form ([models], [datasets], [[iterations]]) or compact "
             f"([defaults], [[generations]]); this one also sets {', '.join(mixed)}")
    _unknown(document, {"schema_version", "run", "comparisons", "datasets", *COMPACT_KEYS},
             "config")
    run = _table(document.get("run"), "config.run")
    run_name = _string(run, "name", "config.run")
    defaults = _table(document.get("defaults", {}), "config.defaults")
    _unknown(defaults, set(GENERATION_SECTIONS), "config.defaults")
    for section, table in defaults.items():
        shared = sorted(DERIVED_SETTINGS.get(section, set()) &
                        set(_table(table, f"config.defaults.{section}")))
        _require(not shared, f"config.defaults.{section} cannot set {', '.join(shared)}; "
                 "each generation derives it")
    generations = document.get("generations")
    _require(isinstance(generations, list) and generations and
             all(isinstance(item, dict) for item in generations),
             "config.generations must be a nonempty array of tables")
    models: dict[str, Any] = {}
    datasets: dict[str, Any] = _imported_datasets(document)
    imported = set(datasets)
    # Each generation's parent, or None for a pinned generation, which has no dataset.
    lineage: dict[str, str | None] = {}
    iterations: list[dict[str, Any]] = []
    for index, generation in enumerate(generations):
        field = f"config.generations[{index}]"
        name = _string(generation, "name", field)
        _require(name not in models, f"generation name is repeated: {name}")
        _require(name not in datasets, f"{field}.name is already a dataset name: {name}")
        if "model" in generation:
            _unknown(generation, {"name", "model"}, field)
            models[name] = _string(generation, "model", field)
            lineage[name] = None
            continue
        _unknown(generation, {"name", "parent", "seed", *GENERATION_SECTIONS}, field)
        parent = _string(generation, "parent", field)
        _require(parent in models, f"{field}.parent must name an earlier generation: {parent}")
        seed = _integer(generation, "seed", field)
        sections = _generation_sections(defaults, generation, field)
        dataset: dict[str, Any] = {
            "kind": "managed", "teacher": parent,
            "source": {"corpus_id": f"poe2-{run_name}-{name}-v1", "seed": seed,
                       **sections.pop("source", {})},
        }
        for section in ("labels", "features"):
            if section in sections:
                dataset[section] = sections.pop(section)
        training = sections.pop("training", {})
        extras, relabeled = _generation_data(
            name, parent, training, dataset, lineage,
            {item for item, owner in lineage.items() if owner is not None} | imported, field)
        training = {key: value for key, value in training.items()
                    if key not in GENERATION_DATA_SETTINGS}
        if extras:
            training["extra_datasets"] = extras
        iteration = {"name": name, "dataset": name, "training": {"seed": seed, **training},
                     **sections}
        iteration["engine_gate"] = {**iteration["engine_gate"], "base_model": parent}
        models[name] = {"iteration": name}
        lineage[name] = parent
        datasets[name] = dataset
        for copy, value in relabeled.items():
            _require(copy not in datasets, f"{field} relabels into an existing name: {copy}")
            datasets[copy] = value
        iterations.append(iteration)
    _require(bool(iterations), "config.generations must include a generation with a parent")
    return {"schema_version": document.get("schema_version"), "run": run,
            "models": models, "datasets": datasets, "iterations": iterations,
            "comparisons": _generation_comparisons(document, defaults, models)}


def _generation_comparisons(document: dict[str, Any], defaults: dict[str, Any],
                            generations: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand compact comparisons, which play like the gates unless an entry overrides them."""
    entries = document.get("comparisons", [])
    _require(isinstance(entries, list) and all(isinstance(item, dict) for item in entries),
             "config.comparisons must be an array of tables")
    for index, entry in enumerate(entries):
        for side in ("new", "base"):
            name = _string(entry, side, f"config.comparisons[{index}]")
            _require(name in generations,
                     f"config.comparisons[{index}].{side} must name a generation: {name}")
    return [{**defaults.get("engine_gate", {}), **entry} for entry in entries]


def _with_positions(datasets: list[DatasetConfig]) -> list[DatasetConfig]:
    """Give each relabeling dataset the source settings and location of its positions."""
    by_name = {dataset.name: dataset for dataset in datasets}
    result: list[DatasetConfig] = []
    for dataset in datasets:
        if isinstance(dataset, ManagedDatasetConfig) and dataset.positions is not None:
            origin = by_name.get(dataset.positions)
            _require(isinstance(origin, ManagedDatasetConfig) and origin.positions is None and
                     origin.name != dataset.name,
                     f"config.datasets.{dataset.name}.positions must name another managed "
                     f"dataset with its own source: {dataset.positions}")
            dataset = replace(dataset, source=origin.source, positions_root=origin.root)
        result.append(dataset)
    return result


def _check_trained_models(config: WorkflowConfig) -> None:
    """Require each trained model to come from a gated iteration that precedes every use."""
    order = {iteration.name: index for index, iteration in enumerate(config.iterations)}
    named: set[str] = set()
    for model in config.models:
        if model.iteration is None:
            continue
        field = f"config.models.{model.name}"
        _require(model.iteration in order,
                 f"{field} names unknown iteration {model.iteration}")
        _require(model.iteration not in named,
                 f"{field} names iteration {model.iteration}, which another entry already names")
        named.add(model.iteration)
        _require(model.name == model.iteration or model.name not in order,
                 f"{field} shares its name with iteration {model.name} but names the model of "
                 f"iteration {model.iteration}")
        _require(config.iteration(model.iteration).engine_gate is not None,
                 f"{field} names iteration {model.iteration}, which has no engine gate to "
                 "accept its model")
    for index, iteration in enumerate(config.iterations):
        for model in trained_models_used(config, iteration):
            _require(order[model.iteration] < index,
                     f"iteration {iteration.name} uses model {model.name}, which iteration "
                     f"{model.iteration} trains, so it must come after {model.iteration}")


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
    if COMPACT_KEYS & set(document):
        document = expand_generations(document)
    _unknown(document, {"schema_version", "run", "models", "datasets", "iterations",
                        "comparisons"}, "config")
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

    models = _parse_models(repository, document.get("models", {}))
    models_by_name = {model.name: model for model in models}

    dataset_values = _table(document.get("datasets"), "config.datasets")
    _require(bool(dataset_values), "config.datasets must contain at least one named dataset")
    datasets: list[DatasetConfig] = []
    for raw_name, raw_value in dataset_values.items():
        dataset_name = _name(raw_name, f"config.datasets.{raw_name}")
        field = f"config.datasets.{dataset_name}"
        table = _table(raw_value, field)
        kind = _string(table, "kind", field)
        if kind == "managed":
            _unknown(table, {"kind", "teacher", "source", "positions", "labels", "features"},
                     field)
            teacher = None
            if "teacher" in table:
                teacher_name = _string(table, "teacher", field)
                _require(teacher_name in models_by_name,
                         f"{field}.teacher must name an entry in config.models: {teacher_name}")
                teacher = models_by_name[teacher_name]
            _require(("source" in table) != ("positions" in table),
                     f"{field} must set exactly one of source and positions")
            positions = (_name(_string(table, "positions", field), f"{field}.positions")
                         if "positions" in table else None)
            datasets.append(ManagedDatasetConfig(
                name=dataset_name, kind="managed", root=output / "datasets" / dataset_name,
                # A relabeling dataset takes its source settings from its positions dataset.
                source=(_parse_source(table["source"], f"{field}.source")
                        if positions is None else None),
                labels=_parse_labels(_value(table, "labels", field), f"{field}.labels"),
                features=_parse_features(table.get("features", {}), f"{field}.features"),
                teacher=teacher, positions=positions,
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
    datasets = _with_positions(datasets)

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
        if isinstance(training, PatternTrainingConfig):
            for item in training.extra_datasets:
                _require(item.dataset in dataset_names and item.dataset != dataset_name,
                         f"{field}.training.extra_datasets must name other datasets: "
                         f"{item.dataset}")
        sealed = (_parse_sealed(table["sealed_evaluation"], f"{field}.sealed_evaluation")
                  if "sealed_evaluation" in table else None)
        candidate = (_parse_candidate(repository, table["candidate_validation"],
                                      f"{field}.candidate_validation")
                     if "candidate_validation" in table else None)
        gate = (_parse_gate(repository, output, table["engine_gate"], f"{field}.engine_gate",
                            models_by_name)
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
        iterations.append(IterationConfig(
            name=name, dataset=dataset_name, depends_on=dependencies,
            root=output / "iterations" / name, training=training,
            sealed_evaluation=sealed, candidate_validation=candidate, engine_gate=gate,
        ))
        seen.add(name)

    for model in models:
        named = next((item for item in iterations if item.name == model.name), None)
        _require(named is None or named.candidate_validation is not None,
                 f"config.models.{model.name} shares its name with iteration {model.name}, "
                 "which validates no candidate")

    referenced_datasets = {iteration.dataset for iteration in iterations}
    referenced_datasets.update(
        name for iteration in iterations
        if iteration.sealed_evaluation is not None
        for name in iteration.sealed_evaluation.exclude_datasets)
    referenced_datasets.update(
        item.dataset for iteration in iterations
        if isinstance(iteration.training, PatternTrainingConfig)
        for item in iteration.training.extra_datasets)
    referenced_datasets.update(
        dataset.positions for dataset in datasets
        if isinstance(dataset, ManagedDatasetConfig) and dataset.positions is not None)
    unreferenced = sorted(dataset_names - referenced_datasets)
    _require(not unreferenced,
             "config.datasets contains unreferenced dataset(s): " + ", ".join(unreferenced))

    raw_comparisons = document.get("comparisons", [])
    _require(isinstance(raw_comparisons, list),
             "config.comparisons must be an array of tables")
    comparisons = [_parse_comparison(repository, output, value, f"config.comparisons[{index}]",
                                     models_by_name)
                   for index, value in enumerate(raw_comparisons)]
    names = [comparison.name for comparison in comparisons]
    repeated = sorted({name for name in names if names.count(name) > 1})
    _require(not repeated, f"comparison name is repeated: {', '.join(repeated)}")

    config = WorkflowConfig(
        path=config_path, contents=contents, digest=hashlib.sha256(contents).hexdigest(),
        repository=repository, name=run_name, output_directory=output,
        build_preset=preset, label_concurrency=label_concurrency,
        datasets=tuple(datasets), iterations=tuple(iterations), models=models,
        comparisons=tuple(comparisons),
    )
    _check_trained_models(config)
    return config


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


PathRenderer = Callable[[Path], str]


def rendered_paths(value: Any, render: PathRenderer) -> Any:
    """Return value with every Path named by render, so a fingerprint does not embed the location."""
    if isinstance(value, Path):
        return render(value)
    if hasattr(value, "__dataclass_fields__"):
        return rendered_paths(asdict(value), render)
    if isinstance(value, dict):
        return {key: rendered_paths(item, render) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [rendered_paths(item, render) for item in value]
    return value


def _trained_model_definition(result: dict[str, dict[str, Any]], iteration: str) -> dict[str, str]:
    """Identify a model this recipe trains by its iteration's gate, which the recipe fixes alone.

    The model's digest exists only once that gate accepts it, so stages lock it as an input.
    """
    return {"iteration": iteration,
            "engine_gate": fingerprint(result[f"iteration:{iteration}:engine-gate"])}


def _dataset_definitions(config: WorkflowConfig, dataset: DatasetConfig,
                         result: dict[str, dict[str, Any]], render: PathRenderer) -> None:
    if isinstance(dataset, ImportedDatasetConfig):
        result[f"dataset:{dataset.name}:features"] = {
            "adapter": "imported-features", "path": render(dataset.feature_directory),
        }
        return
    source_id = f"dataset:{dataset.positions or dataset.name}:source"
    # Teachers are added only when present so stages recorded before teachers existed
    # keep their fingerprints.
    teacher: dict[str, Any] = {}
    if dataset.teacher is not None:
        teacher["teacher"] = (
            dataset.teacher.model_id if dataset.teacher.iteration is None else
            _trained_model_definition(result, dataset.teacher.iteration))
    if dataset.positions is None:
        result[source_id] = {"adapter": "source", "preset": config.build_preset,
                             "settings": dataset.source, **teacher}
    labels_id = f"dataset:{dataset.name}:labels"
    result[labels_id] = {
        "adapter": "labels", "preset": config.build_preset,
        "source": fingerprint(result[source_id]), "settings": dataset.labels,
        "label_concurrency": config.label_concurrency, **teacher,
    }
    result[f"dataset:{dataset.name}:features"] = {
        "adapter": "features", "preset": config.build_preset,
        "source": fingerprint(result[source_id]), "labels": fingerprint(result[labels_id]),
        "settings": dataset.features,
    }


def _iteration_definitions(config: WorkflowConfig, iteration: IterationConfig,
                           result: dict[str, dict[str, Any]], render: PathRenderer) -> None:
    training_id = f"iteration:{iteration.name}:training"
    dataset_id = f"dataset:{iteration.dataset}:features"
    result[training_id] = {
        "adapter": f"training-{iteration.training.type}",
        "dataset": fingerprint(result[dataset_id]),
        "settings": recorded_training_settings(iteration.training),
        "dependencies": [fingerprint(result[f"iteration:{name}:training"])
                         for name in iteration.depends_on],
    }
    extras = extra_training_datasets(config, iteration)
    if extras:
        result[training_id]["extra_datasets"] = [
            fingerprint(result[f"dataset:{dataset.name}:features"]) for dataset in extras]
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
            "settings": rendered_paths(iteration.candidate_validation, render),
        }
    gate = iteration.engine_gate
    if gate is not None:
        stage_id = f"iteration:{iteration.name}:engine-gate"
        result[stage_id] = {
            "adapter": "engine-gate", "preset": config.build_preset,
            "candidate": fingerprint(result[
                f"iteration:{iteration.name}:candidate-validation"]),
            "settings": rendered_paths(recorded_gate_settings(gate), render),
        }
        if gate.base_iteration is not None:
            result[stage_id]["base"] = _trained_model_definition(result, gate.base_iteration)


def _model_identity(config: WorkflowConfig, result: dict[str, dict[str, Any]],
                    name: str) -> Any:
    """Identify a comparison side from the recipe alone: an archived ID or its candidate stage.

    A trained side needs only its candidate, not an accepting gate, so the candidate stage
    rather than the gate identifies it; the digest is locked as a run-time input.
    """
    model = config.model(name)
    if model.iteration is None:
        return model.model_id
    return {"iteration": model.iteration, "candidate_validation": fingerprint(
        result[f"iteration:{model.iteration}:candidate-validation"])}


def stage_definitions(config: WorkflowConfig, *,
                      absolute_paths: bool = False) -> dict[str, dict[str, Any]]:
    """Return normalized settings and upstream config fingerprints for every stage.

    Paths are named relative to the repository, so a run survives moving it. absolute_paths
    reproduces the earlier absolute form, which stages recorded before then were locked to.
    """
    render: PathRenderer = str if absolute_paths else config.recorded_path
    result: dict[str, dict[str, Any]] = {}

    def deferred(dataset: DatasetConfig) -> bool:
        """A dataset that needs a model this recipe trains is defined once that model's gate is."""
        if not isinstance(dataset, ManagedDatasetConfig):
            return False
        return dataset.taught_in_run or (
            dataset.positions is not None and deferred(config.dataset(dataset.positions)))

    defined: set[str] = set()

    def define(dataset: DatasetConfig) -> None:
        if dataset.name not in defined:
            defined.add(dataset.name)
            _dataset_definitions(config, dataset, result, render)

    for dataset in config.datasets:
        if not deferred(dataset):
            define(dataset)
    for iteration in config.iterations:
        for name in datasets_used(config, iteration):
            define(config.dataset(name))
        _iteration_definitions(config, iteration, result, render)
    for comparison in config.comparisons:
        result[comparison_stage(comparison)] = {
            "adapter": "comparison", "preset": config.build_preset,
            "new": _model_identity(config, result, comparison.new),
            "base": _model_identity(config, result, comparison.base),
            "settings": rendered_paths(match_settings(comparison), render),
        }
    return result


def comparison_stage(comparison: ComparisonConfig) -> str:
    return f"comparison:{comparison.name}:match"


def stage_fingerprints(config: WorkflowConfig, *,
                       absolute_paths: bool = False) -> dict[str, str]:
    return {stage: fingerprint(definition) for stage, definition in
            stage_definitions(config, absolute_paths=absolute_paths).items()}
