#!/usr/bin/env python3
"""Compare completed PoE2 minimax label datasets from the same source shard."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import inspect_minimax_labels as auditor


POLICY_NAMES = {
    1: "random",
    2: "immediate_gain",
    3: "opponent_aware",
    4: "noisy_search",
}
PHASE_BUCKETS = (
    (4, 8),
    (9, 14),
    (15, 20),
    (21, 26),
    (27, 32),
    (33, 38),
    (39, 42),
    (43, 46),
)


class ComparisonError(ValueError):
    """Raised when label datasets cannot be compared safely."""


@dataclass(frozen=True)
class LabelRecord:
    player_one: int
    player_two: int
    key_low: int
    key_high: int
    source_id: int
    family_id: int
    trajectory_id: int
    parent_id: int
    trajectory_index: int
    nodes: int
    completed_nodes: int
    source_line: int
    source_ordinal: int
    value: int
    ply: int
    side_to_move: int
    mode: int
    completed_depth: int
    attempted_depth: int
    terminal_depth: int
    best_move: int
    split: int
    policy_id: int
    sample_index: int
    deepest_completed_nodes: int
    deepest_value: int
    deepest_completed_depth: int
    deepest_best_move: int
    previous_completed_nodes: int
    previous_value: int
    previous_completed_depth: int
    previous_best_move: int

    def source_fields(self) -> tuple[int, ...]:
        """Return every field that must not vary between searches."""
        return (
            self.player_one,
            self.player_two,
            self.key_low,
            self.key_high,
            self.source_id,
            self.family_id,
            self.trajectory_id,
            self.parent_id,
            self.trajectory_index,
            self.source_line,
            self.source_ordinal,
            self.ply,
            self.side_to_move,
            self.mode,
            self.terminal_depth,
            self.split,
            self.policy_id,
            self.sample_index,
        )


@dataclass(frozen=True)
class LabelDataset:
    path: Path
    manifest: dict[str, Any]
    records: tuple[LabelRecord, ...]

    @property
    def name(self) -> str:
        return self.path.name


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComparisonError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ComparisonError(f"{path} does not contain a JSON object")
    return value


def load_dataset(path: Path, source_path: Path | None = None) -> LabelDataset:
    """Audit and load one completed label dataset."""
    path = Path(path)
    try:
        auditor.audit_dataset(path, source_path)
    except auditor.DatasetError as error:
        raise ComparisonError(f"{path}: {error}") from error
    manifest = _load_json(path / "manifest.json")
    binary = (path / "labels.bin").read_bytes()
    header = auditor.HEADER.unpack_from(binary)
    record_count = header[5]
    schema_version = manifest["schema_version"]
    record_layout = auditor.record_struct(schema_version)
    records = []
    for index in range(record_count):
        offset = auditor.HEADER.size + index * record_layout.size
        fields = list(record_layout.unpack_from(binary, offset))
        if schema_version == 1:
            fields.extend((fields[10], fields[13], fields[17], fields[20], 0, 0, 0, 0xFF))
        records.append(LabelRecord(*fields))
    records.sort(key=lambda record: record.source_ordinal)
    return LabelDataset(path.resolve(), manifest, tuple(records))


def _series(values: Iterable[int], include_histogram: bool = True) -> dict[str, Any]:
    materialized = list(values)
    if not materialized:
        return {"count": 0}
    result = {
        "count": len(materialized),
        "minimum": min(materialized),
        "median": statistics.median(materialized),
        "mean": statistics.fmean(materialized),
        "maximum": max(materialized),
    }
    if include_histogram:
        counts = Counter(materialized)
        result["histogram"] = {str(value): counts[value] for value in sorted(counts)}
    return result


def _phase_name(ply: int) -> str:
    for lower, upper in PHASE_BUCKETS:
        if lower <= ply <= upper:
            return f"ply_{lower:02d}_{upper:02d}"
    return f"ply_{ply:02d}"


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _require_comparable(reference: LabelDataset, candidate: LabelDataset) -> None:
    reference_corpus = reference.manifest["corpus"]
    candidate_corpus = candidate.manifest["corpus"]
    reference_source = reference.manifest["source"]
    candidate_source = candidate.manifest["source"]
    provenance = (
        ("corpus.id", reference_corpus["id"], candidate_corpus["id"]),
        ("corpus.digest", reference_corpus["digest"], candidate_corpus["digest"]),
        ("corpus.shard_index", reference_corpus["shard_index"],
         candidate_corpus["shard_index"]),
        ("corpus.shard_count", reference_corpus["shard_count"],
         candidate_corpus["shard_count"]),
        ("source.digest", reference_source["digest"], candidate_source["digest"]),
        ("source.positions", reference_source["positions"], candidate_source["positions"]),
    )
    for field, expected, actual in provenance:
        if expected != actual:
            raise ComparisonError(
                f"{candidate.path}: {field} differs from {reference.path}"
            )
    if len(reference.records) != len(candidate.records):
        raise ComparisonError(
            f"{candidate.path}: record count differs from {reference.path}"
        )
    for index, (left, right) in enumerate(zip(reference.records, candidate.records)):
        if left.source_fields() != right.source_fields():
            raise ComparisonError(
                f"{candidate.path}: source fields differ at aligned record {index} "
                f"(source ordinals {left.source_ordinal} and {right.source_ordinal})"
            )


def _metrics(pairs: Iterable[tuple[LabelRecord, LabelRecord]]) -> dict[str, Any]:
    materialized = list(pairs)
    count = len(materialized)
    if count == 0:
        return {"records": 0}
    value_deltas = [candidate.value - reference.value
                    for reference, candidate in materialized]
    absolute_deltas = [abs(value) for value in value_deltas]
    depth_deltas = [candidate.completed_depth - reference.completed_depth
                    for reference, candidate in materialized]
    deepest_value_deltas = [candidate.deepest_value - reference.deepest_value
                            for reference, candidate in materialized]
    deepest_absolute_deltas = [abs(value) for value in deepest_value_deltas]
    deepest_depth_deltas = [candidate.deepest_completed_depth -
                            reference.deepest_completed_depth
                            for reference, candidate in materialized]
    value_agreement = sum(reference.value == candidate.value
                          for reference, candidate in materialized)
    best_move_agreement = sum(reference.best_move == candidate.best_move
                              for reference, candidate in materialized)
    sign_changes = sum(_sign(reference.value) != _sign(candidate.value)
                       for reference, candidate in materialized)
    same_parity = [pair for pair in materialized
                   if pair[0].completed_depth % 2 == pair[1].completed_depth % 2]
    same_depth = [pair for pair in materialized
                  if pair[0].completed_depth == pair[1].completed_depth]
    newly_terminal = [pair for pair in materialized
                      if pair[0].completed_depth < pair[0].terminal_depth
                      and pair[1].completed_depth == pair[1].terminal_depth]
    return {
        "records": count,
        "value_agreement": value_agreement,
        "value_agreement_rate": value_agreement / count,
        "best_move_agreement": best_move_agreement,
        "best_move_agreement_rate": best_move_agreement / count,
        "best_move_changes": count - best_move_agreement,
        "sign_changes": sign_changes,
        "sign_change_rate": sign_changes / count,
        "mean_value_delta": statistics.fmean(value_deltas),
        "mean_absolute_value_delta": statistics.fmean(absolute_deltas),
        "root_mean_square_value_delta": math.sqrt(
            statistics.fmean(value * value for value in value_deltas)
        ),
        "maximum_absolute_value_delta": max(absolute_deltas),
        "completed_depth_delta": _series(depth_deltas),
        "deeper": sum(value > 0 for value in depth_deltas),
        "same_depth": len(same_depth),
        "shallower": sum(value < 0 for value in depth_deltas),
        "terminal_status_changes": sum(
            (reference.completed_depth == reference.terminal_depth)
            != (candidate.completed_depth == candidate.terminal_depth)
            for reference, candidate in materialized
        ),
        "same_parity": {
            "records": len(same_parity),
            "mean_absolute_value_delta": (
                statistics.fmean(abs(candidate.value - reference.value)
                                 for reference, candidate in same_parity)
                if same_parity else None
            ),
            "sign_changes": sum(
                _sign(reference.value) != _sign(candidate.value)
                for reference, candidate in same_parity
            ),
            "best_move_changes": sum(
                reference.best_move != candidate.best_move
                for reference, candidate in same_parity
            ),
        },
        "equal_depth": {
            "records": len(same_depth),
            "value_changes": sum(reference.value != candidate.value
                                 for reference, candidate in same_depth),
            "best_move_changes": sum(reference.best_move != candidate.best_move
                                     for reference, candidate in same_depth),
        },
        "newly_terminal": {
            "records": len(newly_terminal),
            "mean_absolute_value_delta": (
                statistics.fmean(abs(candidate.value - reference.value)
                                 for reference, candidate in newly_terminal)
                if newly_terminal else None
            ),
            "sign_changes": sum(
                _sign(reference.value) != _sign(candidate.value)
                for reference, candidate in newly_terminal
            ),
            "best_move_changes": sum(
                reference.best_move != candidate.best_move
                for reference, candidate in newly_terminal
            ),
        },
        "deepest": {
            "value_agreement": sum(reference.deepest_value == candidate.deepest_value
                                   for reference, candidate in materialized),
            "mean_absolute_value_delta": statistics.fmean(deepest_absolute_deltas),
            "sign_changes": sum(
                _sign(reference.deepest_value) != _sign(candidate.deepest_value)
                for reference, candidate in materialized
            ),
            "best_move_changes": sum(
                reference.deepest_best_move != candidate.deepest_best_move
                for reference, candidate in materialized
            ),
            "completed_depth_delta": _series(deepest_depth_deltas),
        },
    }


def _dataset_summary(dataset: LabelDataset) -> dict[str, Any]:
    search = dataset.manifest["search"]
    records = dataset.records
    terminal_records = sum(record.completed_depth == record.terminal_depth
                           for record in records)
    return {
        "name": dataset.name,
        "path": str(dataset.path),
        "records": len(records),
        "node_limit": search["node_limit"],
        "hash_bytes_requested": search["hash_bytes_requested"],
        "hash_bytes_effective": search["hash_bytes_effective"],
        "hash_capacity": search["hash_capacity"],
        "workers_requested": search["workers_requested"],
        "workers_used": search["workers_used"],
        "target_selection": search.get("target_selection", "deepest_completed"),
        "terminal_records": terminal_records,
        "parity_backoffs": sum(record.completed_depth < record.deepest_completed_depth
                               for record in records),
        "previous_records": sum(record.previous_completed_depth > 0 for record in records),
        "completed_depth": _series(record.completed_depth for record in records),
        "deepest_completed_depth": _series(record.deepest_completed_depth for record in records),
        "total_nodes": _series((record.nodes for record in records), False),
        "completed_nodes": _series((record.completed_nodes for record in records), False),
        "deepest_completed_nodes": _series(
            (record.deepest_completed_nodes for record in records), False
        ),
        "aggregate_total_nodes": sum(record.nodes for record in records),
        "aggregate_completed_nodes": sum(record.completed_nodes for record in records),
        "aggregate_deepest_completed_nodes": sum(record.deepest_completed_nodes
                                                  for record in records),
    }


def compare_datasets(datasets: list[LabelDataset]) -> dict[str, Any]:
    """Return all-pairs comparison metrics for aligned label datasets."""
    if len(datasets) < 2:
        raise ComparisonError("at least two label datasets are required")
    reference = datasets[0]
    for candidate in datasets[1:]:
        _require_comparable(reference, candidate)

    comparisons = []
    for left, right in itertools.combinations(datasets, 2):
        pairs = list(zip(left.records, right.records))
        phases = {}
        for phase in sorted({_phase_name(record.ply) for record in left.records}):
            phases[phase] = _metrics(
                pair for pair in pairs if _phase_name(pair[0].ply) == phase
            )
        policies = {}
        for policy_id in sorted({record.policy_id for record in left.records}):
            name = POLICY_NAMES.get(policy_id, f"policy_{policy_id}")
            policies[name] = _metrics(
                pair for pair in pairs if pair[0].policy_id == policy_id
            )
        comparisons.append({
            "reference": left.name,
            "candidate": right.name,
            "overall": _metrics(pairs),
            "by_phase": phases,
            "by_policy": policies,
        })

    first_manifest = reference.manifest
    return {
        "schema": "poe2-minimax-label-comparison",
        "source": {
            "corpus_id": first_manifest["corpus"]["id"],
            "source_digest": first_manifest["source"]["digest"],
            "shard_index": first_manifest["corpus"]["shard_index"],
            "shard_count": first_manifest["corpus"]["shard_count"],
        },
        "datasets": [_dataset_summary(dataset) for dataset in datasets],
        "comparisons": comparisons,
    }


def _format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _print_metrics(prefix: str, metrics: dict[str, Any]) -> None:
    count = metrics["records"]
    same_parity = metrics["same_parity"]
    newly_terminal = metrics["newly_terminal"]
    deepest = metrics["deepest"]
    print(
        prefix
        + f" records={count}"
        + f" value_same={metrics['value_agreement']}/{count}"
        + f" mean_delta={_format_number(metrics['mean_value_delta'])}"
        + f" mae={_format_number(metrics['mean_absolute_value_delta'])}"
        + f" rmse={_format_number(metrics['root_mean_square_value_delta'])}"
        + f" sign_changes={metrics['sign_changes']}"
        + f" move_changes={metrics['best_move_changes']}"
        + f" depth_delta={_format_number(metrics['completed_depth_delta']['mean'])}"
        + f" depth_d/s/u={metrics['deeper']}/{metrics['same_depth']}/{metrics['shallower']}"
        + f" parity_n/mae/sign/move={same_parity['records']}/"
        f"{_format_number(same_parity['mean_absolute_value_delta'])}/"
        f"{same_parity['sign_changes']}/{same_parity['best_move_changes']}"
        + f" deepest_mae/sign/move={_format_number(deepest['mean_absolute_value_delta'])}/"
        f"{deepest['sign_changes']}/{deepest['best_move_changes']}"
        + f" newly_terminal={newly_terminal['records']}"
    )


def _print_report(report: dict[str, Any], include_strata: bool) -> None:
    source = report["source"]
    print(
        "label_comparison"
        f" corpus={source['corpus_id']}"
        f" shard={source['shard_index']}/{source['shard_count']}"
        f" datasets={len(report['datasets'])}"
    )
    for dataset in report["datasets"]:
        depth = dataset["completed_depth"]
        deepest = dataset["deepest_completed_depth"]
        print(
            "dataset"
            f" name={dataset['name']}"
            f" records={dataset['records']}"
            f" nodes={dataset['node_limit']}"
            f" hash_mib={dataset['hash_bytes_effective'] / (1024 * 1024):g}"
            f" depth_min/median/mean/max={depth['minimum']}/"
            f"{_format_number(depth['median'])}/{_format_number(depth['mean'])}/"
            f"{depth['maximum']}"
            f" deepest_mean={_format_number(deepest['mean'])}"
            f" parity_backoffs={dataset['parity_backoffs']}"
            f" terminal={dataset['terminal_records']}"
        )
    for comparison in report["comparisons"]:
        pair = f" reference={comparison['reference']} candidate={comparison['candidate']}"
        _print_metrics("comparison" + pair, comparison["overall"])
        if include_strata:
            for name, metrics in comparison["by_phase"].items():
                _print_metrics("phase" + pair + f" name={name}", metrics)
            for name, metrics in comparison["by_policy"].items():
                _print_metrics("policy" + pair + f" name={name}", metrics)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", type=Path,
                        help="two or more completed label dataset directories")
    parser.add_argument("--source", type=Path,
                        help="optional raw source shard to verify for every dataset")
    parser.add_argument("--json", action="store_true", help="emit the complete JSON report")
    parser.add_argument("--strata", action="store_true",
                        help="include phase and policy rows in the text report")
    arguments = parser.parse_args()
    if len(arguments.datasets) < 2:
        parser.error("at least two label datasets are required")
    try:
        datasets = [load_dataset(path, arguments.source) for path in arguments.datasets]
        report = compare_datasets(datasets)
    except (ComparisonError, OSError) as error:
        print(f"could not compare label datasets: {error}", file=sys.stderr)
        return 1
    if arguments.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        _print_report(report, arguments.strata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
