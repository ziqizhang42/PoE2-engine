#!/usr/bin/env python3
"""Reusable read-only loader for a complete PoE2 minimax label corpus."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import compare_minimax_labels as labels
import inspect_position_source as source_auditor


class CorpusError(ValueError):
    """Raised when label shards cannot form one leakage-safe corpus."""


@dataclass(frozen=True)
class SelectedRecord:
    label: labels.LabelRecord
    shard_index: int
    duplicate_count: int


@dataclass(frozen=True)
class MinimaxCorpus:
    corpus_id: str
    label_set_digest: str
    source_records: int
    shard_count: int
    duplicate_records: int
    duplicate_groups: int
    maximum_duplicate_count: int
    representative_upgrades: int
    varying_label_groups: int
    records: tuple[SelectedRecord, ...]
    raw_split_counts: dict[str, int]
    selected_split_counts: dict[str, int]
    label_build: dict[str, Any]
    label_search: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "label_set_digest": self.label_set_digest,
            "shards": self.shard_count,
            "source_records": self.source_records,
            "selected_records": len(self.records),
            "duplicates_removed": self.duplicate_records,
            "duplicate_groups": self.duplicate_groups,
            "maximum_duplicate_count": self.maximum_duplicate_count,
            "representative_upgrades": self.representative_upgrades,
            "varying_label_groups": self.varying_label_groups,
            "cross_split_duplicate_groups": 0,
            "family_split_violations": 0,
            "trajectory_split_violations": 0,
            "same_depth_label_conflicts": 0,
            "raw_split_counts": self.raw_split_counts,
            "selected_split_counts": self.selected_split_counts,
            "label_build": self.label_build,
            "label_search": self.label_search,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorpusError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorpusError(f"could not read {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} does not contain a JSON object")
    return value


def _parse_source_records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise CorpusError(f"could not read {path}: {error}") from error
    _require(bool(lines), f"source shard {path} is empty")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines[1:]):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise CorpusError(f"source shard {path} record {index} is invalid: {error}") from error
        _require(isinstance(value, dict), f"source shard {path} record {index} is not an object")
        records.append(value)
    return records


def _hex64(value: Any, field: str) -> int:
    _require(isinstance(value, str) and len(value) == 16, f"{field} is not a 64-bit hex string")
    try:
        return int(value, 16)
    except ValueError as error:
        raise CorpusError(f"{field} is not hexadecimal") from error


def _label_marker_digests(directory: Path) -> tuple[bytes, bytes]:
    try:
        lines = (directory / "COMPLETE").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise CorpusError(f"could not read {directory}/COMPLETE: {error}") from error
    fields: dict[str, str] = {}
    for line in lines[1:]:
        name, separator, value = line.partition("=")
        _require(bool(separator) and name not in fields,
                 f"{directory}/COMPLETE is malformed")
        fields[name] = value
    _require(lines[:1] == ["poe2-minimax-labels"] and
             set(fields) == {"binary_sha256", "manifest_sha256"},
             f"{directory}/COMPLETE has the wrong fields")
    try:
        binary = bytes.fromhex(fields["binary_sha256"])
        manifest = bytes.fromhex(fields["manifest_sha256"])
    except ValueError as error:
        raise CorpusError(f"{directory}/COMPLETE contains a non-hex digest") from error
    _require(len(binary) == 32 and len(manifest) == 32,
             f"{directory}/COMPLETE has a digest of the wrong size")
    return binary, manifest


def _source_fields(value: dict[str, Any], label: str) -> tuple[int, ...]:
    required = {
        "p1", "p2", "source_id", "family_id", "trajectory_id", "parent_id",
        "trajectory_index", "policy_id", "sample_index", "split", "ply",
    }
    _require(set(value) == required, f"{label} has unexpected source fields")
    integers = ("trajectory_index", "policy_id", "sample_index", "split", "ply")
    for name in integers:
        _require(isinstance(value[name], int) and not isinstance(value[name], bool),
                 f"{label}.{name} is not an integer")
    return (
        _hex64(value["p1"], f"{label}.p1"),
        _hex64(value["p2"], f"{label}.p2"),
        _hex64(value["source_id"], f"{label}.source_id"),
        _hex64(value["family_id"], f"{label}.family_id"),
        _hex64(value["trajectory_id"], f"{label}.trajectory_id"),
        _hex64(value["parent_id"], f"{label}.parent_id"),
        value["trajectory_index"], value["policy_id"], value["sample_index"],
        value["split"], value["ply"],
    )


def _record_identity(record: labels.LabelRecord, shard_index: int) -> tuple[int, ...]:
    return (record.trajectory_index, record.sample_index, record.source_id,
            shard_index, record.source_ordinal)


def load_minimax_corpus(label_directory: Path, source_directory: Path) -> MinimaxCorpus:
    """Audit, combine, and deterministically deduplicate every label shard."""
    label_directory = Path(label_directory)
    source_directory = Path(source_directory)
    try:
        source_summary = source_auditor.audit_source(source_directory)
    except source_auditor.SourceError as error:
        raise CorpusError(f"position source is invalid: {error}") from error
    source_manifest = _load_json(source_directory / "manifest.json")
    source_shards_value = source_manifest.get("shards")
    _require(isinstance(source_shards_value, list), "source manifest shards must be an array")
    source_shards: dict[int, Path] = {}
    for value in source_shards_value:
        _require(isinstance(value, dict), "source manifest shard is not an object")
        index = value.get("index")
        name = value.get("name")
        _require(isinstance(index, int) and not isinstance(index, bool) and
                 isinstance(name, str), "source manifest shard coordinates are invalid")
        _require(index not in source_shards, "source manifest repeats a shard index")
        source_shards[index] = source_directory / "shards" / name
    _require(set(source_shards) == set(range(source_summary.shards)),
             "source manifest does not contain every shard")

    shard_root = label_directory / "shards"
    _require(shard_root.is_dir(), "label corpus has no shards directory")
    label_directories: dict[int, Path] = {}
    for directory in shard_root.iterdir():
        _require(directory.is_dir(), "label shards directory contains a non-directory entry")
        manifest = _load_json(directory / "manifest.json")
        corpus = manifest.get("corpus")
        _require(isinstance(corpus, dict), f"{directory} has no corpus object")
        index = corpus.get("shard_index")
        _require(isinstance(index, int) and not isinstance(index, bool),
                 f"{directory} has an invalid shard index")
        _require(index not in label_directories, "label corpus repeats a shard index")
        label_directories[index] = directory
    _require(set(label_directories) == set(range(source_summary.shards)),
             "label corpus does not contain every source shard exactly once")

    reference_build: dict[str, Any] | None = None
    reference_search: dict[str, Any] | None = None
    corpus_id: str | None = None
    aggregate_digest = hashlib.sha256()
    all_records: list[tuple[int, labels.LabelRecord]] = []
    family_splits: dict[int, set[int]] = defaultdict(set)
    trajectory_splits: dict[int, set[int]] = defaultdict(set)

    for shard_index in range(source_summary.shards):
        directory = label_directories[shard_index]
        source_path = source_shards[shard_index]
        try:
            dataset = labels.load_dataset(directory, source_path)
        except labels.ComparisonError as error:
            raise CorpusError(f"label shard {shard_index} is invalid: {error}") from error
        manifest = dataset.manifest
        build = manifest.get("build")
        search = manifest.get("search")
        corpus = manifest.get("corpus")
        _require(isinstance(build, dict) and isinstance(search, dict) and
                 isinstance(corpus, dict), f"label shard {shard_index} manifest is incomplete")
        if reference_build is None:
            reference_build = build
            reference_search = search
            corpus_id = corpus.get("id")
            _require(isinstance(corpus_id, str) and corpus_id,
                     "label corpus ID is missing")
        else:
            _require(build == reference_build, "label shards use different builds")
            _require(search == reference_search, "label shards use different search settings")
            _require(corpus.get("id") == corpus_id, "label shards use different corpus IDs")
        _require(corpus.get("shard_index") == shard_index and
                 corpus.get("shard_count") == source_summary.shards,
                 f"label shard {shard_index} coordinates are inconsistent")

        source_records = _parse_source_records(source_path)
        _require(len(dataset.records) == len(source_records),
                 f"label shard {shard_index} is not complete")
        for ordinal, (record, source_value) in enumerate(zip(dataset.records, source_records,
                                                              strict=True)):
            source_fields = _source_fields(source_value,
                                           f"source shard {shard_index} record {ordinal}")
            expected_side = source_fields[10] % 2
            actual_source = (
                record.player_one, record.player_two, record.source_id, record.family_id,
                record.trajectory_id, record.parent_id, record.trajectory_index,
                record.policy_id, record.sample_index, record.split, record.ply,
            )
            _require(actual_source == source_fields and record.side_to_move == expected_side and
                     record.source_ordinal == ordinal and record.source_line == ordinal + 2,
                     f"label shard {shard_index} record {ordinal} differs from its source")
            family_splits[record.family_id].add(record.split)
            trajectory_splits[record.trajectory_id].add(record.split)
            all_records.append((shard_index, record))
        binary_digest, manifest_digest = _label_marker_digests(directory)
        aggregate_digest.update(binary_digest)
        aggregate_digest.update(manifest_digest)

    _require(len(all_records) == source_summary.records,
             "combined label record count differs from the source")
    _require(all(len(splits) == 1 for splits in family_splits.values()),
             "a family crosses dataset splits")
    _require(all(len(splits) == 1 for splits in trajectory_splits.values()),
             "a trajectory crosses dataset splits")

    by_position: dict[tuple[int, int], list[tuple[int, labels.LabelRecord]]] = defaultdict(list)
    for item in all_records:
        record = item[1]
        by_position[(record.key_low, record.key_high)].append(item)

    selected: list[SelectedRecord] = []
    duplicate_groups = 0
    maximum_duplicate_count = 1
    representative_upgrades = 0
    varying_label_groups = 0
    for key in sorted(by_position):
        group = by_position[key]
        splits = {record.split for _, record in group}
        _require(len(splits) == 1, "a duplicate canonical position crosses dataset splits")
        by_depth: dict[int, set[int]] = defaultdict(set)
        for _, record in group:
            by_depth[record.completed_depth].add(record.value)
        _require(all(len(values) == 1 for values in by_depth.values()),
                 "duplicate labels disagree at the same completed depth")
        best_index = 0
        for index in range(1, len(group)):
            shard, record = group[index]
            selected_shard, selected_record = group[best_index]
            if (record.completed_depth > selected_record.completed_depth or
                    (record.completed_depth == selected_record.completed_depth and
                     _record_identity(record, shard) <
                     _record_identity(selected_record, selected_shard))):
                best_index = index
                representative_upgrades += 1
        shard, record = group[best_index]
        selected.append(SelectedRecord(record, shard, len(group)))
        if len(group) > 1:
            duplicate_groups += 1
        maximum_duplicate_count = max(maximum_duplicate_count, len(group))
        varying_label_groups += len({record.value for _, record in group}) > 1

    split_names = {1: "train", 2: "validation", 3: "test"}
    raw_split_counts = Counter(split_names[record.split] for _, record in all_records)
    selected_split_counts = Counter(split_names[item.label.split] for item in selected)
    _require(len(all_records) - len(selected) == source_summary.duplicate_positions,
             "label duplicate count differs from the source manifest")
    assert reference_build is not None and reference_search is not None and corpus_id is not None
    return MinimaxCorpus(
        corpus_id=corpus_id,
        label_set_digest=f"sha256:{aggregate_digest.hexdigest()}",
        source_records=len(all_records),
        shard_count=source_summary.shards,
        duplicate_records=len(all_records) - len(selected),
        duplicate_groups=duplicate_groups,
        maximum_duplicate_count=maximum_duplicate_count,
        representative_upgrades=representative_upgrades,
        varying_label_groups=varying_label_groups,
        records=tuple(selected),
        raw_split_counts={name: raw_split_counts[name] for name in split_names.values()},
        selected_split_counts={name: selected_split_counts[name] for name in split_names.values()},
        label_build=reference_build,
        label_search=reference_search,
    )
