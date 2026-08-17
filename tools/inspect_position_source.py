#!/usr/bin/env python3
"""Validate and summarize a completed deterministic PoE2 position source."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
GENERATOR_VERSION = 1
RNG_NAME = "splitmix64-rejection-v1"
BOARD_SIZE = 7
CELL_COUNT = BOARD_SIZE * BOARD_SIZE
BOARD_MASK = (1 << CELL_COUNT) - 1
SIDE_TO_MOVE_BIT = 1 << CELL_COUNT
MASK64 = (1 << 64) - 1
FAMILY_SALT = 0x4E8B2D7D34A2C1F9
TRAJECTORY_SALT = 0xD1B54A32D192ED03
SPLIT_SALT = 0x94D049BB133111EB
PHASE_BUCKETS = ((4, 8), (9, 14), (15, 20), (21, 26), (27, 32),
                 (33, 38), (39, 42), (43, 46))
POLICY_NAMES = ("random", "immediate_gain", "opponent_aware", "noisy_search")
SPLIT_NAMES = (None, "train", "validation", "test")
HEX64 = re.compile(r"[0-9a-f]{16}")
SHARD_NAME = re.compile(r"shard-([0-9]{8})-([0-9a-f]{64})\.jsonl")


class SourceError(ValueError):
    """Raised when a source corpus violates its artifact contract."""


@dataclass(frozen=True)
class AuditSummary:
    trajectories: int
    records: int
    shards: int
    duplicate_positions: int
    train: int
    validation: int
    test: int


@dataclass(frozen=True)
class ParsedRecord:
    player_one: int
    player_two: int
    source_id: int
    family_id: int
    trajectory_id: int
    parent_id: int
    trajectory_index: int
    policy_id: int
    sample_index: int
    split: int
    ply: int
    canonical: tuple[int, int]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceError(message)


def _object(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _array(value: Any, field: str) -> list[Any]:
    _require(isinstance(value, list), f"{field} must be an array")
    return value


def _integer(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool),
             f"{field} must be an integer")
    return value


def _nonnegative(value: Any, field: str) -> int:
    result = _integer(value, field)
    _require(result >= 0, f"{field} must be nonnegative")
    return result


def _digest(value: Any, field: str) -> bytes:
    _require(isinstance(value, str) and value.startswith("sha256:"),
             f"{field} must use the sha256: prefix")
    encoded = value.removeprefix("sha256:")
    _require(len(encoded) == 64, f"{field} must contain 64 hexadecimal digits")
    try:
        return bytes.fromhex(encoded)
    except ValueError as error:
        raise SourceError(f"{field} is not hexadecimal") from error


def _hex64(value: Any, field: str) -> int:
    _require(isinstance(value, str) and HEX64.fullmatch(value) is not None,
             f"{field} must contain 16 lowercase hexadecimal digits")
    return int(value, 16)


def _mix64(value: int) -> int:
    value &= MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def _derived_id(seed: int, trajectory_index: int, salt: int) -> int:
    value = _mix64(seed ^ _mix64(trajectory_index ^ salt))
    return value or 1


def _fnv_append(hash_value: int, value: int, byte_count: int) -> int:
    for index in range(byte_count):
        hash_value ^= (value >> (8 * index)) & 0xFF
        hash_value = (hash_value * 1099511628211) & MASK64
    return hash_value


def _source_id(record: ParsedRecord) -> int:
    value = 14695981039346656037
    for field in (record.player_one, record.player_two, record.family_id,
                  record.trajectory_id, record.parent_id, record.trajectory_index):
        value = _fnv_append(value, field, 8)
    value = _fnv_append(value, record.policy_id, 2)
    value = _fnv_append(value, record.sample_index, 2)
    value = _fnv_append(value, record.split, 1)
    return _fnv_append(value, record.ply, 1)


def _transform_square(symmetry: int, row: int, column: int) -> tuple[int, int]:
    last = BOARD_SIZE - 1
    return (
        (row, column),
        (column, last - row),
        (last - row, last - column),
        (last - column, row),
        (last - row, column),
        (row, last - column),
        (column, row),
        (last - column, last - row),
    )[symmetry]


def _transform_bits(bits: int, symmetry: int) -> int:
    transformed = 0
    remaining = bits
    while remaining:
        bit = remaining & -remaining
        row, column = divmod(bit.bit_length() - 1, BOARD_SIZE)
        target_row, target_column = _transform_square(symmetry, row, column)
        transformed |= 1 << (target_row * BOARD_SIZE + target_column)
        remaining ^= bit
    return transformed


def _canonical_key(player_one: int, player_two: int, ply: int) -> tuple[int, int]:
    candidates = []
    for symmetry in range(8):
        low = _transform_bits(player_one, symmetry)
        high = _transform_bits(player_two, symmetry)
        if ply % 2:
            high |= SIDE_TO_MOVE_BIT
        candidates.append((low, high))
    return min(candidates)


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def unite(self, first: int, second: int) -> None:
        first = self.find(first)
        second = self.find(second)
        if first != second:
            self.parent[max(first, second)] = min(first, second)


def _load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceError(f"could not read {field}: {error}") from error
    return _object(value, field)


def _load_json_line(line: bytes, field: str) -> dict[str, Any]:
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceError(f"{field} is not valid UTF-8 JSON: {error}") from error
    return _object(value, field)


def _trajectory_range(trajectory_count: int, shard_count: int,
                      shard_index: int) -> tuple[int, int]:
    base, remainder = divmod(trajectory_count, shard_count)
    begin = shard_index * base + min(shard_index, remainder)
    return begin, begin + base + (1 if shard_index < remainder else 0)


def _parse_record(value: dict[str, Any], label: str) -> ParsedRecord:
    expected = {"p1", "p2", "source_id", "family_id", "trajectory_id", "parent_id",
                "trajectory_index", "policy_id", "sample_index", "split", "ply"}
    _require(set(value) == expected, f"{label} fields are wrong")
    player_one = _hex64(value["p1"], f"{label}.p1")
    player_two = _hex64(value["p2"], f"{label}.p2")
    source_id = _hex64(value["source_id"], f"{label}.source_id")
    family_id = _hex64(value["family_id"], f"{label}.family_id")
    trajectory_id = _hex64(value["trajectory_id"], f"{label}.trajectory_id")
    parent_id = _hex64(value["parent_id"], f"{label}.parent_id")
    trajectory_index = _nonnegative(value["trajectory_index"],
                                    f"{label}.trajectory_index")
    policy_id = _integer(value["policy_id"], f"{label}.policy_id")
    sample_index = _nonnegative(value["sample_index"], f"{label}.sample_index")
    split = _integer(value["split"], f"{label}.split")
    ply = _integer(value["ply"], f"{label}.ply")
    _require((player_one | player_two) & ~BOARD_MASK == 0,
             f"{label} has off-board bits")
    _require(player_one & player_two == 0, f"{label} has overlapping bitboards")
    _require(0 <= ply < CELL_COUNT, f"{label} ply is out of range")
    occupied = player_one | player_two
    _require(occupied.bit_count() == ply, f"{label} ply differs from occupied squares")
    _require(player_one.bit_count() == (ply + 1) // 2,
             f"{label} has the wrong Player 1 count")
    _require(player_two.bit_count() == ply // 2,
             f"{label} has the wrong Player 2 count")
    _require(policy_id in range(1, 5), f"{label} policy is invalid")
    _require(split in range(1, 4), f"{label} split is invalid")
    _require(family_id != 0 and trajectory_id != 0,
             f"{label} family and trajectory IDs must be nonzero")
    return ParsedRecord(
        player_one=player_one,
        player_two=player_two,
        source_id=source_id,
        family_id=family_id,
        trajectory_id=trajectory_id,
        parent_id=parent_id,
        trajectory_index=trajectory_index,
        policy_id=policy_id,
        sample_index=sample_index,
        split=split,
        ply=ply,
        canonical=_canonical_key(player_one, player_two, ply),
    )


def audit_source(directory: Path) -> AuditSummary:
    directory = Path(directory)
    complete = directory / "COMPLETE"
    incomplete = directory / "INCOMPLETE"
    manifest_path = directory / "manifest.json"
    shard_directory = directory / "shards"
    _require(directory.is_dir(), "source directory does not exist")
    _require(complete.is_file(), "source has no COMPLETE marker")
    _require(not incomplete.exists(), "source still has an INCOMPLETE marker")
    _require(manifest_path.is_file(), "source manifest is missing")
    _require(not (directory / "manifest.json.tmp").exists(),
             "temporary source manifest still exists")
    _require(shard_directory.is_dir(), "source shard directory is missing")

    try:
        marker_lines = complete.read_text(encoding="utf-8").splitlines()
        manifest_bytes = manifest_path.read_bytes()
    except (OSError, UnicodeDecodeError) as error:
        raise SourceError(f"could not read source completion metadata: {error}") from error
    _require(marker_lines[:1] == ["poe2-position-source"],
             "COMPLETE marker header is wrong")
    marker_fields = {}
    for line in marker_lines[1:]:
        name, separator, value = line.partition("=")
        _require(bool(separator) and name not in marker_fields,
                 "COMPLETE marker is malformed")
        marker_fields[name] = value
    _require(set(marker_fields) == {"manifest_sha256"},
             "COMPLETE marker fields are wrong")
    _require(marker_fields["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest(),
             "manifest digest does not match the COMPLETE marker")

    manifest = _load_json(manifest_path, "manifest")
    _require(manifest.get("schema") == "poe2-position-source", "unknown source schema")
    _require(_integer(manifest.get("schema_version"), "schema_version") == SCHEMA_VERSION,
             "unsupported source schema version")
    corpus = _object(manifest.get("corpus"), "corpus")
    generator = _object(manifest.get("generator"), "generator")
    build = _object(manifest.get("build"), "build")
    results = _object(manifest.get("results"), "results")
    manifest_shards = _array(manifest.get("shards"), "shards")

    corpus_id = corpus.get("id")
    _require(isinstance(corpus_id, str) and corpus_id, "corpus.id must be nonempty")
    _require(_digest(corpus.get("digest"), "corpus.digest") ==
             hashlib.sha256(corpus_id.encode("utf-8")).digest(),
             "corpus digest does not match corpus.id")
    _require(_integer(generator.get("version"), "generator.version") == GENERATOR_VERSION,
             "unsupported generator version")
    _require(generator.get("rng") == RNG_NAME, "unknown generator RNG")
    seed = _hex64(generator.get("seed"), "generator.seed")
    trajectory_count = _integer(generator.get("trajectory_count"),
                                "generator.trajectory_count")
    samples = _integer(generator.get("samples_per_trajectory"),
                       "generator.samples_per_trajectory")
    shard_count = _integer(generator.get("shard_count"), "generator.shard_count")
    workers_requested = _integer(generator.get("workers_requested"),
                                 "generator.workers_requested")
    workers_used = _integer(generator.get("workers_used"), "generator.workers_used")
    search_nodes = _nonnegative(generator.get("search_nodes"), "generator.search_nodes")
    search_hash_bytes = _nonnegative(generator.get("search_hash_bytes"),
                                     "generator.search_hash_bytes")
    noise_percent = _integer(generator.get("noise_percent"), "generator.noise_percent")
    weights_object = _object(generator.get("policy_weights"), "generator.policy_weights")
    _require(set(weights_object) == set(POLICY_NAMES), "policy weight fields are wrong")
    weights = tuple(_nonnegative(weights_object[name], f"policy_weights.{name}")
                    for name in POLICY_NAMES)
    _require(trajectory_count > 0, "trajectory count must be positive")
    _require(1 <= samples <= len(PHASE_BUCKETS), "sample count must be between one and eight")
    _require(1 <= shard_count <= trajectory_count, "shard count is invalid")
    _require(workers_requested > 0 and workers_used == min(workers_requested, trajectory_count),
             "worker count is inconsistent")
    _require(0 <= noise_percent <= 100, "noise percent is invalid")
    _require(sum(weights) > 0, "all policy weights are zero")
    _require(weights[3] == 0 or search_nodes > 0,
             "noisy-search policy has no node budget")
    _require(search_hash_bytes >= 0, "search hash bytes is invalid")
    _require(len(manifest_shards) == shard_count, "manifest shard count is wrong")
    for field in ("git_commit", "project_version", "compiler_id", "compiler_version",
                  "build_type", "target_processor"):
        _require(isinstance(build.get(field), str), f"build.{field} must be a string")
    _require(isinstance(build.get("git_dirty"), bool), "build.git_dirty must be boolean")
    _require(isinstance(build.get("native_architecture"), bool),
             "build.native_architecture must be boolean")

    all_records: list[ParsedRecord] = []
    expected_names: set[str] = set()
    for shard_index, shard_value in enumerate(manifest_shards):
        shard = _object(shard_value, f"shards[{shard_index}]")
        _require(_integer(shard.get("index"), f"shards[{shard_index}].index") == shard_index,
                 f"shard {shard_index} index is wrong")
        name = shard.get("name")
        _require(isinstance(name, str), f"shard {shard_index} name must be a string")
        match = SHARD_NAME.fullmatch(name)
        _require(match is not None and int(match.group(1)) == shard_index,
                 f"shard {shard_index} name is malformed")
        expected_digest = _digest(shard.get("digest"), f"shards[{shard_index}].digest")
        _require(match.group(2) == expected_digest.hex(),
                 f"shard {shard_index} filename digest is wrong")
        expected_names.add(name)
        path = shard_directory / name
        _require(path.is_file(), f"shard {shard_index} file is missing")
        try:
            shard_bytes = path.read_bytes()
        except OSError as error:
            raise SourceError(f"could not read shard {shard_index}: {error}") from error
        _require(hashlib.sha256(shard_bytes).digest() == expected_digest,
                 f"shard {shard_index} digest is wrong")
        lines = shard_bytes.splitlines()
        _require(lines and shard_bytes.endswith(b"\n"),
                 f"shard {shard_index} is empty or lacks its final newline")
        header = _load_json_line(lines[0], f"shard {shard_index} header")
        expected_header = {"type", "schema", "generator", "rng", "corpus_id_hex", "seed",
                           "trajectory_count", "samples_per_trajectory", "shard_index",
                           "shard_count", "trajectory_begin", "trajectory_end", "search_nodes",
                           "search_hash_bytes", "noise_percent", "policy_weights"}
        _require(set(header) == expected_header, f"shard {shard_index} header fields are wrong")
        begin, end = _trajectory_range(trajectory_count, shard_count, shard_index)
        try:
            header_corpus_id = bytes.fromhex(header["corpus_id_hex"]).decode("utf-8")
        except (KeyError, TypeError, ValueError, UnicodeDecodeError) as error:
            raise SourceError(f"shard {shard_index} corpus ID encoding is invalid") from error
        _require(header_corpus_id == corpus_id and
                 header["corpus_id_hex"] == corpus_id.encode("utf-8").hex(),
                 f"shard {shard_index} corpus ID differs")
        header_weights = _array(header["policy_weights"], "header.policy_weights")
        _require(len(header_weights) == len(weights), "header policy weight count is wrong")
        parsed_header_weights = tuple(
            _nonnegative(value, f"header.policy_weights[{index}]")
            for index, value in enumerate(header_weights)
        )
        _require(header["type"] == "poe2-position-source" and
                 _integer(header["schema"], "header.schema") == SCHEMA_VERSION and
                 _integer(header["generator"], "header.generator") == GENERATOR_VERSION and
                 header["rng"] == RNG_NAME and _hex64(header["seed"], "header.seed") == seed and
                 _integer(header["trajectory_count"], "header.trajectory_count") ==
                 trajectory_count and
                 _integer(header["samples_per_trajectory"], "header.samples") == samples and
                 _integer(header["shard_index"], "header.shard_index") == shard_index and
                 _integer(header["shard_count"], "header.shard_count") == shard_count and
                 _integer(header["trajectory_begin"], "header.trajectory_begin") == begin and
                 _integer(header["trajectory_end"], "header.trajectory_end") == end and
                 _nonnegative(header["search_nodes"], "header.search_nodes") == search_nodes and
                 _nonnegative(header["search_hash_bytes"], "header.search_hash_bytes") ==
                 search_hash_bytes and
                 _integer(header["noise_percent"], "header.noise_percent") == noise_percent and
                 parsed_header_weights == weights,
                 f"shard {shard_index} header differs from the manifest")

        shard_records = [_parse_record(_load_json_line(line, f"shard {shard_index} record {offset}"),
                                       f"shard {shard_index} record {offset}")
                         for offset, line in enumerate(lines[1:])]
        expected_record_count = (end - begin) * samples
        _require(len(shard_records) == expected_record_count,
                 f"shard {shard_index} record count is wrong")
        _require(_integer(shard.get("records"), f"shards[{shard_index}].records") ==
                 expected_record_count, f"shard {shard_index} manifest record count is wrong")
        _require(_integer(shard.get("trajectory_begin"),
                          f"shards[{shard_index}].trajectory_begin") == begin and
                 _integer(shard.get("trajectory_end"),
                          f"shards[{shard_index}].trajectory_end") == end,
                 f"shard {shard_index} manifest trajectory range is wrong")
        for offset, record in enumerate(shard_records):
            trajectory = begin + offset // samples
            sample_index = offset % samples
            label = f"shard {shard_index} record {offset}"
            _require(record.trajectory_index == trajectory and
                     record.sample_index == sample_index,
                     f"{label} is out of trajectory/sample order")
            _require(record.family_id == _derived_id(seed, trajectory, FAMILY_SALT),
                     f"{label} family ID is wrong")
            _require(record.trajectory_id == _derived_id(seed, trajectory, TRAJECTORY_SALT),
                     f"{label} trajectory ID is wrong")
            _require(record.parent_id == 0, f"{label} parent ID is unsupported")
            _require(record.source_id == _source_id(record), f"{label} source ID is wrong")
            _require(weights[record.policy_id - 1] > 0,
                     f"{label} uses a zero-weight policy")
            matching_phases = [phase for phase, limits in enumerate(PHASE_BUCKETS)
                               if limits[0] <= record.ply <= limits[1]]
            _require(len(matching_phases) == 1, f"{label} ply is outside phase buckets")
        for trajectory_offset in range(end - begin):
            group = shard_records[trajectory_offset * samples:(trajectory_offset + 1) * samples]
            _require(len({record.policy_id for record in group}) == 1 and
                     len({record.split for record in group}) == 1 and
                     len({record.family_id for record in group}) == 1 and
                     len({record.trajectory_id for record in group}) == 1,
                     f"shard {shard_index} trajectory metadata changes between samples")
            _require([record.ply for record in group] == sorted(record.ply for record in group),
                     f"shard {shard_index} trajectory samples are not phase ordered")
            _require(len({_canonical_phase(record.ply) for record in group}) == samples,
                     f"shard {shard_index} trajectory repeats a phase bucket")
        all_records.extend(shard_records)

    actual_names = {path.name for path in shard_directory.iterdir() if path.is_file()}
    _require(actual_names == expected_names, "source shard directory has missing or extra files")
    _require(len(all_records) == trajectory_count * samples, "total source record count is wrong")

    components = _DisjointSet(trajectory_count)
    first_by_position: dict[tuple[int, int], int] = {}
    duplicates = 0
    for record in all_records:
        previous = first_by_position.get(record.canonical)
        if previous is None:
            first_by_position[record.canonical] = record.trajectory_index
        else:
            duplicates += 1
            components.unite(previous, record.trajectory_index)

    by_trajectory = [all_records[index * samples:(index + 1) * samples]
                     for index in range(trajectory_count)]
    for trajectory, records in enumerate(by_trajectory):
        root = components.find(trajectory)
        root_family = by_trajectory[root][0].family_id
        bucket = _mix64(seed ^ root_family ^ SPLIT_SALT) % 100
        expected_split = 1 if bucket < 70 else 2 if bucket < 85 else 3
        _require(all(record.split == expected_split for record in records),
                 f"trajectory {trajectory} split assignment is wrong")
    split_by_position: dict[tuple[int, int], int] = {}
    for record in all_records:
        previous_split = split_by_position.setdefault(record.canonical, record.split)
        _require(previous_split == record.split,
                 "a duplicate canonical position crosses dataset splits")

    policy_counts = [sum(record.policy_id == policy for record in all_records)
                     for policy in range(1, 5)]
    split_counts = [sum(record.split == split for record in all_records)
                    for split in range(1, 4)]
    result_policy_counts = _object(results.get("policy_counts"), "results.policy_counts")
    result_split_counts = _object(results.get("split_counts"), "results.split_counts")
    _require(_integer(results.get("records"), "results.records") == len(all_records),
             "manifest result record count is wrong")
    _require(_integer(results.get("duplicate_positions"), "results.duplicate_positions") ==
             duplicates, "manifest duplicate-position count is wrong")
    _require([_integer(result_policy_counts.get(name), f"policy_counts.{name}")
              for name in POLICY_NAMES] == policy_counts,
             "manifest policy counts are wrong")
    _require([_integer(result_split_counts.get(name), f"split_counts.{name}")
              for name in SPLIT_NAMES[1:]] == split_counts,
             "manifest split counts are wrong")
    return AuditSummary(
        trajectories=trajectory_count,
        records=len(all_records),
        shards=shard_count,
        duplicate_positions=duplicates,
        train=split_counts[0],
        validation=split_counts[1],
        test=split_counts[2],
    )


def _canonical_phase(ply: int) -> int:
    for phase, (minimum, maximum) in enumerate(PHASE_BUCKETS):
        if minimum <= ply <= maximum:
            return phase
    raise SourceError("position ply is outside source phase buckets")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="completed source corpus directory")
    arguments = parser.parse_args()
    try:
        summary = audit_source(arguments.source)
    except SourceError as error:
        print(f"invalid position source: {error}", file=sys.stderr)
        return 1
    print(
        "position_source_valid"
        f" trajectories={summary.trajectories} records={summary.records}"
        f" shards={summary.shards} duplicates={summary.duplicate_positions}"
        f" train={summary.train} validation={summary.validation} test={summary.test}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
