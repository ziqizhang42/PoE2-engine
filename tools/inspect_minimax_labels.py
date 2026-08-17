#!/usr/bin/env python3
"""Validate and summarize a completed PoE2 minimax label dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAGIC = b"POE2LBL\0"
SCHEMA_VERSION = 1
HEADER = struct.Struct("<8sIIIIQQ32s32sII")
RECORD = struct.Struct("<11QIIi8BHH")
ENDIAN_MARKER = 0x01020304
BOARD_SIZE = 7
CELL_COUNT = BOARD_SIZE * BOARD_SIZE
BOARD_MASK = (1 << CELL_COUNT) - 1
SIDE_TO_MOVE_BIT = 1 << CELL_COUNT


class DatasetError(ValueError):
    """Raised when a dataset violates its on-disk contract."""


@dataclass(frozen=True)
class AuditSummary:
    inputs: int
    records: int
    exact_records: int
    teacher_records: int
    shard_index: int
    shard_count: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetError(message)


def _digest_bytes(text: Any, field: str) -> bytes:
    _require(isinstance(text, str), f"{field} must be a string")
    prefix = "sha256:"
    _require(text.startswith(prefix), f"{field} must use the sha256: prefix")
    value = text[len(prefix) :]
    _require(len(value) == 64, f"{field} must contain 64 hexadecimal digits")
    try:
        return bytes.fromhex(value)
    except ValueError as error:
        raise DatasetError(f"{field} is not hexadecimal") from error


def _transform_square(symmetry: int, row: int, column: int) -> tuple[int, int]:
    last = BOARD_SIZE - 1
    transforms = (
        (row, column),
        (column, last - row),
        (last - row, last - column),
        (last - column, row),
        (last - row, column),
        (row, last - column),
        (column, row),
        (last - column, last - row),
    )
    return transforms[symmetry]


def _transform_bits(bits: int, symmetry: int) -> int:
    transformed = 0
    remaining = bits & BOARD_MASK
    while remaining:
        bit = remaining & -remaining
        index = bit.bit_length() - 1
        row, column = divmod(index, BOARD_SIZE)
        target_row, target_column = _transform_square(symmetry, row, column)
        transformed |= 1 << (target_row * BOARD_SIZE + target_column)
        remaining ^= bit
    return transformed


def canonical_key(player_one: int, player_two: int, side_to_move: int) -> tuple[int, int]:
    candidates = []
    for symmetry in range(8):
        low = _transform_bits(player_one, symmetry)
        high = _transform_bits(player_two, symmetry)
        if side_to_move == 1:
            high |= SIDE_TO_MOVE_BIT
        candidates.append((low, high))
    return min(candidates)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetError(f"could not read manifest: {error}") from error
    _require(isinstance(value, dict), "manifest root must be an object")
    return value


def _object(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _integer(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{field} must be an integer")
    return value


def audit_dataset(directory: Path, source_path: Path | None = None) -> AuditSummary:
    directory = Path(directory)
    complete = directory / "COMPLETE"
    incomplete = directory / "INCOMPLETE"
    binary_path = directory / "labels.bin"
    manifest_path = directory / "manifest.json"
    _require(directory.is_dir(), "dataset directory does not exist")
    _require(complete.is_file(), "dataset has no COMPLETE marker")
    _require(not incomplete.exists(), "dataset still has an INCOMPLETE marker")
    _require(not (directory / "labels.bin.tmp").exists(), "temporary binary still exists")
    _require(not (directory / "manifest.json.tmp").exists(), "temporary manifest still exists")
    _require(binary_path.is_file(), "labels.bin is missing")
    _require(manifest_path.is_file(), "manifest.json is missing")

    manifest = _load_manifest(manifest_path)
    try:
        marker_lines = complete.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise DatasetError(f"could not read COMPLETE marker: {error}") from error
    marker = {}
    for line in marker_lines[1:]:
        name, separator, value = line.partition("=")
        _require(bool(separator) and name not in marker, "COMPLETE marker is malformed")
        marker[name] = value
    _require(marker_lines[:1] == ["poe2-minimax-labels"], "COMPLETE marker header is wrong")
    _require(set(marker) == {"binary_sha256", "manifest_sha256"},
             "COMPLETE marker digest fields are wrong")
    try:
        marker_binary_digest = bytes.fromhex(marker["binary_sha256"])
        marker_manifest_digest = bytes.fromhex(marker["manifest_sha256"])
    except ValueError as error:
        raise DatasetError("COMPLETE marker digest is not hexadecimal") from error
    _require(len(marker_binary_digest) == 32 and len(marker_manifest_digest) == 32,
             "COMPLETE marker digest has the wrong length")
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise DatasetError(f"could not read manifest bytes: {error}") from error
    _require(hashlib.sha256(manifest_bytes).digest() == marker_manifest_digest,
             "manifest digest does not match the COMPLETE marker")
    _require(manifest.get("schema") == "poe2-minimax-labels", "unknown manifest schema")
    _require(
        _integer(manifest.get("schema_version"), "schema_version") == SCHEMA_VERSION,
        "unsupported schema version",
    )
    format_info = _object(manifest.get("format"), "format")
    _require(_integer(format_info.get("header_bytes"), "format.header_bytes") == HEADER.size,
             "manifest header size is wrong")
    _require(_integer(format_info.get("record_bytes"), "format.record_bytes") == RECORD.size,
             "manifest record size is wrong")

    try:
        binary = binary_path.read_bytes()
    except OSError as error:
        raise DatasetError(f"could not read labels.bin: {error}") from error
    binary_digest = hashlib.sha256(binary).digest()
    _require(binary_digest == marker_binary_digest,
             "labels.bin digest does not match the COMPLETE marker")
    _require(
        binary_digest == _digest_bytes(manifest.get("binary_digest"), "binary_digest"),
        "labels.bin digest does not match the manifest",
    )
    _require(len(binary) >= HEADER.size, "labels.bin is shorter than its header")

    (
        magic,
        schema_version,
        header_size,
        record_size,
        endian_marker,
        record_count,
        input_count,
        source_digest,
        corpus_digest,
        shard_index,
        shard_count,
    ) = HEADER.unpack_from(binary)
    _require(magic == MAGIC, "binary magic is wrong")
    _require(schema_version == SCHEMA_VERSION, "binary schema version is wrong")
    _require(header_size == HEADER.size, "binary header size is wrong")
    _require(record_size == RECORD.size, "binary record size is wrong")
    _require(endian_marker == ENDIAN_MARKER, "binary endian marker is wrong")
    _require(len(binary) == HEADER.size + record_count * RECORD.size,
             "binary length does not match its record count")

    corpus = _object(manifest.get("corpus"), "corpus")
    source = _object(manifest.get("source"), "source")
    search = _object(manifest.get("search"), "search")
    results = _object(manifest.get("results"), "results")
    build = _object(manifest.get("build"), "build")
    corpus_id = corpus.get("id")
    _require(isinstance(corpus_id, str) and corpus_id, "corpus.id must be nonempty")
    _require(corpus_digest == hashlib.sha256(corpus_id.encode("utf-8")).digest(),
             "binary corpus digest does not match corpus.id")
    _require(corpus_digest == _digest_bytes(corpus.get("digest"), "corpus.digest"),
             "corpus digest differs between binary and manifest")
    _require(source_digest == _digest_bytes(source.get("digest"), "source.digest"),
             "source digest differs between binary and manifest")
    if source_path is not None:
        try:
            supplied_source_digest = hashlib.sha256(Path(source_path).read_bytes()).digest()
        except OSError as error:
            raise DatasetError(f"could not read supplied source: {error}") from error
        _require(source_digest == supplied_source_digest,
                 "supplied source digest differs from the dataset")
    _require(shard_count > 0 and shard_index < shard_count, "binary shard coordinates are invalid")
    _require(_integer(corpus.get("shard_index"), "corpus.shard_index") == shard_index,
             "manifest shard index differs from binary")
    _require(_integer(corpus.get("shard_count"), "corpus.shard_count") == shard_count,
             "manifest shard count differs from binary")
    _require(_integer(source.get("positions"), "source.positions") == input_count,
             "manifest input count differs from binary")
    _require(_integer(results.get("records"), "results.records") == record_count,
             "manifest record count differs from binary")
    unsolved = _integer(results.get("unsolved"), "results.unsolved")
    unsolved_lines = results.get("unsolved_source_lines")
    _require(isinstance(unsolved_lines, list), "results.unsolved_source_lines must be an array")
    _require(len(unsolved_lines) == unsolved, "unsolved count does not match its line array")
    _require(record_count + unsolved == input_count, "records plus unsolved does not equal inputs")
    node_limit = _integer(search.get("node_limit"), "search.node_limit")
    _require(node_limit > 0, "search.node_limit must be positive")
    expected_mode = {"exact": 1, "teacher": 2}.get(search.get("mode"))
    _require(expected_mode is not None, "search.mode is unknown")
    _require(search.get("evaluator") == "b", "search evaluator is unknown")
    _require(search.get("symmetry") is True, "symmetry must be enabled")
    _require(search.get("two_ply_closure") is True, "two-ply closure must be enabled")
    _require(_integer(search.get("hash_bytes_requested"), "search.hash_bytes_requested") >= 0,
             "requested hash bytes is negative")
    _require(_integer(search.get("hash_bytes_effective"), "search.hash_bytes_effective") >= 0,
             "effective hash bytes is negative")
    _require(_integer(search.get("hash_capacity"), "search.hash_capacity") >= 0,
             "hash capacity is negative")
    workers_requested = _integer(search.get("workers_requested"), "search.workers_requested")
    workers_used = _integer(search.get("workers_used"), "search.workers_used")
    _require(workers_requested > 0, "requested workers must be positive")
    _require(workers_used == min(workers_requested, input_count),
             "used worker count does not match inputs and requested workers")
    _require(isinstance(search.get("require_all"), bool), "search.require_all must be boolean")
    if search.get("require_all"):
        _require(unsolved == 0, "require-all dataset contains unsolved positions")
    for field in ("git_commit", "project_version", "compiler_id", "compiler_version",
                  "build_type", "target_processor"):
        _require(isinstance(build.get(field), str), f"build.{field} must be a string")
    _require(isinstance(build.get("git_dirty"), bool), "build.git_dirty must be boolean")
    _require(isinstance(build.get("native_architecture"), bool),
             "build.native_architecture must be boolean")

    exact_records = 0
    teacher_records = 0
    terminal_records = 0
    source_ordinals = set()
    for record_index in range(record_count):
        offset = HEADER.size + record_index * RECORD.size
        fields = RECORD.unpack_from(binary, offset)
        (
            player_one,
            player_two,
            key_low,
            key_high,
            _source_id,
            _family_id,
            _trajectory_id,
            _parent_id,
            _trajectory_index,
            nodes,
            completed_nodes,
            source_line,
            source_ordinal,
            _value,
            ply,
            side_to_move,
            mode,
            completed_depth,
            attempted_depth,
            terminal_depth,
            best_move,
            split,
            _policy_id,
            _sample_index,
        ) = fields
        label = f"record {record_index}"
        _require((player_one | player_two) & ~BOARD_MASK == 0, f"{label} has off-board bits")
        _require(player_one & player_two == 0, f"{label} has overlapping bitboards")
        occupied = player_one | player_two
        _require(occupied.bit_count() == ply, f"{label} ply does not match occupied squares")
        _require(player_one.bit_count() == (ply + 1) // 2, f"{label} has the wrong P1 count")
        _require(player_two.bit_count() == ply // 2, f"{label} has the wrong P2 count")
        _require(side_to_move in (0, 1), f"{label} has an invalid side to move")
        _require(side_to_move == ply % 2, f"{label} side to move disagrees with ply")
        _require((key_low, key_high) == canonical_key(player_one, player_two, side_to_move),
                 f"{label} canonical key is wrong")
        _require(source_line > 0, f"{label} source line must be positive")
        _require(source_ordinal < input_count, f"{label} source ordinal is out of range")
        _require(source_ordinal not in source_ordinals, f"{label} source ordinal is duplicated")
        source_ordinals.add(source_ordinal)
        _require(mode == expected_mode, f"{label} mode differs from the manifest")
        _require(nodes <= node_limit, f"{label} exceeds the node limit")
        _require(0 < completed_nodes <= nodes, f"{label} completed-node count is invalid")
        empty_count = CELL_COUNT - ply
        expected_terminal_depth = max(1, empty_count - 2)
        _require(terminal_depth == expected_terminal_depth, f"{label} terminal depth is wrong")
        _require(0 < completed_depth <= attempted_depth <= terminal_depth,
                 f"{label} depth fields are inconsistent")
        _require(best_move < CELL_COUNT and not (occupied & (1 << best_move)),
                 f"{label} best move is illegal")
        _require(split in (1, 2, 3), f"{label} dataset split is invalid")
        if mode == 1:
            exact_records += 1
            _require(completed_depth == terminal_depth and attempted_depth == terminal_depth,
                     f"{label} exact label did not reach terminal depth")
            _require(completed_nodes == nodes,
                     f"{label} exact label has post-completion nodes")
        else:
            teacher_records += 1
            if completed_depth < terminal_depth:
                _require(attempted_depth == completed_depth + 1,
                         f"{label} teacher attempted depth is wrong")
        if completed_depth == terminal_depth:
            terminal_records += 1

    _require(_integer(results.get("terminal_records"), "results.terminal_records") ==
             terminal_records, "terminal record count differs from binary")
    return AuditSummary(
        inputs=input_count,
        records=record_count,
        exact_records=exact_records,
        teacher_records=teacher_records,
        shard_index=shard_index,
        shard_count=shard_count,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="completed dataset directory")
    parser.add_argument("--source", type=Path, help="optional raw source file to verify")
    arguments = parser.parse_args()
    try:
        summary = audit_dataset(arguments.dataset, arguments.source)
    except DatasetError as error:
        print(f"invalid label dataset: {error}", file=sys.stderr)
        return 1
    print(
        "label_dataset_valid"
        f" inputs={summary.inputs} records={summary.records}"
        f" exact={summary.exact_records} teacher={summary.teacher_records}"
        f" shard={summary.shard_index}/{summary.shard_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
