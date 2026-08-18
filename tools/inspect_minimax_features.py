#!/usr/bin/env python3
"""Validate and summarize an exported PoE2 minimax feature dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import inspect_minimax_labels as label_auditor


MAGIC = b"POE2FTR\0"
SCHEMA_VERSION = 1
HEADER = struct.Struct("<8sIIIIQQQ32s32sIIII")
RECORD = struct.Struct("<13QIIiiiiiiIHH16B36H49h49hI")
ENDIAN_MARKER = 0x01020304
BOARD_SIZE = 7
CELL_COUNT = BOARD_SIZE * BOARD_SIZE
BOARD_MASK = (1 << CELL_COUNT) - 1
LINE_COUNT = 36
OCCUPIED_GAIN = -(1 << 15)
TERMINAL_FLAG = 1 << 0
PARITY_BACKOFF_FLAG = 1 << 1
KNOWN_FLAGS = TERMINAL_FLAG | PARITY_BACKOFF_FLAG


class FeatureError(ValueError):
    """Raised when a feature dataset violates its on-disk contract."""


@dataclass(frozen=True)
class FeatureRecord:
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
    deepest_completed_nodes: int
    previous_completed_nodes: int
    source_shard: int
    source_ordinal: int
    teacher_value: int
    deepest_value: int
    previous_value: int
    normalized_value: int
    two_ply_closure_value: int
    residual: int
    duplicate_count: int
    policy_id: int
    sample_index: int
    ply: int
    side_to_move: int
    split: int
    mode: int
    completed_depth: int
    deepest_completed_depth: int
    previous_completed_depth: int
    attempted_depth: int
    terminal_depth: int
    best_move: int
    deepest_best_move: int
    previous_best_move: int
    flags: int
    reserved_one: int
    reserved_two: int
    reserved_three: int
    line_patterns: tuple[int, ...]
    own_gains: tuple[int, ...]
    opponent_gains: tuple[int, ...]
    trailing_reserved: int


@dataclass(frozen=True)
class FeatureDataset:
    path: Path
    manifest: dict[str, Any]
    records: tuple[FeatureRecord, ...]


@dataclass(frozen=True)
class AuditSummary:
    records: int
    source_records: int
    duplicates_removed: int
    terminal_records: int
    parity_backoffs: int
    gain_records_checked: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FeatureError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FeatureError(f"could not read {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} does not contain a JSON object")
    return value


def _object(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _integer(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{field} must be an integer")
    return value


def _digest(value: Any, field: str) -> bytes:
    _require(isinstance(value, str) and value.startswith("sha256:"),
             f"{field} must be a SHA-256 digest")
    try:
        digest = bytes.fromhex(value.removeprefix("sha256:"))
    except ValueError as error:
        raise FeatureError(f"{field} is not hexadecimal") from error
    _require(len(digest) == 32, f"{field} has the wrong digest size")
    return digest


def _complete_digests(path: Path, header: str) -> tuple[bytes, bytes]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise FeatureError(f"could not read {path}: {error}") from error
    fields: dict[str, str] = {}
    for line in lines[1:]:
        name, separator, value = line.partition("=")
        _require(bool(separator) and name not in fields, f"{path} is malformed")
        fields[name] = value
    _require(lines[:1] == [header] and
             set(fields) == {"binary_sha256", "manifest_sha256"},
             f"{path} has the wrong fields")
    try:
        binary = bytes.fromhex(fields["binary_sha256"])
        manifest = bytes.fromhex(fields["manifest_sha256"])
    except ValueError as error:
        raise FeatureError(f"{path} contains a non-hex digest") from error
    _require(len(binary) == 32 and len(manifest) == 32, f"{path} has a wrong-sized digest")
    return binary, manifest


def _scoring_lines() -> tuple[tuple[int, ...], ...]:
    lines: list[tuple[int, ...]] = []

    def add(row: int, column: int, row_step: int, column_step: int) -> None:
        cells = []
        while 0 <= row < BOARD_SIZE and 0 <= column < BOARD_SIZE:
            cells.append(row * BOARD_SIZE + column)
            row += row_step
            column += column_step
        lines.append(tuple(cells))

    for row in range(BOARD_SIZE):
        add(row, 0, 0, 1)
    for column in range(BOARD_SIZE):
        add(0, column, 1, 0)
    for column in range(BOARD_SIZE - 1):
        add(0, column, 1, 1)
    for row in range(1, BOARD_SIZE - 1):
        add(row, 0, 1, 1)
    for column in range(1, BOARD_SIZE):
        add(0, column, 1, -1)
    for row in range(1, BOARD_SIZE - 1):
        add(row, BOARD_SIZE - 1, 1, -1)
    assert len(lines) == LINE_COUNT
    return tuple(lines)


SCORING_LINES = _scoring_lines()
LINE_LENGTHS = tuple(len(line) for line in SCORING_LINES)
PATTERN_OFFSETS: dict[int, int] = {}
_pattern_offset = 0
for _length in range(2, BOARD_SIZE + 1):
    PATTERN_OFFSETS[_length] = _pattern_offset
    _pattern_offset += 3 ** _length
assert _pattern_offset == 3276


def _line_patterns(player_one: int, player_two: int, side_to_move: int) -> tuple[int, ...]:
    patterns = []
    own = player_one if side_to_move == 0 else player_two
    opponent = player_two if side_to_move == 0 else player_one
    for line in SCORING_LINES:
        code = 0
        place = 1
        for cell in line:
            bit = 1 << cell
            digit = 1 if own & bit else 2 if opponent & bit else 0
            code += digit * place
            place *= 3
        patterns.append(PATTERN_OFFSETS[len(line)] + code)
    return tuple(patterns)


def _score_player(pieces: int) -> int:
    total = 0
    pieces_in_lines = 0
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    for row in range(BOARD_SIZE):
        for column in range(BOARD_SIZE):
            index = row * BOARD_SIZE + column
            if not pieces & (1 << index):
                continue
            for row_step, column_step in directions:
                previous_row = row - row_step
                previous_column = column - column_step
                if (0 <= previous_row < BOARD_SIZE and 0 <= previous_column < BOARD_SIZE and
                        pieces & (1 << (previous_row * BOARD_SIZE + previous_column))):
                    continue
                run_bits = 0
                length = 0
                scan_row = row
                scan_column = column
                while (0 <= scan_row < BOARD_SIZE and 0 <= scan_column < BOARD_SIZE and
                       pieces & (1 << (scan_row * BOARD_SIZE + scan_column))):
                    run_bits |= 1 << (scan_row * BOARD_SIZE + scan_column)
                    length += 1
                    scan_row += row_step
                    scan_column += column_step
                if length >= 2:
                    total += 1 << (length - 1)
                    pieces_in_lines |= run_bits
    return total + (pieces & ~pieces_in_lines).bit_count()


def _normalized_value(record: FeatureRecord) -> int:
    p1_score = _score_player(record.player_one)
    p2_score = _score_player(record.player_two)
    p1_stones = (record.ply + 1) // 2
    p2_stones = record.ply // 2
    advantage = (p1_score - p1_stones + 25) * 2 - ((p2_score - p2_stones + 24) * 2 + 11)
    return advantage if record.side_to_move == 0 else -advantage


def _two_ply_closure_value(record: FeatureRecord) -> int:
    legal = [index for index, gain in enumerate(record.own_gains) if gain != OCCUPIED_GAIN]
    if not legal:
        return record.normalized_value
    if len(legal) == 1:
        return record.normalized_value + 2 * (record.own_gains[legal[0]] - 1)
    reply_order = sorted(legal, key=lambda index: (-record.opponent_gains[index], index))
    best_reply = reply_order[0]
    second_reply_gain = record.opponent_gains[reply_order[1]]
    best_reply_gain = record.opponent_gains[best_reply]
    pair_value = max(
        record.own_gains[index] -
        (second_reply_gain if index == best_reply else best_reply_gain)
        for index in legal
    )
    return record.normalized_value + 2 * pair_value


def _decode_record(fields: tuple[int, ...]) -> FeatureRecord:
    patterns_begin = 40
    own_begin = patterns_begin + LINE_COUNT
    opponent_begin = own_begin + CELL_COUNT
    return FeatureRecord(
        *fields[:patterns_begin],
        tuple(fields[patterns_begin:own_begin]),
        tuple(fields[own_begin:opponent_begin]),
        tuple(fields[opponent_begin:opponent_begin + CELL_COUNT]),
        fields[-1],
    )


def _label_set_digest(label_directory: Path, shard_count: int) -> bytes:
    shard_root = Path(label_directory) / "shards"
    _require(shard_root.is_dir(), "supplied label corpus has no shards directory")
    shards: dict[int, Path] = {}
    for directory in shard_root.iterdir():
        _require(directory.is_dir(), "supplied label shards contain a non-directory entry")
        manifest = _load_json(directory / "manifest.json")
        corpus = _object(manifest.get("corpus"), "label corpus")
        index = _integer(corpus.get("shard_index"), "label shard index")
        _require(index not in shards, "supplied label corpus repeats a shard index")
        shards[index] = directory
    _require(set(shards) == set(range(shard_count)),
             "supplied label corpus does not contain every feature-input shard")
    aggregate = hashlib.sha256()
    for index in range(shard_count):
        directory = shards[index]
        binary_digest, manifest_digest = _complete_digests(
            directory / "COMPLETE", "poe2-minimax-labels")
        try:
            binary = (directory / "labels.bin").read_bytes()
            manifest = (directory / "manifest.json").read_bytes()
        except OSError as error:
            raise FeatureError(f"could not read supplied label shard {index}: {error}") from error
        _require(hashlib.sha256(binary).digest() == binary_digest and
                 hashlib.sha256(manifest).digest() == manifest_digest,
                 f"supplied label shard {index} has a bad digest")
        aggregate.update(binary_digest)
        aggregate.update(manifest_digest)
    return aggregate.digest()


def load_feature_dataset(directory: Path) -> FeatureDataset:
    """Read a feature artifact after authenticating its framing and digests."""
    directory = Path(directory)
    _require(directory.is_dir(), "feature dataset directory does not exist")
    _require((directory / "COMPLETE").is_file(), "feature dataset has no COMPLETE marker")
    _require(not (directory / "INCOMPLETE").exists(), "feature dataset remains incomplete")
    binary_digest, manifest_digest = _complete_digests(
        directory / "COMPLETE", "poe2-minimax-features")
    try:
        binary = (directory / "features.bin").read_bytes()
        manifest_bytes = (directory / "manifest.json").read_bytes()
    except OSError as error:
        raise FeatureError(f"could not read feature artifact: {error}") from error
    _require(hashlib.sha256(binary).digest() == binary_digest,
             "feature binary digest differs from COMPLETE")
    _require(hashlib.sha256(manifest_bytes).digest() == manifest_digest,
             "feature manifest digest differs from COMPLETE")
    manifest = _load_json(directory / "manifest.json")
    _require(_digest(manifest.get("binary_digest"), "binary_digest") == binary_digest,
             "feature binary digest differs from manifest")
    _require(len(binary) >= HEADER.size, "feature binary is shorter than its header")
    header = HEADER.unpack_from(binary)
    (magic, schema, header_size, record_size, endian, record_count, source_records,
     duplicates_removed, corpus_digest, label_set_digest, line_count, cell_count,
     shard_count, reserved) = header
    _require(magic == MAGIC and schema == SCHEMA_VERSION and header_size == HEADER.size and
             record_size == RECORD.size and endian == ENDIAN_MARKER,
             "feature binary header is incompatible")
    _require(line_count == LINE_COUNT and cell_count == CELL_COUNT and shard_count > 0 and
             reserved == 0, "feature binary dimensions are invalid")
    _require(source_records == record_count + duplicates_removed,
             "feature binary deduplication accounting is invalid")
    _require(len(binary) == HEADER.size + record_count * RECORD.size,
             "feature binary length differs from its record count")

    corpus = _object(manifest.get("corpus"), "corpus")
    inputs = _object(manifest.get("inputs"), "inputs")
    features = _object(manifest.get("features"), "features")
    results = _object(manifest.get("results"), "results")
    exporter_build = _object(manifest.get("exporter_build"), "exporter_build")
    corpus_id = corpus.get("id")
    _require(manifest.get("schema") == "poe2-minimax-features" and
             _integer(manifest.get("schema_version"), "schema_version") == SCHEMA_VERSION,
             "feature manifest schema is unsupported")
    _require(isinstance(corpus_id, str) and hashlib.sha256(corpus_id.encode()).digest() ==
             corpus_digest and _digest(corpus.get("digest"), "corpus.digest") == corpus_digest,
             "feature corpus identity is inconsistent")
    _require(_digest(inputs.get("label_set_digest"), "inputs.label_set_digest") ==
             label_set_digest, "feature label-set digest differs between binary and manifest")
    _require(_integer(inputs.get("shards"), "inputs.shards") == shard_count and
             _integer(inputs.get("source_records"), "inputs.source_records") == source_records,
             "feature input counts differ between binary and manifest")
    _require(features.get("definition") == "b-residual-line-pattern-gains-v1" and
             features.get("line_order") ==
             "rows-columns-down-diagonals-up-diagonals-v1" and
             features.get("line_pattern_encoding") ==
             "length-offset-plus-base3-empty0-stm1-opponent2-v1" and
             features.get("gain_encoding") ==
             "raw-score-gain-stm-first-int16-min-occupied-v1" and
             features.get("line_lengths") == list(LINE_LENGTHS),
             "feature definition is unsupported")
    _require(_integer(results.get("records"), "results.records") == record_count and
             _integer(results.get("duplicates_removed"), "results.duplicates_removed") ==
             duplicates_removed, "feature result counts differ from binary")
    for field in ("git_commit", "project_version", "compiler_id", "compiler_version",
                  "build_type", "target_processor"):
        _require(isinstance(exporter_build.get(field), str),
                 f"exporter_build.{field} must be a string")
    _require(isinstance(exporter_build.get("git_dirty"), bool) and
             isinstance(exporter_build.get("native_architecture"), bool),
             "exporter build booleans are invalid")

    records = tuple(
        _decode_record(RECORD.unpack_from(binary, HEADER.size + index * RECORD.size))
        for index in range(record_count)
    )
    return FeatureDataset(directory.resolve(), manifest, records)


def audit_feature_dataset(directory: Path, label_directory: Path | None = None,
                          gain_samples: int = 256) -> AuditSummary:
    dataset = load_feature_dataset(directory)
    manifest = dataset.manifest
    inputs = _object(manifest["inputs"], "inputs")
    results = _object(manifest["results"], "results")
    source_records = _integer(inputs["source_records"], "inputs.source_records")
    shard_count = _integer(inputs["shards"], "inputs.shards")
    if label_directory is not None:
        _require(_label_set_digest(label_directory, shard_count) ==
                 _digest(inputs["label_set_digest"], "inputs.label_set_digest"),
                 "supplied label corpus differs from the feature inputs")

    previous_key: tuple[int, int] | None = None
    terminal_records = 0
    parity_backoffs = 0
    split_counts = {1: 0, 2: 0, 3: 0}
    duplicate_total = 0
    for index, record in enumerate(dataset.records):
        label = f"record {index}"
        occupied = record.player_one | record.player_two
        _require((occupied & ~BOARD_MASK) == 0 and
                 (record.player_one & record.player_two) == 0,
                 f"{label} has invalid bitboards")
        _require(occupied.bit_count() == record.ply and
                 record.player_one.bit_count() == (record.ply + 1) // 2 and
                 record.player_two.bit_count() == record.ply // 2 and
                 record.side_to_move == record.ply % 2,
                 f"{label} has inconsistent piece counts")
        _require((record.key_low, record.key_high) == label_auditor.canonical_key(
            record.player_one, record.player_two, record.side_to_move),
            f"{label} has the wrong canonical key")
        key = (record.key_low, record.key_high)
        _require(previous_key is None or previous_key < key,
                 "feature records are not uniquely sorted by canonical key")
        previous_key = key
        _require(record.source_shard < shard_count and record.split in split_counts and
                 1 <= record.policy_id <= 4 and record.duplicate_count > 0,
                 f"{label} has invalid provenance")
        _require(record.mode in (1, 2) and
                 0 < record.completed_depth <= record.deepest_completed_depth <=
                 record.attempted_depth <= record.terminal_depth and
                 record.completed_depth % 2 == record.terminal_depth % 2 and
                 0 < record.completed_nodes <= record.nodes and
                 0 < record.deepest_completed_nodes <= record.nodes,
                 f"{label} has invalid label depth")
        _require(record.best_move < CELL_COUNT and not occupied & (1 << record.best_move),
                 f"{label} has an illegal best move")
        _require(record.deepest_best_move < CELL_COUNT and
                 not occupied & (1 << record.deepest_best_move),
                 f"{label} has an illegal deepest best move")
        if record.previous_completed_depth == 0:
            _require(record.previous_completed_nodes == 0 and record.previous_value == 0 and
                     record.previous_best_move == 0xFF,
                     f"{label} has a noncanonical absent previous result")
        else:
            _require(record.previous_completed_depth + 1 == record.deepest_completed_depth and
                     0 < record.previous_completed_nodes <= record.deepest_completed_nodes and
                     record.previous_best_move < CELL_COUNT and
                     not occupied & (1 << record.previous_best_move),
                     f"{label} has an invalid previous result")
        expected_attempted = (record.deepest_completed_depth if
                              record.deepest_completed_depth == record.terminal_depth else
                              record.deepest_completed_depth + 1)
        _require(record.attempted_depth == expected_attempted,
                 f"{label} has an invalid attempted depth")
        if record.completed_depth == record.deepest_completed_depth:
            _require(record.completed_nodes == record.deepest_completed_nodes and
                     record.teacher_value == record.deepest_value and
                     record.best_move == record.deepest_best_move,
                     f"{label} selected result differs from its deepest result")
        else:
            _require(record.completed_depth == record.previous_completed_depth and
                     record.completed_nodes == record.previous_completed_nodes and
                     record.teacher_value == record.previous_value and
                     record.best_move == record.previous_best_move,
                     f"{label} selected result differs from its previous result")
        _require(record.residual ==
                 record.teacher_value - record.two_ply_closure_value,
                 f"{label} has an inconsistent residual")
        _require(record.flags & ~KNOWN_FLAGS == 0 and record.reserved_one == 0 and
                 record.reserved_two == 0 and record.reserved_three == 0 and
                 record.trailing_reserved == 0,
                 f"{label} has nonzero reserved fields")
        terminal = record.completed_depth == record.terminal_depth
        _require(bool(record.flags & TERMINAL_FLAG) == terminal,
                 f"{label} has the wrong terminal flag")
        _require(bool(record.flags & PARITY_BACKOFF_FLAG) ==
                 (record.completed_depth < record.deepest_completed_depth),
                 f"{label} has the wrong parity-backoff flag")
        _require(record.line_patterns == _line_patterns(
            record.player_one, record.player_two, record.side_to_move),
            f"{label} line patterns differ from its board")
        for cell in range(CELL_COUNT):
            is_occupied = bool(occupied & (1 << cell))
            _require((record.own_gains[cell] == OCCUPIED_GAIN) == is_occupied and
                     (record.opponent_gains[cell] == OCCUPIED_GAIN) == is_occupied,
                     f"{label} gain occupancy sentinel is wrong")
            if not is_occupied:
                _require(record.own_gains[cell] >= 1 and record.opponent_gains[cell] >= 1,
                         f"{label} has a nonpositive legal gain")
        _require(record.normalized_value == _normalized_value(record),
                 f"{label} normalized static value is wrong")
        _require(record.two_ply_closure_value == _two_ply_closure_value(record),
                 f"{label} two-ply closure is wrong")
        terminal_records += terminal
        parity_backoffs += bool(record.flags & PARITY_BACKOFF_FLAG)
        split_counts[record.split] += 1
        duplicate_total += record.duplicate_count - 1

    _require(len(dataset.records) + duplicate_total == source_records,
             "feature record multiplicities do not reconstruct the source count")
    _require(_integer(results["terminal_records"], "results.terminal_records") ==
             terminal_records and
             _integer(results["parity_backoffs"], "results.parity_backoffs") ==
             parity_backoffs, "feature flag counts differ from the manifest")
    manifest_splits = _object(results["split_counts"], "results.split_counts")
    _require([_integer(manifest_splits[name], f"results.split_counts.{name}")
              for name in ("train", "validation", "test")] ==
             [split_counts[index] for index in range(1, 4)],
             "feature split counts differ from the manifest")

    samples = min(max(gain_samples, 0), len(dataset.records))
    sample_indices = ({index * len(dataset.records) // samples for index in range(samples)}
                      if samples else set())
    for index in sample_indices:
        record = dataset.records[index]
        own_bits = record.player_one if record.side_to_move == 0 else record.player_two
        opponent_bits = record.player_two if record.side_to_move == 0 else record.player_one
        own_score = _score_player(own_bits)
        opponent_score = _score_player(opponent_bits)
        occupied = own_bits | opponent_bits
        for cell in range(CELL_COUNT):
            bit = 1 << cell
            if occupied & bit:
                continue
            _require(record.own_gains[cell] == _score_player(own_bits | bit) - own_score and
                     record.opponent_gains[cell] ==
                     _score_player(opponent_bits | bit) - opponent_score,
                     f"record {index} marginal gains differ from independent scoring")

    return AuditSummary(
        records=len(dataset.records),
        source_records=source_records,
        duplicates_removed=source_records - len(dataset.records),
        terminal_records=terminal_records,
        parity_backoffs=parity_backoffs,
        gain_records_checked=len(sample_indices),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="completed feature dataset directory")
    parser.add_argument("--labels", type=Path,
                        help="optional complete label corpus used to verify input identity")
    parser.add_argument("--gain-samples", type=int, default=256,
                        help="deterministic records whose gains are independently recomputed")
    arguments = parser.parse_args()
    if arguments.gain_samples < 0:
        parser.error("--gain-samples must be nonnegative")
    try:
        summary = audit_feature_dataset(arguments.dataset, arguments.labels,
                                        arguments.gain_samples)
    except FeatureError as error:
        print(f"invalid feature dataset: {error}", file=sys.stderr)
        return 1
    print(
        "feature_dataset_valid"
        f" source_records={summary.source_records}"
        f" records={summary.records}"
        f" duplicates_removed={summary.duplicates_removed}"
        f" terminal={summary.terminal_records}"
        f" parity_backoffs={summary.parity_backoffs}"
        f" gain_records_checked={summary.gain_records_checked}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
