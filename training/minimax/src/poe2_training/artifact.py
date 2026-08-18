"""Dependency-free authentication of a minimax feature artifact."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAGIC = b"POE2FTR\0"
SCHEMA_VERSION = 1
HEADER = struct.Struct("<8sIIIIQQQ32s32sIIII")
RECORD_BYTES = 432
ENDIAN_MARKER = 0x01020304
LINE_COUNT = 36
CELL_COUNT = 49
FEATURE_DEFINITION = "b-residual-line-pattern-gains-v1"
COMPLETE_HEADER = "poe2-minimax-features"


class FeatureArtifactError(ValueError):
    """Raised when an input is not an authenticated feature artifact."""


@dataclass(frozen=True)
class FeatureArtifact:
    """Authenticated framing and provenance needed by the training loader."""

    directory: Path
    binary_path: Path
    manifest: dict[str, Any]
    record_count: int
    source_record_count: int
    duplicates_removed: int
    shard_count: int
    split_counts: tuple[int, int, int]
    binary_digest: str
    manifest_digest: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FeatureArtifactError(message)


def _object(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _integer(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool),
             f"{field} must be an integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise FeatureArtifactError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def _read_complete(path: Path) -> tuple[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise FeatureArtifactError(f"could not read {path}: {error}") from error
    _require(lines[:1] == [COMPLETE_HEADER], f"{path} has the wrong header")
    fields: dict[str, str] = {}
    for line in lines[1:]:
        name, separator, value = line.partition("=")
        _require(bool(separator) and name not in fields, f"{path} is malformed")
        fields[name] = value
    _require(set(fields) == {"binary_sha256", "manifest_sha256"},
             f"{path} has the wrong fields")
    for name, value in fields.items():
        try:
            decoded = bytes.fromhex(value)
        except ValueError as error:
            raise FeatureArtifactError(f"{path} contains a non-hex {name}") from error
        _require(len(decoded) == 32, f"{path} contains a wrong-sized {name}")
    return fields["binary_sha256"], fields["manifest_sha256"]


def _read_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        contents = path.read_bytes()
        value = json.loads(contents)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FeatureArtifactError(f"could not read {path}: {error}") from error
    _require(isinstance(value, dict), f"{path} does not contain a JSON object")
    return value, contents


def _digest_text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.startswith("sha256:"),
             f"{field} must be a SHA-256 digest")
    digest = value.removeprefix("sha256:")
    try:
        decoded = bytes.fromhex(digest)
    except ValueError as error:
        raise FeatureArtifactError(f"{field} is not hexadecimal") from error
    _require(len(decoded) == 32, f"{field} has the wrong digest size")
    return digest


def open_feature_artifact(directory: Path | str, *, verify_digest: bool = True,
                          require_clean: bool = True) -> FeatureArtifact:
    """Authenticate a feature directory without loading its records into memory."""
    directory = Path(directory).resolve()
    _require(directory.is_dir(), f"feature directory does not exist: {directory}")
    _require(not (directory / "INCOMPLETE").exists(),
             "feature directory still has an INCOMPLETE marker")
    complete_path = directory / "COMPLETE"
    binary_path = directory / "features.bin"
    manifest_path = directory / "manifest.json"
    _require(complete_path.is_file(), "feature directory has no COMPLETE marker")
    _require(binary_path.is_file(), "feature directory has no features.bin")
    _require(manifest_path.is_file(), "feature directory has no manifest.json")

    binary_digest, manifest_digest = _read_complete(complete_path)
    manifest, manifest_bytes = _read_manifest(manifest_path)
    if verify_digest:
        _require(_sha256(binary_path) == binary_digest,
                 "features.bin digest differs from COMPLETE")
    _require(hashlib.sha256(manifest_bytes).hexdigest() == manifest_digest,
             "manifest.json digest differs from COMPLETE")

    _require(manifest.get("schema") == "poe2-minimax-features" and
             _integer(manifest.get("schema_version"), "schema_version") == SCHEMA_VERSION,
             "feature manifest schema is unsupported")
    _require(_digest_text(manifest.get("binary_digest"), "binary_digest") == binary_digest,
             "manifest binary digest differs from COMPLETE")

    try:
        with binary_path.open("rb") as source:
            header_bytes = source.read(HEADER.size)
    except OSError as error:
        raise FeatureArtifactError(f"could not read {binary_path}: {error}") from error
    _require(len(header_bytes) == HEADER.size, "features.bin is shorter than its header")
    (magic, schema, header_size, record_size, endian, record_count, source_records,
     duplicates_removed, corpus_digest, label_set_digest, line_count, cell_count,
     shard_count, reserved) = HEADER.unpack(header_bytes)
    _require(magic == MAGIC and schema == SCHEMA_VERSION and header_size == HEADER.size and
             record_size == RECORD_BYTES and endian == ENDIAN_MARKER,
             "feature binary header is incompatible")
    _require(record_count > 0 and line_count == LINE_COUNT and cell_count == CELL_COUNT and
             shard_count > 0 and reserved == 0,
             "feature binary dimensions are invalid")
    _require(source_records == record_count + duplicates_removed,
             "feature binary deduplication accounting is invalid")
    try:
        file_size = binary_path.stat().st_size
    except OSError as error:
        raise FeatureArtifactError(f"could not stat {binary_path}: {error}") from error
    _require(file_size == HEADER.size + record_count * RECORD_BYTES,
             "features.bin length differs from its record count")

    corpus = _object(manifest.get("corpus"), "corpus")
    inputs = _object(manifest.get("inputs"), "inputs")
    features = _object(manifest.get("features"), "features")
    results = _object(manifest.get("results"), "results")
    exporter_build = _object(manifest.get("exporter_build"), "exporter_build")
    label_build = _object(inputs.get("label_build"), "inputs.label_build")
    corpus_id = corpus.get("id")
    _require(isinstance(corpus_id, str) and
             hashlib.sha256(corpus_id.encode()).digest() == corpus_digest,
             "feature corpus identity is inconsistent")
    _require(_digest_text(corpus.get("digest"), "corpus.digest") == corpus_digest.hex(),
             "feature corpus digest differs between binary and manifest")
    _require(_digest_text(inputs.get("label_set_digest"), "inputs.label_set_digest") ==
             label_set_digest.hex(),
             "label-set digest differs between binary and manifest")
    _require(_integer(inputs.get("shards"), "inputs.shards") == shard_count and
             _integer(inputs.get("source_records"), "inputs.source_records") == source_records,
             "input counts differ between binary and manifest")
    _require(features.get("definition") == FEATURE_DEFINITION and
             _integer(features.get("line_count"), "features.line_count") == LINE_COUNT and
             _integer(features.get("cell_count"), "features.cell_count") == CELL_COUNT,
             "feature definition is unsupported")
    _require(_integer(results.get("records"), "results.records") == record_count and
             _integer(results.get("duplicates_removed"), "results.duplicates_removed") ==
             duplicates_removed,
             "result counts differ between binary and manifest")
    split_values = _object(results.get("split_counts"), "results.split_counts")
    split_counts = tuple(_integer(split_values.get(name), f"results.split_counts.{name}")
                         for name in ("train", "validation", "test"))
    _require(sum(split_counts) == record_count, "manifest split counts do not sum to records")

    if require_clean:
        _require(exporter_build.get("git_dirty") is False,
                 "feature exporter build was dirty")
        _require(label_build.get("git_dirty") is False,
                 "label generator build was dirty")
    for owner, name in ((exporter_build, "exporter_build"), (label_build, "label_build")):
        commit = owner.get("git_commit")
        _require(isinstance(commit, str) and len(commit) == 40,
                 f"{name}.git_commit must be a full commit ID")

    return FeatureArtifact(
        directory=directory,
        binary_path=binary_path,
        manifest=manifest,
        record_count=record_count,
        source_record_count=source_records,
        duplicates_removed=duplicates_removed,
        shard_count=shard_count,
        split_counts=split_counts,
        binary_digest=binary_digest,
        manifest_digest=manifest_digest,
    )
