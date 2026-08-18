"""Small authenticated feature artifacts for loader tests."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from poe2_training.artifact import CELL_COUNT, HEADER, LINE_COUNT, MAGIC, RECORD_BYTES


@dataclass(frozen=True)
class FixtureRecord:
    key: int
    split: int
    ply: int
    teacher_value: int
    two_ply_closure_value: int
    own_gains: tuple[int, ...] = (1,) * CELL_COUNT
    opponent_gains: tuple[int, ...] = (1,) * CELL_COUNT
    flags: int = 0


def write_feature_fixture(directory: Path, *, dirty: bool = False,
                          rows: Sequence[FixtureRecord] | None = None) -> Path:
    directory.mkdir()
    corpus_id = "training-loader-test"
    corpus_digest = hashlib.sha256(corpus_id.encode()).digest()
    label_digest = hashlib.sha256(b"labels").digest()

    selected_rows = tuple(rows or (
        FixtureRecord(key=1, split=1, ply=0, teacher_value=8, two_ply_closure_value=6),
    ))
    if not selected_rows:
        raise ValueError("fixture requires at least one record")
    if tuple(row.key for row in selected_rows) != tuple(sorted(row.key for row in selected_rows)):
        raise ValueError("fixture records must be sorted by unique key")

    encoded_records: list[bytes] = []
    for row in selected_rows:
        if len(row.own_gains) != CELL_COUNT or len(row.opponent_gains) != CELL_COUNT:
            raise ValueError("fixture gain arrays must have 49 values")
        record = bytearray(RECORD_BYTES)
        struct.pack_into("<Q", record, 16, row.key)  # canonical key low
        struct.pack_into("<i", record, 112, row.teacher_value)
        struct.pack_into("<i", record, 128, row.two_ply_closure_value)
        struct.pack_into("<i", record, 132,
                         row.teacher_value - row.two_ply_closure_value)
        struct.pack_into("<I", record, 136, 1)  # duplicate count
        record[144] = row.ply
        record[146] = row.split
        record[156] = row.flags
        struct.pack_into("<49h", record, 232, *row.own_gains)
        struct.pack_into("<49h", record, 330, *row.opponent_gains)
        encoded_records.append(bytes(record))

    split_counts = tuple(sum(row.split == split for row in selected_rows) for split in (1, 2, 3))

    header = HEADER.pack(
        MAGIC,
        1,
        HEADER.size,
        RECORD_BYTES,
        0x01020304,
        len(selected_rows),
        len(selected_rows),
        0,
        corpus_digest,
        label_digest,
        LINE_COUNT,
        CELL_COUNT,
        1,
        0,
    )
    binary = header + b"".join(encoded_records)
    binary_digest = hashlib.sha256(binary).hexdigest()
    manifest = {
        "schema": "poe2-minimax-features",
        "schema_version": 1,
        "binary_digest": f"sha256:{binary_digest}",
        "corpus": {
            "id": corpus_id,
            "digest": f"sha256:{corpus_digest.hex()}",
        },
        "inputs": {
            "label_set_digest": f"sha256:{label_digest.hex()}",
            "shards": 1,
            "source_records": len(selected_rows),
            "label_build": {
                "git_commit": "1" * 40,
                "git_dirty": dirty,
            },
        },
        "exporter_build": {
            "git_commit": "2" * 40,
            "git_dirty": dirty,
        },
        "features": {
            "definition": "b-residual-line-pattern-gains-v1",
            "line_count": LINE_COUNT,
            "cell_count": CELL_COUNT,
        },
        "results": {
            "records": len(selected_rows),
            "duplicates_removed": 0,
            "split_counts": {
                "train": split_counts[0],
                "validation": split_counts[1],
                "test": split_counts[2],
            },
        },
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) +
                      "\n").encode()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()

    (directory / "features.bin").write_bytes(binary)
    (directory / "manifest.json").write_bytes(manifest_bytes)
    (directory / "COMPLETE").write_text(
        "poe2-minimax-features\n"
        f"binary_sha256={binary_digest}\n"
        f"manifest_sha256={manifest_digest}\n",
        encoding="utf-8",
    )
    return directory
