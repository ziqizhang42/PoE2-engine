"""Small authenticated feature artifacts for loader tests."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from poe2_training.artifact import CELL_COUNT, HEADER, LINE_COUNT, MAGIC, RECORD_BYTES


def write_feature_fixture(directory: Path, *, dirty: bool = False) -> Path:
    directory.mkdir()
    corpus_id = "training-loader-test"
    corpus_digest = hashlib.sha256(corpus_id.encode()).digest()
    label_digest = hashlib.sha256(b"labels").digest()

    record = bytearray(RECORD_BYTES)
    struct.pack_into("<Q", record, 16, 1)  # canonical key low
    struct.pack_into("<i", record, 112, 8)  # teacher value
    struct.pack_into("<i", record, 128, 6)  # B value
    struct.pack_into("<i", record, 132, 2)  # residual
    struct.pack_into("<I", record, 136, 1)  # duplicate count
    record[146] = 1  # train split

    header = HEADER.pack(
        MAGIC,
        1,
        HEADER.size,
        RECORD_BYTES,
        0x01020304,
        1,
        1,
        0,
        corpus_digest,
        label_digest,
        LINE_COUNT,
        CELL_COUNT,
        1,
        0,
    )
    binary = header + record
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
            "source_records": 1,
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
            "records": 1,
            "duplicates_removed": 0,
            "split_counts": {
                "train": 1,
                "validation": 0,
                "test": 0,
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
