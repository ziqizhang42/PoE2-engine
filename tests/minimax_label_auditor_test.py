#!/usr/bin/env python3
"""Independent golden and corruption tests for the label dataset auditor."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import struct
import sys
from pathlib import Path


def load_auditor(source_root: Path):
    path = source_root / "tools" / "inspect_minimax_labels.py"
    spec = importlib.util.spec_from_file_location("inspect_minimax_labels", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load label auditor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def qualified(digest: bytes) -> str:
    return "sha256:" + digest.hex()


def write_golden(auditor, directory: Path) -> None:
    corpus_id = "auditor-golden"
    corpus_digest = hashlib.sha256(corpus_id.encode()).digest()
    source_digest = hashlib.sha256(b"hand-authored source").digest()
    record = auditor.RECORD.pack(
        0,
        0,
        0,
        0,
        11,
        12,
        13,
        0,
        0,
        10,
        5,
        1,
        0,
        -7,
        0,
        0,
        2,
        1,
        2,
        47,
        0,
        1,
        4,
        5,
    )
    header = auditor.HEADER.pack(
        auditor.MAGIC,
        auditor.SCHEMA_VERSION,
        auditor.HEADER.size,
        auditor.RECORD.size,
        auditor.ENDIAN_MARKER,
        1,
        1,
        source_digest,
        corpus_digest,
        0,
        1,
    )
    binary = header + record
    manifest = {
        "schema": "poe2-minimax-labels",
        "schema_version": 1,
        "format": {"header_bytes": auditor.HEADER.size, "record_bytes": auditor.RECORD.size},
        "binary_digest": qualified(hashlib.sha256(binary).digest()),
        "corpus": {"id": corpus_id, "digest": qualified(corpus_digest),
                   "shard_index": 0, "shard_count": 1},
        "source": {"name": "golden.txt", "digest": qualified(source_digest), "positions": 1},
        "build": {"git_commit": "test", "git_dirty": False, "project_version": "test",
                  "compiler_id": "test", "compiler_version": "test", "build_type": "test",
                  "target_processor": "test", "native_architecture": False},
        "search": {"evaluator": "b", "mode": "teacher", "node_limit": 10,
                   "hash_bytes_requested": 0, "hash_bytes_effective": 0,
                   "hash_capacity": 0, "workers_requested": 1, "workers_used": 1,
                   "require_all": True, "symmetry": True, "two_ply_closure": True},
        "results": {"records": 1, "terminal_records": 0, "unsolved": 0,
                    "unsolved_source_lines": []},
    }
    directory.mkdir(parents=True)
    (directory / "labels.bin").write_bytes(binary)
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    (directory / "manifest.json").write_bytes(manifest_bytes)
    marker = (
        "poe2-minimax-labels\n"
        f"binary_sha256={hashlib.sha256(binary).hexdigest()}\n"
        f"manifest_sha256={hashlib.sha256(manifest_bytes).hexdigest()}\n"
    )
    (directory / "COMPLETE").write_text(marker, encoding="utf-8")


def expect_invalid(auditor, directory: Path, expected: str) -> None:
    try:
        auditor.audit_dataset(directory)
    except auditor.DatasetError as error:
        if expected not in str(error):
            raise AssertionError(f"expected {expected!r} in {str(error)!r}") from error
    else:
        raise AssertionError("corrupted dataset was accepted")


def refresh_integrity(directory: Path) -> None:
    binary = (directory / "labels.bin").read_bytes()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["binary_digest"] = qualified(hashlib.sha256(binary).digest())
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    marker = (
        "poe2-minimax-labels\n"
        f"binary_sha256={hashlib.sha256(binary).hexdigest()}\n"
        f"manifest_sha256={hashlib.sha256(manifest_bytes).hexdigest()}\n"
    )
    (directory / "COMPLETE").write_text(marker, encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: minimax_label_auditor_test.py <source-root> <temporary-root>")
    source_root = Path(sys.argv[1])
    temporary_root = Path(sys.argv[2])
    shutil.rmtree(temporary_root, ignore_errors=True)
    auditor = load_auditor(source_root)

    golden = temporary_root / "golden"
    write_golden(auditor, golden)
    summary = auditor.audit_dataset(golden)
    assert summary.records == 1
    assert summary.teacher_records == 1

    incomplete = temporary_root / "incomplete"
    shutil.copytree(golden, incomplete)
    (incomplete / "COMPLETE").rename(incomplete / "INCOMPLETE")
    expect_invalid(auditor, incomplete, "COMPLETE")

    corrupt = temporary_root / "corrupt"
    shutil.copytree(golden, corrupt)
    binary = bytearray((corrupt / "labels.bin").read_bytes())
    binary[-1] ^= 1
    (corrupt / "labels.bin").write_bytes(binary)
    expect_invalid(auditor, corrupt, "digest")

    semantic = temporary_root / "semantic"
    shutil.copytree(golden, semantic)
    binary = bytearray((semantic / "labels.bin").read_bytes())
    binary[auditor.HEADER.size + 16] = 1
    (semantic / "labels.bin").write_bytes(binary)
    refresh_integrity(semantic)
    expect_invalid(auditor, semantic, "canonical key")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
