#!/usr/bin/env python3
"""Controlled tests for the minimax label comparison report."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import struct
import sys
from pathlib import Path


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def qualified(digest: bytes) -> str:
    return "sha256:" + digest.hex()


def write_golden(auditor, directory: Path) -> None:
    corpus_id = "comparison-golden"
    corpus_digest = hashlib.sha256(corpus_id.encode()).digest()
    source_digest = hashlib.sha256(b"comparison source").digest()
    player_one = (1 << 0) | (1 << 2)
    player_two = (1 << 1) | (1 << 3)
    key_low, key_high = auditor.canonical_key(player_one, player_two, 0)
    record = auditor.RECORD.pack(
        player_one, player_two, key_low, key_high, 11, 12, 13, 0, 0,
        10, 5, 1, 0, -7, 4, 0, 2, 1, 3, 43, 4, 1, 4, 0,
        8, 6, 2, 5, 5, -7, 1, 4,
    )
    header = auditor.HEADER.pack(
        auditor.MAGIC, auditor.SCHEMA_VERSION, auditor.HEADER.size,
        auditor.RECORD.size, auditor.ENDIAN_MARKER, 1, 1,
        source_digest, corpus_digest, 0, 1,
    )
    binary = header + record
    manifest = {
        "schema": "poe2-minimax-labels",
        "schema_version": auditor.SCHEMA_VERSION,
        "format": {"header_bytes": auditor.HEADER.size,
                   "record_bytes": auditor.RECORD.size},
        "binary_digest": qualified(hashlib.sha256(binary).digest()),
        "corpus": {"id": corpus_id, "digest": qualified(corpus_digest),
                   "shard_index": 0, "shard_count": 1},
        "source": {"name": "golden.jsonl", "digest": qualified(source_digest),
                   "positions": 1},
        "build": {"git_commit": "test", "git_dirty": False,
                  "project_version": "test", "compiler_id": "test",
                  "compiler_version": "test", "build_type": "test",
                  "target_processor": "test", "native_architecture": False},
        "search": {"evaluator": "b", "mode": "teacher", "node_limit": 10,
                   "hash_bytes_requested": 0, "hash_bytes_effective": 0,
                   "hash_capacity": 0, "workers_requested": 1, "workers_used": 1,
                   "require_all": True, "symmetry": True, "two_ply_closure": True},
        "results": {"records": 1, "terminal_records": 0, "parity_backoffs": 1,
                    "previous_records": 1, "unsolved": 0,
                    "unsolved_source_lines": []},
    }
    manifest["search"]["target_selection"] = "deepest_terminal_parity"
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
        raise SystemExit("usage: minimax_label_comparison_test.py <source-root> <temporary-root>")
    source_root = Path(sys.argv[1])
    temporary_root = Path(sys.argv[2])
    shutil.rmtree(temporary_root, ignore_errors=True)
    tools = source_root / "tools"
    sys.path.insert(0, str(tools))
    auditor = load_module("inspect_minimax_labels", tools / "inspect_minimax_labels.py")
    comparison = load_module("compare_minimax_labels", tools / "compare_minimax_labels.py")

    baseline = temporary_root / "baseline"
    candidate = temporary_root / "candidate"
    write_golden(auditor, baseline)
    shutil.copytree(baseline, candidate)
    binary = bytearray((candidate / "labels.bin").read_bytes())
    fields = list(auditor.RECORD.unpack_from(binary, auditor.HEADER.size))
    fields[13] = -3
    fields[20] = 6
    fields[29] = -3
    fields[31] = 6
    auditor.RECORD.pack_into(binary, auditor.HEADER.size, *fields)
    (candidate / "labels.bin").write_bytes(binary)
    refresh_integrity(candidate)

    datasets = [comparison.load_dataset(path) for path in (baseline, candidate)]
    report = comparison.compare_datasets(datasets)
    metrics = report["comparisons"][0]["overall"]
    assert metrics["records"] == 1
    assert metrics["mean_absolute_value_delta"] == 4
    assert metrics["sign_changes"] == 0
    assert metrics["best_move_changes"] == 1
    assert metrics["completed_depth_delta"]["mean"] == 0
    assert metrics["deepest"]["mean_absolute_value_delta"] == 0
    assert report["comparisons"][0]["by_policy"]["noisy_search"]["records"] == 1
    assert report["comparisons"][0]["by_phase"]["ply_04_08"]["records"] == 1

    incompatible = temporary_root / "incompatible"
    shutil.copytree(baseline, incompatible)
    binary = bytearray((incompatible / "labels.bin").read_bytes())
    struct.pack_into("<Q", binary, auditor.HEADER.size + 32, 99)
    (incompatible / "labels.bin").write_bytes(binary)
    refresh_integrity(incompatible)
    try:
        comparison.compare_datasets(
            [comparison.load_dataset(baseline), comparison.load_dataset(incompatible)]
        )
    except comparison.ComparisonError as error:
        assert "source fields differ" in str(error)
    else:
        raise AssertionError("incompatible label records were accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
