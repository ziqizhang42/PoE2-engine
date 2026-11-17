#!/usr/bin/env python3
"""Convert the models archived before nets were files: run1/g000 (v0.2) and run1/g001.

Each keeps its ID, the hash of its C++ header, and gains its net file and the fields every
archived record now carries. g001's record is regenerated from run1's run folder exactly as
promote-gate archives a model; g000, trained by hand before the workflow existed, is filled in
from its training report and corpus. Run from training/minimax with uv; --apply writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from poe2_training.net_file import PARAMETER_COUNT, TWO_PLY_CLOSURE_TEACHER, decode_net, net_from_header
from poe2_training.shared import atomic_write_bytes
from poe2_training.workflow import (
    MODEL_ARCHIVE_SCHEMA_VERSION,
    _archive_entry,
    _migrate_model_index,
    resolve_trained_models,
)
from poe2_training.workflow_adapters import authenticate_candidate
from poe2_training.workflow_config import load_workflow_config

REPOSITORY = Path(__file__).resolve().parents[1]
MODELS = REPOSITORY / "models"
G000 = "candidate-sha256-633309adb322e026784089d8b8ea175c80864a5941b0f2b912294e94a7d1ade3"
G000_REPORT = (REPOSITORY / "build" / "data" / "training" / "pattern-scale-320k-s20260818" /
               "frozen-c-v1" / "report.json")
G000_SOURCE = (REPOSITORY / "build" / "data" / "position-sources" /
               "pattern-scale-320k-s20260818" / "manifest.json")
# v0.2 was trained and released on 2026-08-18; its training report records no time.
G000_CREATED = "2026-08-18"


def g000_files() -> dict[Path, bytes]:
    header = (MODELS / f"{G000}.hpp").read_text(encoding="utf-8")
    net_bytes = net_from_header(header, name="run1/g000", teacher=TWO_PLY_CLOSURE_TEACHER,
                                created=G000_CREATED)
    net = decode_net(net_bytes)
    record = json.loads((MODELS / f"{G000}.json").read_bytes())
    report = json.loads(G000_REPORT.read_bytes())
    model = next(item for item in report["models"]
                 if item["name"] == report["selection"]["best_model"])
    source = json.loads(G000_SOURCE.read_bytes())
    if record.get("schema_version") != MODEL_ARCHIVE_SCHEMA_VERSION:
        record.update({
            "schema_version": MODEL_ARCHIVE_SCHEMA_VERSION,
            "net": {"name": net.name, "id": net.net_id, "sha256": net.sha256,
                    "design": net.design, "parameters": PARAMETER_COUNT,
                    "file": f"{G000}.poe2net"},
            "lineage": {"parent": None, "teacher": TWO_PLY_CLOSURE_TEACHER},
            "training": {"method": report["training"].get("method", "dense"),
                         "positions": report["input"]["records"],
                         "games": source["generator"]["trajectory_count"],
                         "curve": model["trace"]},
        })
    return {MODELS / f"{G000}.poe2net": net_bytes,
            MODELS / f"{G000}.json": (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()}


def g001_files() -> dict[Path, bytes]:
    config = resolve_trained_models(load_workflow_config(
        REPOSITORY / "training" / "minimax" / "recipes" / "run1.toml"))
    iteration = config.iteration("g001")
    entry = _archive_entry(config, iteration, authenticate_candidate(config, iteration))
    return {entry.net_path: entry.net, entry.metrics_path: entry.metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write the files")
    arguments = parser.parse_args()
    files = g000_files()
    for path, data in files.items():
        if arguments.apply:
            atomic_write_bytes(path, data, replace=True)
        print(f"{'wrote' if arguments.apply else 'would write'} {path.relative_to(REPOSITORY)} "
              f"({len(data)} bytes)")
    # g001's record names g000's net, so g000 is converted first.
    for path, data in g001_files().items():
        if arguments.apply:
            atomic_write_bytes(path, data, replace=True)
        print(f"{'wrote' if arguments.apply else 'would write'} {path.relative_to(REPOSITORY)} "
              f"({len(data)} bytes)")
    if arguments.apply:
        _migrate_model_index(load_workflow_config(
            REPOSITORY / "training" / "minimax" / "recipes" / "run1.toml"))
        print("migrated models/index.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
