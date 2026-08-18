#!/usr/bin/env python3
"""Audit and summarize a complete, leakage-safe PoE2 minimax label corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from minimax_corpus import CorpusError, load_minimax_corpus


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path, help="label corpus directory containing shards/")
    parser.add_argument("--source", type=Path, required=True,
                        help="completed deterministic position-source directory")
    parser.add_argument("--json", action="store_true", help="emit the complete JSON report")
    arguments = parser.parse_args()
    try:
        corpus = load_minimax_corpus(arguments.labels, arguments.source)
    except CorpusError as error:
        print(f"invalid minimax corpus: {error}", file=sys.stderr)
        return 1
    summary = corpus.summary()
    if arguments.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "minimax_corpus_valid"
            f" shards={summary['shards']}"
            f" source_records={summary['source_records']}"
            f" selected_records={summary['selected_records']}"
            f" duplicates_removed={summary['duplicates_removed']}"
            f" duplicate_groups={summary['duplicate_groups']}"
            f" varying_label_groups={summary['varying_label_groups']}"
            f" train={summary['selected_split_counts']['train']}"
            f" validation={summary['selected_split_counts']['validation']}"
            f" test={summary['selected_split_counts']['test']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
