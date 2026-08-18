#!/usr/bin/env python3
"""Benchmark two-ply-closure and pattern/gain search on one opening sample."""

from __future__ import annotations

import argparse
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchResult:
    depth: int
    nodes: int
    wall_seconds: float
    diagnostics: dict[str, int]


def read_openings(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"could not read opening book {path}: {error}") from error
    openings = [line.strip() for line in lines
                if line.strip() and not line.lstrip().startswith("#")]
    if not openings:
        raise ValueError(f"opening book has no positions: {path}")
    return openings


def parse_pairs(line: str) -> dict[str, int]:
    fields = line.split()[1:]
    result: dict[str, int] = {}
    for index in range(0, len(fields) - 1, 2):
        try:
            result[fields[index]] = int(fields[index + 1])
        except ValueError:
            break
    return result


def run_position(binary: Path, evaluator: str, opening: str,
                 move_time_ms: int) -> SearchResult:
    position = "position startpos"
    if opening != "startpos":
        position += f" moves {opening}"
    commands = f"{position}\ngo movetime {move_time_ms}\nquit\n"
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(binary), "--evaluator", evaluator],
            input=commands,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(10.0, move_time_ms / 1000.0 * 5.0),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"{evaluator} search failed to run: {error}") from error
    wall_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise ValueError(
            f"{evaluator} search exited {completed.returncode}: {completed.stderr.strip()}"
        )

    result: dict[str, int] | None = None
    diagnostics: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        if line.startswith("info depth "):
            result = parse_pairs(line)
        elif line.startswith("info ttprobes "):
            diagnostics = parse_pairs(line)
    if result is None or "depth" not in result or "nodes" not in result:
        raise ValueError(
            f"{evaluator} search emitted no complete result: {completed.stdout.strip()}"
        )
    return SearchResult(
        depth=result["depth"],
        nodes=result["nodes"],
        wall_seconds=wall_seconds,
        diagnostics=diagnostics,
    )


def summarize(evaluator: str, results: list[SearchResult], move_time_ms: int) -> dict[str, float]:
    total_nodes = sum(result.nodes for result in results)
    total_wall = sum(result.wall_seconds for result in results)
    depths = [result.depth for result in results]
    nominal_seconds = len(results) * move_time_ms / 1000.0
    summary = {
        "nps": total_nodes / nominal_seconds,
        "wall_nps": total_nodes / total_wall,
        "depth_mean": statistics.fmean(depths),
        "depth_median": statistics.median(depths),
        "depth_min": min(depths),
        "depth_max": max(depths),
    }
    print(
        "search_benchmark"
        f" evaluator={evaluator}"
        f" positions={len(results)}"
        f" movetime_ms={move_time_ms}"
        f" nodes={total_nodes}"
        f" nps={summary['nps']:.0f}"
        f" wall_nps={summary['wall_nps']:.0f}"
        f" depth_mean={summary['depth_mean']:.3f}"
        f" depth_median={summary['depth_median']:.1f}"
        f" depth_range={int(summary['depth_min'])}-{int(summary['depth_max'])}"
        f" staticevals={sum(result.diagnostics.get('staticevals', 0) for result in results)}"
        f" modelevals={sum(result.diagnostics.get('modelevals', 0) for result in results)}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--opening-book", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--movetime-ms", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260818)
    arguments = parser.parse_args()
    if arguments.positions <= 0 or arguments.movetime_ms <= 0:
        parser.error("--positions and --movetime-ms must be positive")
    engine = arguments.engine.resolve()
    if not engine.is_file():
        parser.error(f"engine does not exist: {engine}")

    try:
        openings = read_openings(arguments.opening_book)
        generator = random.Random(arguments.seed)
        sample = generator.sample(openings, min(arguments.positions, len(openings)))
        summaries: dict[str, dict[str, float]] = {}
        for evaluator in ("two-ply-closure", "pattern-gain"):
            results = [run_position(engine, evaluator, opening, arguments.movetime_ms)
                       for opening in sample]
            summaries[evaluator] = summarize(evaluator, results, arguments.movetime_ms)
        print(
            "search_benchmark_comparison"
            " pattern_gain_to_two_ply_closure_nps="
            f"{summaries['pattern-gain']['nps'] / summaries['two-ply-closure']['nps']:.4f}"
            " depth_mean_delta="
            f"{summaries['pattern-gain']['depth_mean'] - summaries['two-ply-closure']['depth_mean']:.3f}"
            " depth_median_delta="
            f"{summaries['pattern-gain']['depth_median'] - summaries['two-ply-closure']['depth_median']:.1f}"
        )
    except ValueError as error:
        print(f"search benchmark failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
