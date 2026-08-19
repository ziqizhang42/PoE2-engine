"""Command-line interface for consolidated PoE2 training runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .workflow import (
    WorkflowError,
    handoff_candidate,
    print_plan,
    print_status,
    promote_gate,
    run_workflow,
    validate_existing,
)
from .workflow_config import WorkflowConfigError, load_workflow_config
from .workflow_state import StateError


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "recipes" / "pilot.toml"


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config", type=Path,
        default=Path(os.environ.get("TRAIN_CONFIG", DEFAULT_CONFIG)),
        help="strict workflow TOML (default: repository lightweight recipe)",
    )


def _iteration_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name", help="configured iteration name")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="poe2-train", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("plan", "print the derived read-only plan"),
        ("validate", "strictly parse and authenticate outputs without writing"),
        ("status", "show read-only run status"),
        ("run", "execute every configured iteration"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        _config_argument(command)
    status = subparsers.choices["status"]
    status.add_argument("--json", action="store_true", help="emit machine-readable status")

    iteration = subparsers.add_parser("iteration", help="execute one named iteration")
    _config_argument(iteration)
    _iteration_argument(iteration)
    iteration.add_argument(
        "--with-dependencies", action="store_true",
        help="execute incomplete earlier iteration dependencies first",
    )

    handoff = subparsers.add_parser("handoff", help="preview or install a candidate header")
    _config_argument(handoff)
    _iteration_argument(handoff)
    handoff.add_argument("--apply", action="store_true",
                         help="atomically replace the tracked generated header")

    promote = subparsers.add_parser(
        "promote-gate", help="preview or append a local gate to eval/results.csv")
    _config_argument(promote)
    _iteration_argument(promote)
    promote.add_argument("--apply", action="store_true",
                         help="append the authenticated saved ledger row")
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        config = load_workflow_config(arguments.config)
        if arguments.command == "plan":
            print_plan(config)
        elif arguments.command == "validate":
            artifacts = validate_existing(config)
            print(f"training_config_valid run={config.name} "
                  f"authenticated_artifacts={len(artifacts)} config={config.path}")
        elif arguments.command == "status":
            print_status(config, json_output=arguments.json)
        elif arguments.command == "run":
            run_workflow(config)
            print(f"training_run_complete run={config.name} output={config.output_directory}")
        elif arguments.command == "iteration":
            run_workflow(config, iteration_name=arguments.name,
                         with_dependencies=arguments.with_dependencies)
            print(f"training_iteration_complete run={config.name} iteration={arguments.name}")
        elif arguments.command == "handoff":
            result = handoff_candidate(config, arguments.name, apply=arguments.apply)
            print(json.dumps(result, indent=2, sort_keys=True))
        elif arguments.command == "promote-gate":
            result = promote_gate(config, arguments.name, apply=arguments.apply)
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            parser.error(f"unknown command: {arguments.command}")
    except (WorkflowConfigError, WorkflowError, StateError,
            OSError, RuntimeError, ValueError) as error:
        print(f"poe2-train {arguments.command} failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
