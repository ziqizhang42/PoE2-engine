from __future__ import annotations

import io
import contextlib
import unittest
from pathlib import Path
from unittest import mock

from fixture import (
    REPOSITORY,
    moved_repository,
    repository_temporary_directory,
    write_feature_fixture,
)
from test_workflow import _write
from poe2_training.workflow import CommandResult, WorkflowError, WorkflowRunner, _evaluator_run
from poe2_training.workflow_adapters import data_binary, data_build_directory
from poe2_training.workflow_config import (
    fingerprint,
    load_workflow_config,
    stage_definitions,
    stage_fingerprints,
)
from poe2_training.workflow_state import RunState, StateError

COMMIT = "1" * 40
RECIPE = """
schema_version = 1
[run]
name = "location"
output_dir = "{root}/run"
build_preset = "debug"

[datasets.main]
kind = "managed"
[datasets.main.source]
corpus_id = "location-corpus"
seed = 7
trajectories = 4
shards = 2
workers = 1
[datasets.main.labels]
nodes = 100
workers = 1

[datasets.old]
kind = "imported"
path = "{root}/features"

[[iterations]]
name = "one"
dataset = "main"
[iterations.training]
type = "pattern"
device = "cpu"
suite = "frozen-pattern-gain"
[iterations.sealed_evaluation]
exclude_datasets = ["old"]
[iterations.candidate_validation]
opening_book = "eval/openings/development.txt"
"""


def _strings(value) -> list[str]:
    if isinstance(value, dict):
        return [item for entry in value.values() for item in _strings(entry)]
    if isinstance(value, list):
        return [item for entry in value for item in _strings(entry)]
    return [value] if isinstance(value, str) else []


class LocationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.enterContext(mock.patch("poe2_training.workflow.catch2_pin", return_value=None))
        self.temporary = repository_temporary_directory()
        self.root = Path(self.temporary.name)
        write_feature_fixture(self.root / "features")
        self.config = load_workflow_config(_write(self.root, RECIPE.format(root=self.root)))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _moved(self):
        destination = repository_temporary_directory()
        self.addCleanup(destination.cleanup)
        return moved_repository(self.config, Path(destination.name))

    def test_fingerprints_name_paths_relative_to_the_repository(self) -> None:
        definitions = stage_definitions(self.config)
        self.assertFalse([value for value in _strings(definitions)
                          if value.startswith(str(REPOSITORY))])
        imported = definitions["dataset:old:features"]["path"]
        self.assertEqual(imported, (self.root / "features").relative_to(REPOSITORY).as_posix())
        self.assertEqual(stage_fingerprints(self._moved()), stage_fingerprints(self.config))
        absolute = stage_fingerprints(self.config, absolute_paths=True)
        changed = {stage for stage, value in stage_fingerprints(self.config).items()
                   if absolute[stage] != value}
        self.assertEqual(changed, {"dataset:old:features", "iteration:one:sealed-evaluation",
                                   "iteration:one:candidate-validation"})

    def test_a_stage_started_with_absolute_paths_validates_only_where_it_started(self) -> None:
        state = RunState(self.config)
        state.accept_config()
        stage = "iteration:one:candidate-validation"
        state.append("stage_started", stage=stage, attempt=1, config_fingerprint=stage_fingerprints(
            self.config, absolute_paths=True)[stage])
        RunState(self.config).validate_evolution()
        moved = self._moved()
        with self.assertRaisesRegex(StateError, f"recorded with its recipe at {self.config.path}"):
            RunState(moved).validate_evolution()

    def test_a_stage_started_with_relative_paths_validates_after_a_move(self) -> None:
        state = RunState(self.config)
        state.accept_config()
        stage = "iteration:one:candidate-validation"
        state.append("stage_started", stage=stage, attempt=1,
                     config_fingerprint=stage_fingerprints(self.config)[stage])
        RunState(self._moved()).validate_evolution()

    def test_locked_import_inputs_accept_the_absolute_form_only_in_place(self) -> None:
        stage = "dataset:old:features"
        features = self.config.dataset("old").feature_directory
        state = RunState(self.config)
        state.accept_config()
        state.append("stage_started", stage=stage, attempt=1,
                     input_fingerprint=fingerprint({"path": str(features)}))
        moved = self._moved()
        runner = WorkflowRunner(self.config, RunState(self.config), COMMIT)
        self.assertEqual(runner._begin(stage, "import", {"path": features}), 2)
        moved_runner = WorkflowRunner(moved, RunState(moved), COMMIT)
        with self.assertRaisesRegex(WorkflowError, "inputs to started stage"):
            moved_runner._begin(stage, "import", {"path": moved.dataset("old").feature_directory})

    def test_locked_import_inputs_hold_after_a_move(self) -> None:
        stage = "dataset:old:features"
        state = RunState(self.config)
        state.accept_config()
        WorkflowRunner(self.config, state, COMMIT)._begin(
            stage, "import", {"path": self.config.dataset("old").feature_directory})
        moved = self._moved()
        runner = WorkflowRunner(moved, RunState(moved), COMMIT)
        self.assertEqual(runner._begin(
            stage, "import", {"path": moved.dataset("old").feature_directory}), 2)

    def test_a_build_tree_configured_elsewhere_is_rebuilt_from_scratch(self) -> None:
        dataset = self.config.dataset("main")
        build = data_build_directory(self.config)
        build.mkdir(parents=True)
        (build / "CMakeCache.txt").write_text(
            "# This is the CMakeCache file.\n"
            "CMAKE_CACHEFILE_DIR:INTERNAL=/elsewhere/build/debug\n"
            f"CMAKE_HOME_DIRECTORY:INTERNAL={REPOSITORY}\n", encoding="utf-8")
        (build / "build.ninja").write_text("stale\n", encoding="utf-8")
        state = RunState(self.config)
        state.accept_config()
        runner = WorkflowRunner(self.config, state, COMMIT)
        fresh: list[bool] = []

        def build_here(_stage: str, operation: str, *_arguments, **_options) -> None:
            if operation == "configure":
                fresh.append(not build.exists())
                build.mkdir(parents=True, exist_ok=True)
                (build / "CMakeCache.txt").write_text(
                    f"CMAKE_CACHEFILE_DIR:INTERNAL={build}\n"
                    f"CMAKE_HOME_DIRECTORY:INTERNAL={REPOSITORY}\n", encoding="utf-8")
            else:
                binary = data_binary(self.config)
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.write_bytes(b"fixture\0" + COMMIT.encode() + b"\0")

        stderr = io.StringIO()
        with mock.patch.object(runner, "_command", side_effect=build_here), \
                contextlib.redirect_stderr(stderr):
            runner._ensure_data_binary("dataset:main:source", 1, dataset)
        self.assertEqual(fresh, [True])
        self.assertIn("configured at /elsewhere/build/debug", stderr.getvalue())
        removed = [event for event in state.events if event["type"] == "moved_build_removed"]
        self.assertEqual([event["configured_at"] for event in removed],
                         ["/elsewhere/build/debug"])

        data_binary(self.config).unlink()
        again = WorkflowRunner(self.config, state, COMMIT)
        with mock.patch.object(again, "_command", side_effect=build_here):
            again._ensure_data_binary("dataset:main:source", 2, dataset)
        self.assertEqual(fresh, [True, False])

    def test_the_evaluator_run_is_found_relative_to_the_repository(self) -> None:
        root = self.root / "gate"
        run = root / "runs" / "one"
        run.mkdir(parents=True)
        (run / "manifest.json").write_text('{"valid": true}\n', encoding="utf-8")
        output = f"eval_run run_dir={run.relative_to(REPOSITORY).as_posix()} ledger=disabled\n"
        result = CommandResult(("fixture",), 0, output, self.root / "log", self.root / "m", {})
        self.assertEqual(_evaluator_run(self.config, result, root, "gate"), run)


if __name__ == "__main__":
    unittest.main()
