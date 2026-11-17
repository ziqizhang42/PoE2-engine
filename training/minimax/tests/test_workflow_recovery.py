from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from fixture import repository_temporary_directory
from test_workflow import _managed_config, _write
from poe2_training.workflow import (
    CommandExecutionError,
    CommandRunner,
    WorkflowError,
    WorkflowRunner,
    crash_leftovers,
    promote_gate,
    recover_after_crash,
    run_lock,
    run_workflow,
    validate_existing,
)
from poe2_training.workflow_adapters import (
    AdapterError,
    catch2_cache_directory,
    catch2_candidates,
    catch2_pin,
    data_binary,
    data_build_directory,
    label_shard_path,
    source_chunk_path,
)
from poe2_training.workflow_config import load_workflow_config
from poe2_training.workflow_state import RunState, StateError
from poe2_training.workflow_ui import WorkflowProgress

REPOSITORY = Path(__file__).resolve().parents[3]
COMMIT = "1" * 40


class Crash(BaseException):
    """Stands in for the orchestrator dying: nothing after it runs, no handler records it."""


def _types(state: RunState) -> list[str]:
    return [event["type"] for event in state.events]


class RecoveryFixture(unittest.TestCase):
    reuse_catch2 = False

    def setUp(self) -> None:
        if not self.reuse_catch2:
            self.enterContext(mock.patch("poe2_training.workflow.catch2_pin", return_value=None))
        self.temporary = repository_temporary_directory()
        self.root = Path(self.temporary.name)
        self.config = load_workflow_config(_write(self.root, _managed_config(self.root)))
        self.dataset = self.config.dataset("main")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _runner(self) -> WorkflowRunner:
        state = RunState(self.config)
        state.accept_config()
        return WorkflowRunner(self.config, state, COMMIT)

    def _recover(self) -> tuple[list[Path], RunState, str]:
        state = RunState(self.config)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            removed = recover_after_crash(self.config, state)
        return removed, state, stderr.getvalue()


class CrashLeftoverTest(RecoveryFixture):
    def test_leftovers_of_a_crash_are_removed_on_the_next_start(self) -> None:
        runner = self._runner()
        shard = label_shard_path(self.dataset, 0)
        runner._start_artifact(shard)
        shard.mkdir(parents=True)
        (shard / "INCOMPLETE").write_text("poe2-minimax-labels\n", encoding="utf-8")
        (shard / "labels.bin.tmp").write_bytes(b"truncated")
        finished = label_shard_path(self.dataset, 1)
        runner._start_artifact(finished)
        finished.mkdir()
        (finished / "COMPLETE").write_text("poe2-minimax-labels\n", encoding="utf-8")
        never_created = source_chunk_path(self.dataset, 0, 4)
        runner._start_artifact(never_created)

        removed, state, printed = self._recover()

        self.assertEqual(removed, [shard])
        self.assertFalse(shard.exists())
        self.assertTrue(shard.parent.is_dir())
        self.assertTrue((finished / "COMPLETE").is_file())
        self.assertIn(f"removed unfinished {shard}", printed)
        outcomes = {event["path"]: event["type"] for event in state.events[-3:]}
        self.assertEqual(outcomes, {
            str(shard): "crash_artifact_removed", str(finished): "artifact_completed",
            str(never_created): "artifact_abandoned"})
        self.assertEqual(state.unfinished_artifacts(), [])
        count = len(state.events)
        self.assertEqual(self._recover()[0], [])
        self.assertEqual(len(RunState(self.config).events), count)

    def test_validation_counts_crash_leftovers_as_not_yet_created(self) -> None:
        runner = self._runner()
        shard = label_shard_path(self.dataset, 0)
        runner._start_artifact(shard)
        shard.mkdir(parents=True)
        (shard / "INCOMPLETE").write_text("poe2-minimax-labels\n", encoding="utf-8")
        chunk = source_chunk_path(self.dataset, 0, 4)
        runner._start_artifact(chunk)
        chunk.mkdir(parents=True)
        (chunk / "INCOMPLETE").write_text("poe2-position-source-chunk\n", encoding="utf-8")
        self.assertEqual(validate_existing(self.config), {})
        self.assertEqual(crash_leftovers(self.config), [shard, chunk])
        self.assertTrue(shard.is_dir() and chunk.is_dir())

    def test_a_recorded_failure_is_kept_for_inspection(self) -> None:
        runner = self._runner()
        shard = label_shard_path(self.dataset, 0)
        runner._start_artifact(shard)
        shard.mkdir(parents=True)
        (shard / "INCOMPLETE").write_text("poe2-minimax-labels\n", encoding="utf-8")
        self.assertEqual(runner.keep_failed_artifacts(), [shard])

        removed, state, printed = self._recover()

        self.assertEqual((removed, printed), ([], ""))
        self.assertTrue((shard / "INCOMPLETE").is_file())
        self.assertEqual(_types(state)[-1], "artifact_failed")

    def test_paths_without_a_start_record_are_never_removed(self) -> None:
        state = RunState(self.config)
        state.accept_config()
        state.append("stage_started", stage="dataset:main:features", attempt=1)
        features = self.dataset.feature_directory
        features.mkdir(parents=True)
        (features / "INCOMPLETE").write_text("fixture\n", encoding="utf-8")

        removed, _, printed = self._recover()

        self.assertEqual((removed, printed), ([], ""))
        self.assertTrue((features / "INCOMPLETE").is_file())
        runner = WorkflowRunner(self.config, RunState(self.config), COMMIT)
        with self.assertRaisesRegex(WorkflowError, f"retry: {features}"):
            runner._run_simple("dataset:main:features", features, "feature artifact", {},
                               mock.Mock(), mock.Mock())

    def test_a_recorded_path_outside_the_run_folder_is_kept(self) -> None:
        state = RunState(self.config)
        outside = self.root / "outside"
        outside.mkdir()
        state.append("artifact_started", path=str(outside), artifact="../outside")
        state.append("artifact_started", path=str(outside), artifact=str(outside))

        removed, state, printed = self._recover()

        self.assertEqual(removed, [])
        self.assertTrue(outside.is_dir())
        self.assertEqual(_types(state)[-2:], ["crash_artifact_kept", "crash_artifact_kept"])
        self.assertIn("not inside the run folder", printed)

    def test_a_half_built_data_build_is_rebuilt_from_scratch(self) -> None:
        runner = self._runner()
        build = data_build_directory(self.config)
        binary = data_binary(self.config)

        def crash_while_building(_stage: str, operation: str, *_arguments, **_options) -> None:
            build.mkdir(parents=True, exist_ok=True)
            (build / "CMakeCache.txt").write_text("fixture\n", encoding="utf-8")
            if operation == "build-data":
                binary.parent.mkdir(parents=True)
                binary.write_bytes(b"truncated")
                raise Crash

        with mock.patch.object(runner, "_command", side_effect=crash_while_building):
            with self.assertRaises(Crash):
                runner._ensure_data_binary("dataset:main:source", 1, self.dataset)

        removed, _, _ = self._recover()
        self.assertEqual(removed, [build])
        self.assertFalse(build.exists())

        fresh_tree: list[bool] = []

        def build_cleanly(_stage: str, operation: str, *_arguments, **_options) -> None:
            if operation == "configure":
                fresh_tree.append(not build.exists())
                build.mkdir(parents=True)
            else:
                binary.parent.mkdir(parents=True)
                binary.write_bytes(b"fixture\0" + COMMIT.encode() + b"\0")

        rebuilt = WorkflowRunner(self.config, RunState(self.config), COMMIT)
        with mock.patch.object(rebuilt, "_command", side_effect=build_cleanly):
            rebuilt._ensure_data_binary("dataset:main:source", 2, self.dataset)
        self.assertEqual(fresh_tree, [True])
        self.assertEqual(rebuilt.state.unfinished_artifacts(), [])

    def test_a_run_resumes_after_a_crash_like_after_a_pause(self) -> None:
        runner = self._runner()
        stage = "iteration:one:training"
        training = self.config.iteration("one").training_directory
        runner._begin(stage, "create", {"input": 1})
        runner._start_artifact(training)
        training.mkdir(parents=True)
        (training / "INCOMPLETE").write_text("fixture\n", encoding="utf-8")
        seen: list[tuple[bool, str]] = []

        def run_iteration(resumed: WorkflowRunner, _iteration) -> None:
            seen.append((training.exists(), resumed.state.stage_state()[stage]["status"]))
            with self.assertRaisesRegex(WorkflowError, "inputs to started stage"):
                resumed._begin(stage, "create", {"input": 2})
            self.assertEqual(resumed._begin(stage, "create", {"input": 1}), 2)

        stderr = io.StringIO()
        with mock.patch("poe2_training.workflow.require_clean_committed", return_value=COMMIT), \
                mock.patch.object(WorkflowRunner, "run_iteration", autospec=True,
                                  side_effect=run_iteration), \
                mock.patch("poe2_training.workflow.WorkflowProgress",
                           lambda config, state: WorkflowProgress(
                               config, state, stream=io.StringIO())), \
                contextlib.redirect_stderr(stderr):
            run_workflow(self.config, iteration_name="one")

        self.assertEqual(seen, [(False, "failed")])
        self.assertIn(f"removed unfinished {training}", stderr.getvalue())
        types = _types(RunState(self.config))
        self.assertLess(types.index("crash_artifact_removed"), types.index("stage_abandoned"))
        self.assertLess(types.index("stage_abandoned"), types.index("run_started"))

    def test_a_run_warns_when_commands_cannot_be_memory_limited(self) -> None:
        self._runner()
        stream = io.StringIO()
        limits = []

        def run_iteration(runner: WorkflowRunner, _iteration) -> None:
            limits.append(runner.commands.scope_limit)

        for problem in ("systemd-run is not usable: missing", None):
            with mock.patch("poe2_training.workflow.require_clean_committed",
                            return_value=COMMIT), \
                    mock.patch("poe2_training.workflow.probe_scope_limit", return_value=problem), \
                    mock.patch.object(WorkflowRunner, "run_iteration", autospec=True,
                                      side_effect=run_iteration), \
                    mock.patch("poe2_training.workflow.WorkflowProgress",
                               lambda config, state: WorkflowProgress(config, state,
                                                                      stream=stream)):
                run_workflow(self.config, iteration_name="one")
        self.assertIn("warning: workflow commands run without a memory limit (systemd-run is "
                      "not usable: missing); they may use swap", stream.getvalue())
        self.assertIsNone(limits[0])
        self.assertIsNotNone(limits[1])
        warnings = [event for event in RunState(self.config).events
                    if event["type"] == "memory_limit_unavailable"]
        self.assertEqual(len(warnings), 1)

    def test_mutating_commands_recover_under_the_lock(self) -> None:
        runner = self._runner()
        gate = self.config.iteration("one").gate_directory
        runner._start_artifact(gate)
        gate.mkdir(parents=True)
        (gate / "INCOMPLETE").write_text("fixture\n", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(AdapterError, "candidate"):
                promote_gate(self.config, "one", apply=True)
        self.assertFalse(gate.exists())
        self.assertTrue(gate.parent.is_dir())

    def test_commands_inherit_the_run_lock(self) -> None:
        commands = CommandRunner(REPOSITORY)
        logs = self.root / "logs"
        with run_lock(self.config) as descriptor:
            check = [sys.executable, "-c",
                     f"import os, sys; sys.exit(not os.path.exists('/proc/self/fd/{descriptor}'))"]
            commands.lock_descriptors = (descriptor,)
            commands.run(check, log_path=logs / "held.log", metrics_path=logs / "held.json")
            commands.lock_descriptors = ()
            with self.assertRaises(CommandExecutionError):
                commands.run(check, log_path=logs / "closed.log",
                             metrics_path=logs / "closed.json")

    def test_a_command_log_reaches_the_disk_before_the_command_returns(self) -> None:
        log = (self.root / "logs" / "synced.log").resolve()
        synced: list[str] = []
        fsync = os.fsync

        def recording_fsync(descriptor: int) -> None:
            synced.append(os.readlink(f"/proc/self/fd/{descriptor}"))
            fsync(descriptor)

        with mock.patch.object(os, "fsync", recording_fsync):
            CommandRunner(REPOSITORY).run([sys.executable, "-c", "print('done')"],
                                          log_path=log, metrics_path=log.with_suffix(".json"),
                                          output_callback=lambda _line, _elapsed: None)
        self.assertIn(str(log), synced)
        self.assertTrue(log.read_text(encoding="utf-8").endswith("done\n"))


class EventLogTest(RecoveryFixture):
    def test_a_torn_final_record_is_dropped_under_the_lock_and_recorded(self) -> None:
        state = RunState(self.config)
        state.accept_config()
        state.append("run_started")
        state.append("run_completed")
        path = state.events_path
        intact = path.read_bytes()
        for tail in (b'{"schema":"poe2-training-ev', b"\0" * 37, b"\n", b'{"sche\0\0\n'):
            with self.subTest(tail=tail):
                path.write_bytes(intact + tail)
                torn = RunState(self.config)
                self.assertEqual(torn.events, state.events)
                self.assertEqual(path.read_bytes(), intact + tail)
                with self.assertRaisesRegex(StateError, "repair the torn final record"):
                    torn.append("run_started")

                _, repaired, printed = self._recover()

                self.assertIn(f"({len(tail)} bytes)", printed)
                event = repaired.events[-1]
                self.assertEqual((event["type"], event["dropped_bytes"]),
                                 ("event_log_repaired", len(tail)))
                self.assertEqual(repaired.events[:-1], state.events)
                reread = RunState(self.config)
                self.assertEqual((reread.events, reread.torn_tail), (repaired.events, b""))
                self.assertTrue(path.read_bytes().startswith(intact))

    def test_damage_before_the_final_record_stays_an_error(self) -> None:
        state = RunState(self.config)
        state.accept_config()
        state.append("run_started")
        lines = state.events_path.read_bytes().splitlines(keepends=True)
        for damaged, message in (
                ([lines[0], b"garbage\n", lines[1]], "malformed event"),
                ([lines[0], b"\n", lines[1]], "malformed event"),
                ([lines[0], lines[0]], "invalid event"),
                ([lines[1], lines[0]], "invalid event")):
            with self.subTest(message=message):
                state.events_path.write_bytes(b"".join(damaged))
                with self.assertRaisesRegex(StateError, message):
                    RunState(self.config)


def _git(path: Path, *arguments: str) -> str:
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    return subprocess.run(
        ["git", "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "init.defaultBranch=main", *arguments],
        cwd=path, env=environment, check=True, capture_output=True, text=True).stdout


class Catch2ReuseTest(RecoveryFixture):
    reuse_catch2 = True

    def setUp(self) -> None:
        super().setUp()
        pin = catch2_pin(REPOSITORY)
        assert pin is not None
        self.pin = pin
        self.source = data_build_directory(self.config) / "_deps" / "catch2-src"
        self._checkout()
        self.cache = catch2_cache_directory(self.config, self.pin)
        self.enterContext(mock.patch("poe2_training.workflow.catch2_candidates",
                                     return_value=[self.root / "absent", self.source]))

    def _checkout(self) -> None:
        shutil.rmtree(self.source, ignore_errors=True)
        self.source.mkdir(parents=True)
        _git(self.source, "init", "-q")
        (self.source / "catch.hpp").write_text("// fixture\n", encoding="utf-8")
        _git(self.source, "add", "catch.hpp")
        _git(self.source, "commit", "-q", "-m", "fixture")
        _git(self.source, "tag", self.pin.tag)
        _git(self.source, "remote", "add", "origin", self.pin.repository)

    def test_the_pin_matches_the_build(self) -> None:
        self.assertEqual((self.pin.name, self.pin.repository, self.pin.tag),
                         ("Catch2", "https://github.com/catchorg/Catch2.git", "v3.15.1"))
        candidates = catch2_candidates(self.config)
        self.assertEqual(candidates[0], self.source)
        self.assertIn(self.config.iteration("one").candidate_directory /
                      "build/_deps/catch2-src", candidates)

    def test_a_verified_checkout_is_copied_into_the_run_and_reused(self) -> None:
        runner = self._runner()
        override = f"-DFETCHCONTENT_SOURCE_DIR_CATCH2={self.cache}"
        self.assertEqual(runner._catch2_arguments(), [override])
        self.assertTrue((self.cache / "catch.hpp").is_file())
        self.assertEqual([event["artifact"] for event in runner.state.events[-2:]],
                         ["build/catch2-v3.15.1"] * 2)
        self.assertEqual(runner.state.unfinished_artifacts(), [])

        _git(self.source, "commit", "-q", "--allow-empty", "-m", "moved on")
        with mock.patch.object(runner, "_command") as command:
            runner._configure("stage", "configure", self.root / "tree", ["-DX=1"], 1)
        self.assertEqual(command.call_args.args[2][-2:], ["-DX=1", override])

        (self.cache / "catch.hpp").write_text("// changed\n", encoding="utf-8")
        self.assertEqual(runner._catch2_arguments(), ["-DFETCHCONTENT_SOURCE_DIR_CATCH2="])

    def test_an_unverifiable_checkout_is_not_used(self) -> None:
        def modified() -> None:
            (self.source / "catch.hpp").write_text("// changed\n", encoding="utf-8")

        def untracked() -> None:
            (self.source / "extra.hpp").write_text("// extra\n", encoding="utf-8")

        def moved() -> None:
            _git(self.source, "commit", "-q", "--allow-empty", "-m", "moved on")

        def other_remote() -> None:
            _git(self.source, "remote", "set-url", "origin", "https://example.invalid/c.git")

        def other_tag() -> None:
            _git(self.source, "tag", "-d", self.pin.tag)

        def no_git() -> None:
            shutil.rmtree(self.source / ".git")

        for change in (modified, untracked, moved, other_remote, other_tag, no_git):
            with self.subTest(change=change.__name__):
                self._checkout()
                change()
                runner = self._runner()
                self.assertEqual(runner._catch2_arguments(),
                                 ["-DFETCHCONTENT_SOURCE_DIR_CATCH2="])
                self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    unittest.main()
