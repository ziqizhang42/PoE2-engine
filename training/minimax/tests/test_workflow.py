from __future__ import annotations

import csv
import fcntl
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest import mock

from fixture import write_feature_fixture
from poe2_training.pattern_suites import pattern_suite
from poe2_training.shared import (
    complete_json_report,
    reserve_report_directory,
    sha256_file,
    write_report_attachment,
)
from poe2_training.workflow import (
    CommandExecutionError,
    CommandResult,
    CommandRunner,
    WorkflowRunner,
    WorkflowError,
    _dependency_order,
    handoff_candidate,
    print_status,
    promote_gate,
    resolved_label_concurrency,
    run_lock,
    validate_existing,
    workflow_snapshot,
)
from poe2_training.workflow_adapters import (
    CANDIDATE_SCHEMA,
    GATE_SCHEMA,
    authenticate_candidate,
    authenticate_gate,
    authenticate_label_shard,
    authenticate_training,
    candidate_build_id,
    data_binary,
    gate_build_directory,
    label_shard_path,
    resolve_eval_binary,
)
from poe2_training.workflow_config import (
    WorkflowConfigError,
    load_workflow_config,
    stage_fingerprints,
)
from poe2_training.workflow_state import RunState, StateError
from poe2_training.workflow_ui import (
    WorkflowProgress,
    format_duration,
    paint,
    progress_detail,
)


REPOSITORY = Path(__file__).resolve().parents[3]


def _managed_config(root: Path, *, extra: str = "", trajectories: int = 4) -> str:
    return textwrap.dedent(f"""
        schema_version = 1

        [run]
        name = "test-run"
        output_dir = "{root / 'run'}"
        build_preset = "debug"

        [datasets.main]
        kind = "managed"
        [datasets.main.source]
        corpus_id = "test-corpus"
        seed = 7
        trajectories = {trajectories}
        shards = 2
        workers = 1
        [datasets.main.labels]
        nodes = 100
        workers = 1

        [[iterations]]
        name = "one"
        dataset = "main"
        [iterations.training]
        type = "baseline"
        device = "cpu"
        {extra}
    """)


def _write(root: Path, contents: str, name: str = "run.toml") -> Path:
    path = root / name
    path.write_text(contents, encoding="utf-8")
    return path


class WorkflowConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY / "build")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_strict_parser_derives_managed_paths_and_references(self) -> None:
        config = load_workflow_config(_write(self.root, _managed_config(self.root)))
        dataset = config.dataset("main")
        self.assertEqual(dataset.source_directory, config.output_directory / "datasets/main/source")
        self.assertEqual(dataset.feature_directory,
                         config.output_directory / "datasets/main/features")
        self.assertEqual(config.iteration("one").training_directory,
                         config.output_directory / "iterations/one/training")
        self.assertIn("dataset:main:features", stage_fingerprints(config))
        self.assertEqual(workflow_snapshot(config)["status"], "pending")

    def test_rejects_unknown_fields_forward_dependencies_and_unsafe_names(self) -> None:
        unknown = _managed_config(self.root).replace(
            'build_preset = "debug"', 'build_preset = "debug"\nunknown = true')
        with self.assertRaisesRegex(WorkflowConfigError, "unknown field"):
            load_workflow_config(_write(self.root, unknown, "unknown.toml"))
        forward = _managed_config(self.root).replace(
            'name = "one"\ndataset',
            'name = "one"\ndepends_on = ["later"]\ndataset')
        with self.assertRaisesRegex(WorkflowConfigError, "earlier"):
            load_workflow_config(_write(self.root, forward, "forward.toml"))
        unsafe = _managed_config(self.root).replace('name = "one"', 'name = "../one"')
        with self.assertRaisesRegex(WorkflowConfigError, "must match"):
            load_workflow_config(_write(self.root, unsafe, "unsafe.toml"))
        unsafe_output = _managed_config(self.root).replace(
            f'output_dir = "{self.root / "run"}"',
            f'output_dir = "{REPOSITORY / ".git/workflow-output"}"')
        with self.assertRaisesRegex(WorkflowConfigError, "inside .git"):
            load_workflow_config(_write(self.root, unsafe_output, "unsafe-output.toml"))

    def test_rejects_overlapping_imports_and_unreferenced_datasets(self) -> None:
        overlapping = textwrap.dedent(f"""
            schema_version = 1
            [run]
            name = "overlap"
            output_dir = "{self.root / 'run'}"
            [datasets.ready]
            kind = "imported"
            path = "{self.root}"
            [[iterations]]
            name = "one"
            dataset = "ready"
            [iterations.training]
            type = "baseline"
            device = "cpu"
        """)
        with self.assertRaisesRegex(WorkflowConfigError, "must not overlap"):
            load_workflow_config(_write(self.root, overlapping, "overlap.toml"))

        unreferenced = _managed_config(self.root) + textwrap.dedent(f"""
            [datasets.unused]
            kind = "imported"
            path = "{self.root / 'unused-features'}"
        """)
        with self.assertRaisesRegex(WorkflowConfigError, "unreferenced dataset"):
            load_workflow_config(_write(self.root, unreferenced, "unreferenced.toml"))

    def test_imported_feature_is_authenticated_without_writes(self) -> None:
        feature = write_feature_fixture(self.root / "imported")
        output = self.root / "run"
        contents = textwrap.dedent(f"""
            schema_version = 1
            [run]
            name = "import-test"
            output_dir = "{output}"
            [datasets.ready]
            kind = "imported"
            path = "{feature}"
            [[iterations]]
            name = "baseline"
            dataset = "ready"
            [iterations.training]
            type = "baseline"
            device = "cpu"
        """)
        config = load_workflow_config(_write(self.root, contents, "import.toml"))
        authenticated = validate_existing(config)
        self.assertIn("dataset:ready:features", authenticated)
        self.assertFalse(output.exists())

    def test_label_concurrency_uses_cpu_budget_and_override(self) -> None:
        config = load_workflow_config(_write(self.root, _managed_config(self.root)))
        with mock.patch("poe2_training.workflow.os.cpu_count", return_value=16):
            self.assertEqual(resolved_label_concurrency(config, config.dataset("main")), 15)
        overridden_text = _managed_config(self.root).replace(
            'build_preset = "debug"', 'build_preset = "debug"\nlabel_concurrency = 3')
        overridden = load_workflow_config(_write(self.root, overridden_text, "override.toml"))
        self.assertEqual(resolved_label_concurrency(overridden, overridden.dataset("main")), 3)

    def test_data_tool_build_tree_is_isolated_per_run(self) -> None:
        first = load_workflow_config(_write(self.root, _managed_config(self.root), "first.toml"))
        second_text = _managed_config(self.root).replace(
            f'output_dir = "{self.root / "run"}"',
            f'output_dir = "{self.root / "other-run"}"')
        second = load_workflow_config(_write(self.root, second_text, "second.toml"))
        self.assertEqual(
            data_binary(first),
            first.output_directory / "build/debug/runner/poe2_minimax_data")
        self.assertNotEqual(data_binary(first), data_binary(second))

    def test_label_authentication_accepts_current_and_legacy_evaluator_names(self) -> None:
        config = load_workflow_config(_write(self.root, _managed_config(self.root)))
        dataset = config.dataset("main")
        shard = label_shard_path(dataset, 0)
        shard.mkdir(parents=True)
        manifest = {
            "schema": "poe2-minimax-labels",
            "corpus": {
                "id": dataset.source.corpus_id,
                "shard_index": 0,
                "shard_count": dataset.source.shards,
            },
            "search": {
                "mode": dataset.labels.mode,
                "node_limit": dataset.labels.nodes,
                "hash_bytes_requested": dataset.labels.hash_mb * 1024 * 1024,
                "workers_requested": dataset.labels.workers,
                "workers_used": dataset.labels.workers,
                "require_all": dataset.labels.require_all,
                "target_selection": "deepest_terminal_parity",
                "evaluator": "two-ply-closure",
                "symmetry": True,
                "two_ply_closure": True,
            },
            "build": {"git_commit": "1" * 40, "git_dirty": False},
            "results": {"records": 1, "unsolved": 0},
            "binary_digest": "sha256:" + "2" * 64,
        }
        manifest_path = shard / "manifest.json"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        self.assertEqual(authenticate_label_shard(dataset, 0)["git_commit"], "1" * 40)

        manifest["search"]["evaluator"] = "b"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        self.assertEqual(authenticate_label_shard(dataset, 0)["records"], 1)

        manifest["search"]["evaluator"] = "two-ply"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "search semantics"):
            authenticate_label_shard(dataset, 0)

    def test_partial_label_resume_reuses_binary_with_original_commit(self) -> None:
        config = load_workflow_config(_write(self.root, _managed_config(self.root)))
        state = RunState(config)
        state.accept_config()
        original_commit = "1" * 40
        binary = data_binary(config)
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"fixture\0" + original_commit.encode() + b"\0")
        workflow = WorkflowRunner(config, state, "2" * 40)

        with mock.patch.object(workflow, "_command") as command:
            workflow._ensure_data_binary(
                "dataset:main:labels", 2, required_commit=original_commit)

        command.assert_not_called()
        self.assertEqual(workflow._data_commit, original_commit)

    def test_pattern_suite_report_remains_authenticatable_after_shared_refactor(self) -> None:
        feature = write_feature_fixture(self.root / "suite-features")
        contents = textwrap.dedent(f"""
            schema_version = 1
            [run]
            name = "suite-auth"
            output_dir = "{self.root / 'suite-run'}"
            [datasets.ready]
            kind = "imported"
            path = "{feature}"
            [[iterations]]
            name = "frozen"
            dataset = "ready"
            [iterations.training]
            type = "pattern"
            device = "cpu"
            suite = "frozen-pattern-gain"
        """)
        config = load_workflow_config(_write(self.root, contents, "suite.toml"))
        iteration = config.iteration("frozen")
        model_configs = json.loads(json.dumps(
            [asdict(item) for item in pattern_suite("frozen-pattern-gain")]))
        selected = model_configs[0]["name"]
        reserve_report_directory(
            iteration.training_directory, "poe2-minimax-pattern-experiment")
        report = {
                "schema": "poe2-minimax-pattern-experiment", "schema_version": 1,
                "training": {"seed": 20260818, "configs": model_configs},
                "input": {
                    "feature_binary_sha256": sha256_file(feature / "features.bin"),
                    "feature_manifest_sha256": sha256_file(feature / "manifest.json"),
                },
                "models": [{
                    "name": selected,
                    "validation": {"overall": {"mae": 1.0, "rmse": 1.0}},
                }],
                "selection": {"best_model": selected},
                "provenance": {
                    "git": {"dirty": False}, "runtime": {"device_type": "cpu"},
                },
            }
        report["attachments"] = {
            "training_metrics": write_report_attachment(
                iteration.training_directory, "training-metrics.svg", b"<svg/>\n",
                media_type="image/svg+xml")
        }
        complete_json_report(
            iteration.training_directory, "poe2-minimax-pattern-experiment", report)
        authenticated = authenticate_training(config.dataset("ready"), iteration)
        self.assertEqual(authenticated["selected_model"], selected)
        self.assertEqual(authenticated["attachments"], report["attachments"])


class WorkflowStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY / "build")
        self.root = Path(self.temporary.name)
        self.path = _write(self.root, _managed_config(self.root))
        self.config = load_workflow_config(self.path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_config_revisions_allow_downstream_additions_but_lock_started_stage(self) -> None:
        state = RunState(self.config)
        state.accept_config()
        stage = "dataset:main:source"
        state.append("stage_started", stage=stage, attempt=1,
                     config_fingerprint=stage_fingerprints(self.config)[stage],
                     input_fingerprint="0" * 64, git_commit="1" * 40)

        changed = _managed_config(self.root, trajectories=5)
        changed_config = load_workflow_config(_write(self.root, changed, "changed.toml"))
        with self.assertRaisesRegex(StateError, "changes started stage"):
            RunState(changed_config).validate_evolution()

        appended = _managed_config(self.root) + textwrap.dedent("""
            [[iterations]]
            name = "two"
            dataset = "main"
            depends_on = ["one"]
            [iterations.training]
            type = "baseline"
            device = "cpu"
        """)
        appended_config = load_workflow_config(_write(self.root, appended, "appended.toml"))
        revision, snapshot, fresh = RunState(appended_config).accept_config()
        self.assertEqual(revision, 2)
        self.assertTrue(fresh)
        self.assertEqual(snapshot.read_bytes(), appended_config.contents)

    def test_event_recovery_and_summary_rebuild(self) -> None:
        state = RunState(self.config)
        state.accept_config()
        stage = "dataset:main:source"
        state.append("stage_started", stage=stage, attempt=1,
                     config_fingerprint=stage_fingerprints(self.config)[stage],
                     input_fingerprint="0" * 64, git_commit="1" * 40)
        recovered = RunState(self.config)
        recovered.recover_running_stages()
        recovered.write_summary(workflow_snapshot(self.config))
        summary = json.loads(recovered.summary_path.read_bytes())
        self.assertEqual(summary["stages"][stage]["status"], "failed")
        self.assertEqual(summary["event_count"], len(recovered.events))

    def test_run_lock_rejects_a_second_orchestrator(self) -> None:
        with run_lock(self.config):
            with self.assertRaisesRegex(WorkflowError, "holds the run lock"):
                with run_lock(self.config):
                    pass

    def test_command_runner_records_wall_cpu_and_peak_rss(self) -> None:
        runner = CommandRunner(REPOSITORY)
        result = runner.run(
            [sys.executable, "-c", "print('workflow-command-ok')"],
            log_path=self.root / "command.log", metrics_path=self.root / "resource.json")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("workflow-command-ok", result.output)
        metrics = json.loads(result.metrics_path.read_bytes())
        self.assertGreaterEqual(metrics["wall_seconds"], 0.0)
        self.assertIn("peak_rss_kb", metrics)

    def test_progress_helpers_reduce_native_output_and_format_time(self) -> None:
        self.assertEqual(format_duration(65), "1m 05s")
        self.assertEqual(
            progress_detail(
                "create-shard-002",
                "label_progress completed=500 total=2000 records=500 unsolved=0"),
            "shard 3 500/2,000 (25%)",
        )
        self.assertEqual(
            progress_detail(
                "create", "source_progress completed_trajectories=100 "
                "total_trajectories=1000"),
            "create 100/1,000 (10%)",
        )

    def test_reused_stage_is_labeled_authenticated(self) -> None:
        output = io.StringIO()
        progress = WorkflowProgress(self.config, RunState(self.config), stream=output)
        stage = "dataset:main:source"
        progress.stage_started(stage, "authenticate")
        progress.stage_completed(stage, created=False)
        progress.close(success=True)
        self.assertIn("dataset main · source  authenticated ·", output.getvalue())

    def test_human_status_has_progress_time_and_iteration_order(self) -> None:
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            print_status(self.config)
        value = output.getvalue()
        self.assertIn("0/4 stages (0%)", value)
        self.assertIn("recorded 0.0s", value)
        self.assertIn("iterations  1 ○ one", value)
        self.assertNotIn("\033[", value)

    def test_color_is_terminal_aware_and_honors_no_color(self) -> None:
        class Terminal(io.StringIO):
            def isatty(self) -> bool:
                return True

        stream = Terminal()
        with mock.patch.dict(os.environ, {"TERM": "xterm"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            self.assertIn("\033[", paint("ok", "green", stream=stream))
        with mock.patch.dict(os.environ, {"TERM": "xterm", "NO_COLOR": "1"}, clear=False):
            self.assertEqual(paint("ok", "green", stream=stream), "ok")

    def test_command_callback_keeps_child_lines_off_the_console(self) -> None:
        runner = CommandRunner(REPOSITORY)
        lines: list[str] = []
        output = io.StringIO()
        with mock.patch("sys.stdout", output):
            result = runner.run(
                [sys.executable, "-c", "print('source_progress completed=1 total=2')"],
                log_path=self.root / "compact.log",
                metrics_path=self.root / "compact.json",
                output_callback=lambda line, _elapsed: lines.append(line),
            )
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(lines, ["source_progress completed=1 total=2\n"])
        self.assertEqual(result.output, lines[0])

    def test_complete_artifact_is_authenticated_without_recreation(self) -> None:
        state = RunState(self.config)
        state.accept_config()
        workflow = WorkflowRunner(self.config, state, "1" * 40)
        artifact = self.root / "reusable"
        artifact.mkdir()
        (artifact / "COMPLETE").write_text("fixture\n", encoding="utf-8")
        create = mock.Mock()
        authenticate = mock.Mock(return_value={"digest": "a" * 64})
        stats = workflow._run_simple(
            "dataset:main:source", artifact, "fixture artifact", {},
            create, authenticate)
        create.assert_not_called()
        authenticate.assert_called_once()
        self.assertEqual(stats["digest"], "a" * 64)

    def test_started_stage_rejects_changed_inputs(self) -> None:
        state = RunState(self.config)
        state.accept_config()
        workflow = WorkflowRunner(self.config, state, "1" * 40)
        artifact = self.root / "input-locked"
        artifact.mkdir()
        (artifact / "COMPLETE").write_text("fixture\n", encoding="utf-8")
        authenticate = mock.Mock(return_value={"digest": "a" * 64})
        workflow._run_simple(
            "dataset:main:source", artifact, "fixture artifact",
            {"upstream": "first"}, mock.Mock(), authenticate)

        with self.assertRaisesRegex(WorkflowError, "inputs to started stage"):
            workflow._run_simple(
                "dataset:main:source", artifact, "fixture artifact",
                {"upstream": "changed"}, mock.Mock(), authenticate)
        self.assertEqual(authenticate.call_count, 1)

    def test_command_failure_can_retry_with_new_create_only_logs(self) -> None:
        runner = CommandRunner(REPOSITORY)
        with self.assertRaises(CommandExecutionError):
            runner.run(
                [sys.executable, "-c", "raise SystemExit(7)"],
                log_path=self.root / "attempt-one.log",
                metrics_path=self.root / "attempt-one.json")
        result = runner.run(
            [sys.executable, "-c", "print('retry-ok')"],
            log_path=self.root / "attempt-two.log",
            metrics_path=self.root / "attempt-two.json")
        self.assertEqual(result.exit_code, 0)
        self.assertTrue((self.root / "attempt-one.log").is_file())
        self.assertTrue((self.root / "attempt-two.log").is_file())

    def test_terminate_all_stops_a_running_subprocess_group(self) -> None:
        runner = CommandRunner(REPOSITORY)
        failures: list[BaseException] = []

        def run() -> None:
            try:
                runner.run(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    log_path=self.root / "long.log", metrics_path=self.root / "long.json")
            except BaseException as error:
                failures.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        deadline = time.monotonic() + 5.0
        while not runner._active and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(runner._active)
        runner.terminate_all()
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(failures)

    def test_concurrent_label_scheduler_commits_only_complete_shards(self) -> None:
        contents = _managed_config(self.root).replace(
            'build_preset = "debug"',
            'build_preset = "debug"\nlabel_concurrency = 2')
        config = load_workflow_config(_write(self.root, contents, "labels.toml"))
        dataset = config.dataset("main")
        source = dataset.source_directory
        (source / "shards").mkdir(parents=True)
        shards = []
        for index in range(dataset.source.shards):
            name = f"shard-{index}.jsonl"
            (source / "shards" / name).write_text("fixture\n", encoding="utf-8")
            shards.append({"index": index, "name": name})
        (source / "manifest.json").write_text(
            json.dumps({"shards": shards}) + "\n", encoding="utf-8")
        state = RunState(config)
        state.accept_config()
        workflow = WorkflowRunner(config, state, "1" * 40)
        active = 0
        maximum = 0
        lock = threading.Lock()

        def fake_command(stage, operation, command, *, attempt=None,
                         accepted_exit_codes=None):
            nonlocal active, maximum
            if operation.startswith("create-shard-"):
                index = int(operation.rsplit("-", 1)[1])
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                time.sleep(0.03)
                path = label_shard_path(dataset, index)
                path.mkdir(parents=True)
                (path / "COMPLETE").write_text("fixture\n", encoding="utf-8")
                with lock:
                    active -= 1
            output = "{}\n" if operation == "preflight" else ""
            return CommandResult(tuple(command), 0, output,
                                 self.root / "fake.log", self.root / "fake.json",
                                 {"wall_seconds": 0.03, "peak_rss_kb": 1})

        def fake_auth(_dataset, index):
            return {"index": index, "manifest_sha256": f"{index:064x}",
                    "records": 1, "terminal_records": 0, "parity_backoffs": 0,
                    "previous_records": 0, "unsolved": 0}

        with mock.patch.object(workflow, "_ensure_data_binary"), \
                mock.patch.object(workflow, "_command", side_effect=fake_command), \
                mock.patch("poe2_training.workflow.authenticate_label_shard",
                           side_effect=fake_auth):
            statistics = workflow._prepare_labels(dataset, {"manifest_sha256": "a" * 64})
        self.assertEqual(maximum, 2)
        self.assertEqual(statistics["shards"], 2)
        self.assertTrue(all((label_shard_path(dataset, index) / "COMPLETE").is_file()
                            for index in range(2)))

    def test_dependency_order_includes_each_earlier_iteration_once(self) -> None:
        appended = _managed_config(self.root) + textwrap.dedent("""
            [[iterations]]
            name = "two"
            dataset = "main"
            depends_on = ["one"]
            [iterations.training]
            type = "baseline"
            device = "cpu"
        """)
        config = load_workflow_config(_write(self.root, appended, "dependencies.toml"))
        self.assertEqual([item.name for item in
                          _dependency_order(config, config.iteration("two"))], ["one", "two"])


class WorkflowPromotionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY / "build")
        self.root = Path(self.temporary.name)
        (self.root / "base-engine").write_text("base\n", encoding="utf-8")
        self.config = self._config()
        self.iteration = self.config.iteration("candidate")
        write_feature_fixture(self.root / "features")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _config(self, *, promotable: bool = True, include_gate: bool = True,
                base: str | None = None,
                new_engine: str = "minimax/poe2_minimax",
                new_engine_args: str = "--evaluator pattern-gain",
                sequential_alpha: float = 0.05):
        base = base or str(self.root / "base-engine")
        gate = textwrap.dedent(f"""
            [iterations.engine_gate]
            base = "{base}"
            new_engine = "{new_engine}"
            new_engine_args = "{new_engine_args}"
            games = 2
            sequential_alpha = {sequential_alpha!r}
        """) if include_gate else ""
        contents = textwrap.dedent(f"""
            schema_version = 1
            [run]
            name = "promotion-test"
            output_dir = "{self.root / 'run'}"
            [datasets.ready]
            kind = "imported"
            path = "{self.root / 'features'}"
            [[iterations]]
            name = "candidate"
            dataset = "ready"
            [iterations.training]
            type = "pattern"
            device = "cpu"
            suite = "frozen-pattern-gain"
            [iterations.candidate_validation]
            promotable = {str(promotable).lower()}
            {gate}
        """)
        return load_workflow_config(_write(self.root, contents, "promotion.toml"))

    def _candidate(self) -> None:
        candidate = self.iteration.candidate_validation
        assert candidate is not None
        training = self.iteration.training_directory
        reserve_report_directory(training, "poe2-minimax-pattern-experiment")
        complete_json_report(training, "poe2-minimax-pattern-experiment", {
            "schema": "poe2-minimax-pattern-experiment", "schema_version": 1})
        root = self.iteration.candidate_directory
        reserve_report_directory(root, CANDIDATE_SCHEMA)
        log = self.config.output_directory / "logs" / "candidate-fixture.log"
        log.parent.mkdir(parents=True)
        log.write_text("candidate fixture\n", encoding="utf-8")
        artifacts = {}
        for name, relative in {
            "model_header": "frozen_pattern_gain_model.hpp",
            "inference_binary": "build/engines/minimax/poe2_minimax_infer",
            "engine_binary": "build/engines/minimax/poe2_minimax",
            "runner_binary": "build/runner/poe2_runner",
        }.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
            artifacts[name] = {"path": relative, "sha256": sha256_file(path)}
        feature_digest = sha256_file(self.root / "features" / "features.bin")
        complete_json_report(root, CANDIDATE_SCHEMA, {
            "schema": CANDIDATE_SCHEMA, "schema_version": 1,
            "promotable": candidate.promotable,
            "git_commit": "1" * 40,
            "settings": {
                "promotable": candidate.promotable, "samples": candidate.samples,
                "symmetry_samples": candidate.symmetry_samples,
                "benchmark_iterations": candidate.benchmark_iterations,
                "opening_book": str(candidate.opening_book),
                "search_positions": candidate.search_positions,
                "search_movetime_ms": candidate.search_movetime_ms,
            },
            "opening_book_sha256": sha256_file(candidate.opening_book),
            "training_report_sha256": sha256_file(training / "report.json"),
            "sealed_evaluation_report_sha256": None,
            "feature_binary_sha256": feature_digest,
            "model": {
                "name": "fixture",
                "float": {"validation": {"mae": 1.0}, "sealed_test": None},
                "quantized": {"validation": {"mae": 1.25}, "sealed_test": None},
            },
            "parity": {
                "passed": True, "base_positions": 1, "transformed_positions": 7,
                "feature_binary_sha256": feature_digest,
            },
            "evaluator_benchmark": {
                "two-ply-closure": {"evaluator": "two-ply-closure"},
                "pattern-gain": {"evaluator": "pattern-gain"},
            },
            "search_benchmark": {
                "evaluators": {
                    "two-ply-closure": {"evaluator": "two-ply-closure"},
                    "pattern-gain": {"evaluator": "pattern-gain"},
                },
                "comparison": {"pattern_gain_to_two_ply_closure_nps": 1.0},
            },
            "artifacts": artifacts,
            "commands": [{
                "command": ["fixture"],
                "log": log.relative_to(self.config.output_directory).as_posix(),
                "log_sha256": sha256_file(log),
                "resources": {"wall_seconds": 0.0, "peak_rss_kb": 1},
            }],
        })

    def _gate(self, *, manifest_base_path: str | None = None,
              ledger_overrides: dict[str, str] | None = None) -> tuple[list[str], list[str]]:
        master = REPOSITORY / "eval" / "results.csv"
        with master.open(newline="", encoding="utf-8") as source:
            header = next(csv.reader(source))
        row = [""] * len(header)
        row[header.index("run_id")] = "workflow-promotion-test-run"
        root = self.iteration.gate_directory
        reserve_report_directory(root, GATE_SCHEMA)
        header_digest = sha256_file(
            self.iteration.candidate_directory / "frozen_pattern_gain_model.hpp")
        candidate_stats = {
            "artifacts": {"model_header": {"sha256": header_digest}}}
        build_id = candidate_build_id(candidate_stats)
        build = gate_build_directory(self.config, self.iteration, candidate_stats)
        new_engine_binary = build / "engines/minimax/poe2_minimax"
        runner_binary = build / "runner/poe2_runner"
        for path in (new_engine_binary, runner_binary):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(path.name, encoding="utf-8")
        run = root / "runs" / "one"
        run.mkdir(parents=True)
        gate = self.iteration.engine_gate
        assert gate is not None
        base_engine = resolve_eval_binary(self.config, gate.base, gate.base_engine)
        manifest = {
            "run_id": row[header.index("run_id")],
            "created_at_utc": "20260819T000000Z", "valid": True,
            "kind": "training-gate", "new_id": build_id,
            "new_engine": gate.new_engine,
            "new_engine_path": str(new_engine_binary.resolve()),
            "new_engine_args": gate.new_engine_args, "base_engine": gate.base_engine,
            "base_engine_args": gate.base_engine_args,
            "base_engine_path": manifest_base_path or str(base_engine),
            "base_id": "fixture-base", "opening_book": str(gate.opening_book),
            "opening_book_digest": "fnv1a64:fixture", "opening_count": 1,
            "opening_seed": 123, "opening_seed_source": "derived",
            "opening_sampling": "deterministic_shuffle_without_replacement",
            "shuffle_openings": True, "unique_openings": 1,
            "games": gate.games, "games_played": gate.games,
            "workers_requested": gate.workers, "workers_used": 1,
            "games_discarded": 0, "invalid_reason": None,
            "timeout_ms": gate.timeout_ms, "go_depth": None,
            "go_movetime_ms": gate.go_movetime_ms, "go_nodes": None,
            "analysis_version": 2, "statistical_unit": "opening_pair",
            "confidence_method": "paired_betting_confidence_sequence",
            "sequential_test_method": "paired_normalized_elo_gsprt",
            "sequential_stop": gate.sequential_stop,
            "sequential_bound_unit": "normalized_elo",
            "sequential_null": gate.sequential_null,
            "sequential_alt": gate.sequential_alt,
            "sequential_alpha": gate.sequential_alpha,
            "sequential_beta": gate.sequential_beta,
            "sequential_llr": 1.0, "sequential_lower_bound": -2.0,
            "sequential_upper_bound": 2.0, "normalized_elo": 20.0,
            "betting_log_evidence_above_even": 3.0,
            "betting_log_evidence_below_even": -3.0,
            "sequential_decision": "accept_alt",
        }
        summary = {
            "analysis_version": 2, "valid": True, "invalid_reason": None,
            "games_requested": gate.games, "games_played": gate.games,
            "workers_requested": gate.workers, "workers_used": 1,
            "games_discarded": 0, "unique_openings": 1,
            "engine_one_wins": gate.games, "engine_two_wins": 0, "no_winner": 0,
            "statistical_unit": "opening_pair", "statistical_samples": 1,
            "statistical_games": gate.games,
            "statistical_score_rate_counts": {
                "0": 0, "0.25": 0, "0.5": 0, "0.75": 0, "1": 1,
            },
        }
        ledger_values = {
            "created_at_utc": manifest["created_at_utc"],
            "kind": "training-gate", "new_id": build_id,
            "new_engine": gate.new_engine,
            "new_engine_args": gate.new_engine_args, "base_id": manifest["base_id"],
            "base_engine": gate.base_engine, "base_engine_args": gate.base_engine_args,
            "games_requested": str(gate.games), "games_played": str(gate.games),
            "engine_one_wins": str(gate.games), "engine_two_wins": "0", "no_winner": "0",
            "engine_one_score_pct": "100.000", "confidence_low_pct": "50.000",
            "confidence_high_pct": "100.000",
            "sequential_decision": manifest["sequential_decision"],
            "sequential_null": str(gate.sequential_null),
            "sequential_alt": str(gate.sequential_alt),
            "opening_book": str(gate.opening_book),
            "opening_book_digest": manifest["opening_book_digest"],
            "opening_count": "1", "go_movetime_ms": str(gate.go_movetime_ms),
            "timeout_ms": str(gate.timeout_ms), "run_dir": str(run.resolve()),
            "analysis_version": "2", "statistical_unit": "opening_pair",
            "statistical_samples": "1", "statistical_games": str(gate.games),
            "pair_score_0": "0", "pair_score_0_5": "0", "pair_score_1": "0",
            "pair_score_1_5": "0", "pair_score_2": "1",
            "confidence_method": manifest["confidence_method"],
            "sequential_test_method": manifest["sequential_test_method"],
            "analysis_note": "paired normalized-Elo GSPRT", "valid": "true",
            "opening_seed": "123", "opening_seed_source": "derived",
            "opening_sampling": manifest["opening_sampling"], "unique_openings": "1",
            "sequential_model": manifest["sequential_test_method"],
            "sequential_bound_unit": manifest["sequential_bound_unit"],
            "sequential_llr": "1.000", "sequential_lower_bound": "-2.000",
            "sequential_upper_bound": "2.000", "normalized_elo": "20.000",
            "betting_log_evidence_above_even": "3.000",
            "betting_log_evidence_below_even": "-3.000",
            "workers_requested": str(gate.workers), "workers_used": "1",
            "games_discarded": "0",
        }
        ledger_values.update(ledger_overrides or {})
        for name, value in ledger_values.items():
            row[header.index(name)] = value
        (run / "manifest.json").write_text(
            json.dumps(manifest) + "\n", encoding="utf-8")
        (run / "summary.json").write_text(
            json.dumps(summary) + "\n", encoding="utf-8")
        (run / "games.csv").write_text("game\n", encoding="utf-8")
        with (run / "ledger-row.csv").open("w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination, lineterminator="\n")
            writer.writerow(header)
            writer.writerow(row)
        complete_json_report(root, GATE_SCHEMA, {
            "schema": GATE_SCHEMA, "schema_version": 1,
            "git_commit": "2" * 40,
            "settings": {
                **asdict(gate),
                "opening_book": str(gate.opening_book),
            },
            "opening_book_sha256": sha256_file(gate.opening_book),
            "run_id": row[header.index("run_id")], "run_directory": "runs/one",
            "candidate_report_sha256": sha256_file(
                self.iteration.candidate_directory / "report.json"),
            "candidate_build_id": build_id,
            "build_directory": build.relative_to(root).as_posix(),
            "tracked_model_header_sha256": header_digest,
            "base_engine_artifact": {
                "path": str(base_engine),
                "sha256": sha256_file(base_engine),
            },
            "build_artifacts": {
                "new_engine_binary": {
                    "path": new_engine_binary.relative_to(root).as_posix(),
                    "sha256": sha256_file(new_engine_binary),
                },
                "runner_binary": {
                    "path": runner_binary.relative_to(root).as_posix(),
                    "sha256": sha256_file(runner_binary),
                },
            },
            "artifacts": {name: sha256_file(run / name) for name in
                          ("manifest.json", "summary.json", "games.csv", "ledger-row.csv")},
        })
        return header, row

    def test_non_promotable_candidate_can_be_previewed_but_not_applied(self) -> None:
        self.config = self._config(promotable=False, include_gate=False)
        self.iteration = self.config.iteration("candidate")
        self._candidate()
        preview = handoff_candidate(self.config, "candidate")
        self.assertFalse(preview["promotable"])
        with self.assertRaisesRegex(WorkflowError, "non-promotable"):
            handoff_candidate(self.config, "candidate", apply=True)

    def test_engine_gate_rejects_a_non_promotable_candidate(self) -> None:
        with self.assertRaisesRegex(WorkflowConfigError, "promotable candidate"):
            self._config(promotable=False, include_gate=True)

    def test_engine_gate_rejects_commands_that_bypass_the_candidate(self) -> None:
        with self.assertRaisesRegex(WorkflowConfigError, "candidate evaluator"):
            self._config(new_engine_args="--evaluator two-ply-closure")
        with self.assertRaisesRegex(WorkflowConfigError, "candidate model"):
            self._config(new_engine="poe2_greedy")

    def test_candidate_distinguishes_float_and_exported_quantized_metrics(self) -> None:
        self._candidate()
        authenticated = authenticate_candidate(self.config, self.iteration)
        self.assertEqual(authenticated["model"]["float"]["validation"]["mae"], 1.0)
        self.assertEqual(
            authenticated["model"]["quantized"]["validation"]["mae"], 1.25)

    def test_gate_authenticates_relative_base_and_candidate_build_identity(self) -> None:
        base_parent = REPOSITORY / "build/by-commit"
        base_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
                dir=base_parent, prefix="workflow-base-") as base_directory:
            base_root = Path(base_directory)
            base_binary = base_root / "release/engines/minimax/poe2_minimax"
            base_binary.parent.mkdir(parents=True)
            base_binary.write_text("base\n", encoding="utf-8")
            self.config = self._config(
                base=base_root.name, sequential_alpha=0.05123456789)
            self.iteration = self.config.iteration("candidate")
            self._candidate()
            self._gate(manifest_base_path=base_binary.relative_to(REPOSITORY).as_posix())
            authenticated = authenticate_gate(self.config, self.iteration)
            header_digest = sha256_file(
                self.iteration.candidate_directory / "frozen_pattern_gain_model.hpp")
            self.assertEqual(
                authenticated["candidate_build_id"],
                f"candidate-sha256-{header_digest}")
            manifest = json.loads(Path(authenticated["run_directory"])
                                  .joinpath("manifest.json").read_bytes())
            self.assertEqual(manifest["new_id"], authenticated["candidate_build_id"])

    def test_gate_promotion_rejects_a_ledger_row_that_disagrees_with_manifest(self) -> None:
        self._candidate()
        self._gate(ledger_overrides={"base_id": "wrong-base"})
        with self.assertRaisesRegex(WorkflowError, "base_id"):
            promote_gate(self.config, "candidate")

    def test_apply_handoff_and_promotion_respect_the_run_lock(self) -> None:
        self._candidate()
        self._gate()
        with run_lock(self.config):
            with self.assertRaisesRegex(WorkflowError, "holds the run lock"):
                handoff_candidate(self.config, "candidate", apply=True)
            with self.assertRaisesRegex(WorkflowError, "holds the run lock"):
                promote_gate(self.config, "candidate", apply=True)

    def test_dependency_reuse_authenticates_all_configured_outputs(self) -> None:
        workflow = WorkflowRunner(self.config, RunState(self.config), "1" * 40)
        with mock.patch(
                "poe2_training.workflow.authenticate_training",
                return_value={"report_sha256": "a" * 64}) as training, \
                mock.patch(
                    "poe2_training.workflow.authenticate_candidate") as candidate, \
                mock.patch("poe2_training.workflow.authenticate_gate") as gate:
            result = workflow._authenticate_iteration_outputs(self.iteration)
        self.assertEqual(result["report_sha256"], "a" * 64)
        training.assert_called_once()
        candidate.assert_called_once()
        gate.assert_called_once()

    def test_engine_gate_requires_the_committed_header_to_match_candidate(self) -> None:
        self._candidate()
        workflow = WorkflowRunner(self.config, RunState(self.config), "1" * 40)
        candidate_header = self.iteration.candidate_directory / "frozen_pattern_gain_model.hpp"
        with self.assertRaisesRegex(WorkflowError, "handoff --apply"):
            workflow._run_gate(self.iteration, {
                "report_sha256": sha256_file(
                    self.iteration.candidate_directory / "report.json"),
                "artifacts": {"model_header": {"sha256": sha256_file(candidate_header)}},
            })

    def test_promotable_handoff_applies_only_the_tracked_header(self) -> None:
        self.config = self._config(promotable=True)
        self.iteration = self.config.iteration("candidate")
        self._candidate()
        fake_repository = self.root / "handoff-repository"
        header = fake_repository / "engines/minimax/src/frozen_pattern_gain_model.hpp"
        header.parent.mkdir(parents=True)
        header.write_text("tracked header\n", encoding="utf-8")
        fake_config = fake_repository / "promotion.toml"
        fake_config.write_bytes(self.config.contents)
        subprocess.run(["git", "init", "-q"], cwd=fake_repository, check=True)
        subprocess.run(
            ["git", "add", "engines/minimax/src/frozen_pattern_gain_model.hpp",
             "promotion.toml"], cwd=fake_repository, check=True)
        subprocess.run([
            "git", "-c", "user.name=Workflow Test", "-c",
            "user.email=test@example.invalid", "commit", "-qm", "fixture"],
            cwd=fake_repository, check=True)
        isolated = replace(self.config, repository=fake_repository, path=fake_config)

        preview = handoff_candidate(isolated, "candidate")
        self.assertTrue(preview["different"])
        result = handoff_candidate(isolated, "candidate", apply=True)
        self.assertTrue(result["applied"])
        self.assertEqual(
            header.read_bytes(),
            (self.iteration.candidate_directory /
             "frozen_pattern_gain_model.hpp").read_bytes())
        status = subprocess.run(
            ["git", "status", "--short"], cwd=fake_repository,
            check=True, capture_output=True, text=True).stdout.splitlines()
        self.assertEqual(
            status, [" M engines/minimax/src/frozen_pattern_gain_model.hpp"])

    def test_gate_preview_apply_and_duplicate_rejection(self) -> None:
        self._candidate()
        header, _ = self._gate()
        preview = promote_gate(self.config, "candidate")
        self.assertFalse(preview["duplicate"])

        fake_repository = self.root / "repository"
        (fake_repository / "eval").mkdir(parents=True)
        with (fake_repository / "eval" / "results.csv").open(
                "w", newline="", encoding="utf-8") as destination:
            csv.writer(destination, lineterminator="\n").writerow(header)
        (fake_repository / ".gitignore").write_text(
            "eval/results.csv.lock\n", encoding="utf-8")
        fake_config = fake_repository / "promotion.toml"
        fake_config.write_bytes(self.config.contents)
        subprocess.run(["git", "init", "-q"], cwd=fake_repository, check=True)
        subprocess.run(
            ["git", "add", ".gitignore", "eval/results.csv", "promotion.toml"],
            cwd=fake_repository, check=True)
        subprocess.run([
            "git", "-c", "user.name=Workflow Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", "fixture"], cwd=fake_repository, check=True)
        isolated = replace(self.config, repository=fake_repository, path=fake_config)
        lock_path = fake_repository / "eval/results.csv.lock"
        lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        results: list[dict[str, object]] = []
        errors: list[BaseException] = []

        def apply_promotion() -> None:
            try:
                results.append(promote_gate(isolated, "candidate", apply=True))
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=apply_promotion)
        thread.start()
        try:
            time.sleep(0.05)
            self.assertTrue(thread.is_alive())
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
        thread.join(timeout=5.0)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        applied = results[0]
        self.assertTrue(applied["applied"])
        self.assertTrue((fake_repository / "eval/results.csv.lock").is_file())
        with (fake_repository / "eval/results.csv").open(
                newline="", encoding="utf-8") as source:
            promoted_rows = list(csv.reader(source))
        self.assertEqual(len(promoted_rows), 2)
        self.assertTrue(all(len(item) == len(header) for item in promoted_rows))
        with self.assertRaisesRegex(WorkflowError, "already contains"):
            promote_gate(isolated, "candidate")


if __name__ == "__main__":
    unittest.main()
