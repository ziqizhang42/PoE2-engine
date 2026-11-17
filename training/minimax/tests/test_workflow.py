from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import os
import shutil
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

from fixture import archive_net, net_bytes, moved_repository, repository_temporary_directory, write_feature_fixture
from test_workflow_generations import V02
from poe2_training.pattern_suites import pattern_suite
from poe2_training.shared import (
    complete_json_report,
    reserve_report_directory,
    sha256_file,
    write_report_attachment,
)
from poe2_training.workflow import (
    MODEL_INDEX_HEADER,
    CommandExecutionError,
    CommandResult,
    CommandRunner,
    WorkflowRunner,
    WorkflowError,
    _comparison_models,
    _dependency_order,
    _verified_ledger_row,
    check_model_lineage,
    handoff_candidate,
    print_status,
    promote_gate,
    record_comparison,
    resolve_trained_models,
    resolved_label_concurrency,
    run_lock,
    validate_existing,
    workflow_snapshot,
)
from poe2_training.net_file import decode_net
from poe2_training.workflow_adapters import (
    CANDIDATE_NET,
    CANDIDATE_SCHEMA,
    CANDIDATE_SCHEMA_VERSION,
    COMPARISON_KIND,
    COMPARISON_SCHEMA,
    COMPARISON_SCHEMA_VERSION,
    GATE_KIND,
    GATE_SCHEMA,
    GATE_SCHEMA_VERSION,
    HEADER_GATE_SCHEMA_VERSION,
    AdapterError,
    authenticate_candidate,
    authenticate_comparison,
    authenticate_features,
    authenticate_gate,
    authenticate_label_shard,
    authenticate_training,
    candidate_build_id,
    candidate_net_name,
    candidate_net_path,
    comparison_side,
    data_binary,
    data_build_directory,
    engine_build_id,
    engine_gate_command,
    gate_base_binary,
    gate_base_model_id,
    gate_build_directory,
    label_command,
    label_shard_path,
    model_net,
    resolve_eval_binary,
    source_assemble_command,
    source_chunk_command,
    source_chunk_path,
    source_chunk_ranges,
)
from poe2_training.workflow_config import (
    ModelReference,
    WorkflowConfigError,
    fingerprint,
    load_workflow_config,
    match_settings,
    recorded_gate_settings,
    stage_definitions,
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
        # Private builds here are mocked; test_workflow_recovery covers the Catch2 reuse.
        self.enterContext(mock.patch("poe2_training.workflow.catch2_pin", return_value=None))
        self.temporary = repository_temporary_directory()
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
        with self.assertRaisesRegex(ValueError, "evaluator is unsupported"):
            authenticate_label_shard(dataset, 0)

    def _teacher_repository(self) -> tuple[Path, str]:
        repository = self.root / "teacher-repository"
        (repository / ".git").mkdir(parents=True)
        (repository / "CMakePresets.json").write_bytes(
            (REPOSITORY / "CMakePresets.json").read_bytes())
        (repository / "eval/openings").mkdir(parents=True)
        for book in ("development.txt", "holdout.txt"):
            (repository / "eval/openings" / book).write_bytes(
                (REPOSITORY / "eval/openings" / book).read_bytes())
        return repository, archive_net(repository, "fixture/g000")

    @staticmethod
    def _teacher_text(repository: Path, model_id: str, *, teacher: str = "g000") -> str:
        return _managed_config(repository).replace(
            '[datasets.main]\nkind = "managed"',
            f'[models]\ng000 = "{model_id}"\n\n'
            f'[datasets.main]\nkind = "managed"\nteacher = "{teacher}"')

    @staticmethod
    def _label_manifest(dataset, *, evaluator: str = "two-ply-closure",
                        model: str | None = None) -> dict[str, object]:
        return {
            "schema": "poe2-minimax-labels",
            "corpus": {"id": dataset.source.corpus_id, "shard_index": 0,
                       "shard_count": dataset.source.shards},
            "search": {
                "mode": dataset.labels.mode, "node_limit": dataset.labels.nodes,
                "hash_bytes_requested": dataset.labels.hash_mb * 1024 * 1024,
                "workers_requested": dataset.labels.workers,
                "workers_used": dataset.labels.workers,
                "require_all": dataset.labels.require_all,
                "target_selection": "deepest_terminal_parity", "evaluator": evaluator,
                **({} if model is None else {"model_header_sha256": model}),
                "symmetry": True, "two_ply_closure": True,
            },
            "build": {"git_commit": "1" * 40, "git_dirty": False},
            "results": {"records": 1, "unsolved": 0},
            "binary_digest": "sha256:" + "2" * 64,
        }

    def test_teacher_dataset_uses_an_archived_model_and_changes_only_its_stages(self) -> None:
        repository, model_id = self._teacher_repository()
        digest = model_id.removeprefix("candidate-sha256-")
        taught = load_workflow_config(_write(
            repository, self._teacher_text(repository, model_id), "taught.toml"))
        plain = load_workflow_config(_write(repository, _managed_config(repository), "plain.toml"))
        dataset, plain_dataset = taught.dataset("main"), plain.dataset("main")
        self.assertEqual(dataset.teacher.model_id, model_id)
        net = repository / "models" / f"{model_id}.poe2net"
        self.assertEqual(dataset.teacher.net, net)
        self.assertEqual(dataset.teacher.net_name, "fixture/g000")
        self.assertIsNone(dataset.teacher.header)
        self.assertIsNone(plain_dataset.teacher)

        taught_stages, plain_stages = stage_definitions(taught), stage_definitions(plain)
        for stage in ("dataset:main:source", "dataset:main:labels"):
            self.assertEqual(taught_stages[stage]["teacher"], model_id)
            self.assertNotIn("teacher", plain_stages[stage])
            self.assertNotEqual(fingerprint(taught_stages[stage]), fingerprint(plain_stages[stage]))

        # Teachers are nets the one run-local data tool loads; no build is private to a model.
        self.assertEqual(data_binary(taught),
                         taught.output_directory / "build/debug/runner/poe2_minimax_data")
        for command in (source_chunk_command(taught, dataset, 0, 4),
                        source_assemble_command(taught, dataset),
                        label_command(taught, dataset, 0)):
            self.assertEqual(command[0], str(data_binary(taught)))
            self.assertEqual(command[command.index("--evaluator") + 1], "pattern-gain")
            self.assertEqual(command[command.index("--net") + 1], str(net))
        for command in (source_chunk_command(plain, plain_dataset, 0, 4),
                        source_assemble_command(plain, plain_dataset),
                        label_command(plain, plain_dataset, 0)):
            self.assertNotIn("--evaluator", command)

    @staticmethod
    def _generation_text(repository: Path, model_id: str, gate: str,
                         *, iteration: str = "g001", candidate: bool = True) -> str:
        validation = ("[iterations.candidate_validation]\npromotable = true\n"
                      if candidate else "")
        return textwrap.dedent(f"""
            schema_version = 1
            [run]
            name = "generation-test"
            output_dir = "{repository / 'run'}"
            build_preset = "debug"
            [models]
            g000 = "{model_id}"
            [datasets.g001]
            kind = "managed"
            teacher = "g000"
            [datasets.g001.source]
            corpus_id = "generation-test"
            seed = 7
            trajectories = 4
            [datasets.g001.labels]
            mode = "teacher"
            nodes = 100
            [[iterations]]
            name = "{iteration}"
            dataset = "g001"
            [iterations.training]
            type = "pattern"
            device = "cpu"
            suite = "frozen-pattern-gain"
        """) + validation + gate

    def test_gate_base_model_names_an_archived_model(self) -> None:
        repository, model_id = self._teacher_repository()
        config = load_workflow_config(_write(repository, self._generation_text(
            repository, model_id, '[iterations.engine_gate]\nbase_model = "g000"\n'),
            "generation.toml"))
        gate = config.iteration("g001").engine_gate
        self.assertIsNone(gate.base)
        self.assertEqual(gate.base_model, model_id)
        self.assertEqual(
            stage_definitions(config)["iteration:g001:engine-gate"]["settings"]["base_model"],
            model_id)
        for lines, message in (
                ('base = "000074-aa1f911c27b6"\nbase_model = "g000"\n', "exactly one"),
                ('games = 2\n', "exactly one"),
                ('base_model = "g009"\n', "must name an entry")):
            with self.subTest(lines=lines), self.assertRaisesRegex(WorkflowConfigError, message):
                load_workflow_config(_write(repository, self._generation_text(
                    repository, model_id, "[iterations.engine_gate]\n" + lines), "bad.toml"))
        with self.assertRaisesRegex(WorkflowConfigError, "validates no candidate"):
            load_workflow_config(_write(repository, self._generation_text(
                repository, model_id, "", iteration="g000", candidate=False), "shadow.toml"))

    def test_teacher_must_name_an_intact_archived_model(self) -> None:
        repository, model_id = self._teacher_repository()
        with self.assertRaisesRegex(WorkflowConfigError, "must name an entry"):
            load_workflow_config(_write(repository, self._teacher_text(
                repository, model_id, teacher="g001"), "unknown.toml"))
        with self.assertRaisesRegex(WorkflowConfigError, "archived model ID"):
            load_workflow_config(_write(repository, self._teacher_text(
                repository, "v0.2"), "malformed.toml"))
        (repository / "models" / f"{model_id}.json").unlink()
        with self.assertRaisesRegex(WorkflowConfigError, "is not archived"):
            load_workflow_config(_write(repository, self._teacher_text(
                repository, model_id), "unarchived.toml"))
        (repository / "models" / f"{model_id}.json").write_text("{}\n", encoding="utf-8")
        net = repository / "models" / f"{model_id}.poe2net"
        net.write_bytes(net_bytes("fixture/other"))
        with self.assertRaisesRegex(WorkflowConfigError, "does not match its model ID"):
            load_workflow_config(_write(repository, self._teacher_text(
                repository, model_id), "changed.toml"))
        net.write_bytes(b"not a net\n")
        with self.assertRaisesRegex(WorkflowConfigError, "unusable"):
            load_workflow_config(_write(repository, self._teacher_text(
                repository, model_id), "damaged.toml"))

    def test_label_and_feature_authentication_require_the_configured_teacher(self) -> None:
        repository, model_id = self._teacher_repository()
        digest = model_id.removeprefix("candidate-sha256-")
        taught = load_workflow_config(_write(
            repository, self._teacher_text(repository, model_id), "taught.toml"))
        plain = load_workflow_config(_write(repository, _managed_config(repository), "plain.toml"))
        dataset = taught.dataset("main")
        shard = label_shard_path(dataset, 0)
        shard.mkdir(parents=True)
        manifest_path = shard / "manifest.json"

        manifest_path.write_text(json.dumps(self._label_manifest(
            dataset, evaluator="pattern-gain", model=digest)) + "\n", encoding="utf-8")
        self.assertEqual(authenticate_label_shard(dataset, 0)["teacher"], model_id)
        # Data a file-loaded net made records the file's hash under its own key.
        net_manifest = self._label_manifest(dataset, evaluator="pattern-gain")
        net_manifest["search"]["net_sha256"] = dataset.teacher.net_sha256
        manifest_path.write_text(json.dumps(net_manifest) + "\n", encoding="utf-8")
        self.assertEqual(authenticate_label_shard(dataset, 0)["teacher"], model_id)
        net_manifest["search"]["model_header_sha256"] = digest
        manifest_path.write_text(json.dumps(net_manifest) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "evaluator is unsupported"):
            authenticate_label_shard(dataset, 0)
        manifest_path.write_text(json.dumps(self._label_manifest(
            dataset, evaluator="pattern-gain", model=digest)) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "teacher differs"):
            authenticate_label_shard(plain.dataset("main"), 0)
        for manifest in (self._label_manifest(dataset),
                         self._label_manifest(dataset, evaluator="pattern-gain", model="f" * 64)):
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "teacher differs"):
                authenticate_label_shard(dataset, 0)

        dataset.feature_directory.parent.mkdir(parents=True, exist_ok=True)
        write_feature_fixture(dataset.feature_directory, teacher_sha256=dataset.teacher.net_sha256,
                              teacher_key="net_sha256")
        self.assertEqual(authenticate_features(dataset)["teacher"], model_id)
        with self.assertRaisesRegex(ValueError, "teacher differs"):
            authenticate_features(plain.dataset("main"))

    def test_teacher_data_binary_is_the_shared_build_checked_against_its_net(self) -> None:
        repository, model_id = self._teacher_repository()
        config = load_workflow_config(_write(
            repository, self._teacher_text(repository, model_id), "taught.toml"))
        dataset = config.dataset("main")
        commit = "3" * 40
        binary = data_binary(config)
        commands: list[list[str]] = []

        def build(contents: bytes):
            def run(_stage: str, operation: str, command, **_options) -> None:
                commands.append(list(command))
                if operation == "build-data":
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    binary.write_bytes(contents)
            return run

        workflow = WorkflowRunner(config, RunState(config), commit)
        with mock.patch.object(workflow, "_command", side_effect=build(
                b"fixture\0" + commit.encode())):
            workflow._ensure_data_binary("dataset:main:source", 1, dataset)
        self.assertFalse(any("POE2_MINIMAX_MODEL_HEADER" in item for item in commands[0]))
        self.assertIn(str(data_build_directory(config)), commands[0])
        self.assertEqual(workflow._data_commits[binary], commit)

        binary.unlink()
        dataset.teacher.net.write_bytes(b"changed\n")
        other = WorkflowRunner(config, RunState(config), commit)
        with mock.patch.object(other, "_command",
                               side_effect=build(b"fixture\0" + commit.encode())):
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                other._ensure_data_binary("dataset:main:source", 1, dataset)

    def test_source_chunks_have_a_fixed_size_and_a_short_last_chunk(self) -> None:
        config = load_workflow_config(_write(self.root, _managed_config(
            self.root, trajectories=4500)))
        self.assertEqual(source_chunk_ranges(config.dataset("main")),
                         [(0, 2000), (2000, 4000), (4000, 4500)])
        small = load_workflow_config(_write(self.root, _managed_config(self.root), "small.toml"))
        self.assertEqual(source_chunk_ranges(small.dataset("main")), [(0, 4)])

    def test_source_resume_generates_only_missing_chunks_with_their_build(self) -> None:
        config = load_workflow_config(_write(self.root, _managed_config(
            self.root, trajectories=4500)))
        dataset = config.dataset("main")
        state = RunState(config)
        state.accept_config()
        finished = source_chunk_path(dataset, 2000, 4000)
        finished.mkdir(parents=True)
        (finished / "COMPLETE").write_text("fixture\n", encoding="utf-8")
        (finished / "chunk.jsonl").write_text(json.dumps({
            "type": "poe2-position-source-chunk", "git_commit": "4" * 40}) + "\n",
            encoding="utf-8")
        workflow = WorkflowRunner(config, state, "5" * 40)
        operations: list[str] = []

        def record(_stage: str, operation: str, *_arguments, **_options) -> None:
            operations.append(operation)
            if operation == "assemble":
                raise KeyboardInterrupt

        with mock.patch.object(workflow, "_ensure_data_binary") as binary, \
                mock.patch.object(workflow, "_command", side_effect=record):
            with self.assertRaises(KeyboardInterrupt):
                workflow._prepare_source(dataset)
        self.assertEqual(binary.call_args.kwargs["required_commit"], "4" * 40)
        self.assertEqual(operations, [
            "create-chunk-000000000", "create-chunk-000004000", "assemble"])
        relative = [path.relative_to(config.output_directory).as_posix() for path in (
            dataset.source_directory, source_chunk_path(dataset, 0, 2000),
            source_chunk_path(dataset, 4000, 4500))]
        self.assertEqual([event["artifact"] for event in state.events
                          if event["type"] == "artifact_started"], relative)
        self.assertEqual(state.unfinished_artifacts(), relative[:1])

        (finished.parent / "stray").mkdir()
        with self.assertRaisesRegex(WorkflowError, "unexpected output"):
            WorkflowRunner(config, state, "5" * 40)._prepare_source(dataset)
        (finished.parent / "stray").rmdir()
        unfinished = source_chunk_path(dataset, 0, 2000)
        unfinished.mkdir()
        (unfinished / "INCOMPLETE").write_text("fixture\n", encoding="utf-8")
        with self.assertRaisesRegex(WorkflowError, "source chunk path exists"):
            WorkflowRunner(config, state, "5" * 40)._prepare_source(dataset)

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
                "dataset:main:labels", 2, config.dataset("main"),
                required_commit=original_commit)

        command.assert_not_called()
        self.assertEqual(workflow._data_commits[binary], original_commit)

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
        # Private builds here are mocked; test_workflow_recovery covers the Catch2 reuse.
        self.enterContext(mock.patch("poe2_training.workflow.catch2_pin", return_value=None))
        self.temporary = repository_temporary_directory()
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

    def test_paused_stage_is_labeled_paused_not_failed(self) -> None:
        output = io.StringIO()
        progress = WorkflowProgress(self.config, RunState(self.config), stream=output)
        progress.stage_started("dataset:main:source", "create")
        progress.stage_failed("dataset:main:source", KeyboardInterrupt())
        progress.stage_started("dataset:main:labels", "create")
        progress.stage_failed("dataset:main:labels", WorkflowError("broken"))
        progress.close(success=False)
        self.assertIn("dataset main · source  paused after", output.getvalue())
        self.assertIn("dataset main · labels  failed after", output.getvalue())

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

    def test_interruption_discards_only_unfinished_artifacts_this_run_started(self) -> None:
        state = RunState(self.config)
        state.accept_config()
        runner = WorkflowRunner(self.config, state, "1" * 40)
        interrupted = self.config.iteration("one").training_directory

        def interrupt(_attempt: int) -> None:
            reserve_report_directory(interrupted, "fixture-report")
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            runner._run_simple("iteration:one:training", interrupted, "training report",
                               {}, interrupt, lambda _attempt: {})
        finished = self.config.output_directory / "finished"
        reserve_report_directory(finished, "fixture-report")
        complete_json_report(finished, "fixture-report", {"schema": "fixture-report",
                                                          "schema_version": 1})
        runner._start_artifact(finished)
        foreign = self.config.dataset("main").source_directory
        foreign.mkdir(parents=True)
        (foreign / "INCOMPLETE").write_text("fixture\n", encoding="utf-8")

        self.assertEqual(runner.discard_interrupted_artifacts(), [interrupted])
        self.assertFalse(interrupted.exists())
        self.assertTrue((finished / "COMPLETE").is_file())
        self.assertTrue((foreign / "INCOMPLETE").is_file())
        removals = [event["path"] for event in RunState(self.config).events
                    if event["type"] == "interrupted_artifact_removed"]
        self.assertEqual(removals, [str(interrupted)])

        with self.assertRaisesRegex(WorkflowError, "outside the run"):
            runner._start_artifact(self.root / "outside")

    def test_interrupted_label_shard_is_discarded(self) -> None:
        state = RunState(self.config)
        state.accept_config()
        runner = WorkflowRunner(self.config, state, "1" * 40)
        dataset = self.config.dataset("main")
        shard = label_shard_path(dataset, 1)

        def interrupted_command(*_arguments, **_options) -> None:
            shard.mkdir(parents=True)
            (shard / "INCOMPLETE").write_text("poe2-minimax-labels\n", encoding="utf-8")
            raise KeyboardInterrupt

        with mock.patch.object(runner, "_command", side_effect=interrupted_command):
            with self.assertRaises(KeyboardInterrupt):
                runner._create_label_shard("dataset:main:labels", dataset, 1, 1)
        self.assertEqual(runner.discard_interrupted_artifacts(), [shard])
        self.assertFalse(shard.exists())
        self.assertTrue(shard.parent.is_dir())

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
    # Fixtures record paths absolutely, as runs did before paths were relative to the repository.
    relative_records = False

    def _named(self, path: Path) -> str:
        return self.config.recorded_path(path) if self.relative_records else str(path)

    def setUp(self) -> None:
        # Private builds here are mocked; test_workflow_recovery covers the Catch2 reuse.
        self.enterContext(mock.patch("poe2_training.workflow.catch2_pin", return_value=None))
        self.temporary = repository_temporary_directory()
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
                sequential_alpha: float = 0.05, gate_extra: str = ""):
        base = base or str(self.root / "base-engine")
        gate = textwrap.dedent(f"""
            [iterations.engine_gate]
            base = "{base}"
            new_engine = "{new_engine}"
            new_engine_args = "{new_engine_args}"
            games = 2
            sequential_alpha = {sequential_alpha!r}
            {gate_extra}
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

    def _training_report(self) -> None:
        training = self.iteration.training_directory
        reserve_report_directory(training, "poe2-minimax-pattern-experiment")
        complete_json_report(training, "poe2-minimax-pattern-experiment", {
            "schema": "poe2-minimax-pattern-experiment", "schema_version": 1,
            "training": {"method": "lookup"}, "input": {"records": 12},
            "selection": {"best_model": "fixture"},
            "models": [{"name": "fixture", "trace": [{"epoch": 1, "validation_mae": 1.0}]}],
        })

    def _candidate(self, features: Path | None = None, *, legacy: bool = False,
                   teacher: str = "two-ply-closure", parent: str | None = None) -> None:
        """Write a validated candidate: a net file, or a header as before nets were files."""
        candidate = self.iteration.candidate_validation
        assert candidate is not None
        self._training_report()
        training = self.iteration.training_directory
        root = self.iteration.candidate_directory
        reserve_report_directory(root, CANDIDATE_SCHEMA)
        log = self.config.output_directory / "logs" / f"{self.iteration.name}-fixture.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("candidate fixture\n", encoding="utf-8")
        feature_digest = sha256_file((features or self.root / "features") / "features.bin")
        artifacts: dict[str, dict[str, str]] = {}
        provenance: dict[str, object] = {}
        if legacy:
            # Its net is converted when it is archived, dated by the training stage's completion.
            RunState(self.config).append(
                "stage_completed", stage=f"iteration:{self.iteration.name}:training")
            for name, relative in {
                "model_header": "frozen_pattern_gain_model.hpp",
                "inference_binary": "build/engines/minimax/poe2_minimax_infer",
                "engine_binary": "build/engines/minimax/poe2_minimax",
                "runner_binary": "build/runner/poe2_runner",
            }.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if name == "model_header":
                    header = (REPOSITORY / "models" / f"{V02}.hpp").read_text(encoding="utf-8")
                    path.write_text(f"{header}// {self.iteration.name} fixture\n",
                                    encoding="utf-8")
                else:
                    path.write_text(f"{self.iteration.name} {name}", encoding="utf-8")
                artifacts[name] = {"path": relative, "sha256": sha256_file(path)}
        else:
            data = net_bytes(
                candidate_net_name(self.config, self.iteration), teacher=teacher,
                **({} if parent is None else {"parent": parent}), created="2026-08-19",
                training_report_sha256=sha256_file(training / "report.json"),
                feature_binary_sha256=feature_digest)
            candidate_net_path(self.iteration).write_bytes(data)
            net = decode_net(data)
            artifacts["net"] = {"path": CANDIDATE_NET, "sha256": net.sha256, "id": net.net_id,
                                "name": net.name, "design": net.design}
            provenance["tools"] = {
                name: {"path": f"run/build/release/engines/minimax/{binary}",
                       "sha256": hashlib.sha256(binary.encode()).hexdigest()}
                for name, binary in (("inference_binary", "poe2_minimax_infer"),
                                     ("engine_binary", "poe2_minimax"))}
        complete_json_report(root, CANDIDATE_SCHEMA, {
            "schema": CANDIDATE_SCHEMA, "schema_version": 1 if legacy else CANDIDATE_SCHEMA_VERSION,
            **provenance,
            "promotable": candidate.promotable,
            "git_commit": "1" * 40,
            "settings": {
                "promotable": candidate.promotable, "samples": candidate.samples,
                "symmetry_samples": candidate.symmetry_samples,
                "benchmark_iterations": candidate.benchmark_iterations,
                "opening_book": self._named(candidate.opening_book),
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

    def _isolated_repository(self, name: str, *, ledger_header: list[str] | None = None,
                             tracked_header: bytes | None = None):
        repository = self.root / name
        repository.mkdir()
        (repository / ".gitignore").write_text("eval/results.csv.lock\n", encoding="utf-8")
        if ledger_header is not None:
            (repository / "eval").mkdir()
            with (repository / "eval" / "results.csv").open(
                    "w", newline="", encoding="utf-8") as destination:
                csv.writer(destination, lineterminator="\n").writerow(ledger_header)
        if tracked_header is not None:
            path = repository / "engines/minimax/src/frozen_pattern_gain_model.hpp"
            path.parent.mkdir(parents=True)
            path.write_bytes(tracked_header)
        config_path = repository / "promotion.toml"
        config_path.write_bytes(self.config.contents)
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        self._commit_all(repository, "fixture")
        return repository, replace(self.config, repository=repository, path=config_path)

    @staticmethod
    def _commit_all(repository: Path, message: str) -> None:
        subprocess.run(["git", "add", "-A"], cwd=repository, check=True)
        subprocess.run([
            "git", "-c", "user.name=Workflow Test", "-c", "user.email=test@example.invalid",
            "commit", "-qm", message], cwd=repository, check=True)

    def _write_match_run(self, root: Path, settings, *, kind: str, new_id: str, base_id: str,
                         new_engine_path: Path, base_engine_path: str, run_id: str,
                         decision: str, ledger_overrides: dict[str, str] | None = None,
                         nets: dict[str, Path | None] | None = None):
        """Write one evaluator run under root/runs/one, as the native runner records it.

        nets maps each side to the net file it loaded; without it, the run is one recorded
        before engines reported their nets, with the ledger columns of that time.
        """
        master = REPOSITORY / "eval" / "results.csv"
        with master.open(newline="", encoding="utf-8") as source:
            header = next(csv.reader(source))
        if nets is None:
            header = header[:header.index("games_discarded") + 1]
        row = [""] * len(header)
        row[header.index("run_id")] = run_id
        run = root / "runs" / "one"
        run.mkdir(parents=True)
        identities: dict[str, object] = {}
        for side, net in (nets or {}).items():
            decoded = None if net is None else decode_net(net.read_bytes())
            identities.update({
                f"{side}_net": None if net is None else self._named(net),
                f"{side}_engine_version": "0.3.0",
                f"{side}_net_name": "" if decoded is None else decoded.name,
                f"{side}_net_id": "" if decoded is None else decoded.net_id,
                f"{side}_net_sha256": "" if decoded is None else decoded.sha256,
            })
        manifest = {
            **identities,
            "run_id": run_id,
            "created_at_utc": "20260819T000000Z", "valid": True,
            "kind": kind, "new_id": new_id,
            "new_engine": settings.new_engine,
            "new_engine_path": self._named(new_engine_path.resolve()),
            "new_engine_args": settings.new_engine_args, "base_engine": settings.base_engine,
            "base_engine_args": settings.base_engine_args,
            "base_engine_path": base_engine_path,
            "base_id": base_id,
            "opening_book": self._named(settings.opening_book),
            "opening_book_digest": "fnv1a64:fixture", "opening_count": 1,
            "opening_seed": 123, "opening_seed_source": "derived",
            "opening_sampling": "deterministic_shuffle_without_replacement",
            "shuffle_openings": True, "unique_openings": 1,
            "games": settings.games, "games_played": settings.games,
            "workers_requested": settings.workers, "workers_used": 1,
            "games_discarded": 0, "invalid_reason": None,
            "timeout_ms": settings.timeout_ms, "go_depth": None,
            "go_movetime_ms": settings.go_movetime_ms, "go_nodes": None,
            "analysis_version": 2, "statistical_unit": "opening_pair",
            "confidence_method": "paired_betting_confidence_sequence",
            "sequential_test_method": "paired_normalized_elo_gsprt",
            "sequential_stop": settings.sequential_stop,
            "sequential_bound_unit": "normalized_elo",
            "sequential_null": settings.sequential_null,
            "sequential_alt": settings.sequential_alt,
            "sequential_alpha": settings.sequential_alpha,
            "sequential_beta": settings.sequential_beta,
            "sequential_llr": 1.0, "sequential_lower_bound": -2.0,
            "sequential_upper_bound": 2.0, "normalized_elo": 20.0,
            "betting_log_evidence_above_even": 3.0,
            "betting_log_evidence_below_even": -3.0,
            "sequential_decision": decision,
        }
        summary = {
            "analysis_version": 2, "valid": True, "invalid_reason": None,
            "games_requested": settings.games, "games_played": settings.games,
            "workers_requested": settings.workers, "workers_used": 1,
            "games_discarded": 0, "unique_openings": 1,
            "engine_one_wins": settings.games, "engine_two_wins": 0, "no_winner": 0,
            "statistical_unit": "opening_pair", "statistical_samples": 1,
            "statistical_games": settings.games,
            "statistical_score_rate_counts": {
                "0": 0, "0.25": 0, "0.5": 0, "0.75": 0, "1": 1,
            },
        }
        ledger_values = {
            "created_at_utc": manifest["created_at_utc"],
            "kind": kind, "new_id": new_id,
            "new_engine": settings.new_engine,
            "new_engine_args": settings.new_engine_args, "base_id": manifest["base_id"],
            "base_engine": settings.base_engine, "base_engine_args": settings.base_engine_args,
            "games_requested": str(settings.games), "games_played": str(settings.games),
            "engine_one_wins": str(settings.games), "engine_two_wins": "0", "no_winner": "0",
            "engine_one_score_pct": "100.000", "confidence_low_pct": "50.000",
            "confidence_high_pct": "100.000",
            "sequential_decision": manifest["sequential_decision"],
            "sequential_null": str(settings.sequential_null),
            "sequential_alt": str(settings.sequential_alt),
            "opening_book": self._named(settings.opening_book),
            "opening_book_digest": manifest["opening_book_digest"],
            "opening_count": "1", "go_movetime_ms": str(settings.go_movetime_ms),
            "timeout_ms": str(settings.timeout_ms), "run_dir": self._named(run.resolve()),
            "analysis_version": "2", "statistical_unit": "opening_pair",
            "statistical_samples": "1", "statistical_games": str(settings.games),
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
            "workers_requested": str(settings.workers), "workers_used": "1",
            "games_discarded": "0",
            **{name: value for name, value in identities.items()
               if name in header and isinstance(value, str)},
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
        return header, row, run

    def _gate(self, *, manifest_base_path: str | None = None,
              run_id: str = "workflow-promotion-test-run",
              ledger_overrides: dict[str, str] | None = None,
              decision: str = "accept_alt") -> tuple[list[str], list[str]]:
        """Write a gate as _run_gate does: one engine build playing the candidate's net."""
        root = self.iteration.gate_directory
        candidate = authenticate_candidate(self.config, self.iteration)
        reserve_report_directory(root, GATE_SCHEMA)
        model_id = candidate_build_id(candidate)
        build_id = engine_build_id(1, "2" * 40)
        build = root / "builds" / build_id / self.config.build_preset
        new_engine_binary = build / "engines/minimax/poe2_minimax"
        runner_binary = build / "runner/poe2_runner"
        for path in (new_engine_binary, runner_binary):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(path.name, encoding="utf-8")
        gate = self.iteration.engine_gate
        assert gate is not None
        new_net = candidate_net_path(self.iteration)
        base_model_id = gate_base_model_id(self.config, self.iteration)
        if base_model_id is None:
            base_engine = resolve_eval_binary(self.config, gate.base or "", gate.base_engine)
            base_net = None
        else:
            base_engine = new_engine_binary
            base_net = model_net(next(model for model in resolve_trained_models(
                self.config).models if model.model_id == base_model_id))
        header, row, run = self._write_match_run(
            root, gate, kind=GATE_KIND, new_id=build_id,
            base_id="fixture-base" if base_net is None else build_id,
            new_engine_path=new_engine_binary,
            base_engine_path=manifest_base_path or self._named(base_engine), run_id=run_id,
            decision=decision, ledger_overrides=ledger_overrides,
            nets={"new": new_net, "base": base_net})
        complete_json_report(root, GATE_SCHEMA, {
            "schema": GATE_SCHEMA, "schema_version": GATE_SCHEMA_VERSION,
            "git_commit": "2" * 40,
            "settings": {
                **recorded_gate_settings(gate),
                "opening_book": self._named(gate.opening_book),
            },
            "opening_book_sha256": sha256_file(gate.opening_book),
            "run_id": row[header.index("run_id")], "run_directory": "runs/one",
            "candidate_report_sha256": candidate["report_sha256"],
            "candidate_build_id": model_id, "engine_build_id": build_id,
            "build_directory": build.relative_to(root).as_posix(),
            "new_net": {"path": self._named(new_net), "sha256": sha256_file(new_net),
                        "model_id": model_id},
            "base_net": None if base_net is None else {
                "path": self._named(base_net), "sha256": sha256_file(base_net),
                "model_id": base_model_id},
            "base_engine_artifact": {
                "path": self._named(base_engine), "sha256": sha256_file(base_engine)},
            "build_artifacts": {
                name: {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
                for name, path in (("new_engine_binary", new_engine_binary),
                                   ("runner_binary", runner_binary))},
            "artifacts": {name: sha256_file(run / name) for name in
                          ("manifest.json", "summary.json", "games.csv", "ledger-row.csv")},
        })
        return header, row

    def _header_gate(self, *, run_id: str = "workflow-promotion-test-run",
                     decision: str = "accept_alt") -> tuple[list[str], list[str]]:
        """Write a gate as it was played before nets were files: one private build per model."""
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
        gate = self.iteration.engine_gate
        assert gate is not None
        base_engine = gate_base_binary(self.config, self.iteration)
        base_model_id = gate_base_model_id(self.config, self.iteration)
        if base_model_id is not None:
            base_engine.parent.mkdir(parents=True, exist_ok=True)
            base_engine.write_text("base model engine\n", encoding="utf-8")
        header, row, run = self._write_match_run(
            root, gate, kind=GATE_KIND, new_id=build_id,
            base_id=base_model_id or "fixture-base", new_engine_path=new_engine_binary,
            base_engine_path=self._named(base_engine), run_id=run_id, decision=decision)
        complete_json_report(root, GATE_SCHEMA, {
            "schema": GATE_SCHEMA, "schema_version": HEADER_GATE_SCHEMA_VERSION,
            "git_commit": "2" * 40,
            "settings": {
                **recorded_gate_settings(gate),
                "opening_book": self._named(gate.opening_book),
            },
            "opening_book_sha256": sha256_file(gate.opening_book),
            "run_id": row[header.index("run_id")], "run_directory": "runs/one",
            "candidate_report_sha256": sha256_file(
                self.iteration.candidate_directory / "report.json"),
            "candidate_build_id": build_id,
            "build_directory": build.relative_to(root).as_posix(),
            "model_header_sha256": header_digest,
            "base_engine_artifact": {
                "path": self._named(base_engine),
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
        self._candidate(legacy=True)
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

    def test_candidate_net_must_be_the_one_its_report_names(self) -> None:
        self._candidate()
        net = candidate_net_path(self.iteration)
        original = net.read_bytes()
        self.assertEqual(authenticate_candidate(self.config, self.iteration)["artifacts"]["net"],
                         {"path": CANDIDATE_NET, "sha256": sha256_file(net),
                          "id": sha256_file(net)[:12], "name": "promotion-test/candidate",
                          "design": "pattern-gain-v1"})
        for reason, data in {"differs from its report": net_bytes("promotion-test/candidate"),
                             "unusable": original[:100]}.items():
            with self.subTest(reason=reason):
                net.write_bytes(data)
                with self.assertRaisesRegex(AdapterError, reason):
                    authenticate_candidate(self.config, self.iteration)
        net.write_bytes(original)
        authenticate_candidate(self.config, self.iteration)

    def test_a_candidate_and_gate_from_before_nets_still_authenticate_and_promote(self) -> None:
        self._candidate(legacy=True)
        ledger_header, _ = self._header_gate()
        header = self.iteration.candidate_directory / "frozen_pattern_gain_model.hpp"
        candidate = authenticate_candidate(self.config, self.iteration)
        model_id = f"candidate-sha256-{sha256_file(header)}"
        self.assertEqual(candidate_build_id(candidate), model_id)
        gate = authenticate_gate(self.config, self.iteration)
        self.assertEqual(gate["new_id"], model_id)
        repository, isolated = self._isolated_repository(
            "legacy-repository", ledger_header=ledger_header)
        with (repository / "eval/results.csv").open("a", newline="", encoding="utf-8") as ledger:
            csv.writer(ledger, lineterminator="\n").writerow(["earlier"] * len(ledger_header))
        self._commit_all(repository, "record an earlier match")
        applied = promote_gate(isolated, "candidate", apply=True)
        self.assertTrue(applied["applied"])
        self.assertEqual((repository / "models" / f"{model_id}.hpp").read_bytes(),
                         header.read_bytes())
        net = decode_net((repository / "models" / f"{model_id}.poe2net").read_bytes())
        self.assertEqual((net.name, net.net_id, net.metadata["header_sha256"]),
                         ("promotion-test/candidate", model_id[17:29], model_id[17:]))
        with (repository / "eval/results.csv").open(newline="", encoding="utf-8") as source:
            rows = list(csv.reader(source))
        net_columns = rows[0][len(ledger_header):]
        self.assertEqual(net_columns[:2], ["new_engine_version", "new_net_name"])
        self.assertEqual(len(net_columns), 8)
        self.assertEqual([row[len(ledger_header):] for row in rows[1:]], [[""] * 8] * 2)

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
            net_digest = sha256_file(candidate_net_path(self.iteration))
            self.assertEqual(authenticated["candidate_build_id"], f"candidate-sha256-{net_digest}")
            self.assertEqual(authenticated["new_net_sha256"], net_digest)
            self.assertIsNone(authenticated["base_net_sha256"])
            manifest = json.loads(Path(authenticated["run_directory"])
                                  .joinpath("manifest.json").read_bytes())
            self.assertEqual(manifest["new_id"], authenticated["engine_build_id"])
            self.assertEqual(manifest["new_net_sha256"], net_digest)

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

    def _gate_commands(self) -> dict[str, list[str]]:
        """Run the gate stage until its match and return the commands it issued by operation."""
        candidate_stats = authenticate_candidate(self.config, self.iteration)
        workflow = WorkflowRunner(self.config, RunState(self.config), "1" * 40)
        commands: dict[str, list[str]] = {}

        def stop_at_the_match(_stage: str, operation: str, command, **_options) -> None:
            commands[operation] = list(command)
            if operation == "evaluate":
                raise WorkflowError("fixture stops the gate")

        with mock.patch.object(workflow, "_command", side_effect=stop_at_the_match), \
                mock.patch.object(workflow, "_engine_build_id",
                                  return_value=engine_build_id(7, "1" * 40)):
            with self.assertRaisesRegex(WorkflowError, "fixture stops the gate"):
                workflow._run_gate(self.iteration, candidate_stats)
        return commands

    def test_engine_gate_builds_its_commit_once_and_plays_the_candidate_net(self) -> None:
        self._candidate()
        tracked = REPOSITORY / "engines/minimax/src/frozen_pattern_gain_model.hpp"
        tracked_digest = sha256_file(tracked)
        commands = self._gate_commands()
        build = (self.iteration.gate_directory / "builds" / "000007-111111111111" /
                 self.config.build_preset)
        self.assertEqual(commands["configure"][:4],
                         ["cmake", "--preset", self.config.build_preset, "-B"])
        self.assertEqual(commands["configure"][4], str(build))
        self.assertFalse(any(item.startswith("-DPOE2_MINIMAX_MODEL_HEADER")
                             for item in commands["configure"]))
        evaluate = commands["evaluate"]
        self.assertEqual(evaluate[evaluate.index("--new-net") + 1],
                         candidate_net_path(self.iteration).relative_to(REPOSITORY).as_posix())
        self.assertEqual(evaluate[evaluate.index("--base") + 1], str(self.root / "base-engine"))
        self.assertNotIn("--base-net", evaluate)
        self.assertEqual(sha256_file(tracked), tracked_digest)

    def test_engine_gate_rejects_a_changed_candidate_net(self) -> None:
        self._candidate()
        candidate_stats = authenticate_candidate(self.config, self.iteration)
        candidate_net_path(self.iteration).write_bytes(net_bytes("promotion-test/candidate"))
        workflow = WorkflowRunner(self.config, RunState(self.config), "1" * 40)
        with self.assertRaisesRegex(WorkflowError, "missing or changed"):
            workflow._run_gate(self.iteration, candidate_stats)

    def test_engine_gate_refuses_a_candidate_validated_before_nets_were_files(self) -> None:
        self._candidate(legacy=True)
        candidate_stats = authenticate_candidate(self.config, self.iteration)
        workflow = WorkflowRunner(self.config, RunState(self.config), "1" * 40)
        with self.assertRaisesRegex(WorkflowError, "validated before nets were files"):
            workflow._run_gate(self.iteration, candidate_stats)

    def _with_base_model(self, data: bytes | None = None) -> ModelReference:
        net = self.root / "base-model.poe2net"
        net.write_bytes(data or net_bytes("promotion-test/g000", teacher="two-ply-closure"))
        decoded = decode_net(net.read_bytes())
        model = ModelReference(
            name="g000", model_id=f"candidate-sha256-{decoded.identity_sha256}", header=None,
            net=net, net_sha256=decoded.sha256, net_name=decoded.name)
        gate = self.iteration.engine_gate
        assert gate is not None
        self.iteration = replace(
            self.iteration, engine_gate=replace(gate, base=None, base_model=model.model_id))
        self.config = replace(self.config, iterations=(self.iteration,), models=(model,))
        return model

    def _moved(self):
        """Copy the repository files this test uses to another folder; return the config there."""
        destination = repository_temporary_directory()
        self.addCleanup(destination.cleanup)
        return moved_repository(self.config, Path(destination.name))

    def _saved_gate_row(self, config) -> dict[str, str]:
        iteration = config.iteration("candidate")
        gate = authenticate_gate(config, iteration)
        return _verified_ledger_row(config, gate, iteration.engine_gate, kind=GATE_KIND,
                                    new_id=gate["new_id"], label="gate").values

    def test_relative_records_authenticate_after_the_repository_moves(self) -> None:
        self.relative_records = True
        self._candidate()
        self._with_base_model()
        self._gate()
        runs = (self.iteration.gate_directory / "runs/one").relative_to(REPOSITORY).as_posix()
        moved = self._moved()
        for config in (self.config, moved):
            values = self._saved_gate_row(config)
            self.assertEqual((values["run_dir"], values["opening_book"]),
                             (runs, "eval/openings/holdout.txt"))
        self.assertEqual(stage_fingerprints(moved), stage_fingerprints(self.config))

    def test_absolute_records_authenticate_only_where_they_were_written(self) -> None:
        self._candidate()
        self._with_base_model()
        self._gate()
        values = self._saved_gate_row(self.config)
        self.assertEqual(values["run_dir"], (self.iteration.gate_directory / "runs/one")
                         .relative_to(REPOSITORY).as_posix())
        moved = self._moved()
        with self.assertRaisesRegex(AdapterError, "candidate validation settings differ"):
            authenticate_candidate(moved, moved.iteration("candidate"))

    def test_engine_gate_plays_its_base_model_net_on_the_same_build(self) -> None:
        self._candidate()
        model = self._with_base_model()
        commands = self._gate_commands()
        build = (self.iteration.gate_directory / "builds" / "000007-111111111111" /
                 self.config.build_preset)
        self.assertEqual(commands["configure"][4], str(build))
        self.assertIn("ctest", commands["test"])
        evaluate = commands["evaluate"]
        self.assertEqual(evaluate[evaluate.index("--base") + 1],
                         build.relative_to(REPOSITORY).as_posix())
        self.assertEqual(evaluate[evaluate.index("--new-build") + 1],
                         build.relative_to(REPOSITORY).as_posix())
        self.assertEqual(evaluate[evaluate.index("--base-net") + 1],
                         model.net.relative_to(REPOSITORY).as_posix())

    def test_engine_gate_rejects_a_base_model_equal_to_its_candidate(self) -> None:
        self._candidate()
        candidate_stats = authenticate_candidate(self.config, self.iteration)
        self._with_base_model(candidate_net_path(self.iteration).read_bytes())
        workflow = WorkflowRunner(self.config, RunState(self.config), "1" * 40)
        with self.assertRaisesRegex(WorkflowError, "against itself"):
            workflow._run_gate(self.iteration, candidate_stats)

    def test_gate_against_a_base_model_authenticates_and_indexes_its_short_id(self) -> None:
        self._candidate()
        model = self._with_base_model()
        ledger_header, _ = self._gate()
        authenticated = authenticate_gate(self.config, self.iteration)
        self.assertEqual(authenticated["manifest"]["base_id"], authenticated["engine_build_id"])
        self.assertEqual(authenticated["base_net_sha256"], model.net_sha256)
        self.assertEqual(authenticated["manifest"]["base_net_name"], "promotion-test/g000")
        _, isolated = self._isolated_repository("base-model-repository", ledger_header=ledger_header)
        preview = promote_gate(isolated, "candidate")
        self.assertEqual(preview["index_row"]["base"], "promotion-test/g000")
        self.assertEqual(preview["index_row"]["base_model"], model.sha256[:12])

    def test_model_lineage_requires_entries_to_match_their_iteration(self) -> None:
        named = ModelReference(name="candidate", model_id="candidate-sha256-" + "a" * 64,
                               header=self.root / "unused.hpp")
        config = replace(self.config, models=(named,))
        check_model_lineage(config)
        self._candidate()
        with self.assertRaisesRegex(WorkflowError, "but iteration candidate produced"):
            check_model_lineage(config)
        produced = candidate_build_id(authenticate_candidate(self.config, self.iteration))
        check_model_lineage(replace(self.config, models=(replace(named, model_id=produced),)))

    def test_engine_gate_records_a_decision_that_is_not_a_win(self) -> None:
        self._candidate()
        self._gate(decision="accept_null")
        authenticated = authenticate_gate(self.config, self.iteration)
        self.assertEqual(authenticated["sequential_decision"], "accept_null")
        command = engine_gate_command(
            self.config, self.iteration, self.root / "build", self.root / "runner",
            new_net=candidate_net_path(self.iteration), base_model=None)
        self.assertNotIn("--require-accept-alt", command)

    def test_engine_gate_rejects_the_removed_require_accept_alt_field(self) -> None:
        with self.assertRaisesRegex(WorkflowConfigError, "require_accept_alt"):
            self._config(gate_extra="require_accept_alt = false")

    def test_handoff_requires_an_engine_gate(self) -> None:
        self.config = self._config(include_gate=False)
        self.iteration = self.config.iteration("candidate")
        self._candidate(legacy=True)
        with self.assertRaisesRegex(WorkflowError, "no engine gate"):
            handoff_candidate(self.config, "candidate", apply=True)

    def test_handoff_rejects_a_gate_that_was_not_won(self) -> None:
        self._candidate(legacy=True)
        self._header_gate(decision="accept_null")
        preview = handoff_candidate(self.config, "candidate")
        self.assertEqual(preview["gate_decision"], "accept_null")
        with self.assertRaisesRegex(WorkflowError, "did not win"):
            handoff_candidate(self.config, "candidate", apply=True)

    def test_handoff_installs_only_a_recorded_gate_winner(self) -> None:
        self._candidate(legacy=True)
        ledger_header, _ = self._header_gate()
        repository, isolated = self._isolated_repository(
            "handoff-repository", ledger_header=ledger_header,
            tracked_header=b"tracked header\n")
        candidate_header = self.iteration.candidate_directory / "frozen_pattern_gain_model.hpp"
        tracked = repository / "engines/minimax/src/frozen_pattern_gain_model.hpp"

        with self.assertRaisesRegex(WorkflowError, "record the gate first"):
            handoff_candidate(isolated, "candidate", apply=True)
        promote_gate(isolated, "candidate", apply=True)
        with self.assertRaisesRegex(WorkflowError, "clean committed worktree"):
            handoff_candidate(isolated, "candidate", apply=True)
        self._commit_all(repository, "record gate")

        preview = handoff_candidate(isolated, "candidate")
        self.assertTrue(preview["different"])
        self.assertTrue(preview["archived"])
        result = handoff_candidate(isolated, "candidate", apply=True)
        self.assertTrue(result["applied"])
        self.assertEqual(tracked.read_bytes(), candidate_header.read_bytes())
        status = subprocess.run(
            ["git", "status", "--short"], cwd=repository,
            check=True, capture_output=True, text=True).stdout.splitlines()
        self.assertEqual(
            status, [" M engines/minimax/src/frozen_pattern_gain_model.hpp"])

    def test_promotion_archives_the_model_and_its_metrics(self) -> None:
        self._candidate()
        ledger_header, _ = self._gate(decision="accept_null")
        repository, isolated = self._isolated_repository(
            "archive-repository", ledger_header=ledger_header)
        candidate_net = candidate_net_path(self.iteration)
        net_digest = sha256_file(candidate_net)
        model_id = f"candidate-sha256-{net_digest}"

        preview = promote_gate(isolated, "candidate")
        self.assertFalse(preview["archived"])
        self.assertEqual(preview["sequential_decision"], "accept_null")
        self.assertFalse((repository / "models").exists())
        applied = promote_gate(isolated, "candidate", apply=True)
        self.assertTrue(applied["archived"])
        self.assertEqual((repository / "models" / f"{model_id}.poe2net").read_bytes(),
                         candidate_net.read_bytes())
        metrics = json.loads((repository / "models" / f"{model_id}.json").read_bytes())
        self.assertEqual(metrics["model_id"], model_id)
        self.assertNotIn("model_header_sha256", metrics)
        self.assertEqual(metrics["net"], {
            "name": "promotion-test/candidate", "id": net_digest[:12], "sha256": net_digest,
            "design": "pattern-gain-v1", "parameters": 5293, "file": f"{model_id}.poe2net"})
        self.assertEqual(metrics["lineage"], {"parent": None, "teacher": "two-ply-closure"})
        self.assertEqual(metrics["training"], {
            "method": "lookup", "positions": 12, "games": None,
            "curve": [{"epoch": 1, "validation_mae": 1.0}]})
        self.assertEqual(metrics["recipe"], {
            "path": "promotion.toml", "run": "promotion-test", "iteration": "candidate"})
        self.assertEqual(metrics["metrics"]["quantized"]["validation"]["mae"], 1.25)
        self.assertEqual(metrics["corpus_id"], "training-loader-test")
        self.assertEqual(metrics["teacher"], {"evaluator": "two-ply-closure"})
        with (repository / "models" / "index.csv").open(newline="", encoding="utf-8") as source:
            index = list(csv.DictReader(source))
        self.assertEqual(index, [{
            "recorded_utc": "20260819T000000Z", "run": "promotion-test",
            "iteration": "candidate", "model": net_digest[:12],
            "base": "fixture-base", "base_model": "", "decision": "accept_null", "games": "2",
            "score_pct": "100.0", "sealed_mae": "",
            "gate_run_id": "workflow-promotion-test-run", "net_sha256": net_digest,
            "teacher": "two-ply-closure", "method": "lookup", "data_positions": "12",
            "data_games": "",
        }])
        self.assertTrue(applied["indexed"])
        status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"], cwd=repository,
            check=True, capture_output=True, text=True).stdout.splitlines()
        self.assertEqual(sorted(status), [
            " M eval/results.csv", f"?? models/{model_id}.json",
            f"?? models/{model_id}.poe2net", "?? models/index.csv"])

    def test_promotion_reuses_an_identical_archive_after_an_interruption(self) -> None:
        self._candidate()
        ledger_header, _ = self._gate()
        repository, isolated = self._isolated_repository(
            "retry-repository", ledger_header=ledger_header)
        real_writer = csv.writer

        def fail_on_ledger(destination, *arguments, **options):
            if getattr(destination, "name", "").endswith("results.csv"):
                raise OSError("fixture interruption")
            return real_writer(destination, *arguments, **options)

        with mock.patch("poe2_training.workflow.csv.writer", side_effect=fail_on_ledger):
            with self.assertRaisesRegex(WorkflowError, "fixture interruption"):
                promote_gate(isolated, "candidate", apply=True)
        self._commit_all(repository, "keep interrupted archive")
        preview = promote_gate(isolated, "candidate")
        self.assertTrue(preview["archived"])
        self.assertTrue(preview["indexed"])
        self.assertTrue(promote_gate(isolated, "candidate", apply=True)["applied"])
        with (repository / "models" / "index.csv").open(newline="", encoding="utf-8") as source:
            self.assertEqual(len(list(csv.DictReader(source))), 1)

    def test_promotion_rejects_a_conflicting_index_row(self) -> None:
        self._candidate()
        ledger_header, _ = self._gate()
        repository, isolated = self._isolated_repository(
            "index-conflict-repository", ledger_header=ledger_header)
        (repository / "models").mkdir()
        with (repository / "models" / "index.csv").open(
                "w", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination, lineterminator="\n")
            writer.writerow(MODEL_INDEX_HEADER)
            writer.writerow(["workflow-promotion-test-run" if name == "gate_run_id" else "other"
                             for name in MODEL_INDEX_HEADER])
        with self.assertRaisesRegex(WorkflowError, "different values"):
            promote_gate(isolated, "candidate")

    def test_promotion_rejects_a_conflicting_archive(self) -> None:
        self._candidate()
        ledger_header, _ = self._gate()
        repository, isolated = self._isolated_repository(
            "conflict-repository", ledger_header=ledger_header)
        model_id = candidate_build_id(authenticate_candidate(isolated, self.iteration))
        (repository / "models").mkdir()
        (repository / "models" / f"{model_id}.poe2net").write_text("other\n", encoding="utf-8")
        with self.assertRaisesRegex(WorkflowError, "archived net differs from the candidate"):
            promote_gate(isolated, "candidate")

    def _chain_config(self):
        """Return a parent iteration and a child taught by and gated against its model."""
        pattern = ('[iterations.training]\ntype = "pattern"\ndevice = "cpu"\n'
                   'suite = "frozen-pattern-gain"\n'
                   '[iterations.candidate_validation]\npromotable = true\n')
        contents = textwrap.dedent(f"""
            schema_version = 1
            [run]
            name = "promotion-test"
            output_dir = "{self.root / 'run'}"
            [models]
            candidate = {{ iteration = "candidate" }}
            [datasets.ready]
            kind = "imported"
            path = "{self.root / 'features'}"
            [datasets.child]
            kind = "managed"
            teacher = "candidate"
            [datasets.child.source]
            corpus_id = "child"
            seed = 1
            trajectories = 4
            [datasets.child.labels]
            nodes = 100
        """) + (
            '[[iterations]]\nname = "candidate"\ndataset = "ready"\n' + pattern +
            f'[iterations.engine_gate]\nbase = "{self.root / "base-engine"}"\ngames = 2\n'
            '[[iterations]]\nname = "child"\ndataset = "child"\n' + pattern +
            '[iterations.engine_gate]\nbase_model = "candidate"\ngames = 2\n')
        return load_workflow_config(_write(self.root, contents, "chain.toml"))

    def test_a_child_is_promoted_only_after_its_parent_is_archived(self) -> None:
        chain = self._chain_config()
        self.config, self.iteration = chain, chain.iteration("candidate")
        self._candidate()
        ledger_header, _ = self._gate(run_id="parent-gate")
        parent_digest = sha256_file(candidate_net_path(self.iteration))
        resolved = resolve_trained_models(chain)
        parent = resolved.trained_model("candidate")
        self.assertEqual(parent.model_id, f"candidate-sha256-{parent_digest}")
        self.assertEqual((parent.net, parent.net_name),
                         (candidate_net_path(self.iteration), "promotion-test/candidate"))
        features = resolved.dataset("child").feature_directory
        features.parent.mkdir(parents=True)
        write_feature_fixture(features, teacher_sha256=parent_digest, teacher_key="net_sha256")
        self.config, self.iteration = resolved, resolved.iteration("child")
        self._candidate(features, teacher="promotion-test/candidate",
                        parent="promotion-test/candidate")
        self._gate(run_id="child-gate", decision="accept_null")
        self.assertEqual(authenticate_gate(self.config, self.iteration)["base_net_sha256"],
                         parent_digest)
        self.config = chain

        repository, isolated = self._isolated_repository(
            "chain-repository", ledger_header=ledger_header)
        with self.assertRaisesRegex(WorkflowError, "promote candidate first"):
            promote_gate(isolated, "child")
        promote_gate(isolated, "candidate", apply=True)
        self._commit_all(repository, "record parent gate")
        preview = promote_gate(isolated, "child")
        self.assertEqual(preview["index_row"]["base"], "promotion-test/candidate")
        self.assertEqual(preview["index_row"]["base_model"], parent_digest[:12])
        applied = promote_gate(isolated, "child", apply=True)
        metrics = json.loads(
            (repository / "models" / f"{applied['model_id']}.json").read_bytes())
        self.assertEqual(metrics["teacher"], {
            "evaluator": "pattern-gain", "model_id": f"candidate-sha256-{parent_digest}"})
        self.assertEqual(metrics["lineage"], {
            "parent": "promotion-test/candidate", "teacher": "promotion-test/candidate"})

    def _comparison_config(self):
        """Return a gated iteration and a comparison of its model with the archived v0.2."""
        contents = textwrap.dedent(f"""
            schema_version = 1
            [run]
            name = "promotion-test"
            output_dir = "{self.root / 'run'}"
            [models]
            g000 = "{V02}"
            candidate = {{ iteration = "candidate" }}
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
            promotable = true
            [iterations.engine_gate]
            base = "{self.root / 'base-engine'}"
            games = 2
            [[comparisons]]
            new = "candidate"
            base = "g000"
            games = 2
        """)
        return load_workflow_config(_write(self.root, contents, "comparison.toml"))

    def _comparison_artifacts(self, comparison, new, base) -> None:
        """Write a complete comparison as run_comparison records it."""
        root = comparison.root
        reserve_report_directory(root, COMPARISON_SCHEMA)
        sides = {side: {**comparison_side(self.config, model), "net": self._named(model_net(model))}
                 for side, model in (("new", new), ("base", base))}
        build_id = engine_build_id(3, "3" * 40)
        build = root / "builds" / build_id / self.config.build_preset
        binaries = {
            "new_engine_binary": build / "engines/minimax/poe2_minimax",
            "base_engine_binary": build / "engines/minimax/poe2_minimax",
            "runner_binary": build / "runner/poe2_runner",
        }
        for name, binary in binaries.items():
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text(binary.name, encoding="utf-8")
        _, _, run = self._write_match_run(
            root, comparison, kind=COMPARISON_KIND, new_id=build_id, base_id=build_id,
            new_engine_path=binaries["new_engine_binary"],
            base_engine_path=self._named(binaries["base_engine_binary"].resolve()),
            run_id="comparison-run", decision="accept_alt",
            nets={"new": model_net(new), "base": model_net(base)})
        complete_json_report(root, COMPARISON_SCHEMA, {
            "schema": COMPARISON_SCHEMA, "schema_version": COMPARISON_SCHEMA_VERSION,
            "git_commit": "3" * 40,
            "settings": {**match_settings(comparison),
                         "opening_book": self._named(comparison.opening_book)},
            "opening_book_sha256": sha256_file(comparison.opening_book),
            "run_id": "comparison-run", "run_directory": "runs/one",
            "engine_build_id": build_id, "build_directory": build.relative_to(root).as_posix(),
            **sides,
            "build_artifacts": {name: {"path": binary.relative_to(root).as_posix(),
                                       "sha256": sha256_file(binary)}
                                for name, binary in binaries.items()},
            "artifacts": {name: sha256_file(run / name) for name in
                          ("manifest.json", "summary.json", "games.csv", "ledger-row.csv")},
        })

    def test_a_comparison_is_recorded_once_both_models_are_archived(self) -> None:
        self.config = self._comparison_config()
        self.iteration = self.config.iteration("candidate")
        self._candidate()
        ledger_header, _ = self._gate(run_id="parent-gate", decision="accept_null")
        comparison = self.config.comparison("candidate-vs-g000")
        with self.assertRaisesRegex(WorkflowError, "has not finished"):
            record_comparison(self.config, "candidate-vs-g000")
        new, base = _comparison_models(self.config, comparison)
        self._comparison_artifacts(comparison, new, base)
        self.assertEqual(authenticate_comparison(
            self.config, comparison, new, base)["sequential_decision"], "accept_alt")

        repository, isolated = self._isolated_repository(
            "comparison-repository", ledger_header=ledger_header)
        (repository / "models").mkdir()
        for suffix in (".poe2net", ".json"):
            shutil.copy(REPOSITORY / "models" / f"{V02}{suffix}", repository / "models")
        self._commit_all(repository, "archive g000")
        with self.assertRaisesRegex(WorkflowError, "promote candidate first"):
            record_comparison(isolated, "candidate-vs-g000")
        promote_gate(isolated, "candidate", apply=True)
        self._commit_all(repository, "record the losing gate")
        preview = record_comparison(isolated, "candidate-vs-g000")
        self.assertEqual(preview["index_row"], {
            "recorded_utc": "20260819T000000Z", "run": "promotion-test",
            "comparison": "candidate-vs-g000", "new": "promotion-test/candidate",
            "new_model": new.sha256[:12], "base": "run1/g000",
            "base_model": V02.removeprefix("candidate-sha256-")[:12], "decision": "accept_alt",
            "games": "2", "score_pct": "100.0", "run_id": "comparison-run",
            "new_net_sha256": sha256_file(candidate_net_path(self.iteration)),
            "base_net_sha256": sha256_file(REPOSITORY / "models" / f"{V02}.poe2net")})
        self.assertTrue(record_comparison(isolated, "candidate-vs-g000", apply=True)["applied"])
        with (repository / "eval/results.csv").open(newline="", encoding="utf-8") as source:
            self.assertEqual([row["kind"] for row in csv.DictReader(source)],
                             ["training-gate", "training-comparison"])
        for name, rows in (("comparisons.csv", 1), ("index.csv", 1)):
            with (repository / "models" / name).open(newline="", encoding="utf-8") as source:
                self.assertEqual(len(list(csv.DictReader(source))), rows, name)
        with self.assertRaisesRegex(WorkflowError, "already contains run ID comparison-run"):
            record_comparison(isolated, "candidate-vs-g000")

    def test_gate_preview_apply_and_duplicate_rejection(self) -> None:
        self._candidate()
        header, _ = self._gate()
        preview = promote_gate(self.config, "candidate")
        self.assertFalse(preview["duplicate"])

        fake_repository, isolated = self._isolated_repository(
            "repository", ledger_header=header)
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
