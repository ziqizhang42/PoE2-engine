from __future__ import annotations

import io
import json
import tempfile
import textwrap
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from poe2_training.workflow import (
    WorkflowError,
    WorkflowRunner,
    _dependency_order,
    blocked_iterations,
    describe_not_started,
    print_plan,
    print_status,
    run_workflow,
    validate_existing,
    workflow_snapshot,
)
from poe2_training.workflow_adapters import data_binary, label_command, source_chunk_command
from poe2_training.workflow_config import (
    WorkflowConfigError,
    fingerprint,
    load_workflow_config,
    stage_definitions,
    stage_fingerprints,
    with_trained_model,
)
from poe2_training.workflow_state import RunState
from poe2_training.workflow_ui import WorkflowProgress


REPOSITORY = Path(__file__).resolve().parents[3]
RECIPES = REPOSITORY / "training/minimax/recipes"
V02 = "candidate-sha256-633309adb322e026784089d8b8ea175c80864a5941b0f2b912294e94a7d1ade3"
# Stage fingerprints of the long-form run1 recipe before the compact form existed. Finished
# runs are tied to these, so neither form may change them.
RUN1_FINGERPRINTS = {
    "dataset:g001:source": "dd37bbd5caf89213a81b5f533151ae4d767c3b3cb31fa5c5673069e259381259",
    "dataset:g001:labels": "0ae64cafbc93bc101d42d923123159a857cf144f6ef5f6105f6e37c07b250a7b",
    "dataset:g001:features": "27cef920a37a4b60056d13ec786363d845301a0024ef1dcc2a040a0c0bfe264e",
    "iteration:g001:training":
        "a1c5b9e6e8d1c2d3d2be5a7f6dba9dbfd27bcdd40df1347f6a8d1f0c12fa0b52",
    "iteration:g001:sealed-evaluation":
        "668a696e5e802e40399131c5aa7f5aebae7bb7957dc46e7393a3e35cadecd1ae",
    "iteration:g001:candidate-validation":
        "03a9c86d6d017800916f660f7ffae1c2a2a46db83d1330a0e367423a9836cb8c",
    "iteration:g001:engine-gate":
        "1379d1abb9a1b6b769a2c48e93312163f206250951b9f9eb7e5ab2a9cb943612",
}
DEFAULTS = textwrap.dedent("""
    [defaults.source]
    trajectories = 4
    workers = 1
    search_weight = 30
    [defaults.labels]
    mode = "teacher"
    nodes = 100
    workers = 1
    [defaults.training]
    type = "pattern"
    device = "cpu"
    suite = "frozen-pattern-gain"
    [defaults.candidate_validation]
    promotable = true
    [defaults.engine_gate]
    games = 2
""")
CHAIN = textwrap.dedent("""
    [[generations]]
    name = "g000"
    model = "%s"
    [[generations]]
    name = "g001"
    parent = "g000"
    seed = 11
    [[generations]]
    name = "g002"
    parent = "g001"
    seed = 12
    [generations.source]
    search_weight = 60
    [[generations]]
    name = "g003"
    parent = "g002"
    seed = 13
    [generations.training]
    seed = 99
    [generations.source]
    corpus_id = "custom-g003"
""" % V02)


class ChainFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY / "build")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _recipe(self, body: str, *, defaults: str = DEFAULTS, name: str = "chain.toml") -> Path:
        path = self.root / name
        path.write_text(textwrap.dedent(f"""
            schema_version = 1
            [run]
            name = "chain"
            output_dir = "{self.root / 'run'}"
            build_preset = "debug"
        """) + textwrap.dedent(defaults) + textwrap.dedent(body), encoding="utf-8")
        return path

    def _chain(self, body: str = CHAIN, **options):
        return load_workflow_config(self._recipe(body, **options))


class CompactRecipeTest(ChainFixture):
    def test_compact_run1_has_the_long_form_stage_fingerprints(self) -> None:
        compact = load_workflow_config(RECIPES / "run1.toml")
        long_form = load_workflow_config(
            REPOSITORY / "training/minimax/tests/recipes/run1-long-form.toml")
        self.assertEqual(stage_fingerprints(long_form), RUN1_FINGERPRINTS)
        self.assertEqual(stage_fingerprints(compact), RUN1_FINGERPRINTS)
        self.assertEqual(stage_definitions(compact), stage_definitions(long_form))
        self.assertEqual(compact.datasets, long_form.datasets)
        self.assertEqual(compact.iterations, long_form.iterations)

    def test_generations_expand_to_datasets_iterations_and_models(self) -> None:
        config = self._chain()
        self.assertEqual([item.name for item in config.datasets], ["g001", "g002", "g003"])
        self.assertEqual([item.name for item in config.iterations], ["g001", "g002", "g003"])
        self.assertEqual([(item.name, item.model_id, item.iteration) for item in config.models], [
            ("g000", V02, None), ("g001", None, "g001"), ("g002", None, "g002"),
            ("g003", None, "g003")])
        first, second, third = config.datasets
        self.assertEqual((first.source.corpus_id, first.source.seed), ("poe2-chain-g001-v1", 11))
        self.assertEqual((third.source.corpus_id, third.source.seed), ("custom-g003", 13))
        self.assertEqual([item.training.seed for item in config.iterations], [11, 12, 99])
        self.assertEqual([item.source.search_weight for item in config.datasets], [30, 60, 30])
        self.assertEqual({item.source.trajectories for item in config.datasets}, {4})
        self.assertEqual([item.teacher.name for item in config.datasets], ["g000", "g001", "g002"])
        gates = [item.engine_gate for item in config.iterations]
        self.assertEqual([(gate.base, gate.base_model, gate.base_iteration) for gate in gates], [
            (None, V02, None), (None, None, "g001"), (None, None, "g002")])
        self.assertTrue(all(item.sealed_evaluation is None for item in config.iterations))

    def test_generations_reject_invalid_entries(self) -> None:
        trained = '[[generations]]\nname = "g001"\nparent = "g000"\nseed = 1\n'
        pinned = f'[[generations]]\nname = "g000"\nmodel = "{V02}"\n'
        cases = [
            (pinned + trained + '[datasets.extra]\nkind = "managed"\n', DEFAULTS,
             "either long-form"),
            (pinned + '[[generations]]\nname = "g001"\nseed = 1\n', DEFAULTS, "parent is required"),
            (trained + pinned, DEFAULTS, "earlier generation: g000"),
            (pinned + trained.replace('"g000"', '"g001"'), DEFAULTS, "earlier generation"),
            (pinned + trained.replace("seed = 1\n", ""), DEFAULTS, "seed is required"),
            (pinned + trained + trained, DEFAULTS, "repeated: g001"),
            (pinned, DEFAULTS, "with a parent"),
            (pinned.replace('model = ', 'seed = 1\nmodel = '), DEFAULTS, "unknown field"),
            (pinned + trained.replace("seed = 1", "seed = 1\nwindow = 2"), DEFAULTS,
             "unknown field"),
            (pinned + trained + '[generations.engine_gate]\nbase_model = "g000"\n', DEFAULTS,
             "gate base is its parent"),
            (pinned + trained, DEFAULTS.replace("trajectories = 4", 'corpus_id = "x"\ntrajectories = 4'),
             "defaults.source cannot set corpus_id"),
            (pinned + trained, DEFAULTS.replace('type = "pattern"', 'seed = 4\ntype = "pattern"'),
             "defaults.training cannot set seed"),
            (pinned + trained, DEFAULTS + 'base = "000074-aa1f911c27b6"\n',
             "defaults.engine_gate cannot set base"),
            (pinned + trained, DEFAULTS.replace("[defaults.engine_gate]\ngames = 2", ""),
             "needs \\[defaults.engine_gate\\]"),
            (pinned + trained, DEFAULTS + "[defaults.comparisons]\n", "unknown field"),
        ]
        for index, (body, defaults, message) in enumerate(cases):
            with self.subTest(index=index, message=message):
                with self.assertRaisesRegex(WorkflowConfigError, message):
                    self._chain(body, defaults=defaults, name=f"bad-{index}.toml")

    def test_child_fingerprints_follow_the_parent_recipe_not_its_model(self) -> None:
        config = self._chain()
        definitions = stage_definitions(config)
        parent_gate = fingerprint(definitions["iteration:g001:engine-gate"])
        expected = {"iteration": "g001", "engine_gate": parent_gate}
        self.assertEqual(definitions["dataset:g002:source"]["teacher"], expected)
        self.assertEqual(definitions["dataset:g002:labels"]["teacher"], expected)
        self.assertEqual(definitions["iteration:g002:engine-gate"]["base"], expected)
        self.assertEqual(definitions["iteration:g002:engine-gate"]["settings"]["base_iteration"],
                         "g001")
        self.assertEqual(definitions["dataset:g001:source"]["teacher"], V02)

        model = replace(config.trained_model("g001"), model_id="candidate-sha256-" + "a" * 64,
                        header=self.root / "g001.hpp")
        self.assertEqual(stage_fingerprints(with_trained_model(config, model)),
                         stage_fingerprints(config))

        changed = self._chain(CHAIN.replace(
            'parent = "g000"\nseed = 11', 'parent = "g000"\nseed = 11\n'
            '[generations.engine_gate]\ngames = 4'), name="changed.toml")
        before, after = stage_fingerprints(config), stage_fingerprints(changed)
        for stage in ("dataset:g002:source", "iteration:g002:engine-gate",
                      "iteration:g003:training"):
            self.assertNotEqual(before[stage], after[stage], stage)
        for stage in ("dataset:g001:source", "iteration:g001:candidate-validation"):
            self.assertEqual(before[stage], after[stage], stage)

    def test_a_trained_teacher_has_a_data_tool_named_by_its_iteration(self) -> None:
        config = self._chain()
        dataset = config.dataset("g002")
        binary = data_binary(config, dataset)
        self.assertEqual(binary, config.output_directory /
                         "build/teacher-iteration-g001/debug/runner/poe2_minimax_data")
        for command in (source_chunk_command(config, dataset, 0, 4),
                        label_command(config, dataset, 0)):
            self.assertEqual(command[0], str(binary))
            self.assertEqual(command[command.index("--evaluator") + 1], "pattern-gain")

    def test_dependency_order_includes_the_parent_that_trains_a_model(self) -> None:
        config = self._chain()
        self.assertEqual([item.name for item in _dependency_order(config, config.iteration("g003"))],
                         ["g001", "g002", "g003"])
        runner = WorkflowRunner(config, RunState(config), "1" * 40)
        with self.assertRaisesRegex(WorkflowError, "run g001 first or rerun with --with-depend"):
            runner.run_iteration(config.iteration("g002"))


class LongFormTrainedModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(dir=REPOSITORY / "build")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _load(self, models: str, *, child_teacher: str = "first", gate: bool = True,
              child_base: str = "first", order: tuple[str, str] = ("first", "second")):
        def iteration(name: str) -> str:
            dataset = "taught" if name == "second" else "plain"
            text = textwrap.dedent(f"""
                [[iterations]]
                name = "{name}"
                dataset = "{dataset}"
                [iterations.training]
                type = "pattern"
                device = "cpu"
                suite = "frozen-pattern-gain"
                [iterations.candidate_validation]
                promotable = true
            """)
            if gate and name == "first":
                text += '[iterations.engine_gate]\nbase = "000074-aa1f911c27b6"\n'
            if name == "second":
                text += f'[iterations.engine_gate]\nbase_model = "{child_base}"\n'
            return text

        text = textwrap.dedent(f"""
            schema_version = 1
            [run]
            name = "long"
            output_dir = "{self.root / 'run'}"
            build_preset = "debug"
            [models]
            g000 = "{V02}"
        """) + textwrap.dedent(models) + textwrap.dedent(f"""
            [datasets.plain]
            kind = "managed"
            teacher = "g000"
            [datasets.plain.source]
            corpus_id = "plain"
            seed = 1
            trajectories = 4
            [datasets.plain.labels]
            nodes = 100
            [datasets.taught]
            kind = "managed"
            teacher = "{child_teacher}"
            [datasets.taught.source]
            corpus_id = "taught"
            seed = 2
            trajectories = 4
            [datasets.taught.labels]
            nodes = 100
        """) + "".join(iteration(name) for name in order)
        path = self.root / "long.toml"
        path.write_text(text, encoding="utf-8")
        return load_workflow_config(path)

    def test_a_trained_model_is_named_with_its_iteration(self) -> None:
        config = self._load('first = { iteration = "first" }\n')
        self.assertEqual(config.dataset("taught").teacher.iteration, "first")
        self.assertEqual(config.iteration("second").engine_gate.base_iteration, "first")
        self.assertIsNone(config.iteration("second").engine_gate.base_model)
        archived = self._load('first = { iteration = "first" }\n', child_teacher="g000",
                              child_base="g000")
        self.assertNotIn("base_iteration",
                         stage_definitions(archived)["iteration:second:engine-gate"]["settings"])

    def test_a_trained_model_reference_is_checked(self) -> None:
        for options, message in (
                ({"models": 'first = { iteration = "missing" }\n'}, "unknown iteration"),
                ({"models": 'first = { iteration = "first" }\n', "gate": False}, "no engine gate"),
                ({"models": 'first = { iteration = "first" }\n',
                  "order": ("second", "first")}, "must come after first"),
                ({"models": 'first = { iteration = "first" }\nalso = { iteration = "first" }\n',
                  "child_teacher": "also", "child_base": "also"}, "already names"),
                ({"models": 'second = { iteration = "first" }\n', "child_teacher": "second",
                  "child_base": "second"}, "shares its name with iteration second"),
                ({"models": 'first = { model = "first" }\n'}, "unknown field")):
            with self.subTest(message=message), self.assertRaisesRegex(WorkflowConfigError, message):
                self._load(**options)


def _record_gate(iteration, decision: str) -> None:
    """Write the marker-level gate record that status and blocking read without authenticating."""
    run = iteration.gate_directory / "runs" / "one"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"sequential_decision": decision}) + "\n",
                                       encoding="utf-8")
    (iteration.gate_directory / "report.json").write_text(
        json.dumps({"run_directory": "runs/one"}) + "\n", encoding="utf-8")
    (iteration.gate_directory / "COMPLETE").write_text("fixture\n", encoding="utf-8")


def _complete(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "COMPLETE").write_text("fixture\n", encoding="utf-8")


class GenerationChainTest(ChainFixture):
    def test_descendants_of_a_gate_that_did_not_win_are_blocked(self) -> None:
        config = self._chain()
        self.assertEqual(blocked_iterations(config), {})
        _record_gate(config.iteration("g001"), "continue")
        blocked = blocked_iterations(config)
        self.assertEqual(sorted(blocked), ["g002", "g003"])
        self.assertEqual({item.loser for item in blocked.values()}, {"g001"})
        self.assertEqual(blocked["g003"].reason,
                         "g001 did not win its engine gate (decision continue)")

        first = config.iteration("g001")
        for path in (first.training_directory, first.candidate_directory):
            _complete(path)
        snapshot = workflow_snapshot(config)
        self.assertEqual(snapshot["iterations"]["g002"]["status"], "blocked")
        self.assertEqual(snapshot["iterations"]["g003"]["blocked_by"], blocked["g003"].reason)
        self.assertEqual(snapshot["status"], "stopped")
        for show in (print_status, print_plan):
            output = io.StringIO()
            with mock.patch("sys.stdout", output):
                show(config)
            self.assertIn(f"not started: {blocked['g002'].reason}", output.getvalue())

    def test_validate_skips_blocked_generations_but_rejects_their_artifacts(self) -> None:
        config = self._chain()
        _record_gate(config.iteration("g001"), "accept_null")
        with mock.patch("poe2_training.workflow.authenticate_gate", return_value={}):
            self.assertEqual(list(validate_existing(config)), ["iteration:g001:engine-gate"])
            config.dataset("g002").root.mkdir(parents=True)
            with self.assertRaisesRegex(WorkflowError, "dataset g002 has artifacts"):
                validate_existing(config)

    def _run(self, config, decisions: dict[str, str], **options):
        ran: list[tuple[str, str | None]] = []

        def run_iteration(runner, iteration) -> None:
            parent = runner.config.dataset(iteration.dataset).teacher
            ran.append((iteration.name, parent.model_id))
            if iteration.name in decisions:
                _record_gate(iteration, decisions[iteration.name])

        def accepted(_config, model):
            return replace(model, model_id=f"candidate-sha256-{model.name}",
                           header=self.root / f"{model.name}.hpp")

        self.progress = io.StringIO()
        with mock.patch("poe2_training.workflow.require_clean_committed", return_value="1" * 40), \
                mock.patch.object(WorkflowRunner, "run_iteration", autospec=True,
                                  side_effect=run_iteration), \
                mock.patch("poe2_training.workflow._accepted_model", side_effect=accepted), \
                mock.patch("poe2_training.workflow.WorkflowProgress",
                           lambda config, state: WorkflowProgress(
                               config, state, stream=self.progress)):
            return run_workflow(config, **options), ran

    def test_a_chain_continues_with_the_accepted_parent_as_teacher(self) -> None:
        config = self._chain()
        not_started, ran = self._run(
            config, {"g001": "accept_alt", "g002": "accept_alt", "g003": "accept_null"})
        self.assertEqual(not_started, [])
        self.assertEqual(ran, [("g001", V02), ("g002", "candidate-sha256-g001"),
                               ("g003", "candidate-sha256-g002")])

    def test_a_chain_stops_after_a_gate_that_did_not_win(self) -> None:
        config = self._chain()
        not_started, ran = self._run(config, {"g001": "accept_null"})
        self.assertEqual([name for name, _ in ran], ["g001"])
        self.assertEqual([item.name for item in not_started], ["g002", "g003"])
        self.assertEqual(describe_not_started(not_started), [
            "g001 did not win its engine gate (decision accept_null); not started: g002, g003"])
        self.assertIn("iteration 3/3  g003 not started: g001 did not win", self.progress.getvalue())
        events = RunState(config).events
        self.assertEqual([event["iteration"] for event in events
                          if event["type"] == "iteration_not_started"], ["g002", "g003"])
        self.assertEqual(events[-1]["type"], "run_completed")
        self.assertEqual(events[-1]["not_started"], ["g002", "g003"])

        with self.assertRaisesRegex(
                WorkflowError, "iteration g003 is not started: g001 did not win"):
            self._run(config, {}, iteration_name="g003")

    def test_a_requested_generation_whose_parent_loses_in_the_same_command_fails(self) -> None:
        config = self._chain()
        with self.assertRaisesRegex(WorkflowError, "iteration g002 is not started"):
            self._run(config, {"g001": "continue"}, iteration_name="g002",
                      with_dependencies=True)

    def test_a_started_child_locks_the_parent_model_it_was_taught_by(self) -> None:
        config = self._chain()
        dataset = config.dataset("g002")
        _complete(dataset.source_directory)
        state = RunState(config)
        state.accept_config()

        def prepare(model_id: str) -> None:
            model = replace(config.trained_model("g001"), model_id=model_id,
                            header=self.root / "g001.hpp")
            runner = WorkflowRunner(with_trained_model(config, model), state, "1" * 40)
            with mock.patch.object(runner, "_command"), \
                    mock.patch("poe2_training.workflow.authenticate_source",
                               return_value={"teacher": model_id}):
                runner._prepare_source(runner.config.dataset("g002"))

        prepare("candidate-sha256-" + "a" * 64)
        prepare("candidate-sha256-" + "a" * 64)
        with self.assertRaisesRegex(WorkflowError, "inputs to started stage dataset:g002:source"):
            prepare("candidate-sha256-" + "b" * 64)


if __name__ == "__main__":
    unittest.main()
