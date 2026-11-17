from __future__ import annotations

import io
import textwrap
import unittest
from dataclasses import replace
from unittest import mock

from fixture import net_bytes
from test_workflow_generations import (
    CHAIN,
    RECIPE_FIXTURES,
    REPOSITORY,
    RUN1_FINGERPRINTS,
    V02,
    ChainFixture,
    _complete,
    _record_gate,
)
from poe2_training.shared import sha256_file
from poe2_training.workflow import (
    WorkflowError,
    WorkflowRunner,
    blocked_comparisons,
    describe_not_started,
    print_plan,
    print_status,
    run_workflow,
    validate_existing,
    workflow_snapshot,
)
from poe2_training.workflow_adapters import engine_build_id
from poe2_training.workflow_config import (
    CANDIDATE_ENGINE,
    CANDIDATE_ENGINE_ARGS,
    WorkflowConfigError,
    fingerprint,
    load_workflow_config,
    stage_definitions,
    stage_fingerprints,
)
from poe2_training.workflow_state import RunState
from poe2_training.workflow_ui import WorkflowProgress


COMPARISONS = textwrap.dedent("""
    [[comparisons]]
    new = "g002"
    base = "g000"
    [[comparisons]]
    new = "g001"
    base = "g000"
    games = 4
    [[comparisons]]
    name = "latest"
    new = "g003"
    base = "g000"
""")


class ComparisonConfigTest(ChainFixture):
    def test_compact_comparisons_play_like_the_gates_unless_overridden(self) -> None:
        config = self._chain(CHAIN + COMPARISONS)
        self.assertEqual([(item.name, item.new, item.base, item.games)
                          for item in config.comparisons],
                         [("g002-vs-g000", "g002", "g000", 2), ("g001-vs-g000", "g001", "g000", 4),
                          ("latest", "g003", "g000", 2)])
        comparison = config.comparison("latest")
        self.assertEqual(comparison.root, config.output_directory / "comparisons/latest")
        self.assertEqual((comparison.new_engine, comparison.base_engine,
                          comparison.new_engine_args, comparison.base_engine_args),
                         (CANDIDATE_ENGINE, CANDIDATE_ENGINE, CANDIDATE_ENGINE_ARGS,
                          CANDIDATE_ENGINE_ARGS))
        self.assertIn("comparison:latest:match", stage_fingerprints(config))

    def test_adding_a_comparison_changes_no_other_fingerprint(self) -> None:
        compared = self.root / "run1-compared.toml"
        compared.write_text((RECIPE_FIXTURES / "run1-compact.toml").read_text(encoding="utf-8") + textwrap.dedent(
            """
            [[comparisons]]
            new = "g001"
            base = "g000"
            """), encoding="utf-8")
        fingerprints = stage_fingerprints(load_workflow_config(compared))
        self.assertIn("comparison:g001-vs-g000:match", fingerprints)
        del fingerprints["comparison:g001-vs-g000:match"]
        self.assertEqual(fingerprints, RUN1_FINGERPRINTS)

        plain = stage_fingerprints(self._chain())
        with_comparisons = stage_fingerprints(self._chain(CHAIN + COMPARISONS, name="cmp.toml"))
        self.assertEqual({stage: value for stage, value in with_comparisons.items()
                          if not stage.startswith("comparison:")}, plain)

    def test_a_trained_side_is_identified_by_its_candidate_not_its_gate(self) -> None:
        config = self._chain(CHAIN + COMPARISONS)
        definitions = stage_definitions(config)
        self.assertEqual(definitions["comparison:g002-vs-g000:match"]["new"], {
            "iteration": "g002",
            "candidate_validation": fingerprint(
                definitions["iteration:g002:candidate-validation"])})
        self.assertEqual(definitions["comparison:g002-vs-g000:match"]["base"], V02)

        stage = "comparison:g002-vs-g000:match"
        gated = self._chain(CHAIN.replace(
            "[generations.source]\nsearch_weight = 60",
            "[generations.source]\nsearch_weight = 60\n[generations.engine_gate]\ngames = 6")
            + COMPARISONS, name="gated.toml")
        self.assertEqual(stage_fingerprints(gated)[stage], stage_fingerprints(config)[stage])
        reseeded = self._chain(CHAIN.replace("seed = 12", "seed = 22") + COMPARISONS,
                               name="reseeded.toml")
        self.assertNotEqual(stage_fingerprints(reseeded)[stage], stage_fingerprints(config)[stage])

    def test_comparison_entries_are_checked(self) -> None:
        same = CHAIN + f'[[generations]]\nname = "copy"\nmodel = "{V02}"\n'
        cases = [
            (CHAIN + '[[comparisons]]\nnew = "g009"\nbase = "g000"\n', "must name a generation"),
            (CHAIN + '[[comparisons]]\nnew = "g001"\nbase = "g001"\n', "the same model"),
            (same + '[[comparisons]]\nnew = "copy"\nbase = "g000"\n', "the same model"),
            (CHAIN + COMPARISONS + '[[comparisons]]\nnew = "g002"\nbase = "g000"\n',
             "name is repeated: g002-vs-g000"),
            (CHAIN + '[[comparisons]]\nnew = "g001"\nbase = "g000"\n'
                     'base_engine_args = "--evaluator static"\n', "pattern-gain engine"),
            (CHAIN + '[[comparisons]]\nnew = "g001"\nbase = "g000"\nwindow = 2\n',
             "unknown field"),
            (CHAIN + '[[comparisons]]\nnew = "g001"\nbase = "g000"\ngames = 3\n',
             "games must be even"),
        ]
        for index, (body, message) in enumerate(cases):
            with self.subTest(message=message), self.assertRaisesRegex(WorkflowConfigError, message):
                self._chain(body, name=f"bad-{index}.toml")


class ComparisonRunTest(ChainFixture):
    def _config(self):
        return self._chain(CHAIN + COMPARISONS)

    def _accepted(self, _config, model):
        return replace(model, model_id=f"candidate-sha256-{model.name}",
                       header=self.root / f"{model.name}.hpp")

    def test_a_comparison_whose_side_never_started_is_not_started(self) -> None:
        config = self._config()
        _record_gate(config.iteration("g001"), "accept_alt")
        _record_gate(config.iteration("g002"), "accept_null")
        blocked = blocked_comparisons(config)
        self.assertEqual(list(blocked), ["latest"])
        self.assertEqual(blocked["latest"].reason,
                         "g002 did not win its engine gate (decision accept_null)")
        snapshot = workflow_snapshot(config)
        self.assertEqual(snapshot["comparisons"]["latest"]["status"], "blocked")
        self.assertEqual(snapshot["comparisons"]["g002-vs-g000"]["status"], "pending")
        for show in (print_status, print_plan):
            output = io.StringIO()
            with mock.patch("sys.stdout", output):
                show(config)
            self.assertIn(f"not started: {blocked['latest'].reason}", output.getvalue())
        with mock.patch("poe2_training.workflow._accepted_model", side_effect=self._accepted), \
                mock.patch("poe2_training.workflow.authenticate_gate", return_value={}):
            self.assertEqual(list(validate_existing(config)),
                             ["iteration:g001:engine-gate", "iteration:g002:engine-gate"])
            config.comparison("latest").root.mkdir(parents=True)
            with self.assertRaisesRegex(WorkflowError, "comparison latest has artifacts"):
                validate_existing(config)

    def _run(self, config, decisions: dict[str, str], **options):
        played: list[str] = []

        def run_iteration(runner, iteration) -> None:
            played.append(iteration.name)
            _complete(iteration.candidate_directory)
            if iteration.name in decisions:
                _record_gate(iteration, decisions[iteration.name])

        def run_comparison(_runner, comparison) -> None:
            played.append(comparison.name)

        self.progress = io.StringIO()
        with mock.patch("poe2_training.workflow.require_clean_committed", return_value="1" * 40), \
                mock.patch.object(WorkflowRunner, "run_iteration", autospec=True,
                                  side_effect=run_iteration), \
                mock.patch.object(WorkflowRunner, "run_comparison", autospec=True,
                                  side_effect=run_comparison), \
                mock.patch("poe2_training.workflow._accepted_model", side_effect=self._accepted), \
                mock.patch("poe2_training.workflow.WorkflowProgress",
                           lambda config, state: WorkflowProgress(
                               config, state, stream=self.progress)):
            return run_workflow(config, **options), played

    def test_run_plays_each_comparison_as_soon_as_both_models_exist(self) -> None:
        config = self._config()
        not_started, played = self._run(config, {"g001": "accept_alt", "g002": "accept_null"})
        self.assertEqual(played, ["g001", "g001-vs-g000", "g002", "g002-vs-g000"])
        self.assertEqual([item.label for item in not_started], ["g003", "comparison latest"])
        self.assertEqual(describe_not_started(not_started), [
            "g002 did not win its engine gate (decision accept_null); "
            "not started: g003, comparison latest"])
        self.assertIn("comparison latest not started: g002 did not win", self.progress.getvalue())
        events = RunState(config).events
        self.assertEqual([event["comparison"] for event in events
                          if event["type"] == "comparison_completed"],
                         ["g001-vs-g000", "g002-vs-g000"])
        self.assertEqual(events[-1]["not_started"], ["g003", "comparison latest"])

    def test_a_comparison_by_name_runs_alone_or_with_its_dependencies(self) -> None:
        config = self._config()
        with self.assertRaisesRegex(WorkflowError, "rerun with --with-dependencies"):
            self._run(config, {}, comparison_name="g002-vs-g000")
        _, played = self._run(config, {"g001": "accept_alt", "g002": "accept_null"},
                              comparison_name="g002-vs-g000", with_dependencies=True)
        self.assertEqual(played, ["g001", "g002", "g002-vs-g000"])
        _, played = self._run(config, {}, comparison_name="g001-vs-g000")
        self.assertEqual(played, ["g001-vs-g000"])
        with self.assertRaisesRegex(
                WorkflowError, "comparison latest is not started: g002 did not win"):
            self._run(config, {}, comparison_name="latest", with_dependencies=True)

    def _sides(self):
        models = []
        for name in ("g001", "g000"):
            net = self.root / f"{name}.poe2net"
            net.write_bytes(net_bytes(f"chain/{name}"))
            models.append(replace(self._config().model(name), header=None, net=net,
                                  net_sha256=sha256_file(net), net_name=f"chain/{name}",
                                  model_id=f"candidate-sha256-{sha256_file(net)}"))
        return tuple(models)

    def test_a_comparison_builds_its_commit_once_and_a_pause_discards_it(self) -> None:
        config = self._config()
        comparison = config.comparison("g001-vs-g000")
        new, base = self._sides()
        state = RunState(config)
        state.accept_config()
        runner = WorkflowRunner(config, state, "1" * 40)
        commands: dict[str, list[str]] = {}

        def record(_stage: str, operation: str, command, **_options) -> None:
            commands[operation] = list(command)
            if operation == "evaluate":
                raise KeyboardInterrupt

        with mock.patch("poe2_training.workflow._comparison_models", return_value=(new, base)), \
                mock.patch.object(runner, "_engine_build_id",
                                  return_value=engine_build_id(5, "1" * 40)), \
                mock.patch.object(runner, "_command", side_effect=record):
            with self.assertRaises(KeyboardInterrupt):
                runner.run_comparison(comparison)
        build = comparison.root / "builds" / "000005-111111111111" / "debug"
        self.assertEqual(commands["configure"][4], str(build))
        self.assertFalse(any(item.startswith("-DPOE2_MINIMAX_MODEL_HEADER")
                             for item in commands["configure"]))
        self.assertEqual(commands["build"][-2:], ["poe2_minimax", "poe2_runner"])
        evaluate = commands["evaluate"]
        self.assertEqual(evaluate[0], str(build / "runner/poe2_runner"))
        relative = build.relative_to(REPOSITORY).as_posix()
        for flag, value in (("--kind", "training-comparison"), ("--new-build", relative),
                            ("--base", relative),
                            ("--new-net", new.net.relative_to(REPOSITORY).as_posix()),
                            ("--base-net", base.net.relative_to(REPOSITORY).as_posix()),
                            ("--run-root", (comparison.root / "runs").relative_to(REPOSITORY)
                             .as_posix())):
            self.assertEqual(evaluate[evaluate.index(flag) + 1], value)
        started = next(event for event in state.events if event["type"] == "stage_started")
        self.assertEqual(started["input_fingerprint"], fingerprint({
            "new": new.model_id, "base": base.model_id,
            "opening_book_sha256": sha256_file(comparison.opening_book)}))
        self.assertEqual(runner.discard_interrupted_artifacts(), [comparison.root])
        self.assertFalse(comparison.root.exists())

    def test_a_comparison_of_one_model_with_itself_is_refused(self) -> None:
        config = self._config()
        new, _ = self._sides()
        with mock.patch("poe2_training.workflow._comparison_model", return_value=new):
            with self.assertRaisesRegex(WorkflowError, "against itself"):
                WorkflowRunner(config, RunState(config), "1" * 40).run_comparison(
                    config.comparison("g001-vs-g000"))


if __name__ == "__main__":
    unittest.main()
