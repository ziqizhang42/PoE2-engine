from __future__ import annotations

import hashlib
import json
import textwrap
import unittest
from dataclasses import replace
from unittest import mock

from fixture import net_bytes, write_feature_fixture
from test_workflow_generations import CHAIN, DEFAULTS, V02, ChainFixture, _record_gate
from poe2_training.shared import complete_json_report, reserve_report_directory
from poe2_training.workflow import (
    WorkflowError,
    WorkflowRunner,
    _archive_entry,
    blocked_iterations,
)
from poe2_training.workflow_adapters import (
    CANDIDATE_NET,
    AdapterError,
    _authenticate_training_data,
    candidate_net_path,
    label_command,
    training_command,
)
from poe2_training.workflow_config import (
    TrainingDataset,
    WorkflowConfigError,
    datasets_used,
    extra_training_datasets,
    fingerprint,
    recorded_training_settings,
    stage_definitions,
    stage_fingerprints,
    trained_models_used,
)
from poe2_training.workflow_state import RunState


WINDOW = DEFAULTS.replace('suite = "frozen-pattern-gain"',
                          'suite = "frozen-pattern-gain"\nwindow = 3\nage_weight = 0.5')
G003_OPTIONS = CHAIN.replace('[generations.training]\nseed = 99', textwrap.dedent('''\
    [generations.training]
    seed = 99
    refresh = 1
    max_records = 5000
    piece_records = 1000
    extra = [{ dataset = "v02", weight = 0.1 }]'''))


class TrainingDataRecipeTest(ChainFixture):
    def setUp(self) -> None:
        super().setUp()
        self.imported = write_feature_fixture(self.root / "v02-features", corpus_id="v02")
        self.declared = f'[datasets.v02]\nkind = "imported"\npath = "{self.imported}"\n'

    def _windowed(self, body: str = G003_OPTIONS, name: str = "window.toml"):
        return self._chain(self.declared + body, defaults=WINDOW, name=name)

    def test_window_refresh_and_extras_expand_to_extra_datasets(self) -> None:
        config = self._windowed()
        self.assertEqual(config.iteration("g001").training.extra_datasets, ())
        self.assertEqual(config.iteration("g002").training.extra_datasets,
                         (TrainingDataset("g001", 0.5),))
        g003 = config.iteration("g003").training
        self.assertEqual(g003.extra_datasets, (
            TrainingDataset("g003-refresh-g002", 0.5), TrainingDataset("g001", 0.25),
            TrainingDataset("v02", 0.1)))
        self.assertEqual((g003.max_records, g003.piece_records), (5000, 1000))
        relabeled = config.dataset("g003-refresh-g002")
        self.assertEqual((relabeled.positions, relabeled.teacher.name), ("g002", "g002"))
        self.assertEqual(relabeled.source, config.dataset("g002").source)
        self.assertEqual(relabeled.source_directory, config.dataset("g002").source_directory)
        self.assertEqual(relabeled.labels, config.dataset("g003").labels)
        self.assertEqual(datasets_used(config, config.iteration("g003")),
                         ("g003", "g002", "g003-refresh-g002", "g001", "v02"))
        self.assertEqual([model.name for model in trained_models_used(
            config, config.iteration("g003"))], ["g002", "g001"])

        definitions = stage_definitions(config)
        self.assertNotIn("dataset:g003-refresh-g002:source", definitions)
        self.assertEqual(definitions["dataset:g003-refresh-g002:labels"]["source"],
                         fingerprint(definitions["dataset:g002:source"]))
        training = definitions["iteration:g003:training"]
        self.assertEqual(training["extra_datasets"], [
            fingerprint(definitions[f"dataset:{name}:features"])
            for name in ("g003-refresh-g002", "g001", "v02")])
        self.assertNotIn("extra_datasets", definitions["iteration:g001:training"])
        self.assertNotIn("max_records", definitions["iteration:g001:training"]["settings"])

    def test_training_data_options_change_only_the_stages_that_use_them(self) -> None:
        plain = stage_fingerprints(self._chain(CHAIN, name="plain.toml"))
        windowed = stage_fingerprints(self._windowed())
        # g002 learns from g001's data too, so its model changes, and so does g003's data,
        # which g002 teaches; g001 and g002's own data do not.
        for stage, value in plain.items():
            if stage.startswith(("iteration:g002", "iteration:g003", "dataset:g003")):
                self.assertNotEqual(windowed[stage], value, stage)
            else:
                self.assertEqual(windowed[stage], value, stage)
        self.assertEqual(sorted(set(windowed) - set(plain)), [
            "dataset:g003-refresh-g002:features", "dataset:g003-refresh-g002:labels",
            "dataset:v02:features"])
        settings = recorded_training_settings(self._windowed().iteration("g001").training)
        self.assertEqual(sorted(settings), ["device", "seed", "suite", "type"])

    def test_the_lookup_method_is_fingerprinted_and_passed_only_when_used(self) -> None:
        plain = self._chain(CHAIN, name="plain.toml")
        lookup = self._chain(CHAIN.replace('seed = 99', 'seed = 99\nmethod = "lookup"'),
                             name="lookup.toml")
        before, after = stage_fingerprints(plain), stage_fingerprints(lookup)
        self.assertEqual(sorted(stage for stage in before if before[stage] != after[stage]), [
            "iteration:g003:candidate-validation", "iteration:g003:engine-gate",
            "iteration:g003:training"])
        self.assertEqual(recorded_training_settings(lookup.iteration("g003").training)["method"],
                         "lookup")
        self.assertNotIn("method", recorded_training_settings(lookup.iteration("g002").training))
        self.assertNotIn("--method", training_command(
            "python", lookup.dataset("g002"), lookup.iteration("g002")))
        command = training_command("python", lookup.dataset("g003"), lookup.iteration("g003"))
        self.assertEqual(command[command.index("--method") + 1], "lookup")
        everywhere = self._chain(CHAIN, defaults=DEFAULTS.replace(
            'suite = "frozen-pattern-gain"', 'suite = "frozen-pattern-gain"\nmethod = "lookup"'),
            name="everywhere.toml")
        self.assertEqual({item.training.method for item in everywhere.iterations}, {"lookup"})
        for body, message in (
                (CHAIN.replace('seed = 99', 'seed = 99\nmethod = "sparse"'),
                 "method must be dense or lookup"),
                (CHAIN.replace('seed = 99', 'seed = 99\nmethod = "lookup"\npiece_records = 10'),
                 "which lookup never builds")):
            with self.subTest(message=message), self.assertRaisesRegex(WorkflowConfigError,
                                                                       message):
                self._chain(body, name="bad-method.toml")

    def test_training_data_settings_are_checked(self) -> None:
        cases = [
            (CHAIN.replace('seed = 99', 'seed = 99\nrefresh = 3'), "refresh is outside"),
            (CHAIN.replace('seed = 99', 'seed = 99\nextra = [{ dataset = "g000" }]'),
             "must name a trained generation or an imported dataset"),
            (CHAIN.replace('seed = 99', 'seed = 99\nextra = [{ dataset = "nope" }]'),
             "must name a trained generation"),
            (CHAIN.replace('seed = 99', 'seed = 99\nextra = [{ dataset = "g001", weight = 0 }]'),
             "weight must be positive"),
            (CHAIN.replace('seed = 99', 'seed = 99\nextra = [{ dataset = "g002" }]'),
             "names a dataset twice"),
            (CHAIN.replace('seed = 99', 'seed = 99\nage_weight = -1'), "age_weight must be"),
            (CHAIN.replace('seed = 99', 'seed = 99\npiece_records = 0'), "piece_records is out"),
        ]
        for index, (body, message) in enumerate(cases):
            with self.subTest(message=message), self.assertRaisesRegex(WorkflowConfigError,
                                                                       message):
                self._chain(body, defaults=WINDOW, name=f"bad-{index}.toml")

    def test_long_form_training_data_rules(self) -> None:
        base = textwrap.dedent(f"""
            schema_version = 1
            [run]
            name = "long"
            output_dir = "{self.root / 'long-run'}"
            build_preset = "debug"
            [datasets.main]
            kind = "managed"
            [datasets.main.source]
            corpus_id = "main"
            seed = 1
            trajectories = 4
            [datasets.main.labels]
            nodes = 100
            [datasets.other]
            kind = "imported"
            path = "{self.imported}"
            [[iterations]]
            name = "one"
            dataset = "main"
            [iterations.training]
            type = "pattern"
            device = "cpu"
            suite = "frozen-pattern-gain"
        """)
        relabel = ('[datasets.again]\nkind = "managed"\npositions = "main"\n'
                   '[datasets.again.labels]\nnodes = 200\n')
        good = base + 'extra_datasets = [{ dataset = "other", weight = 0.5 }, ' \
                      '{ dataset = "again" }]\n' + relabel
        path = self.root / "long.toml"
        path.write_text(good, encoding="utf-8")
        from poe2_training.workflow_config import load_workflow_config
        config = load_workflow_config(path)
        self.assertEqual([item.name for item in extra_training_datasets(
            config, config.iteration("one"))], ["other", "again"])
        self.assertEqual(config.dataset("again").source, config.dataset("main").source)
        cases = [
            (base + 'extra_datasets = [{ dataset = "nope" }]\n', "must name other datasets"),
            (base + 'extra_datasets = [{ dataset = "main" }]\n', "must name other datasets"),
            (base.replace('type = "pattern"\ndevice = "cpu"\nsuite = "frozen-pattern-gain"',
                          'type = "baseline"\ndevice = "cpu"') + 'max_records = 3\n',
             "unknown field"),
            (base + 'extra_datasets = [{ dataset = "again" }]\n' +
             relabel.replace('positions = "main"', 'positions = "other"'),
             "another managed dataset with its own source"),
            (base + 'extra_datasets = [{ dataset = "again" }]\n' +
             relabel.replace('positions = "main"', 'positions = "main"\n'
                                                    '[datasets.again.source]\ncorpus_id = "x"\n'
                                                    'seed = 1\ntrajectories = 4'),
             "exactly one of source and positions"),
        ]
        for index, (text, message) in enumerate(cases):
            with self.subTest(message=message), self.assertRaisesRegex(WorkflowConfigError,
                                                                       message):
                bad = self.root / f"long-bad-{index}.toml"
                bad.write_text(text, encoding="utf-8")
                load_workflow_config(bad)

    def test_a_needed_dataset_taught_by_a_loser_blocks_the_iteration(self) -> None:
        body = CHAIN + textwrap.dedent('''
            [[generations]]
            name = "g004"
            parent = "g001"
            seed = 14
            [generations.training]
            extra = [{ dataset = "g003", weight = 0.5 }]
        ''')
        config = self._chain(body, name="sibling.toml")
        _record_gate(config.iteration("g001"), "accept_alt")
        _record_gate(config.iteration("g002"), "accept_null")
        blocked = blocked_iterations(config)
        self.assertEqual(blocked["g004"].reason,
                         "g002 did not win its engine gate (decision accept_null)")

    def test_a_relabeled_dataset_reads_its_positions_source_and_locks_its_teacher(self) -> None:
        config = self._windowed()
        model = replace(config.trained_model("g002"), model_id="candidate-sha256-" + "c" * 64,
                        header=self.root / "g002.hpp")
        from poe2_training.workflow_config import with_trained_model
        config = with_trained_model(config, model)
        relabeled = config.dataset("g003-refresh-g002")
        command = label_command(config, relabeled, 0)
        self.assertEqual(command[command.index("--input") + 1],
                         str(config.dataset("g002").source_directory))
        self.assertEqual(command[command.index("--output-dir") + 1],
                         str(relabeled.labels_directory / "shards" / "shard-000"))
        state = RunState(config)
        state.accept_config()
        runner = WorkflowRunner(config, state, "1" * 40)
        with mock.patch.object(runner, "_prepare_source", return_value={"source": 1}) as source, \
                mock.patch.object(runner, "_begin", side_effect=WorkflowError("stop")) as begin:
            with self.assertRaisesRegex(WorkflowError, "stop"):
                runner.prepare_dataset("g003-refresh-g002")
        self.assertEqual(source.call_args.args[0].name, "g002")
        self.assertEqual(begin.call_args.args[2], {"source": {"source": 1},
                                                   "teacher": model.model_id})

    def test_training_command_and_authentication_follow_the_options(self) -> None:
        config = self._windowed()
        iteration = config.iteration("g003")
        extras = extra_training_datasets(config, iteration)
        command = training_command("python", config.dataset("g003"), iteration, extras)
        pairs = [command[index + 1:index + 3] for index, item in enumerate(command)
                 if item == "--extra-dataset"]
        self.assertEqual(pairs, [[str(dataset.feature_directory), repr(weight)] for dataset, weight
                                 in zip(extras, (0.5, 0.25, 0.1))])
        self.assertEqual(command[command.index("--max-records") + 1], "5000")
        self.assertEqual(command[command.index("--piece-records") + 1], "1000")
        plain = training_command("python", config.dataset("g001"), config.iteration("g001"))
        self.assertFalse({"--extra-dataset", "--max-records", "--piece-records"} & set(plain))

        features = [{"binary_sha256": f"{index}" * 64, "manifest_sha256": f"{index + 4}" * 64}
                    for index in range(4)]
        datasets = [{"feature_binary_sha256": item["binary_sha256"],
                     "feature_manifest_sha256": item["manifest_sha256"], "weight": weight}
                    for item, weight in zip(features, (1.0, 0.5, 0.25, 0.1))]
        report = {"training_data": {"datasets": datasets, "max_records": 5000,
                                    "piece_records": 1000}}
        self.assertEqual(_authenticate_training_data(report, iteration.training, features),
                         datasets)
        for change in ({"max_records": 4000}, {"datasets": datasets[:3]},
                       {"datasets": [*datasets[:3], {**datasets[3], "weight": 0.2}]}):
            with self.subTest(change=change), self.assertRaisesRegex(AdapterError, "differ"):
                _authenticate_training_data(
                    {"training_data": {**report["training_data"], **change}},
                    iteration.training, features)

    def test_the_archive_lists_training_datasets_only_when_there_are_several(self) -> None:
        config = self._windowed()
        iteration = config.iteration("g003")
        iteration.candidate_directory.mkdir(parents=True)
        net = net_bytes("chain/g003", teacher="chain/g002", parent="chain/g002")
        candidate_net_path(iteration).write_bytes(net)
        reserve_report_directory(iteration.training_directory, "poe2-minimax-pattern-experiment")
        complete_json_report(iteration.training_directory, "poe2-minimax-pattern-experiment", {
            "schema": "poe2-minimax-pattern-experiment", "schema_version": 1,
            "training": {"method": "lookup"}, "training_data": {"records": 40},
            "selection": {"best_model": "fixture"},
            "models": [{"name": "fixture", "trace": [{"epoch": 1}]}],
        })
        candidate = {
            "artifacts": {"net": {"sha256": hashlib.sha256(net).hexdigest(),
                                  "path": CANDIDATE_NET}},
            "model": {"name": "fixture", "float": {}, "quantized": {}}, "git_commit": "1" * 40,
            "created_at_utc": "now", "corpus_id": "c", "teacher": None,
            "feature_binary_sha256": "e" * 64, "training_report_sha256": "f" * 64,
            "sealed_evaluation_report_sha256": None, "report_sha256": "a" * 64,
            "parity": {}, "evaluator_benchmark": {}, "search_benchmark": {},
        }
        plain = json.loads(_archive_entry(config, iteration, candidate).metrics)
        self.assertNotIn("training_datasets", plain)
        # v02 is imported features, whose games are not known here.
        self.assertEqual(plain["training"], {"method": "lookup", "positions": 40, "games": None,
                                             "curve": [{"epoch": 1}]})
        self.assertEqual(plain["lineage"], {"parent": "chain/g002", "teacher": "chain/g002"})
        listed = json.loads(_archive_entry(config, iteration, {
            **candidate, "training_datasets": [{"corpus_id": "a"}, {"corpus_id": "b"}]}).metrics)
        self.assertEqual(listed["training_datasets"], [{"corpus_id": "a"}, {"corpus_id": "b"}])


if __name__ == "__main__":
    unittest.main()
