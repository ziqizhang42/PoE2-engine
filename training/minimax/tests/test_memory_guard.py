from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from fixture import REPOSITORY, write_feature_fixture
from test_training_data import learnable_rows
from poe2_training import memory_guard
from poe2_training.memory_guard import (
    GIB,
    EVALUATION_PROFILE,
    HostMemory,
    MemoryGuardError,
    ScopeLimit,
    check_memory,
    command_memory_limit,
    estimate,
    parse_size,
    probe_scope_limit,
    training_estimate,
)
from poe2_training.pattern_evaluation import run_pattern_evaluation
from poe2_training.pattern_experiment import frozen_pattern_gain_suite, run_pattern_experiment
from poe2_training.workflow import CommandExecutionError, CommandRunner

VALIDATION = 74_157
# Peaks measured on the v0.2 corpus and tiled copies (GiB): host and device for a method,
# train records, and piece size.
MEASURED = {
    ("dense", 189_246, None): (5.43, 1.79), ("dense", 567_738, None): (12.88, 4.27),
    ("dense", 946_230, None): (20.33, 6.75),
    ("lookup", 189_246, None): (2.40, 0.32), ("lookup", 946_230, None): (4.25, 1.14),
    ("lookup", 1_513_968, None): (5.90, 1.76), ("lookup", 2_081_706, None): (7.68, 2.38),
    ("lookup", 4_163_412, None): (13.92, 4.64),
    ("dense", 189_246, 50_000): (2.12, 1.09), ("dense", 946_230, 50_000): (2.43, 1.23),
    ("dense", 2_081_706, 50_000): (2.96, 1.44), ("dense", 189_246, 20_000): (2.25, 0.50),
}
SCOPES = probe_scope_limit(GIB) is None


def _roomy_host(available: float) -> HostMemory:
    return HostMemory(total=30 * GIB, available=int(available * GIB), cgroup_headroom=None,
                      resident=0)


class EstimateTest(unittest.TestCase):
    def test_estimates_cover_the_measured_peaks_closely(self) -> None:
        for (method, records, pieces), (host, device) in MEASURED.items():
            with self.subTest(method=method, records=records, pieces=pieces):
                needed = training_estimate(method, records, VALIDATION, pieces)
                self.assertGreaterEqual(needed.host / GIB, host * 0.99)
                self.assertLessEqual(needed.host / GIB, host * 1.15)
                self.assertGreaterEqual(needed.device / GIB, device * 0.99)
                self.assertLessEqual(needed.device / GIB, device * 1.15)
        evaluation = estimate(EVALUATION_PROFILE, 39_979).host / GIB
        self.assertTrue(1.82 <= evaluation <= 1.82 * 1.15)

    def test_a_refusal_names_the_estimate_what_is_free_and_the_ways_out(self) -> None:
        needed = training_estimate("dense", 1_513_968, VALIDATION)
        with self.assertRaises(MemoryGuardError) as raised:
            check_memory("pattern training (dense, 1,513,968 train records)", needed,
                         remedies="Ways out: lookup.", host=_roomy_host(27.0),
                         gpu=(15 * GIB, 16 * GIB))
        self.assertEqual(str(raised.exception), (
            "pattern training (dense, 1,513,968 train records) needs about 31.5 GiB of host "
            "memory, but 25.5 GiB is free after a 1.5 GiB margin; nothing was allocated. "
            "Ways out: lookup."))
        with self.assertRaisesRegex(MemoryGuardError, "about 10.5 GiB of GPU memory, but 7.2 GiB "
                                    "is free after a 0.8 GiB margin"):
            check_memory("dense", needed, host=_roomy_host(40.0), gpu=(8 * GIB, 16 * GIB))
        check_memory("lookup", training_estimate("lookup", 1_513_968, VALIDATION),
                     host=_roomy_host(27.0), gpu=(15 * GIB, 16 * GIB))
        with self.assertRaisesRegex(MemoryGuardError, "within this command's memory limit"):
            check_memory("dense", needed, host=replace(_roomy_host(27.0),
                                                       cgroup_headroom=20 * GIB))
        with mock.patch.dict(os.environ, {memory_guard.ESTIMATE_VARIABLE: "off"}):
            check_memory("dense", needed, host=_roomy_host(1.0))

    def test_sizes_and_the_machine_limit(self) -> None:
        self.assertEqual([parse_size(text) for text in ("20G", "512M", "1.5GiB", "4096")],
                         [20 * GIB, 512 << 20, 3 * GIB // 2, 4096])
        with self.assertRaisesRegex(MemoryGuardError, "not a memory size"):
            parse_size("lots")
        total = memory_guard._meminfo()["MemTotal"]
        self.assertEqual(command_memory_limit({}),
                         max(total - memory_guard.SYSTEM_RESERVE, total // 2))
        self.assertEqual(command_memory_limit({memory_guard.LIMIT_VARIABLE: "12G"}), 12 * GIB)
        self.assertIsNone(command_memory_limit({memory_guard.LIMIT_VARIABLE: "off"}))


class GuardedCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, runner: CommandRunner, name: str, code: str, **options):
        return runner.run([sys.executable, "-c", code], log_path=self.root / f"{name}.log",
                          metrics_path=self.root / f"{name}.json", **options)

    def test_the_wrapper_runs_the_command_itself_in_a_limited_scope(self) -> None:
        self.assertEqual(ScopeLimit(1234).command("poe2-x.scope", ["a", "--b"]), [
            "systemd-run", "--user", "--scope", "--quiet", "--unit=poe2-x.scope",
            "-p", "MemoryMax=1234", "-p", "MemorySwapMax=0", "--", "a", "--b"])

    def test_without_scopes_commands_run_unlimited_with_a_warning(self) -> None:
        runner = CommandRunner(REPOSITORY)
        with mock.patch("poe2_training.workflow.probe_scope_limit",
                        return_value="systemd-run is not usable: missing"):
            warning = runner.limit_memory()
        self.assertIn("without a memory limit (systemd-run is not usable: missing)", warning)
        self.assertIn("may use swap", warning)
        self.assertIsNone(runner.scope_limit)
        with mock.patch.dict(os.environ, {memory_guard.LIMIT_VARIABLE: "off"}):
            self.assertIn("=off", runner.limit_memory())
        with mock.patch("builtins.print"):
            result = self._run(runner, "plain", "print(open('/proc/self/cgroup').read())")
        self.assertNotIn("poe2-", result.output)
        self.assertNotIn("memory_limit_bytes", result.metrics)

    @unittest.skipUnless(SCOPES, "systemd user scopes with memory limits are unavailable")
    def test_scoped_commands_keep_exit_codes_and_descriptors_and_report_the_limit(self) -> None:
        runner = CommandRunner(REPOSITORY)
        with mock.patch.dict(os.environ, {memory_guard.LIMIT_VARIABLE: "256M"}):
            self.assertIsNone(runner.limit_memory())
        self.assertEqual(runner.scope_limit, ScopeLimit(256 << 20))
        read, write = os.pipe()
        self.addCleanup(os.close, read)
        self.addCleanup(os.close, write)
        runner.lock_descriptors = (write,)
        check = ("import os; p = '/sys/fs/cgroup' + open('/proc/self/cgroup').read()"
                 "[3:].strip(); print(open(p + '/memory.max').read().strip(), "
                 f"open(p + '/memory.swap.max').read().strip(), os.path.exists('/proc/self/fd/"
                 f"{write}'))")
        with mock.patch("builtins.print"):
            result = self._run(runner, "limited", check)
            self.assertEqual(result.output.split(), [str(256 << 20), "0", "True"])
            self.assertEqual(result.metrics["memory_limit_bytes"], 256 << 20)
            self.assertFalse(result.metrics["memory_limit_exceeded"])
            with self.assertRaisesRegex(CommandExecutionError, r"command exited 3 \(stopping\)"):
                self._run(runner, "failing", "print('stopping'); raise SystemExit(3)")
            with self.assertRaises(CommandExecutionError) as raised:
                self._run(runner, "greedy", "b = bytearray(1 << 30); print('survived')")
        self.assertIn("command was stopped for using more than its 0.2 GiB memory limit",
                      str(raised.exception))
        self.assertTrue(raised.exception.result.metrics["memory_limit_exceeded"])


class RefusalBeforeAllocationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.features = write_feature_fixture(self.root / "features",
                                              rows=learnable_rows(600, 4), corpus_id="guard")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_training_and_sealed_evaluation_refuse_before_creating_anything(self) -> None:
        tiny = mock.patch("poe2_training.memory_guard.host_memory",
                          return_value=_roomy_host(0.5))
        config = [replace(frozen_pattern_gain_suite()[0], max_steps=10)]
        with tiny, self.assertRaisesRegex(MemoryGuardError, r"Ways out: method = \"lookup\""):
            run_pattern_experiment(self.features, self.root / "refused", requested_device="cpu",
                                   configs=config)
        self.assertFalse((self.root / "refused").exists())
        with mock.patch("sys.stdout", io.StringIO()):
            run_pattern_experiment(self.features, self.root / "training",
                                   requested_device="cpu", configs=config, method="lookup")
        with tiny, self.assertRaisesRegex(MemoryGuardError, "sealed evaluation"):
            run_pattern_evaluation(self.features, self.root / "training",
                                   self.root / "evaluation-refused")
        self.assertFalse((self.root / "evaluation-refused").exists())
        with mock.patch("sys.stdout", io.StringIO()):
            run_pattern_evaluation(self.features, self.root / "training", self.root / "evaluation")
        self.assertTrue((self.root / "evaluation" / "COMPLETE").is_file())


if __name__ == "__main__":
    unittest.main()
