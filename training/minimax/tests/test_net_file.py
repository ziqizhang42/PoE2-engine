from __future__ import annotations

import unittest
from dataclasses import replace

from fixture import REPOSITORY
from poe2_training.net_file import (
    GAIN_KNOTS,
    PATTERN_GAIN_DESIGN,
    NetFileError,
    decode_net,
    encode_net,
    parameters_from_header,
    read_net,
)

REFERENCE_NET = REPOSITORY / "tests" / "fixtures" / "nets" / PATTERN_GAIN_DESIGN / "reference.poe2net"
MODEL_HEADER = REPOSITORY / "engines" / "minimax" / "src" / "frozen_pattern_gain_model.hpp"


class NetFileTest(unittest.TestCase):
    def test_reference_net_round_trips_to_identical_bytes(self) -> None:
        data = REFERENCE_NET.read_bytes()
        net = read_net(REFERENCE_NET)
        self.assertEqual(net.design, PATTERN_GAIN_DESIGN)
        self.assertEqual(net.name, "run1/g000")
        self.assertEqual(net.net_id, "633309adb322")
        self.assertEqual(encode_net(net.parameters, net.metadata), data)

    def test_reference_net_holds_the_built_in_header_parameters(self) -> None:
        parameters, strings = parameters_from_header(MODEL_HEADER.read_text())
        self.assertEqual(read_net(REFERENCE_NET).parameters, parameters)
        self.assertIn("kArtifactModelId", strings)

    def test_metadata_order_does_not_change_the_bytes(self) -> None:
        parameters = read_net(REFERENCE_NET).parameters
        forward = encode_net(parameters, {"name": "run9/g001", "parent": "run9/g000"})
        backward = encode_net(parameters, {"parent": "run9/g000", "name": "run9/g001"})
        self.assertEqual(forward, backward)
        net = decode_net(forward)
        self.assertEqual(net.net_id, net.sha256[:12])
        self.assertEqual(net.metadata, {"name": "run9/g001", "parent": "run9/g000"})

    def test_unusable_files_are_refused_with_a_reason(self) -> None:
        parameters = read_net(REFERENCE_NET).parameters
        valid = encode_net(parameters, {"name": "test/net"})
        cases = {
            b"X" + valid[1:]: "not a PoE2 net file",
            valid[:40]: "truncated",
            valid[:200] + bytes([valid[200] ^ 1]) + valid[201:]: "integrity check",
            valid[:8] + b"\x02" + valid[9:]: "container format 2",
        }
        for data, reason in cases.items():
            with self.subTest(reason=reason), self.assertRaisesRegex(NetFileError, reason):
                decode_net(data)
        with self.assertRaisesRegex(NetFileError, "needs a name"):
            encode_net(parameters, {"teacher": "x"})
        with self.assertRaisesRegex(NetFileError, "control character"):
            encode_net(parameters, {"name": "bad\tvalue"})
        with self.assertRaisesRegex(NetFileError, "gain knots must be 0, 12, 24, 36, 49"):
            encode_net(replace(parameters, gain_knots=(0, 10, 24, 36, 49)), {"name": "x"})
        self.assertEqual(parameters.gain_knots, GAIN_KNOTS)
        with self.assertRaisesRegex(NetFileError, "intercepts"):
            encode_net(replace(parameters, intercept=(1 << 25,) + parameters.intercept[1:]),
                       {"name": "x"})


if __name__ == "__main__":
    unittest.main()
