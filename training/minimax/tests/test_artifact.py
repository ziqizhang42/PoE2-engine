from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from poe2_training.artifact import FeatureArtifactError, open_feature_artifact

from fixture import write_feature_fixture


class FeatureArtifactTest(unittest.TestCase):
    def test_authenticates_complete_clean_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = open_feature_artifact(write_feature_fixture(Path(temporary) / "data"))
            self.assertEqual(artifact.record_count, 1)
            self.assertEqual(artifact.source_record_count, 1)
            self.assertEqual(artifact.split_counts, (1, 0, 0))

    def test_rejects_corrupted_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = write_feature_fixture(Path(temporary) / "data")
            binary = directory / "features.bin"
            contents = bytearray(binary.read_bytes())
            contents[-1] = 1
            binary.write_bytes(contents)
            with self.assertRaisesRegex(FeatureArtifactError, "digest differs"):
                open_feature_artifact(directory)

    def test_rejects_incomplete_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = write_feature_fixture(Path(temporary) / "data")
            (directory / "INCOMPLETE").write_text("test\n", encoding="utf-8")
            with self.assertRaisesRegex(FeatureArtifactError, "INCOMPLETE"):
                open_feature_artifact(directory)

    def test_rejects_dirty_provenance_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = write_feature_fixture(Path(temporary) / "data", dirty=True)
            with self.assertRaisesRegex(FeatureArtifactError, "dirty"):
                open_feature_artifact(directory)
            artifact = open_feature_artifact(directory, require_clean=False)
            self.assertEqual(artifact.record_count, 1)


if __name__ == "__main__":
    unittest.main()
