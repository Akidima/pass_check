"""Tests for opt-in, non-blocking production example capture."""

import json
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from training_capture import capture_inference_example


class TrainingCaptureTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRAINING_CAPTURE_ENABLED", None)
            self.assertIsNone(capture_inference_example(b"bytes", {}))

    def test_capture_writes_original_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"TRAINING_CAPTURE_ENABLED": "1", "TRAINING_CAPTURE_DIR": directory},
        ):
            capture_id = capture_inference_example(
                b"original-bytes",
                {"overall_passed": False},
                identity_id="application-2026-1",
                attire_policy="not_applicable",
                file_extension=".png",
            )
            self.assertIsNotNone(capture_id)
            image = pathlib.Path(directory) / f"{capture_id}.png"
            metadata = pathlib.Path(directory) / f"{capture_id}.json"
            self.assertEqual(image.read_bytes(), b"original-bytes")
            document = json.loads(metadata.read_text())
            self.assertEqual(document["identity_id"], "application-2026-1")
            self.assertIsNone(document["label"])
            self.assertIsNone(document["label_source"])

    def test_bad_extension_is_not_used_as_a_path(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"TRAINING_CAPTURE_ENABLED": "1", "TRAINING_CAPTURE_DIR": directory},
        ):
            capture_id = capture_inference_example(
                b"original", {}, file_extension="../../unsafe"
            )
            self.assertTrue((pathlib.Path(directory) / f"{capture_id}.jpg").exists())


if __name__ == "__main__":
    unittest.main()
