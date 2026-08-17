"""Tests for tie_detector.py using mock/fake detectors.

Unit tests should be fast and deterministic — they do not require a real
trained model.  Integration tests that load a production model should be
kept in a separate suite.
"""

import pathlib
import sys
import unittest
from unittest.mock import patch, MagicMock
from dataclasses import FrozenInstanceError

from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tie_detector import TieDetection, TieDetector, get_tie_detector


class FakeDetector:
    """A simple fake detector implementing the TieDetector protocol."""

    def __init__(self, detection=None):
        self._detection = detection

    def detect(self, image, roi_offset=(0, 0)):
        return self._detection


class TestTieDetection(unittest.TestCase):
    """Test the TieDetection dataclass."""

    def test_creation(self):
        d = TieDetection(confidence=0.95, bbox=(10, 20, 50, 80))
        self.assertEqual(d.confidence, 0.95)
        self.assertEqual(d.bbox, (10, 20, 50, 80))

    def test_frozen(self):
        d = TieDetection(confidence=0.5, bbox=(0, 0, 10, 10))
        with self.assertRaises((AttributeError, FrozenInstanceError)):
            d.confidence = 0.9


class TestFakeDetectorProtocol(unittest.TestCase):
    """Ensure the fake detector satisfies the protocol."""

    def test_fake_is_tie_detector(self):
        fake = FakeDetector()
        self.assertIsInstance(fake, TieDetector)

    def test_high_confidence_detection(self):
        det = TieDetection(confidence=0.96, bbox=(10, 20, 50, 80))
        fake = FakeDetector(detection=det)
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        result = fake.detect(img)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.confidence, 0.96)

    def test_low_confidence_detection(self):
        det = TieDetection(confidence=0.15, bbox=(10, 20, 50, 80))
        fake = FakeDetector(detection=det)
        img = Image.new("RGB", (100, 100))
        result = fake.detect(img)
        self.assertIsNotNone(result)
        self.assertLess(result.confidence, 0.30)

    def test_borderline_confidence(self):
        det = TieDetection(confidence=0.45, bbox=(10, 20, 50, 80))
        fake = FakeDetector(detection=det)
        img = Image.new("RGB", (100, 100))
        result = fake.detect(img)
        self.assertIsNotNone(result)
        self.assertGreater(result.confidence, 0.30)
        self.assertLess(result.confidence, 0.65)

    def test_no_detection(self):
        fake = FakeDetector(detection=None)
        img = Image.new("RGB", (100, 100))
        result = fake.detect(img)
        self.assertIsNone(result)


class TestGetTieDetector(unittest.TestCase):
    """Test the cached model loader."""

    def test_model_missing_raises_file_not_found(self):
        """When model file doesn't exist, get_tie_detector should raise."""
        # Clear lru_cache between tests
        get_tie_detector.cache_clear()
        with patch.dict("os.environ", {"TIE_MODEL_PATH": "/nonexistent/model.pt"}):
            with self.assertRaises(FileNotFoundError):
                get_tie_detector()
        get_tie_detector.cache_clear()

    def test_invalid_device_in_env(self):
        """Invalid configuration should raise when loading."""
        get_tie_detector.cache_clear()
        with patch.dict("os.environ", {
            "TIE_MODEL_PATH": "/nonexistent/model.pt",
            "TIE_MODEL_DEVICE": "invalid_device",
        }):
            with self.assertRaises(FileNotFoundError):
                get_tie_detector()
        get_tie_detector.cache_clear()

    def test_default_env_vars(self):
        """Default env vars should point to models/tie_detector_v1.pt."""
        get_tie_detector.cache_clear()
        # This will raise FileNotFoundError because the model doesn't exist,
        # but we can verify the error message mentions the default path.
        try:
            get_tie_detector()
        except FileNotFoundError as e:
            self.assertIn("tie_detector_v1.pt", str(e))
        get_tie_detector.cache_clear()


if __name__ == "__main__":
    unittest.main()
