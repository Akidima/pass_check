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
from tie_detector import TieDetection, TieDetector, get_tie_detector, validate_tie_detection


GEOMETRY = {
    "min_width_face_ratio": 0.04,
    "max_width_face_ratio": 0.85,
    "min_height_face_ratio": 0.12,
    "max_height_face_ratio": 1.60,
    "min_top_offset_face_ratio": -0.30,
    "max_top_offset_face_ratio": 1.35,
    "max_center_offset_face_ratio": 0.60,
}


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


class TestTieDetectionLocalization(unittest.TestCase):
    def test_centered_chest_box_is_valid(self):
        detection = TieDetection(confidence=0.95, bbox=(220, 310, 280, 490))
        valid, reason = validate_tie_detection(
            detection, {"x": 150, "y": 80, "w": 200, "h": 200}, 500, 700, GEOMETRY
        )
        self.assertTrue(valid)
        self.assertEqual(reason, "valid")

    def test_lapel_or_background_box_is_rejected(self):
        detection = TieDetection(confidence=0.99, bbox=(5, 310, 65, 490))
        valid, reason = validate_tie_detection(
            detection, {"x": 150, "y": 80, "w": 200, "h": 200}, 500, 700, GEOMETRY
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "implausible_horizontal_position")

    def test_out_of_image_box_is_rejected(self):
        detection = TieDetection(confidence=0.99, bbox=(220, 310, 550, 490))
        valid, reason = validate_tie_detection(
            detection, {"x": 150, "y": 80, "w": 200, "h": 200}, 500, 700, GEOMETRY
        )
        self.assertFalse(valid)
        self.assertEqual(reason, "bbox_out_of_image")


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
        with patch.dict("os.environ", {
            "TIE_MODEL_PATH": "/nonexistent/model.pt",
            "TIE_DETECTOR_BACKEND": "custom",
        }):
            with self.assertRaises(FileNotFoundError):
                get_tie_detector()
        get_tie_detector.cache_clear()

    def test_invalid_device_in_env(self):
        """Invalid configuration should raise when loading."""
        get_tie_detector.cache_clear()
        with patch.dict("os.environ", {
            "TIE_MODEL_PATH": "/nonexistent/model.pt",
            "TIE_MODEL_DEVICE": "invalid_device",
            "TIE_DETECTOR_BACKEND": "custom",
        }):
            with self.assertRaises(FileNotFoundError):
                get_tie_detector()
        get_tie_detector.cache_clear()

    @patch("tie_detector.CocoTieDetector")
    def test_default_auto_backend_uses_coco_when_custom_model_is_absent(self, mock_coco):
        """The app must remain usable before a custom checkpoint is trained."""
        get_tie_detector.cache_clear()
        with patch.dict("os.environ", {"TIE_DETECTOR_BACKEND": "auto"}, clear=False):
            get_tie_detector()
        mock_coco.assert_called_once()
        get_tie_detector.cache_clear()


if __name__ == "__main__":
    unittest.main()
