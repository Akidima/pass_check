"""Tests for check_no_tie() in verify.py.

All tests use a mock detector so they run fast without a trained model.
"""

import pathlib
import sys
import unittest
from unittest.mock import patch, MagicMock

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tie_detector import TieDetection
from verify import check_no_tie, run_checks


def _make_image(width=500, height=700, color=(200, 200, 200)):
    """Create a synthetic BGR image."""
    return np.full((height, width, 3), color, dtype=np.uint8)


def _make_faces(x=150, y=80, w=200, h=200):
    """Create a face list with one face that has sufficient below-face space."""
    return [{"x": x, "y": y, "w": w, "h": h, "score": 1.0, "keypoints": None}]


class TestCheckNoTie(unittest.TestCase):
    """Unit tests for the check_no_tie function."""

    # --- Tie present -> passed=False ---

    @patch("verify.get_tie_detector")
    def test_tie_present_rejects(self, mock_get_detector):
        """High-confidence tie detection should reject (passed=False)."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = TieDetection(
            confidence=0.96, bbox=(200, 350, 280, 550)
        )
        mock_get_detector.return_value = mock_detector

        bgr = _make_image()
        faces = _make_faces()
        result = check_no_tie(bgr, faces, {})

        self.assertFalse(result["passed"])
        self.assertEqual(result["meta"]["decision"], "reject")
        self.assertEqual(result["meta"]["tie_status"], "tie_present")
        self.assertTrue(result["meta"]["tie_detected"])
        self.assertIn("model_version", result["meta"])

    # --- Tie absent -> passed=True ---

    @patch("verify.get_tie_detector")
    def test_tie_absent_accepts(self, mock_get_detector):
        """No detection should accept (passed=True)."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = None
        mock_get_detector.return_value = mock_detector

        bgr = _make_image()
        faces = _make_faces()
        result = check_no_tie(bgr, faces, {})

        self.assertTrue(result["passed"])
        self.assertEqual(result["meta"]["decision"], "accept")
        self.assertEqual(result["meta"]["tie_status"], "tie_absent")
        self.assertFalse(result["meta"]["tie_detected"])

    # --- Low confidence detection -> treat as absent ---

    @patch("verify.get_tie_detector")
    def test_low_confidence_accepts(self, mock_get_detector):
        """Detection below accept_threshold should be treated as absent."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = TieDetection(
            confidence=0.10, bbox=(200, 350, 280, 550)
        )
        mock_get_detector.return_value = mock_detector

        bgr = _make_image()
        faces = _make_faces()
        result = check_no_tie(bgr, faces, {"tie_accept_threshold": 0.30})

        self.assertTrue(result["passed"])
        self.assertEqual(result["meta"]["tie_status"], "tie_absent")

    # --- Uncertain -> passed=False + manual_review ---

    @patch("verify.get_tie_detector")
    def test_uncertain_requires_manual_review(self, mock_get_detector):
        """Confidence between accept and reject thresholds -> manual_review."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = TieDetection(
            confidence=0.45, bbox=(200, 350, 280, 550)
        )
        mock_get_detector.return_value = mock_detector

        bgr = _make_image()
        faces = _make_faces()
        result = check_no_tie(bgr, faces, {
            "tie_accept_threshold": 0.30,
            "tie_reject_threshold": 0.65,
        })

        self.assertFalse(result["passed"])
        self.assertEqual(result["meta"]["decision"], "manual_review")
        self.assertEqual(result["meta"]["tie_status"], "uncertain")
        self.assertIsNone(result["meta"]["tie_detected"])

    # --- Insufficient visibility -> passed=False + manual_review ---

    def test_insufficient_visibility_tight_crop(self):
        """When the image is tightly cropped, should flag insufficient visibility."""
        bgr = _make_image(width=300, height=150)
        # Face near bottom — not enough space below (10px < 0.35 * 120 = 42px)
        faces = _make_faces(x=50, y=20, w=100, h=120)
        result = check_no_tie(bgr, faces, {})

        self.assertFalse(result["passed"])
        self.assertEqual(result["meta"]["decision"], "manual_review")
        self.assertEqual(result["meta"]["tie_status"], "insufficient_upper_body_visibility")
        self.assertFalse(result["meta"]["upper_body_visible"])

    # --- No face -> cannot make tie decision ---

    def test_no_face_returns_insufficient(self):
        """No face detected -> cannot determine tie status."""
        bgr = _make_image()
        result = check_no_tie(bgr, [], {})

        self.assertFalse(result["passed"])
        self.assertEqual(result["meta"]["decision"], "manual_review")
        self.assertFalse(result["meta"]["upper_body_visible"])

    # --- Model unavailable -> manual_review ---

    @patch("verify.get_tie_detector", side_effect=FileNotFoundError("Model not found"))
    def test_model_unavailable_returns_manual_review(self, mock_get_detector):
        """When the model file is missing, should fail safe with manual_review."""
        bgr = _make_image()
        faces = _make_faces()
        result = check_no_tie(bgr, faces, {})

        self.assertFalse(result["passed"])
        self.assertEqual(result["meta"]["decision"], "manual_review")
        self.assertEqual(result["meta"]["tie_status"], "uncertain")
        self.assertEqual(result["meta"]["error"], "model_unavailable")

    # --- Model version always present ---

    @patch("verify.get_tie_detector")
    def test_model_version_present(self, mock_get_detector):
        """Every response must include model_version."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = None
        mock_get_detector.return_value = mock_detector

        bgr = _make_image()
        faces = _make_faces()
        result = check_no_tie(bgr, faces, {})
        self.assertIn("model_version", result["meta"])


class TestNoTieHardComplianceGate(unittest.TestCase):
    """Test that no_tie=False blocks overall approval in run_checks."""

    def _make_image_bytes(self, width=500, height=700, color=(255, 255, 255)):
        """Create a valid JPEG from synthetic data."""
        import io
        from PIL import Image as PILImage
        img = PILImage.new("RGB", (width, height), color)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    MOCK_FACE = [{"x": 150, "y": 80, "w": 200, "h": 200, "score": 1.0, "keypoints": None}]

    @patch("verify._detect_faces", return_value=MOCK_FACE)
    @patch("verify.get_tie_detector")
    def test_tie_detected_blocks_overall_pass(self, mock_get_detector, mock_detect_faces):
        """When tie is detected and no_tie is enabled, overall_passed must be False."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = TieDetection(
            confidence=0.96, bbox=(200, 350, 280, 550)
        )
        mock_get_detector.return_value = mock_detector

        image_bytes = self._make_image_bytes()
        criteria = {"no_tie": True}
        result = run_checks(image_bytes, criteria, {"min_pass_criteria": 1})

        self.assertFalse(result["overall_passed"])
        self.assertIn("no_tie", result["results"])
        self.assertFalse(result["results"]["no_tie"]["passed"])

    @patch("verify._detect_faces", return_value=MOCK_FACE)
    @patch("verify.get_tie_detector")
    def test_no_tie_absent_allows_overall_pass(self, mock_get_detector, mock_detect_faces):
        """When no tie is detected and no_tie is enabled, it should pass."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = None
        mock_get_detector.return_value = mock_detector

        image_bytes = self._make_image_bytes()
        criteria = {"no_tie": True}
        result = run_checks(image_bytes, criteria, {"min_pass_criteria": 1})

        self.assertTrue(result["overall_passed"])
        self.assertIn("no_tie", result["results"])
        self.assertTrue(result["results"]["no_tie"]["passed"])

    @patch("verify._detect_faces", return_value=MOCK_FACE)
    @patch("verify.get_tie_detector")
    def test_uncertain_blocks_overall_pass(self, mock_get_detector, mock_detect_faces):
        """Uncertain tie status must block overall approval."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = TieDetection(
            confidence=0.45, bbox=(200, 350, 280, 550)
        )
        mock_get_detector.return_value = mock_detector

        image_bytes = self._make_image_bytes()
        criteria = {"no_tie": True}
        result = run_checks(image_bytes, criteria, {
            "min_pass_criteria": 1,
            "tie_accept_threshold": 0.30,
            "tie_reject_threshold": 0.65,
        })

        self.assertFalse(result["overall_passed"])

    @patch("verify._detect_faces", return_value=MOCK_FACE)
    @patch("verify.get_tie_detector")
    def test_pass_count_cannot_override_no_tie_failure(self, mock_get_detector, mock_detect_faces):
        """Even if pass count is met, no_tie failure must block approval."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = TieDetection(
            confidence=0.96, bbox=(200, 350, 280, 550)
        )
        mock_get_detector.return_value = mock_detector

        image_bytes = self._make_image_bytes()
        # Enable many criteria but set min_pass_criteria low
        criteria = {
            "no_tie": True,
            "min_resolution": True,
            "brightness": True,
        }
        result = run_checks(image_bytes, criteria, {"min_pass_criteria": 1})

        # Even though min_resolution and brightness may pass,
        # no_tie failure should block overall
        self.assertFalse(result["overall_passed"])

    def test_no_tie_disabled_does_not_affect_result(self):
        """When no_tie is not enabled, it should not appear in results."""
        image_bytes = self._make_image_bytes()
        criteria = {"min_resolution": True}
        result = run_checks(image_bytes, criteria, {"min_pass_criteria": 1})

        self.assertNotIn("no_tie", result["results"])


if __name__ == "__main__":
    unittest.main()
