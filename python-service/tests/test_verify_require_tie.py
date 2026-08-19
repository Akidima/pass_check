"""Tests for check_tie() in verify.py (require_tie criterion).

All tests use a mock detector so they run fast without a trained model.
"""

import io
import os
import pathlib
import sys
import unittest
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tie_detector import TieDetection
from verify import check_tie, run_checks, _detect_faces


def _make_image(width=500, height=700, color=(200, 200, 200)):
    """Create a synthetic BGR image."""
    return np.full((height, width, 3), color, dtype=np.uint8)


def _make_faces(x=150, y=80, w=200, h=200):
    """Create a face list with one face that has sufficient below-face space."""
    return [{"x": x, "y": y, "w": w, "h": h, "score": 1.0, "keypoints": None}]


class TestCheckRequireTie(unittest.TestCase):
    """Unit tests for the check_tie function."""

    # --- Tie present -> passed=True ---

    @patch("verify.get_tie_detector")
    def test_tie_present_accepts(self, mock_get_detector):
        """High-confidence tie detection should accept (passed=True)."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = TieDetection(
            confidence=0.96, bbox=(200, 350, 280, 550)
        )
        mock_get_detector.return_value = mock_detector

        bgr = _make_image()
        faces = _make_faces()
        result = check_tie(bgr, faces, {})

        self.assertTrue(result["passed"])
        self.assertEqual(result["meta"]["decision"], "accept")
        self.assertEqual(result["meta"]["tie_status"], "tie_present")
        self.assertTrue(result["meta"]["tie_detected"])
        self.assertIn("model_version", result["meta"])

    # --- No detection -> manual review (a detector cannot prove absence) ---

    @patch("verify.get_tie_detector")
    def test_tie_absent_requires_manual_review(self, mock_get_detector):
        """No box is ambiguous, not proof that a required tie is absent."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = None
        mock_get_detector.return_value = mock_detector

        bgr = _make_image()
        faces = _make_faces()
        result = check_tie(bgr, faces, {})

        self.assertFalse(result["passed"])
        self.assertEqual(result["meta"]["decision"], "manual_review")
        self.assertEqual(result["meta"]["tie_status"], "uncertain")
        self.assertIsNone(result["meta"]["tie_detected"])

    # --- Low confidence detection -> manual review ---

    @patch("verify.get_tie_detector")
    def test_low_confidence_requires_manual_review(self, mock_get_detector):
        """Weak positive evidence cannot prove a tie is present or absent."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = TieDetection(
            confidence=0.10, bbox=(200, 350, 280, 550)
        )
        mock_get_detector.return_value = mock_detector

        bgr = _make_image()
        faces = _make_faces()
        result = check_tie(bgr, faces, {"tie_accept_threshold": 0.30})

        self.assertFalse(result["passed"])
        self.assertEqual(result["meta"]["decision"], "manual_review")
        self.assertEqual(result["meta"]["tie_status"], "uncertain")

    @patch("verify.get_tie_detector")
    def test_off_chest_detection_requires_manual_review(self, mock_get_detector):
        """A confident object away from the neck/chest cannot be a valid tie."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = TieDetection(
            confidence=0.99, bbox=(5, 350, 80, 550)
        )
        mock_get_detector.return_value = mock_detector

        result = check_tie(_make_image(), _make_faces(), {})

        self.assertFalse(result["passed"])
        self.assertEqual(result["meta"]["decision"], "manual_review")
        self.assertEqual(result["meta"]["reason"], "implausible_horizontal_position")

    # --- Uncertain -> passed=False + manual_review ---

    @patch("verify.get_tie_detector")
    def test_uncertain_requires_manual_review(self, mock_get_detector):
        """Confidence between accept and require thresholds -> manual_review."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = TieDetection(
            confidence=0.45, bbox=(200, 350, 280, 550)
        )
        mock_get_detector.return_value = mock_detector

        bgr = _make_image()
        faces = _make_faces()
        result = check_tie(bgr, faces, {
            "tie_accept_threshold": 0.30,
            "tie_require_threshold": 0.65,
        })

        self.assertFalse(result["passed"])
        self.assertEqual(result["meta"]["decision"], "manual_review")
        self.assertEqual(result["meta"]["tie_status"], "uncertain")
        self.assertIsNone(result["meta"]["tie_detected"])

    # --- Insufficient visibility -> passed=False + manual_review ---

    def test_insufficient_visibility_tight_crop(self):
        """When the image is tightly cropped, should flag insufficient visibility."""
        bgr = _make_image(width=300, height=150)
        faces = _make_faces(x=50, y=20, w=100, h=120)
        result = check_tie(bgr, faces, {})

        self.assertFalse(result["passed"])
        self.assertEqual(result["meta"]["decision"], "manual_review")
        self.assertEqual(result["meta"]["tie_status"], "insufficient_upper_body_visibility")
        self.assertFalse(result["meta"]["upper_body_visible"])

    # --- No face -> cannot make tie decision ---

    def test_no_face_returns_insufficient(self):
        """No face detected -> cannot determine tie status."""
        bgr = _make_image()
        result = check_tie(bgr, [], {})

        self.assertFalse(result["passed"])
        self.assertEqual(result["meta"]["decision"], "manual_review")
        self.assertFalse(result["meta"]["upper_body_visible"])

    # --- Model unavailable -> manual_review / fail-safe ---

    @patch("verify.get_tie_detector", side_effect=FileNotFoundError("Model not found"))
    def test_model_unavailable_fails_safe(self, mock_get_detector):
        """When the model file is missing, should fail safe with passed=False."""
        bgr = _make_image()
        faces = _make_faces()
        result = check_tie(bgr, faces, {})

        self.assertFalse(result["passed"])
        self.assertIn(result["meta"]["decision"], ("manual_review", "reject"))


class TestRequireTieHardComplianceGate(unittest.TestCase):
    """Test that require_tie=False blocks overall approval in run_checks."""

    def _make_image_bytes(self, width=500, height=700, color=(255, 255, 255)):
        """Create a valid JPEG from synthetic data."""
        img = PILImage.new("RGB", (width, height), color)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    MOCK_FACE = [{"x": 150, "y": 80, "w": 200, "h": 200, "score": 1.0, "keypoints": None}]

    @patch("verify._detect_faces", return_value=MOCK_FACE)
    @patch("verify.get_tie_detector")
    def test_tie_detected_allows_overall_pass(self, mock_get_detector, mock_detect_faces):
        """When tie is detected and require_tie is enabled, it should pass."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = TieDetection(
            confidence=0.96, bbox=(200, 350, 280, 550)
        )
        mock_get_detector.return_value = mock_detector

        image_bytes = self._make_image_bytes()
        criteria = {"require_tie": True}
        result = run_checks(image_bytes, criteria, {"min_pass_criteria": 1})

        self.assertTrue(result["overall_passed"])
        self.assertIn("require_tie", result["results"])
        self.assertTrue(result["results"]["require_tie"]["passed"])

    @patch("verify._detect_faces", return_value=MOCK_FACE)
    @patch("verify.get_tie_detector")
    def test_missing_detection_blocks_overall_pass(self, mock_get_detector, mock_detect_faces):
        """Ambiguous missing detection must block auto-approval."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = None
        mock_get_detector.return_value = mock_detector

        image_bytes = self._make_image_bytes()
        criteria = {"require_tie": True}
        result = run_checks(image_bytes, criteria, {"min_pass_criteria": 1})

        self.assertFalse(result["overall_passed"])
        self.assertIn("require_tie", result["results"])
        self.assertFalse(result["results"]["require_tie"]["passed"])

    @patch("verify._detect_faces", return_value=MOCK_FACE)
    @patch("verify.get_tie_detector")
    def test_pass_count_cannot_override_require_tie_failure(self, mock_get_detector, mock_detect_faces):
        """Even if pass count is met, require_tie failure must block overall approval."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = None
        mock_get_detector.return_value = mock_detector

        image_bytes = self._make_image_bytes()
        # Enable multiple criteria and set min_pass_criteria low
        criteria = {
            "require_tie": True,
            "min_resolution": True,
            "brightness": True,
        }
        result = run_checks(image_bytes, criteria, {"min_pass_criteria": 1})

        # Even though min_resolution and brightness pass,
        # require_tie failure must block overall approval
        self.assertFalse(result["overall_passed"])
        self.assertFalse(result["results"]["require_tie"]["passed"])

    def test_require_tie_disabled_does_not_affect_result(self):
        """When require_tie is not enabled, it should not appear in results."""
        image_bytes = self._make_image_bytes()
        criteria = {"min_resolution": True}
        result = run_checks(image_bytes, criteria, {"min_pass_criteria": 1})

        self.assertNotIn("require_tie", result["results"])


class TestRequireTieRealWorldImage(unittest.TestCase):
    """Tests against real-world sample photos."""

    def test_open_collar_photo_fails_require_tie(self):
        """test_curl.jpg has a subject in suit with open collar (no tie); must fail require_tie."""
        test_path = pathlib.Path(__file__).resolve().parents[2] / "test_curl.jpg"
        if not test_path.exists():
            self.skipTest("test_curl.jpg not found")

        image_bytes = test_path.read_bytes()
        result = run_checks(image_bytes, {
            "require_tie": True,
            "white_background": True,
            "single_face": True,
            "face_framing": True,
        }, {"min_pass_criteria": 1})

        self.assertIn("require_tie", result["results"])
        self.assertFalse(result["results"]["require_tie"]["passed"])
        self.assertFalse(result["overall_passed"])

    def test_user_uploaded_striped_tie_photo_passes_with_coco_backend(self):
        """The bundled COCO backend detects a real traditional necktie."""
        candidate_paths = [
            pathlib.Path("/Users/georgeakidima/.gemini/antigravity-ide/brain/8508fd68-841b-46b0-9288-a4066302f061/.user_uploaded/media_1786962279199.jpg"),
            pathlib.Path("/Users/georgeakidima/.gemini/antigravity-ide/brain/8508fd68-841b-46b0-9288-a4066302f061/.user_uploaded/media_1786962271625.jpg"),
            pathlib.Path("/Users/georgeakidima/.gemini/antigravity-ide/brain/841200b8-059b-423c-a86f-52b1962ab545/.user_uploaded/media_1786960848824.jpg"),
        ]
        test_path = next((p for p in candidate_paths if p.exists()), None)
        if test_path is None:
            self.skipTest("User uploaded tie test images not found")

        image_bytes = test_path.read_bytes()
        bgr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        faces = _detect_faces(bgr)
        self.assertTrue(len(faces) >= 1)

        tie_res = check_tie(bgr, faces, {})
        self.assertTrue(tie_res["passed"], f"Tie check failed: {tie_res}")
        self.assertEqual(tie_res["meta"]["decision"], "accept")
        self.assertTrue(tie_res["meta"]["upper_body_visible"])
        self.assertTrue(tie_res["meta"]["tie_detected"])

    def test_solid_white_shirt_without_tie_is_not_accepted(self):
        """A no-tie image must not pass the required-tie criterion."""
        # White shirt without tie
        bgr = np.full((600, 400, 3), 250, dtype=np.uint8)
        # Add head/face
        cv2.ellipse(bgr, (200, 180), (80, 100), 0, 0, 360, (140, 160, 200), -1)
        faces = [{"x": 120, "y": 80, "w": 160, "h": 200, "score": 1.0, "keypoints": None}]
        tie_res = check_tie(bgr, faces, {})
        self.assertFalse(tie_res["passed"])
        self.assertIn(tie_res["meta"]["decision"], ("reject", "manual_review"))
        self.assertNotEqual(tie_res["meta"]["tie_detected"], True)


if __name__ == "__main__":
    unittest.main()
