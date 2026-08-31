"""API-level integration tests for /verify with the no_tie criterion.

Uses the Flask test client — no external HTTP requests required.
"""

import io
import json
import pathlib
import sys
import unittest
from unittest.mock import patch, MagicMock

from PIL import Image as PILImage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from service_client import authenticated_client
from tie_detector import TieDetection


def _make_jpeg_bytes(width=500, height=700, color=(255, 255, 255)):
    """Create a valid JPEG file in memory."""
    img = PILImage.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


class TestVerifyEndpoint(unittest.TestCase):
    """Test the /verify endpoint with no_tie criterion."""

    def setUp(self):
        from app import app
        self.client = authenticated_client(app)

    def _post_verify(self, photo_buf, criteria, params=None):
        data = {
            "photo": (photo_buf, "test.jpg", "image/jpeg"),
            "criteria": json.dumps(criteria),
        }
        if params:
            data["params"] = json.dumps(params)
        return self.client.post(
            "/verify",
            data=data,
            content_type="multipart/form-data",
        )

    MOCK_FACE = [{"x": 150, "y": 80, "w": 200, "h": 200, "score": 1.0, "keypoints": None}]

    @patch("verify._detect_faces", return_value=MOCK_FACE)
    @patch("verify.get_tie_detector")
    def test_no_tie_present_in_results(self, mock_get_detector, mock_detect_faces):
        """When no_tie is enabled, the result should include a no_tie key."""
        mock_detector = MagicMock()
        mock_detector.supports_absence_decision = True
        mock_detector.detect.return_value = None
        mock_get_detector.return_value = mock_detector

        photo = _make_jpeg_bytes()
        response = self._post_verify(photo, {"no_tie": True}, {"min_pass_criteria": 1})

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("results", data)
        self.assertIn("no_tie", data["results"])

    @patch("verify._detect_faces", return_value=MOCK_FACE)
    @patch("verify.get_tie_detector")
    def test_no_tie_label_present(self, mock_get_detector, mock_detect_faces):
        """The no_tie result should have the correct label."""
        mock_detector = MagicMock()
        mock_detector.supports_absence_decision = True
        mock_detector.detect.return_value = None
        mock_get_detector.return_value = mock_detector

        photo = _make_jpeg_bytes()
        response = self._post_verify(photo, {"no_tie": True}, {"min_pass_criteria": 1})

        data = response.get_json()
        self.assertEqual(data["results"]["no_tie"]["label"], "No Necktie")

    @patch("verify._detect_faces", return_value=MOCK_FACE)
    @patch("verify.get_tie_detector")
    def test_timing_metadata_present(self, mock_get_detector, mock_detect_faces):
        """Response should include timing metadata."""
        mock_detector = MagicMock()
        mock_detector.supports_absence_decision = True
        mock_detector.detect.return_value = None
        mock_get_detector.return_value = mock_detector

        photo = _make_jpeg_bytes()
        response = self._post_verify(photo, {"no_tie": True}, {"min_pass_criteria": 1})

        data = response.get_json()
        self.assertIn("timings_ms", data)
        self.assertIn("total", data["timings_ms"])
        self.assertIn("checks", data["timings_ms"])
        self.assertIn("no_tie", data["timings_ms"]["checks"])

    @patch("verify._detect_faces", return_value=MOCK_FACE)
    @patch("verify.get_tie_detector")
    def test_tie_detected_overall_fails(self, mock_get_detector, mock_detect_faces):
        """When a tie is detected, overall_passed should be False."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = TieDetection(
            confidence=0.92, bbox=(200, 350, 280, 550)
        )
        mock_get_detector.return_value = mock_detector

        photo = _make_jpeg_bytes()
        response = self._post_verify(photo, {"no_tie": True}, {"min_pass_criteria": 1})

        data = response.get_json()
        self.assertFalse(data["overall_passed"])
        self.assertFalse(data["results"]["no_tie"]["passed"])

    @patch("verify._detect_faces", return_value=MOCK_FACE)
    @patch("verify.get_tie_detector")
    def test_no_tie_absent_overall_passes(self, mock_get_detector, mock_detect_faces):
        """When no tie is detected, overall_passed should be True."""
        mock_detector = MagicMock()
        mock_detector.supports_absence_decision = True
        mock_detector.detect.return_value = None
        mock_get_detector.return_value = mock_detector

        photo = _make_jpeg_bytes()
        response = self._post_verify(photo, {"no_tie": True}, {"min_pass_criteria": 1})

        data = response.get_json()
        self.assertTrue(data["overall_passed"])
        self.assertTrue(data["results"]["no_tie"]["passed"])

    def test_endpoint_without_no_tie_still_works(self):
        """Existing criteria should work without no_tie."""
        photo = _make_jpeg_bytes()
        response = self._post_verify(photo, {"min_resolution": True}, {"min_pass_criteria": 1})

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertNotIn("no_tie", data["results"])

    def test_missing_photo_returns_400(self):
        """Request without photo should return 400."""
        response = self.client.post(
            "/verify",
            data={"criteria": json.dumps({"no_tie": True})},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)

    def test_invalid_criteria_json_returns_400(self):
        """Invalid JSON in criteria should return 400."""
        photo = _make_jpeg_bytes()
        data = {
            "photo": (photo, "test.jpg", "image/jpeg"),
            "criteria": "not-valid-json{{{",
        }
        response = self.client.post(
            "/verify",
            data=data,
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)

    @patch("verify.get_tie_detector")
    def test_meta_includes_model_version(self, mock_get_detector):
        """Result meta should always include model_version."""
        mock_detector = MagicMock()
        mock_detector.supports_absence_decision = True
        mock_detector.detect.return_value = None
        mock_get_detector.return_value = mock_detector

        photo = _make_jpeg_bytes()
        response = self._post_verify(photo, {"no_tie": True}, {"min_pass_criteria": 1})

        data = response.get_json()
        meta = data["results"]["no_tie"]["meta"]
        self.assertIn("model_version", meta)

    def test_health_endpoint_includes_no_tie(self):
        """The /health endpoint should list no_tie in available checks."""
        response = self.client.get("/health")
        data = response.get_json()
        self.assertIn("no_tie", data["checks_available"])
        self.assertIn("require_tie", data["checks_available"])

    @patch("verify._detect_faces", return_value=MOCK_FACE)
    @patch("verify.get_tie_detector")
    def test_require_tie_present_passes(self, mock_get_detector, mock_detect_faces):
        """When require_tie is enabled and tie is present, it should pass."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = TieDetection(
            confidence=0.92, bbox=(200, 350, 280, 550)
        )
        mock_get_detector.return_value = mock_detector

        photo = _make_jpeg_bytes()
        response = self._post_verify(photo, {"require_tie": True}, {"min_pass_criteria": 1})

        data = response.get_json()
        self.assertTrue(data["overall_passed"])
        self.assertTrue(data["results"]["require_tie"]["passed"])
        self.assertEqual(data["results"]["require_tie"]["label"], "Tie / Formal Neckwear Required")

    @patch("verify._detect_faces", return_value=MOCK_FACE)
    @patch("verify.get_tie_detector")
    def test_require_tie_absent_fails_overall(self, mock_get_detector, mock_detect_faces):
        """When require_tie is enabled and tie is absent, overall_passed should be False."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = None
        mock_get_detector.return_value = mock_detector

        photo = _make_jpeg_bytes()
        response = self._post_verify(photo, {"require_tie": True}, {"min_pass_criteria": 1})

        data = response.get_json()
        self.assertFalse(data["overall_passed"])
        self.assertFalse(data["results"]["require_tie"]["passed"])


class TestEditPhotoUnchanged(unittest.TestCase):
    """Verify that /edit-photo is not affected by tie detection changes."""

    def setUp(self):
        from app import app
        self.client = authenticated_client(app)

    def test_edit_photo_endpoint_exists(self):
        """The /edit-photo endpoint should still be accessible."""
        # Send without photo to get expected error
        response = self.client.post("/edit-photo")
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)


if __name__ == "__main__":
    unittest.main()
