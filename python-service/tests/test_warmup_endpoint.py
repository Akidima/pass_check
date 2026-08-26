"""Tests for the /warmup endpoint that pre-loads lazy ML models.

The endpoint must be idempotent, cheap when already warm, survive a broken
detector backend (report the error, still answer 200), and accept GET+POST.
"""
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import app as service_app  # noqa: E402


class WarmupEndpointTests(unittest.TestCase):
    def setUp(self):
        service_app.app.config["TESTING"] = True
        self.client = service_app.app.test_client()

    def test_warmup_returns_ok_and_reports_loaded_models(self):
        with patch("app.get_tie_detector") as fake_det, \
             patch("app.get_rembg_session", return_value=object()):
            fake_det.return_value = object()
            resp = self.client.post("/warmup")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["tie_detector_loaded"])
        self.assertTrue(data["rembg_session_loaded"])

    def test_warmup_survives_detector_failure(self):
        with patch("app.get_tie_detector", side_effect=RuntimeError("boom")), \
             patch("app.get_rembg_session", return_value=None):
            resp = self.client.post("/warmup")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertFalse(data["tie_detector_loaded"])
        self.assertIn("tie_detector_error", data)
        self.assertFalse(data["rembg_session_loaded"])

    def test_warmup_accepts_get_too(self):
        with patch("app.get_tie_detector"), \
             patch("app.get_rembg_session", return_value=object()):
            resp = self.client.get("/warmup")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()