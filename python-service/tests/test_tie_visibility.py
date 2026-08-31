"""Tests for UpperBodyVisibilityEstimator (tie_visibility.py)."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tie_visibility import UpperBodyVisibilityEstimator, VisibilityResult


class TestUpperBodyVisibility(unittest.TestCase):
    """Test the face-relative upper-body visibility estimator."""

    def setUp(self):
        self.estimator = UpperBodyVisibilityEstimator(
            min_face_height_px=80,
            min_visible_below_face_ratio=1.25,
            horizontal_face_padding=0.85,
        )

    # --- Sufficient visibility ---

    def test_sufficient_visibility_passport_portrait(self):
        """Standard passport photo: face in upper third, plenty of chest visible."""
        face = {"x": 150, "y": 80, "w": 200, "h": 250}
        result = self.estimator.estimate(face, image_width=500, image_height=700)
        self.assertTrue(result.sufficient)
        self.assertIsNotNone(result.roi)
        x1, y1, x2, y2 = result.roi
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 500)
        self.assertLessEqual(y2, 700)

    def test_roi_includes_collar_seat(self):
        """The ROI must start above the face bottom so the knot is in-crop."""
        face = {"x": 100, "y": 50, "w": 150, "h": 200}
        result = self.estimator.estimate(face, image_width=400, image_height=600)
        self.assertTrue(result.sufficient)
        _, y1, _, y2 = result.roi
        face_bottom = face["y"] + face["h"]
        self.assertLess(y1, face_bottom)
        self.assertGreater(y2, face_bottom)

    # --- Insufficient visibility ---

    def test_insufficient_visibility_tight_crop(self):
        """Image ends at the chin — no collar/knot band visible."""
        face = {"x": 50, "y": 20, "w": 100, "h": 120}
        # Face bottom = 140, image height = 145 → only 5px below the chin
        result = self.estimator.estimate(face, image_width=200, image_height=145)
        self.assertFalse(result.sufficient)
        self.assertIn("insufficient", result.reason.lower())

    def test_collar_knot_band_is_sufficient(self):
        """A crop that shows the knot but not the blade is still analyzable."""
        estimator = UpperBodyVisibilityEstimator(
            min_face_height_px=80,
            min_visible_below_face_ratio=0.35,
            horizontal_face_padding=0.85,
        )
        face = {"x": 100, "y": 80, "w": 200, "h": 200}
        # Face bottom = 280; 0.12 * 200 = 24px of collar is enough for a knot
        result = estimator.estimate(face, image_width=400, image_height=310)
        self.assertTrue(result.sufficient)
        self.assertIn("collar/knot", result.reason.lower())

    def test_face_at_bottom_edge(self):
        """Face is at the very bottom — no visible area below."""
        face = {"x": 100, "y": 450, "w": 100, "h": 100}
        result = self.estimator.estimate(face, image_width=300, image_height=550)
        self.assertFalse(result.sufficient)

    # --- Tiny face ---

    def test_tiny_face_rejected(self):
        """A face smaller than min_face_height_px should be rejected."""
        face = {"x": 50, "y": 50, "w": 40, "h": 50}  # h=50 < 80
        result = self.estimator.estimate(face, image_width=500, image_height=700)
        self.assertFalse(result.sufficient)
        self.assertIn("below the minimum", result.reason.lower())

    def test_face_exactly_at_minimum(self):
        """Face exactly at min_face_height_px should be accepted if visibility is OK."""
        face = {"x": 100, "y": 50, "w": 80, "h": 80}
        # Face bottom = 130, need 100px below → image height >= 230
        result = self.estimator.estimate(face, image_width=300, image_height=300)
        self.assertTrue(result.sufficient)

    # --- ROI clipping ---

    def test_roi_clipped_to_image_bounds(self):
        """The ROI should be clamped within image boundaries."""
        # Face near left edge — horizontal ROI might try to go negative
        face = {"x": 5, "y": 50, "w": 100, "h": 100}
        result = self.estimator.estimate(face, image_width=200, image_height=400)
        self.assertTrue(result.sufficient)
        x1, y1, x2, y2 = result.roi
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 200)
        self.assertLessEqual(y2, 400)

    def test_roi_clipped_right_edge(self):
        """Face near right edge — ROI should not exceed image width."""
        face = {"x": 170, "y": 50, "w": 100, "h": 100}
        result = self.estimator.estimate(face, image_width=200, image_height=400)
        self.assertTrue(result.sufficient)
        _, _, x2, _ = result.roi
        self.assertLessEqual(x2, 200)

    # --- Invalid dimensions ---

    def test_zero_image_dimensions(self):
        """Zero-dimension image should return insufficient."""
        face = {"x": 0, "y": 0, "w": 100, "h": 100}
        result = self.estimator.estimate(face, image_width=0, image_height=0)
        self.assertFalse(result.sufficient)
        self.assertIn("invalid", result.reason.lower())

    def test_negative_image_dimensions(self):
        face = {"x": 0, "y": 0, "w": 100, "h": 100}
        result = self.estimator.estimate(face, image_width=-100, image_height=-100)
        self.assertFalse(result.sufficient)

    def test_missing_face_keys(self):
        """Face dict without required keys should return insufficient."""
        result = self.estimator.estimate({}, image_width=400, image_height=600)
        self.assertFalse(result.sufficient)
        self.assertIn("invalid", result.reason.lower())

    def test_partial_face_keys(self):
        """Face dict with only some keys should return insufficient."""
        result = self.estimator.estimate({"x": 10, "y": 20}, image_width=400, image_height=600)
        self.assertFalse(result.sufficient)

    # --- VisibilityResult is frozen ---

    def test_result_is_immutable(self):
        result = VisibilityResult(sufficient=True, reason="OK", roi=(0, 0, 100, 100))
        with self.assertRaises(AttributeError):
            result.sufficient = False


if __name__ == "__main__":
    unittest.main()
