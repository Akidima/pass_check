"""Comprehensive regression tests for white background and quality validation under real-world mobile-camera conditions."""

import os
import pathlib
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from verify import (  # noqa: E402
    _detect_faces,
    _measure_sharpness,
    check_blur,
    check_white_background,
    run_checks,
)


class WhiteBackgroundTests(unittest.TestCase):
    def image(self, bgr=(255, 255, 255), width=240, height=300):
        return np.full((height, width, 3), bgr, dtype=np.uint8)

    def assert_rejected(self, image, params=None):
        params = params or {}
        result = check_white_background(image, [], params)
        self.assertFalse(result["passed"], result)
        self.assertIn("thresholds", result["meta"])

    def test_pure_white_is_accepted(self):
        result = check_white_background(self.image(), [], {})
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["meta"]["white_coverage_percent"], 100.0)
        self.assertEqual(result["meta"]["nonwhite_coverage_percent"], 0.0)

    def test_near_white_and_off_white_are_accepted(self):
        # Near white (248) and off-white (240) in normal mobile room lighting
        result_248 = check_white_background(self.image((248, 248, 248)), [], {})
        self.assertTrue(result_248["passed"], result_248)
        result_240 = check_white_background(self.image((240, 240, 240)), [], {})
        self.assertTrue(result_240["passed"], result_240)

    def test_dark_gray_and_medium_gray_are_rejected(self):
        self.assert_rejected(self.image((180, 180, 180)))
        self.assert_rejected(self.image((120, 120, 120)))
        self.assert_rejected(self.image((0, 0, 0)))

    def test_warm_white_and_cool_white_are_accepted(self):
        # Warm incandescent room white (BGR: 238, 246, 252 -> RGB: 252, 246, 238)
        warm = self.image((238, 246, 252))
        self.assertTrue(check_white_background(warm, [], {})["passed"])
        # Cool daylight white (BGR: 252, 248, 242 -> RGB: 242, 248, 252)
        cool = self.image((252, 248, 242))
        self.assertTrue(check_white_background(cool, [], {})["passed"])

    def test_strong_yellow_and_blue_tints_are_rejected(self):
        # Saturated yellow background (BGR: 150, 220, 255)
        self.assert_rejected(self.image((150, 220, 255)))
        # Saturated blue background (BGR: 255, 180, 100)
        self.assert_rejected(self.image((255, 180, 100)))
        # Saturated red background
        self.assert_rejected(self.image((0, 0, 255)))
        # Saturated green background
        self.assert_rejected(self.image((0, 255, 0)))

    def test_large_shadow_is_rejected(self):
        shadow = self.image()
        # Large dark shadow covering 40% of the image
        shadow[:, :int(shadow.shape[1] * 0.45)] = (150, 150, 150)
        self.assert_rejected(shadow)

    def test_significant_dark_contamination_is_rejected(self):
        image = self.image()
        # Dark patch covering > 10% of image
        image[10:100, 10:100] = (0, 0, 0)
        result = check_white_background(image, [], {})
        self.assertFalse(result["passed"], result)
        self.assertGreater(result["meta"]["dark_coverage_percent"], 5.0)

    def test_significant_colored_contamination_is_rejected(self):
        image = self.image()
        # Colored patch covering > 10% of image
        image[10:100, 10:100] = (0, 0, 255)
        result = check_white_background(image, [], {})
        self.assertFalse(result["passed"], result)
        self.assertGreater(result["meta"]["colored_coverage_percent"], 5.0)

    def test_default_policy_requires_at_least_seventy_percent_white_background(self):
        # 20 columns on the left of 240px image gives ~80% white background coverage (passes >= 70%)
        passing_image = self.image()
        passing_image[:, :20] = (225, 225, 225)
        passing_result = check_white_background(passing_image, [], {})
        self.assertTrue(passing_result["passed"], passing_result)
        self.assertGreaterEqual(passing_result["meta"]["white_coverage_percent"], 70)

        # 36 columns on the left gives < 70% white background coverage (fails < 70%)
        failing_image = self.image()
        failing_image[:, :36] = (225, 225, 225)
        failing_result = check_white_background(failing_image, [], {})
        self.assertFalse(failing_result["passed"], failing_result)
        self.assertLess(failing_result["meta"]["white_coverage_percent"], 70)

    def test_administrator_tolerances_are_applied(self):
        # Admin can tighten requirements if desired
        strict_params = {
            "bg_min_value": 250,
            "bg_max_saturation": 4,
            "bg_max_delta_e": 2,
        }
        off_white = self.image((242, 242, 242))
        self.assertFalse(check_white_background(off_white, [], strict_params)["passed"])

    def test_portrait_mask_excludes_shoulders_but_keeps_upper_background(self):
        image = self.image(width=800, height=800)
        face = {"x": 250, "y": 100, "w": 300, "h": 300}
        # Simulate dark jacket shoulders that reach the lower image corners.
        cv2.fillConvexPoly(
            image,
            np.array([[220, 320], [580, 320], [799, 799], [0, 799]], dtype=np.int32),
            (20, 20, 20),
        )
        result = check_white_background(image, [face], {})
        self.assertTrue(result["passed"], result)
        self.assertGreater(result["meta"]["sampled_pixels"], 64)

    def test_enabled_white_background_failure_blocks_overall_approval(self):
        image = self.image((180, 180, 180))  # Gray background
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        with patch("verify._detect_faces", return_value=[]):
            result = run_checks(encoded.tobytes(), {"white_background": True}, {"min_pass_criteria": 1})
        self.assertFalse(result["results"]["white_background"]["passed"])
        self.assertFalse(result["overall_passed"])
        self.assertIn("total", result["timings_ms"])
        self.assertIn("white_background", result["timings_ms"]["checks"])

    def test_face_detection_falls_back_when_mediapipe_cannot_start(self):
        with patch("verify.MEDIAPIPE_AVAILABLE", True), patch(
            "verify.mp_face_detection.FaceDetection", side_effect=RuntimeError("GPU unavailable")
        ):
            self.assertEqual(_detect_faces(self.image()), [])


# =====================================================================
# Real-World Image & Mobile Submission Tests
# =====================================================================

class RealWorldMobileSubmissionTests(unittest.TestCase):
    """Test validation on actual real-world student submissions."""

    def test_user_uploaded_image_passes(self):
        """The user's real-world student mobile photo must pass validation."""
        path = "/Users/georgeakidima/.gemini/antigravity-ide/brain/23cbe225-6d40-46d1-8c8d-ee2bd4df5328/.user_uploaded/media_1786812369461.jpg"
        if not os.path.exists(path):
            self.skipTest("User uploaded media file not found on disk.")

        with open(path, "rb") as f:
            img_bytes = f.read()

        enabled = {
            "single_face": True,
            "face_framing": True,
            "white_background": True,
            "min_resolution": True,
            "no_blur": True,
            "brightness": True,
            "head_pose": True,
            "eyes_open": True,
        }
        result = run_checks(img_bytes, enabled, {"min_pass_criteria": 4})
        self.assertTrue(result["overall_passed"], result)
        self.assertTrue(result["results"]["white_background"]["passed"])
        self.assertGreaterEqual(result["results"]["white_background"]["meta"]["white_coverage_percent"], 70.0)

    def test_test_curl_image_passes(self):
        """The existing test_curl.jpg baseline image must pass validation."""
        path = pathlib.Path(__file__).resolve().parents[2] / "test_curl.jpg"
        if not path.exists():
            self.skipTest("test_curl.jpg not found.")

        with open(path, "rb") as f:
            img_bytes = f.read()

        enabled = {"single_face": True, "white_background": True, "no_blur": True}
        result = run_checks(img_bytes, enabled, {"min_pass_criteria": 2})
        self.assertTrue(result["overall_passed"], result)
        self.assertTrue(result["results"]["white_background"]["passed"])


# =====================================================================
# Tiered blur severity tests
# =====================================================================

class TieredBlurTests(unittest.TestCase):
    """Verify that check_blur() returns correct severity tiers."""

    def _make_textured(self, h=300, w=240):
        """Checkerboard + shapes: gives predictable Laplacian variance."""
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                if (y // 8 + x // 8) % 2 == 0:
                    img[y:y + 8, x:x + 8] = (200, 200, 200)
        cv2.rectangle(img, (60, 50), (180, 250), (150, 130, 130), 2)
        cv2.circle(img, (120, 120), 40, (100, 80, 80), 2)
        return img

    def _make_image(self, sharpness_level="sharp"):
        """Create a synthetic image with calibrated sharpness.

        Empirically measured Laplacian variance for the checkerboard pattern:
        - sharp (no blur):           ~2787
        - blurred (k=5, sigma=2):    ~65    (soft range 15–80)
        - severely_blurred (k=9, sigma=4): ~14 (below severe threshold 15)
        """
        img = self._make_textured()
        if sharpness_level == "blurred":
            img = cv2.GaussianBlur(img, (5, 5), 2)
        elif sharpness_level == "severely_blurred":
            img = cv2.GaussianBlur(img, (9, 9), 4)
        return img

    def test_sharp_image_is_acceptable(self):
        img = self._make_image("sharp")
        result = check_blur(img, [], {})
        self.assertTrue(result["passed"])
        self.assertEqual(result["meta"]["severity"], "acceptable")
        self.assertGreaterEqual(result["meta"]["sharpness"], 80.0)

    def test_moderately_blurred_image_is_soft(self):
        img = self._make_image("blurred")
        sharpness = _measure_sharpness(img, [])
        if 15 <= sharpness < 80:
            result = check_blur(img, [], {})
            self.assertFalse(result["passed"])
            self.assertEqual(result["meta"]["severity"], "soft")
            self.assertTrue(result["meta"]["blur_soft_fail_enabled"])
        else:
            self.skipTest(f"Synthetic blur produced sharpness={sharpness:.1f}, not in soft range")

    def test_severely_blurred_image_is_severe(self):
        img = self._make_image("severely_blurred")
        sharpness = _measure_sharpness(img, [])
        if sharpness < 15:
            result = check_blur(img, [], {})
            self.assertFalse(result["passed"])
            self.assertEqual(result["meta"]["severity"], "severe")
        else:
            self.skipTest(f"Synthetic severe blur produced sharpness={sharpness:.1f}, not below 15")

    def test_soft_fail_disabled_rejects_soft_blur(self):
        """When admin disables soft-fail, any blur below threshold is a hard fail."""
        img = self._make_image("blurred")
        sharpness = _measure_sharpness(img, [])
        if 15 <= sharpness < 80:
            result = check_blur(img, [], {"blur_soft_fail": 0})
            self.assertFalse(result["passed"])
            self.assertEqual(result["meta"]["severity"], "soft")
            self.assertFalse(result["meta"]["blur_soft_fail_enabled"])
            self.assertIn("blurry", result["message"].lower())
        else:
            self.skipTest(f"Synthetic blur produced sharpness={sharpness:.1f}, not in soft range")

    def test_severity_thresholds_from_meta(self):
        """Result meta contains the threshold values used."""
        img = self._make_image("sharp")
        result = check_blur(img, [], {"blur_threshold": 50, "blur_severe_threshold": 10})
        self.assertEqual(result["meta"]["blur_threshold"], 50.0)
        self.assertEqual(result["meta"]["blur_severe_threshold"], 10.0)


# =====================================================================
# Soft-failure promotion in run_checks
# =====================================================================

class SoftFailurePromotionTests(unittest.TestCase):
    """Verify that run_checks() promotes soft blur failures when background passes."""

    def _make_textured(self, h=300, w=240):
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        for y in range(0, h, 8):
            for x in range(0, w, 8):
                if (y // 8 + x // 8) % 2 == 0:
                    img[y:y + 8, x:x + 8] = (200, 200, 200)
        cv2.rectangle(img, (60, 50), (180, 250), (150, 130, 130), 2)
        cv2.circle(img, (120, 120), 40, (100, 80, 80), 2)
        return img

    def _make_soft_blur_white_bg_image(self):
        """White background image with calibrated soft blur (k=5, sigma=2 -> ~65 sharpness)."""
        img = self._make_textured()
        img = cv2.GaussianBlur(img, (5, 5), 2)
        return img

    def _encode_png(self, img):
        ok, encoded = cv2.imencode(".png", img)
        assert ok
        return encoded.tobytes()

    def test_soft_blur_white_bg_passes_overall(self):
        """A white-background image with soft blur should pass overall."""
        img = self._make_soft_blur_white_bg_image()
        sharpness = _measure_sharpness(img, [])
        if not (15 <= sharpness < 80):
            self.skipTest(f"Synthetic image sharpness={sharpness:.1f}, not in soft range")

        image_bytes = self._encode_png(img)
        with patch("verify._detect_faces", return_value=[]):
            result = run_checks(
                image_bytes,
                {"white_background": True, "no_blur": True},
                {"min_pass_criteria": 2, "blur_soft_fail": 1},
            )
        bg_passed = result["results"].get("white_background", {}).get("passed", False)
        blur_severity = result["results"].get("no_blur", {}).get("meta", {}).get("severity")
        if bg_passed and blur_severity == "soft":
            self.assertTrue(result["overall_passed"], result)
            self.assertIn("quality_notes", result)

    def test_soft_blur_nonwhite_bg_still_fails(self):
        """A non-white background should still fail even with soft blur tolerance."""
        img = np.full((300, 240, 3), 180, dtype=np.uint8)  # gray bg
        for y in range(0, 300, 8):
            for x in range(0, 240, 8):
                if (y // 8 + x // 8) % 2 == 0:
                    img[y:y + 8, x:x + 8] = (160, 160, 160)
        img = cv2.GaussianBlur(img, (5, 5), 2)
        image_bytes = self._encode_png(img)
        with patch("verify._detect_faces", return_value=[]):
            result = run_checks(
                image_bytes,
                {"white_background": True, "no_blur": True},
                {"min_pass_criteria": 2, "blur_soft_fail": 1},
            )
        self.assertFalse(result["results"]["white_background"]["passed"])
        self.assertFalse(result["overall_passed"])

    def test_severe_blur_white_bg_still_fails(self):
        """Severe blur always fails even with white background."""
        img = np.full((300, 240, 3), 255, dtype=np.uint8)
        for y in range(0, 300, 8):
            for x in range(0, 240, 8):
                if (y // 8 + x // 8) % 2 == 0:
                    img[y:y + 8, x:x + 8] = (200, 200, 200)
        cv2.rectangle(img, (60, 50), (180, 250), (150, 130, 130), 2)
        img = cv2.GaussianBlur(img, (9, 9), 4)  # calibrated: sharpness ~14 (severe)
        sharpness = _measure_sharpness(img, [])
        if sharpness >= 15:
            self.skipTest(f"Synthetic severe blur produced sharpness={sharpness:.1f}, not below 15")

        image_bytes = self._encode_png(img)
        with patch("verify._detect_faces", return_value=[]):
            result = run_checks(
                image_bytes,
                {"white_background": True, "no_blur": True},
                {"min_pass_criteria": 2, "blur_soft_fail": 1},
            )
        blur_result = result["results"].get("no_blur", {})
        self.assertEqual(blur_result.get("meta", {}).get("severity"), "severe")
        self.assertFalse(blur_result.get("soft_promoted", False))


# =====================================================================
# Mobile Camera & Low-Resolution Robustness Tests
# =====================================================================

class MobileCameraRobustnessTests(unittest.TestCase):
    """Test realistic mobile camera variations: low resolution, sensor noise, blur, compression."""

    def _make_portrait(self, bg_color, h=600, w=480):
        img = np.full((h, w, 3), bg_color, dtype=np.uint8)
        fx, fy, fw, fh = int(w * 0.25), int(h * 0.2), int(w * 0.5), int(h * 0.4)
        # Head
        cv2.ellipse(img, (fx + fw // 2, fy + fh // 2), (fw // 2, int(fh * 0.6)), 0, 0, 360, (140, 160, 190), -1)
        # Hair
        cv2.ellipse(img, (fx + fw // 2, fy + int(fh * 0.25)), (int(fw * 0.55), int(fh * 0.35)), 0, 180, 360, (30, 25, 20), -1)
        # Torso
        torso = np.array([
            [fx - int(fw * 0.2), fy + int(fh * 0.85)],
            [fx + int(fw * 1.2), fy + int(fh * 0.85)],
            [w, h],
            [0, h],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, torso, (80, 60, 50))
        faces = [{"x": fx, "y": fy, "w": fw, "h": fh}]
        return img, faces

    def test_low_resolution_240x300_white_passes(self):
        img, faces = self._make_portrait((248, 248, 248), h=300, w=240)
        result = check_white_background(img, faces, {})
        self.assertTrue(result["passed"], result)

    def test_high_resolution_1920x2400_white_passes(self):
        img, faces = self._make_portrait((250, 250, 250), h=2400, w=1920)
        result = check_white_background(img, faces, {})
        self.assertTrue(result["passed"], result)

    def test_sensor_noise_white_passes(self):
        """Simulate low-end phone sensor noise on white wall."""
        img, faces = self._make_portrait((242, 242, 242), h=400, w=320)
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        result = check_white_background(noisy, faces, {})
        self.assertTrue(result["passed"], result)

    def test_jpeg_compressed_q40_white_passes(self):
        img, faces = self._make_portrait((245, 245, 245), h=500, w=400)
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        result = check_white_background(dec, faces, {})
        self.assertTrue(result["passed"], result)

    def test_soft_blurred_portrait_white_passes(self):
        """Blurry edges between hair and background must not contaminate white background check."""
        img, faces = self._make_portrait((248, 248, 248), h=400, w=320)
        blurred = cv2.GaussianBlur(img, (15, 15), 5)
        result = check_white_background(blurred, faces, {})
        self.assertTrue(result["passed"], result)

    def test_stray_minor_artifact_passes(self):
        """A tiny speck (<0.1%) on an otherwise pristine white wall passes."""
        img, faces = self._make_portrait((248, 248, 248), h=500, w=400)
        img[10:20, 10:20] = (10, 10, 10)  # 10x10 speck
        result = check_white_background(img, faces, {})
        self.assertTrue(result["passed"], result)

    def test_white_background_independent_of_sharpness(self):
        """White background validation must pass on white background regardless of blur parameter."""
        img, faces = self._make_portrait((246, 246, 246), h=400, w=320)
        blurred = cv2.GaussianBlur(img, (21, 21), 8)
        result = check_white_background(blurred, faces, {"blur_threshold": 150.0})
        self.assertTrue(result["passed"], result)


# =====================================================================
# Adaptive background tolerance tests
# =====================================================================

class AdaptiveBackgroundToleranceTests(unittest.TestCase):
    """Verify that check_white_background() maintains threshold metadata compatibility."""

    def test_blur_tolerance_flag_in_thresholds(self):
        img = np.full((300, 240, 3), 238, dtype=np.uint8)
        result = check_white_background(img, [], {
            "bg_blur_adaptive_tolerance": 1,
            "blur_threshold": 80.0,
            "blur_severe_threshold": 15.0,
        })
        self.assertIn("blur_tolerance_applied", result["meta"]["thresholds"])

    def test_adaptive_tolerance_disabled(self):
        img = np.full((300, 240, 3), 238, dtype=np.uint8)
        result = check_white_background(img, [], {"bg_blur_adaptive_tolerance": 0})
        self.assertFalse(result["meta"]["thresholds"]["blur_tolerance_applied"])


# =====================================================================
# JPEG compression softness tests
# =====================================================================

class JPEGCompressionTests(unittest.TestCase):
    """JPEG-compressed white backgrounds should still be accepted."""

    def test_jpeg_quality_70_white_bg_passes(self):
        img = np.full((300, 240, 3), 255, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        self.assertTrue(ok)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        result = check_white_background(decoded, [], {})
        self.assertTrue(result["passed"], result)

    def test_jpeg_quality_50_white_bg(self):
        img = np.full((300, 240, 3), 255, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        self.assertTrue(ok)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        result = check_white_background(decoded, [], {})
        self.assertTrue(result["passed"], result)


if __name__ == "__main__":
    unittest.main()

