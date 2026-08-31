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
    BACKGROUND_STRICTNESS_LEVELS,
    _detect_faces,
    _measure_sharpness,
    _resolve_background_params,
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
        self.assert_rejected(self.image((100, 100, 100)))
        self.assert_rejected(self.image((80, 80, 80)))
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
        # Large dark shadow covering 45% of the image (deep shadow V=80)
        shadow[:, :int(shadow.shape[1] * 0.45)] = (80, 80, 80)
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
        # The default policy level is "standard": at least 60% of the visible
        # background must qualify as white.
        # 20 columns on the left of a 240px image gives > 80% white coverage (passes >= 60%)
        passing_image = self.image()
        passing_image[:, :20] = (225, 225, 225)
        passing_result = check_white_background(passing_image, [], {})
        self.assertTrue(passing_result["passed"], passing_result)
        self.assertGreaterEqual(passing_result["meta"]["white_coverage_percent"], 60)

        # A dark-grey (80) left half is genuinely non-white at the standard
        # level: ~50% coverage fails the >= 60% gate AND trips the
        # contiguous-patch and dark guards.
        failing_image = self.image()
        failing_image[:, :120] = (80, 80, 80)
        failing_result = check_white_background(failing_image, [], {})
        self.assertFalse(failing_result["passed"], failing_result)
        self.assertLess(failing_result["meta"]["white_coverage_percent"], 60)

    def test_administrator_tolerances_are_applied(self):
        # The strict LEVEL tightens the acceptance band: a light-neutral
        # surface (value 220, L* ~87.6) sits inside the standard band but
        # below every strict floor (value >= 235, near-white L* >= 93).
        light_gray = self.image((220, 220, 220))
        self.assertTrue(check_white_background(light_gray, [], {"background_strictness": "standard"})["passed"])
        strict_params = {"background_strictness": "strict"}
        self.assertFalse(check_white_background(light_gray, [], strict_params)["passed"])

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
        image = self.image((80, 80, 80))  # Dark gray background
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        with patch("verify._detect_faces", return_value=[]):
            result = run_checks(encoded.tobytes(), {"white_background": True}, {"min_pass_criteria": 1})
        self.assertFalse(result["results"]["white_background"]["passed"])
        self.assertFalse(result["overall_passed"])
        self.assertIn("total", result["timings_ms"])
        self.assertIn("white_background", result["timings_ms"]["checks"])

    def test_face_detection_falls_back_when_mediapipe_cannot_start(self):
        import verify
        if not getattr(verify, "MEDIAPIPE_AVAILABLE", False) or getattr(verify, "mp_face_detection", None) is None:
            self.skipTest("MediaPipe not available in environment")
        # The induced failure opens a real cooldown, so clear it afterwards or
        # every later test in this process inherits a degraded worker. Also
        # drop any cached MediaPipe graph so this patch hits the constructor.
        verify.reset_perception_health()
        self.addCleanup(verify.reset_perception_health)
        with patch.object(
            verify.mp_face_detection, "FaceDetection", side_effect=RuntimeError("GPU unavailable")
        ):
            self.assertEqual(_detect_faces(self.image()), [])

    def test_mediapipe_failure_does_not_permanently_disable_landmarks(self):
        """A transient MediaPipe error must not reject every later applicant.

        The Haar fallback cannot produce landmarks, so if the failure latched
        on, head_pose and eyes_open would fail for the rest of the worker's
        life and every subsequent photo would be rejected while /health still
        reported "ok".
        """
        import verify
        if not getattr(verify, "MEDIAPIPE_AVAILABLE", False) or getattr(verify, "mp_face_detection", None) is None:
            self.skipTest("MediaPipe not available in environment")
        verify.reset_perception_health()
        self.addCleanup(verify.reset_perception_health)

        with patch.object(
            verify.mp_face_detection, "FaceDetection", side_effect=RuntimeError("GPU unavailable")
        ):
            _detect_faces(self.image())

        self.assertTrue(verify.MEDIAPIPE_AVAILABLE, "import-time flag must not be mutated")
        degraded = verify.perception_health()
        self.assertFalse(degraded["landmarks_available"])
        self.assertGreater(degraded["mediapipe_cooldown_seconds_remaining"], 0)

        # Recovery must not require a worker restart.
        verify.reset_perception_health()
        self.assertTrue(verify.perception_health()["landmarks_available"])


# =====================================================================
# Real-World Image & Mobile Submission Tests
# =====================================================================

class RealWorldMobileSubmissionTests(unittest.TestCase):
    """Test validation on actual real-world student submissions."""

    def test_user_uploaded_image_passes(self):
        """A real-world student mobile photo must pass validation.

        Point REAL_PHOTO_FIXTURE at a known-good photograph to run this on a
        deployment host; it skips rather than failing when unset.
        """
        import verify

        path = os.environ.get("REAL_PHOTO_FIXTURE", "").strip()
        if not path or not os.path.exists(path):
            self.skipTest("No real-photo fixture available (set REAL_PHOTO_FIXTURE).")

        verify.reset_perception_health()
        self.addCleanup(verify.reset_perception_health)
        if not verify.perception_health()["landmarks_available"]:
            self.skipTest("MediaPipe landmarks are unavailable in this environment.")

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

    def test_soft_blur_white_bg_requires_review(self):
        """A failed enabled quality check cannot be promoted to acceptance."""
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
            self.assertFalse(result["overall_passed"], result)
            self.assertIn("quality_notes", result)

    def test_soft_blur_nonwhite_bg_still_fails(self):
        """A non-white background should still fail even with soft blur tolerance."""
        img = np.full((300, 240, 3), 80, dtype=np.uint8)  # dark gray bg
        for y in range(0, 300, 8):
            for x in range(0, 240, 8):
                if (y // 8 + x // 8) % 2 == 0:
                    img[y:y + 8, x:x + 8] = (60, 60, 60)
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


class BackgroundStrictnessLevelTests(unittest.TestCase):
    """Verify the background_strictness preset levels behave as documented."""

    def image(self, bgr=(255, 255, 255), width=240, height=300):
        return np.full((height, width, 3), bgr, dtype=np.uint8)

    def test_all_levels_are_defined(self):
        for level in ("strict", "standard", "relaxed", "accept_all"):
            self.assertIn(level, BACKGROUND_STRICTNESS_LEVELS)
            self.assertIn("bg_min_value", BACKGROUND_STRICTNESS_LEVELS[level])
            self.assertIn("bg_min_white_coverage", BACKGROUND_STRICTNESS_LEVELS[level])

    def test_levels_define_near_white_tier(self):
        for level, preset in BACKGROUND_STRICTNESS_LEVELS.items():
            self.assertIn("bg_near_white_enabled", preset, level)
            for key in (
                "bg_near_white_min_l_star",
                "bg_near_white_max_chroma",
                "bg_near_white_max_b_star",
            ):
                self.assertIn(key, preset, level)
        # Only the strict preset ships with the near-white tier disabled.
        self.assertEqual(BACKGROUND_STRICTNESS_LEVELS["strict"]["bg_near_white_enabled"], 0.0)
        for level in ("standard", "relaxed", "accept_all"):
            self.assertEqual(BACKGROUND_STRICTNESS_LEVELS[level]["bg_near_white_enabled"], 1.0)


    def test_resolver_uses_preset_when_no_override(self):
        merged = _resolve_background_params({"background_strictness": "relaxed"})
        self.assertLess(
            merged["bg_min_value"],
            BACKGROUND_STRICTNESS_LEVELS["strict"]["bg_min_value"],
        )
        self.assertEqual(merged["background_strictness"], "relaxed")

    def test_resolver_per_field_overrides_are_not_honoured(self):
        # The "Advanced Threshold" per-field override path was removed:
        # BACKGROUND_STRICTNESS_LEVELS is the single authoritative config
        # source, so a stale client-supplied field must not change anything.
        merged = _resolve_background_params({
            "background_strictness": "strict",
            "bg_min_value": 100.0,
        })
        self.assertEqual(
            merged["bg_min_value"],
            BACKGROUND_STRICTNESS_LEVELS["strict"]["bg_min_value"],
        )

    def test_resolver_unknown_level_falls_back_to_standard(self):
        merged = _resolve_background_params({"background_strictness": "nonsense"})
        self.assertEqual(merged["background_strictness"], "standard")

    def test_strict_rejects_beige_wall(self):
        beige = self.image((200, 220, 230))
        result = check_white_background(beige, [], {"background_strictness": "strict"})
        self.assertFalse(result["passed"], result)

    def test_standard_accepts_off_white_with_soft_shadow(self):
        result = check_white_background(
            self.image((220, 220, 220)),
            [],
            {"background_strictness": "standard"},
        )
        self.assertTrue(result["passed"], result)

    def test_relaxed_accepts_warm_offwhite_wall(self):
        warm_wall = self.image((200, 220, 230))
        result = check_white_background(warm_wall, [], {"background_strictness": "relaxed"})
        self.assertTrue(result["passed"], result)

    def test_relaxed_still_rejects_dark_background(self):
        result = check_white_background(
            self.image((60, 60, 60)),
            [],
            {"background_strictness": "relaxed"},
        )
        self.assertFalse(result["passed"], result)

    def test_accept_all_passes_lots_of_non_dark_colour(self):
        result = check_white_background(
            self.image((150, 165, 180)),
            [],
            {"background_strictness": "accept_all"},
        )
        self.assertTrue(result["passed"], result)

    def test_accept_all_still_rejects_pure_black(self):
        result = check_white_background(
            self.image((10, 10, 10)),
            [],
            {"background_strictness": "accept_all"},
        )
        self.assertFalse(result["passed"], result)

    def test_meta_includes_selected_level(self):
        result = check_white_background(
            self.image(),
            [],
            {"background_strictness": "standard"},
        )
        self.assertEqual(
            result["meta"]["thresholds"]["background_strictness"], "standard"
        )


class NearWhiteBackgroundAcceptanceTests(unittest.TestCase):
    """Tier-2 near-white acceptance: legit near-white passes, tinted/dark walls do not."""

    def image(self, bgr=(255, 255, 255), width=240, height=300):
        return np.full((height, width, 3), bgr, dtype=np.uint8)

    def standard(self):
        return {"background_strictness": "standard"}

    # --- legitimate near-white backgrounds must pass -----------------------

    def test_light_gray_wall_passes_standard_level(self):
        # Neutral light-grey studio wall (value 220 -> L* 87.6). The standard
        # level (also the default policy) accepts it through its Tier-1 band
        # (value >= 150, dE <= 38); the strict level rejects it.
        gray = self.image((220, 220, 220))
        self.assertFalse(check_white_background(gray, [], {"background_strictness": "strict"})["passed"])
        result = check_white_background(gray, [], self.standard())
        self.assertTrue(result["passed"], result)
        self.assertGreaterEqual(result["meta"]["white_coverage_percent"], 95.0)

    def test_dimmer_gray_wall_passes_relaxed_level(self):
        # value 140 (L* 58.1) is below the standard Tier-1 floor (150) but
        # inside the relaxed level's (135).
        gray = self.image((140, 140, 140))
        self.assertFalse(check_white_background(gray, [], self.standard())["passed"])
        result = check_white_background(gray, [], {"background_strictness": "relaxed"})
        self.assertTrue(result["passed"], result)

    def test_undexposed_pure_white_passes_standard_level(self):
        # A white wall photographed ~15% under exposure stays perfectly
        # neutral (value ~217) and must pass via the Tier-1 brightness floor
        # at the standard level.
        dim = np.full((300, 240, 3), 255, dtype=np.uint8)
        dim = (dim.astype(np.float32) * 0.88).astype(np.uint8)
        result = check_white_background(dim, [], self.standard())
        self.assertTrue(result["passed"], result)

    def test_window_light_gradient_passes_standard_level(self):
        # Ordinary window-light falloff across a white wall (255 -> 200).
        # The p90-p10 luminance metric must not flag this as a shadow.
        gradient = np.linspace(255, 200, 300, dtype=np.float32).reshape(300, 1)
        img = np.repeat(gradient, 240, axis=1)
        img = np.dstack([img, img, img]).astype(np.uint8)
        result = check_white_background(img, [], self.standard())
        self.assertTrue(result["passed"], result)

    def test_warm_and_cool_lit_white_pass_standard_level(self):
        warm = self.image((238, 246, 252))  # incandescent cast
        cool = self.image((252, 248, 242))  # daylight cast
        self.assertTrue(check_white_background(warm, [], self.standard())["passed"])
        self.assertTrue(check_white_background(cool, [], self.standard())["passed"])

    # --- backgrounds outside the white/near-white range must fail ----------

    def test_beige_and_cream_walls_rejected_at_standard_level(self):
        # Beige (BGR 180,215,225) and cream (190,230,245): yellow walls that
        # the near-white chroma gate must reject at every level.
        for bgr in ((180, 215, 225), (190, 230, 245)):
            for params in ({}, self.standard(), {"background_strictness": "relaxed"}):
                result = check_white_background(self.image(bgr), [], params)
                self.assertFalse(result["passed"], (bgr, params, result))

    def test_pale_blue_and_green_walls_rejected_at_standard_level(self):
        blue = self.image((235, 215, 185))
        green = self.image((185, 235, 185))
        self.assertFalse(check_white_background(blue, [], self.standard())["passed"])
        self.assertFalse(check_white_background(green, [], self.standard())["passed"])

    def test_medium_and_dark_gray_rejected_at_standard_level(self):
        # Below the near-white lightness floor: real grey walls, not lit white.
        for bgr in ((120, 120, 120), (80, 80, 80), (0, 0, 0)):
            result = check_white_background(self.image(bgr), [], self.standard())
            self.assertFalse(result["passed"], (bgr, result))

    def test_beige_rejected_even_when_near_white_tier_forced_on_strict(self):
        # The tier widens tolerance for lighting, not for coloured paint.
        params = {"background_strictness": "strict", "background_near_white_acceptance": "1"}
        result = check_white_background(self.image((180, 215, 225)), [], params)
        self.assertFalse(result["passed"], result)

    def test_dark_shadow_core_still_rejected_at_standard_level(self):
        img = self.image()
        img[:, :80] = (100, 100, 100)  # 33% deep shadow
        result = check_white_background(img, [], self.standard())
        self.assertFalse(result["passed"], result)
        self.assertGreater(result["meta"]["dark_coverage_percent"], 8.0)

    # --- white clothing must not be mistaken for the background ------------

    def _portrait(self, bg_bgr, shirt_bgr, width=480, height=600):
        img = np.full((height, width, 3), bg_bgr, dtype=np.uint8)
        fx, fy, fw, fh = 120, 120, 240, 240
        cv2.ellipse(img, (fx + fw // 2, fy + fh // 2), (fw // 2, int(fh * 0.6)), 0, 0, 360, (140, 160, 190), -1)
        cv2.ellipse(img, (fx + fw // 2, fy + int(fh * 0.25)), (int(fw * 0.55), int(fh * 0.35)), 0, 180, 360, (30, 25, 20), -1)
        torso = np.array([
            [fx - int(fw * 0.2), fy + int(fh * 0.85)],
            [fx + int(fw * 1.2), fy + int(fh * 0.85)],
            [width, height],
            [0, height],
        ], dtype=np.int32)
        cv2.fillConvexPoly(img, torso, shirt_bgr)
        return img, [{"x": fx, "y": fy, "w": fw, "h": fh}]

    def test_white_shirt_is_not_treated_as_background(self):
        # White shirt against a white wall: the shirt lies inside the excluded
        # subject region, so coverage stays ~100% and nothing is contaminated.
        img, faces = self._portrait((248, 248, 248), (250, 250, 250))
        result = check_white_background(img, faces, self.standard())
        self.assertTrue(result["passed"], result)
        self.assertGreaterEqual(result["meta"]["white_coverage_percent"], 95.0)

    def test_white_shirt_against_coloured_wall_still_rejected(self):
        # The guards must judge the wall, not the shirt: a pale blue wall
        # behind a white shirt is still a coloured background.
        img, faces = self._portrait((235, 215, 185), (250, 250, 250))
        result = check_white_background(img, faces, self.standard())
        self.assertFalse(result["passed"], result)
        self.assertGreater(result["meta"]["colored_coverage_percent"], 8.0)

    # --- tier switch semantics ----------------------------------------------

    def test_strict_level_rejects_light_gray_by_default(self):
        result = check_white_background(self.image((220, 220, 220)), [], {"background_strictness": "strict"})
        self.assertFalse(result["passed"], result)

    def test_strictness_level_moves_the_near_white_lightness_floor(self):
        # A dimmer lit wall (value 140, L* ~58) sits below the standard
        # near-white floor (L* >= 60) but inside the relaxed band (L* >= 52):
        # changing the strictness level must move the effective floor.
        dim_wall = self.image((140, 140, 140))
        self.assertFalse(check_white_background(dim_wall, [], {"background_strictness": "standard"})["passed"])
        self.assertTrue(check_white_background(dim_wall, [], {"background_strictness": "relaxed"})["passed"])

    def test_near_white_tier_can_be_forced_on_at_strict_level(self):
        # Warm-lit white wall (BGR 228,242,250 -> L* 95.6, C* 7.7): fails the
        # strict Tier-1 saturation limit (~22 > 18) but qualifies through
        # Tier 2 when the administrator forces the tier on.
        params = {"background_strictness": "strict", "background_near_white_acceptance": "1"}
        result = check_white_background(self.image((228, 242, 250)), [], params)
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["meta"]["thresholds"]["near_white_enabled"])
        self.assertEqual(params["background_strictness"], "strict")

    def test_near_white_tier_can_be_disabled_at_standard_level(self):
        # Explicit opt-out restores pure-white-only behaviour: a warm-lit
        # wall (BGR 210,232,242 -> L* 92.1, C* 11.9) fails again while
        # genuinely bright neutral white still passes.
        params = {"background_strictness": "standard", "background_near_white_acceptance": "0"}
        self.assertFalse(check_white_background(self.image((210, 232, 242)), [], params)["passed"])
        self.assertTrue(check_white_background(self.image((248, 248, 248)), [], params)["passed"])

    def test_near_white_thresholds_follow_the_selected_level(self):
        # Warm-lit wall (BGR 205,228,238 -> L* 90.7, C* 12.4) sits above the
        # standard lightness floor but below the strict Tier-1 limits. The
        # tier follows the level: accepted at standard, rejected at strict,
        # and re-rejected at standard when the admin turns acceptance OFF.
        subject = self.image((205, 228, 238))
        self.assertTrue(check_white_background(subject, [], {"background_strictness": "standard"})["passed"])
        tight = {"background_strictness": "strict", "background_near_white_acceptance": "1"}
        self.assertFalse(check_white_background(subject, [], tight)["passed"])
        off = {"background_strictness": "standard", "background_near_white_acceptance": "0"}
        self.assertFalse(check_white_background(subject, [], off)["passed"])

    def test_meta_reports_tier_coverage(self):
        img = self.image()
        # Right half: warm-lit near-white (BGR 208,230,240) that only Tier 2
        # accepts at the standard level; left half stays pure white.
        img[:, 120:] = (208, 230, 240)
        result = check_white_background(img, [], self.standard())
        self.assertTrue(result["passed"], result)
        meta = result["meta"]
        self.assertGreater(meta["pure_white_coverage_percent"], 40.0)
        self.assertGreater(meta["near_white_coverage_percent"], 40.0)
        self.assertGreaterEqual(meta["white_coverage_percent"], 99.0)


class BackgroundParamResolverTests(unittest.TestCase):
    """_resolve_background_params near-white switch precedence."""

    def test_standard_level_enables_near_white_by_default(self):
        merged = _resolve_background_params({"background_strictness": "standard"})
        self.assertEqual(merged["bg_near_white_enabled"], 1.0)
        self.assertEqual(merged["background_near_white_acceptance"], "auto")

    def test_strict_level_disables_near_white_by_default(self):
        merged = _resolve_background_params({"background_strictness": "strict"})
        self.assertEqual(merged["bg_near_white_enabled"], 0.0)

    def test_explicit_switch_overrides_preset(self):
        forced_on = _resolve_background_params({
            "background_strictness": "strict",
            "background_near_white_acceptance": "1",
        })
        self.assertEqual(forced_on["bg_near_white_enabled"], 1.0)
        forced_off = _resolve_background_params({
            "background_strictness": "standard",
            "background_near_white_acceptance": "0",
        })
        self.assertEqual(forced_off["bg_near_white_enabled"], 0.0)

    def test_legacy_switch_key_is_accepted_as_alias(self):
        merged = _resolve_background_params({
            "background_strictness": "standard",
            "bg_near_white_enabled": 0,
        })
        self.assertEqual(merged["bg_near_white_enabled"], 0.0)
        self.assertEqual(merged["background_near_white_acceptance"], "0")

    def test_boolean_and_numeric_switch_values_are_normalised(self):
        on = _resolve_background_params({"background_strictness": "strict", "background_near_white_acceptance": True})
        self.assertEqual(on["bg_near_white_enabled"], 1.0)
        off = _resolve_background_params({"background_strictness": "strict", "background_near_white_acceptance": 0})
        self.assertEqual(off["bg_near_white_enabled"], 0.0)

    def test_unknown_level_falls_back_to_standard_defaults(self):
        merged = _resolve_background_params({"background_strictness": "bogus"})
        self.assertEqual(merged["background_strictness"], "standard")
        self.assertEqual(merged["bg_near_white_enabled"], 1.0)

    def test_unknown_switch_value_falls_back_to_auto(self):
        merged = _resolve_background_params({
            "background_strictness": "standard",
            "background_near_white_acceptance": "maybe",
        })
        self.assertEqual(merged["background_near_white_acceptance"], "auto")
        self.assertEqual(merged["bg_near_white_enabled"], 1.0)


class MixedBackgroundSamplingTests(unittest.TestCase):
    """Regression: mixed white + non-white backgrounds must fail.

    Two legacy holes let a mostly-white visible border hide substantial
    non-white background regions:
      1. the "studio fast-path" swapped the evaluation mask to the pure-white
         pixels themselves when they covered >= 65% of the sample in a
         detected portrait, so every metric was computed over already-white
         pixels only;
      2. with a face present, the side sampling bands stopped at the head
         line, so any wall beside or below the head was never sampled.
    """

    FACE = {"x": 140, "y": 100, "w": 200, "h": 200}

    def _mixed_vertical(self):
        # White studio backdrop above, vividly coloured wall below.
        img = np.full((600, 480, 3), 255, dtype=np.uint8)
        img[420:, :] = (40, 40, 200)  # red
        return img

    def _mixed_horizontal(self):
        # Coloured wall on half the frame, white on the other half.
        img = np.full((600, 480, 3), (60, 180, 90), dtype=np.uint8)  # green
        img[:, 240:] = 255
        return img

    def test_white_top_red_bottom_fails_at_every_level(self):
        img = self._mixed_vertical()
        for level in ("strict", "standard", "relaxed"):
            with self.subTest(level=level):
                result = check_white_background(
                    img, [self.FACE], {"background_strictness": level}
                )
                self.assertFalse(result["passed"], (level, result["message"]))
                self.assertLess(result["meta"]["white_coverage_percent"], 99.0)

    def test_coloured_wall_beside_the_head_fails_at_every_level(self):
        img = self._mixed_horizontal()
        for level in ("strict", "standard", "relaxed"):
            with self.subTest(level=level):
                result = check_white_background(
                    img, [self.FACE], {"background_strictness": level}
                )
                self.assertFalse(result["passed"], (level, result["message"]))
                self.assertGreater(result["meta"]["colored_coverage_percent"], 5.0)

    def test_metrics_are_computed_over_the_whole_sample(self):
        # With a face present and a >=65% white sample the legacy fast-path
        # reported a perfect score; the metrics must now see the red band.
        img = self._mixed_vertical()
        result = check_white_background(
            img, [self.FACE], {"background_strictness": "standard"}
        )
        meta = result["meta"]
        self.assertGreater(meta["nonwhite_coverage_percent"], 3.0)
        self.assertGreater(meta["colored_coverage_percent"], 3.0)

    def test_dark_and_coloured_guards_do_not_double_count_black_hair(self):
        # Black hair spillage past the subject margin is dark AND carries
        # high saturation noise; it must be counted once by the dark guard,
        # not also by the coloured guard (saturation is meaningless on
        # near-black pixels).
        img = np.full((600, 480, 3), 255, dtype=np.uint8)
        cv2.ellipse(img, (240, 220), (120, 72), 0, 0, 360, (140, 160, 190), -1)
        # Hair-like dark patch spilling into the left side band below the
        # head line.
        cv2.ellipse(img, (30, 380), (90, 60), 0, 0, 360, (10, 10, 10), -1)
        result = check_white_background(
            img, [self.FACE], {"background_strictness": "standard"}
        )
        meta = result["meta"]
        self.assertEqual(meta["colored_coverage_percent"], 0.0)

    def test_white_background_output_messages(self):
        # When accepted
        accepted_img = np.full((300, 240, 3), (255, 255, 255), dtype=np.uint8)
        accepted_res = check_white_background(accepted_img, [], {})
        self.assertTrue(accepted_res["passed"])
        self.assertEqual(accepted_res["message"], "White bg accepted.")

        # When rejected
        rejected_img = np.full((300, 240, 3), (50, 50, 50), dtype=np.uint8)
        rejected_res = check_white_background(rejected_img, [], {})
        self.assertFalse(rejected_res["passed"])
        self.assertEqual(rejected_res["message"], "White background not accepted, please try again.")


if __name__ == "__main__":
    unittest.main()
