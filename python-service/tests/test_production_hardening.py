"""Regression tests for the production-hardening fixes.

Each test here maps to a specific defect found during the pre-deployment
audit. They are deliberately fast and hermetic: no network, no model
downloads, no user-specific fixture paths.
"""

import io
import json
import os
import pathlib
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image as PILImage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import verify  # noqa: E402
from service_client import (  # noqa: E402
    TEST_SERVICE_TOKEN,
    authenticated_client,
    unauthenticated_client,
)
from verify import ImageValidationError, run_checks  # noqa: E402


def _image_bytes(width=500, height=700, color=(255, 255, 255), fmt="JPEG"):
    buf = io.BytesIO()
    PILImage.new("RGB", (width, height), color).save(buf, format=fmt)
    return buf.getvalue()


def _photo_field(width=500, height=700, fmt="JPEG", name="test.jpg"):
    return (io.BytesIO(_image_bytes(width, height, fmt=fmt)), name)


# ---------------------------------------------------------------------------
# 1. run_checks() aggregation (the crash that blocked every verification)
# ---------------------------------------------------------------------------

class RunChecksAggregationTests(unittest.TestCase):
    """verify.py referenced `response` and `k` before assignment."""

    def test_run_checks_returns_a_response_instead_of_raising(self):
        result = run_checks(_image_bytes(), {"min_resolution": True}, {"min_pass_criteria": 1})
        self.assertIn("overall_passed", result)
        self.assertIn("results", result)
        self.assertIn("masked_failures", result)

    def test_enabled_failures_cannot_be_masked(self):
        """Every enabled admissions criterion must pass."""
        result = run_checks(
            _image_bytes(width=100, height=100),
            {"min_resolution": True, "passport_ratio": True},
            {"min_pass_criteria": 1},
        )
        self.assertFalse(result["results"]["min_resolution"]["passed"])
        self.assertFalse(result["overall_passed"])
        self.assertEqual(result["masked_failures"], [])
        self.assertIn("min_resolution", result["failed_criteria"])

    def test_strict_mode_blocks_any_failure(self):
        result = run_checks(
            _image_bytes(width=100, height=100),
            {"min_resolution": True},
            {"min_pass_criteria": 0, "strict_all_criteria": 1},
        )
        self.assertFalse(result["overall_passed"])
        self.assertEqual(result["masked_failures"], [])

    def test_zero_enabled_criteria_passes(self):
        result = run_checks(_image_bytes(), {}, {"min_pass_criteria": 4})
        self.assertTrue(result["overall_passed"])
        self.assertEqual(result["total_criteria"], 0)

    def test_failing_check_does_not_leak_internals(self):
        with patch.dict(verify.CHECK_REGISTRY, {"min_resolution": lambda *a: 1 / 0}):
            result = run_checks(_image_bytes(), {"min_resolution": True}, {})
        message = result["results"]["min_resolution"]["message"]
        self.assertNotIn("division", message.lower())
        self.assertEqual(result["results"]["min_resolution"]["meta"]["error"], "check_failed")


# ---------------------------------------------------------------------------
# 2. Image validation / decompression-bomb protection
# ---------------------------------------------------------------------------

class ImageValidationTests(unittest.TestCase):

    def test_empty_payload_rejected(self):
        with self.assertRaises(ImageValidationError):
            verify._load_image(b"")

    def test_corrupt_payload_rejected(self):
        with self.assertRaises(ImageValidationError):
            verify._load_image(b"this is definitely not an image")

    def test_disallowed_format_rejected(self):
        with self.assertRaises(ImageValidationError) as ctx:
            verify._load_image(_image_bytes(fmt="BMP"))
        self.assertIn("Unsupported image format", str(ctx.exception))

    def test_allowed_formats_accepted(self):
        for fmt in ("JPEG", "PNG", "WEBP"):
            with self.subTest(fmt=fmt):
                bgr = verify._load_image(_image_bytes(60, 80, fmt=fmt))
                self.assertEqual(bgr.shape[:2], (80, 60))

    def test_pixel_budget_enforced(self):
        with patch.object(verify, "MAX_IMAGE_PIXELS", 1000):
            with self.assertRaises(ImageValidationError) as ctx:
                verify._load_image(_image_bytes(500, 700))
        self.assertIn("too large", str(ctx.exception).lower())

    def test_run_checks_reports_validation_error_cleanly(self):
        result = run_checks(b"not an image", {"min_resolution": True}, {})
        self.assertFalse(result["overall_passed"])
        self.assertEqual(result["error"], "Unsupported or corrupt image file.")
        self.assertNotIn("Traceback", result["error"])


# ---------------------------------------------------------------------------
# 3. Service authentication contract
# ---------------------------------------------------------------------------

class ServiceAuthTests(unittest.TestCase):

    def setUp(self):
        from app import app
        self.app = app
        self.client = authenticated_client(app)
        self.anon = unauthenticated_client(app)

    def test_health_is_public(self):
        self.assertEqual(self.anon.get("/health").status_code, 200)

    def test_untrusted_host_is_rejected(self):
        response = self.client.get("/health", headers={"Host": "attacker.example"})
        self.assertEqual(response.status_code, 400)

    def test_verify_requires_token(self):
        response = self.anon.post("/verify", data={"photo": _photo_field()},
                                  content_type="multipart/form-data")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.get_json()["error"], "Unauthorized")

    def test_wrong_token_rejected(self):
        client = self.app.test_client()
        client.environ_base["HTTP_AUTHORIZATION"] = "Bearer wrong-token"
        self.assertEqual(client.get("/warmup").status_code, 401)

    def test_missing_server_secret_returns_401(self):
        """An unconfigured process must reject callers, not disable auth."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PORTAL_SHARED_SECRET", None)
            response = self.client.get("/warmup")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.get_json()["error"], "Unauthorized")
        os.environ["PORTAL_SHARED_SECRET"] = TEST_SERVICE_TOKEN

    def test_authenticated_request_is_accepted(self):
        with patch("app.get_tie_detector"), patch("app.get_rembg_session", return_value=None):
            self.assertEqual(self.client.get("/warmup").status_code, 200)


# ---------------------------------------------------------------------------
# 4. Request payload validation at the HTTP boundary
# ---------------------------------------------------------------------------

class RequestValidationTests(unittest.TestCase):

    def setUp(self):
        from app import app
        self.app = app
        self.client = authenticated_client(app)

    def _post(self, **fields):
        data = {"photo": _photo_field()}
        data.update(fields)
        return self.client.post("/verify", data=data, content_type="multipart/form-data")

    def test_non_object_criteria_rejected(self):
        for raw in ("[]", "null", '"white_background"', "42"):
            with self.subTest(raw=raw):
                response = self._post(criteria=raw)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["code"], "INVALID_JSON_TYPE")

    def test_malformed_criteria_rejected(self):
        response = self._post(criteria="not-json{{{")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "INVALID_JSON")

    def test_non_object_params_fall_back_to_defaults(self):
        response = self._post(criteria='{"min_resolution": true}', params="[]")
        self.assertEqual(response.status_code, 200)
        self.assertIn("min_resolution", response.get_json()["results"])

    def test_oversized_json_field_rejected(self):
        response = self._post(criteria=json.dumps({"x" * 50: True for _ in range(1)}) + " " * 30_000)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "FIELD_TOO_LARGE")

    def test_oversized_upload_returns_json_not_html(self):
        original = self.app.config["MAX_CONTENT_LENGTH"]
        self.app.config["MAX_CONTENT_LENGTH"] = 1024
        try:
            response = self.client.post(
                "/verify",
                data={"photo": (io.BytesIO(b"0" * 5000), "big.jpg")},
                content_type="multipart/form-data",
            )
        finally:
            self.app.config["MAX_CONTENT_LENGTH"] = original
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.content_type.split(";")[0], "application/json")
        self.assertIn("too large", response.get_json()["error"].lower())

    def test_unknown_route_returns_json(self):
        response = self.client.get("/does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.content_type.split(";")[0], "application/json")

    def test_corrupt_image_returns_422_json(self):
        response = self.client.post(
            "/verify",
            data={"photo": (io.BytesIO(b"garbage"), "x.jpg"), "criteria": '{"min_resolution": true}'},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("error", response.get_json())


# ---------------------------------------------------------------------------
# 5. /edit-photo resource bounds and error hygiene
# ---------------------------------------------------------------------------

class EditPhotoBoundsTests(unittest.TestCase):

    def setUp(self):
        from app import app
        self.client = authenticated_client(app)

    def _edit(self, **fields):
        data = {"photo": _photo_field(120, 160)}
        data.update(fields)
        return self.client.post("/edit-photo", data=data, content_type="multipart/form-data")

    def test_oversized_resize_rejected(self):
        response = self._edit(width="100000", height="100000")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "INVALID_IMAGE")

    def test_negative_resize_rejected(self):
        response = self._edit(width="-10", height="-10")
        self.assertEqual(response.status_code, 400)

    def test_non_numeric_resize_rejected(self):
        response = self._edit(width="abc", height="def")
        self.assertEqual(response.status_code, 400)

    def test_reasonable_resize_succeeds(self):
        response = self._edit(width="200", height="260")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        self.assertEqual(PILImage.open(io.BytesIO(response.data)).size, (200, 260))

    def test_corrupt_image_returns_400_without_stack_details(self):
        response = self.client.post(
            "/edit-photo",
            data={"photo": (io.BytesIO(b"nope"), "x.jpg")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("Traceback", response.get_json()["error"])

    def test_edit_photo_without_token_is_forbidden(self):
        from app import app
        response = unauthenticated_client(app).post(
            "/edit-photo",
            data={"photo": _photo_field(80, 100)},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 401)

    def test_bg_color_is_applied_to_transparent_regions(self):
        import app as service_app
        photo = PILImage.new("RGB", (80, 100), (0, 180, 0))
        cutout = PILImage.new("RGBA", (80, 100), (20, 30, 40, 255))
        pixels = cutout.load()
        for y in range(100):
            for x in range(80):
                if x < 8 or x > 71 or y < 8 or y > 91:
                    pixels[x, y] = (0, 0, 0, 0)
        with patch("app.get_subject_cutout", return_value=cutout):
            result = service_app.replace_background_color(photo, "#ef4444")
        corner = result.getpixel((2, 2))
        center = result.getpixel((40, 50))
        self.assertGreater(corner[0], 180)
        self.assertLess(corner[1], 90)
        self.assertLess(abs(center[0] - 20) + abs(center[1] - 30) + abs(center[2] - 40), 40)

    def test_edit_photo_forwards_bg_color(self):
        sentinel = PILImage.new("RGB", (120, 160), (10, 20, 30))
        with patch("app.replace_background_color", return_value=sentinel) as replace:
            response = self._edit(bg_color="#3b82f6")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        replace.assert_called_once()
        self.assertEqual(replace.call_args.args[1], "#3b82f6")


# ---------------------------------------------------------------------------
# 6. Readiness reporting
# ---------------------------------------------------------------------------

class ReadinessTests(unittest.TestCase):

    def setUp(self):
        from app import app
        self.client = authenticated_client(app)

    def test_ready_when_models_load(self):
        healthy = {
            "mediapipe_imported": True,
            "landmarks_available": True,
            "mediapipe_failures": 0,
            "mediapipe_cooldown_seconds_remaining": 0.0,
            "mediapipe_last_error": None,
        }
        with patch("app.get_tie_detector"), \
             patch("app.get_rembg_session", return_value=object()), \
             patch("app.perception_health", return_value=healthy):
            response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ready")

    def test_not_ready_when_tie_model_missing(self):
        with patch("app.get_tie_detector", side_effect=RuntimeError("no model")), \
             patch("app.get_rembg_session", return_value=None), \
             patch.dict(os.environ, {"REQUIRE_TIE_MODEL_READY": "1"}):
            response = self.client.get("/ready")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["status"], "not_ready")

    def test_readiness_can_be_relaxed(self):
        with patch("app.get_tie_detector", side_effect=RuntimeError("no model")), \
             patch("app.get_rembg_session", return_value=None), \
             patch.dict(os.environ, {
                 "REQUIRE_TIE_MODEL_READY": "0",
                 "REQUIRE_REMBG_READY": "0",
                 "REQUIRE_LANDMARKS_READY": "0",
             }):
            response = self.client.get("/ready")
        self.assertEqual(response.status_code, 200)

    def test_ready_requires_authentication(self):
        from app import app
        self.assertEqual(unauthenticated_client(app).get("/ready").status_code, 401)


# ---------------------------------------------------------------------------
# 7. Environment configuration validation
# ---------------------------------------------------------------------------

class EnvValidationTests(unittest.TestCase):

    def test_tie_detector_rejects_non_numeric_threshold(self):
        from tie_detector import _float_env
        with patch.dict(os.environ, {"TIE_COCO_THRESHOLD": "banana"}):
            with self.assertRaises(ValueError):
                _float_env("TIE_COCO_THRESHOLD", 0.5, minimum=0.01, maximum=0.99)

    def test_tie_detector_rejects_out_of_range_threshold(self):
        from tie_detector import _float_env
        for bad in ("-1", "2.5", "nan", "inf"):
            with self.subTest(bad=bad), patch.dict(os.environ, {"TIE_COCO_THRESHOLD": bad}):
                with self.assertRaises(ValueError):
                    _float_env("TIE_COCO_THRESHOLD", 0.5, minimum=0.01, maximum=0.99)

    def test_tie_detector_accepts_valid_threshold(self):
        from tie_detector import _float_env
        with patch.dict(os.environ, {"TIE_COCO_THRESHOLD": "0.75"}):
            self.assertAlmostEqual(
                _float_env("TIE_COCO_THRESHOLD", 0.5, minimum=0.01, maximum=0.99), 0.75
            )

    def test_visibility_estimator_falls_back_on_bad_env(self):
        from tie_visibility import UpperBodyVisibilityEstimator
        with patch.dict(os.environ, {"TIE_MIN_FACE_HEIGHT": "not-a-number"}):
            self.assertEqual(UpperBodyVisibilityEstimator().min_face_height_px, 80)

    def test_visibility_estimator_rejects_bad_padding(self):
        from tie_visibility import UpperBodyVisibilityEstimator
        with self.assertRaises(ValueError):
            UpperBodyVisibilityEstimator(horizontal_face_padding=0)

    def test_int_env_falls_back_on_invalid_value(self):
        with patch.dict(os.environ, {"MAX_IMAGE_PIXELS": "huge"}):
            self.assertEqual(
                verify._int_env("MAX_IMAGE_PIXELS", 123, minimum=1, maximum=1000), 123
            )

    def test_int_env_rejects_out_of_range(self):
        with patch.dict(os.environ, {"MAX_IMAGE_PIXELS": "999999"}):
            self.assertEqual(
                verify._int_env("MAX_IMAGE_PIXELS", 123, minimum=1, maximum=1000), 123
            )


# ---------------------------------------------------------------------------
# 8. Detector loader concurrency
# ---------------------------------------------------------------------------

class DetectorLoaderTests(unittest.TestCase):

    def test_concurrent_first_calls_build_only_one_detector(self):
        import threading
        import tie_detector

        tie_detector.get_tie_detector.cache_clear()
        with patch("tie_detector.CocoTieDetector") as mock_coco, \
             patch.dict(os.environ, {"TIE_DETECTOR_BACKEND": "coco"}):
            results = []

            def _load():
                results.append(tie_detector.get_tie_detector())

            threads = [threading.Thread(target=_load) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(mock_coco.call_count, 1)
        self.assertEqual(len(results), 8)
        tie_detector.get_tie_detector.cache_clear()

    def test_cache_clear_api_is_preserved(self):
        import tie_detector
        self.assertTrue(hasattr(tie_detector.get_tie_detector, "cache_clear"))
        self.assertTrue(hasattr(tie_detector.get_tie_detector, "cache_info"))


class EditPhotoSpeedPathTests(unittest.TestCase):
    """The portal blocks on /edit-photo after a pass; that path must stay cheap."""

    def test_default_rembg_model_is_the_portable_u2netp(self):
        import app as service_app
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("REMBG_MODEL", None)
            self.assertEqual(service_app._rembg_model_name(), "u2netp")

    def test_unknown_rembg_model_falls_back_to_u2netp(self):
        import app as service_app
        with patch.dict("os.environ", {"REMBG_MODEL": "not-a-model"}):
            self.assertEqual(service_app._rembg_model_name(), "u2netp")

    def test_cutout_inference_downscales_large_phone_photos(self):
        import app as service_app
        big = PILImage.new("RGB", (4000, 3000), (240, 240, 240))
        with patch.dict("os.environ", {"EDIT_INFERENCE_MAX_SIDE": "768"}):
            small, orig = service_app._image_for_cutout_inference(big)
        self.assertEqual(orig, (4000, 3000))
        self.assertLessEqual(max(small.size), 768)

    def test_cutout_inference_leaves_passport_sized_photos_alone(self):
        import app as service_app
        photo = PILImage.new("RGB", (600, 750), (240, 240, 240))
        with patch.dict("os.environ", {"EDIT_INFERENCE_MAX_SIDE": "768"}):
            out, orig = service_app._image_for_cutout_inference(photo)
        self.assertIsNone(orig)
        self.assertEqual(out.size, (600, 750))

    def test_white_background_edit_does_not_run_a_second_face_detector(self):
        import app as service_app
        photo = PILImage.new("RGB", (200, 260), (255, 255, 255))
        fake_cutout = PILImage.new("RGBA", (200, 260), (10, 20, 30, 255))
        with patch("app.get_subject_cutout", return_value=fake_cutout), \
             patch("app._detect_faces") as detect:
            service_app.replace_background_color(photo, "#ffffff")
        detect.assert_not_called()


class LandmarkReuseTests(unittest.TestCase):
    def test_face_mesh_runs_once_per_image_for_pose_and_eyes(self):
        import verify
        if not getattr(verify, "MEDIAPIPE_AVAILABLE", False):
            self.skipTest("MediaPipe not available")
        verify.reset_perception_health()
        self.addCleanup(verify.reset_perception_health)
        image = np.full((240, 180, 3), 220, dtype=np.uint8)
        with patch.object(verify, "_ensure_face_mesh_locked") as ensure:
            fake = MagicMock()
            fake.process.return_value.multi_face_landmarks = None
            ensure.return_value = fake
            verify._face_mesh_landmarks(image)
            verify._face_mesh_landmarks(image)
        self.assertEqual(fake.process.call_count, 1)


class TieInferenceReuseTests(unittest.TestCase):
    """Both tie criteria must share the expensive model pass per request."""

    def test_require_and_no_tie_run_detector_once(self):
        import verify
        from tie_detector import TieDetection

        bgr = np.full((700, 500, 3), 220, dtype=np.uint8)
        faces = [{"x": 150, "y": 80, "w": 200, "h": 200}]
        detector = MagicMock()
        detector.detect.return_value = TieDetection(0.95, (220, 310, 280, 490))
        detector.policy = None
        detector.positive_threshold = 0.50
        detector.geometry = {
            "min_width_face_ratio": 0.04,
            "max_width_face_ratio": 0.85,
            "min_height_face_ratio": 0.06,
            "max_height_face_ratio": 1.60,
            "min_top_offset_face_ratio": -0.45,
            "max_top_offset_face_ratio": 1.35,
            "max_center_offset_face_ratio": 0.40,
        }
        with patch.object(verify, "_detect_faces", return_value=faces), \
             patch.object(verify, "get_tie_detector", return_value=detector), \
             patch.object(verify, "_face_mesh_landmarks", return_value=None):
            result = verify.run_checks(
                _image_bytes(),
                {"require_tie": True, "no_tie": True},
                {},
            )

        self.assertIn("require_tie", result["results"])
        self.assertIn("no_tie", result["results"])
        self.assertEqual(detector.detect.call_count, 1)


if __name__ == "__main__":
    unittest.main()
