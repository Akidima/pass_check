"""Authentication-layer tests for the Python verification service.

These tests cover the service-to-service credential only. They do not
exercise tie detection, face detection, or other computer-vision paths.
"""

from __future__ import annotations

import io
import logging
import os
import pathlib
import sys
import unittest
from unittest.mock import patch

from PIL import Image as PILImage

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from service_auth import (  # noqa: E402
    MIN_SECRET_LENGTH,
    ServiceAuthConfigError,
    _log_auth_failure,
    extract_bearer_token,
    reset_auth_failure_log_state_for_tests,
    secrets_match,
    validate_service_auth_config,
)
from service_client import (  # noqa: E402
    TEST_SERVICE_TOKEN,
    authenticated_client,
    unauthenticated_client,
)


def _photo_field(width=80, height=100, name="test.jpg"):
    buf = io.BytesIO()
    PILImage.new("RGB", (width, height), (255, 255, 255)).save(buf, format="JPEG")
    return (io.BytesIO(buf.getvalue()), name)


def _auth_body(response):
    payload = response.get_json()
    return payload if payload is not None else {}, response.get_data(as_text=True)


class BearerParsingTests(unittest.TestCase):
    def test_valid_bearer(self):
        self.assertEqual(extract_bearer_token("Bearer secret-value"), "secret-value")

    def test_scheme_is_case_insensitive(self):
        self.assertEqual(extract_bearer_token("bearer secret-value"), "secret-value")
        self.assertEqual(extract_bearer_token("BEARER secret-value"), "secret-value")

    def test_wrong_scheme(self):
        self.assertIsNone(extract_bearer_token("Basic secret-value"))

    def test_missing_token(self):
        self.assertIsNone(extract_bearer_token("Bearer"))
        self.assertIsNone(extract_bearer_token("Bearer   "))

    def test_empty_or_absent(self):
        self.assertIsNone(extract_bearer_token(""))
        self.assertIsNone(extract_bearer_token(None))


class ConstantTimeCompareTests(unittest.TestCase):
    def test_matching_secret(self):
        self.assertTrue(secrets_match("abc", ("abc",)))

    def test_non_matching_secret(self):
        self.assertFalse(secrets_match("abc", ("xyz",)))

    def test_different_lengths_do_not_match(self):
        self.assertFalse(secrets_match("short", ("a-much-longer-expected-secret",)))

    def test_empty_expected_never_matches(self):
        self.assertFalse(secrets_match("", ()))
        self.assertFalse(secrets_match("anything", ()))

    def test_rotation_accepts_previous_secret(self):
        self.assertTrue(secrets_match("old-secret", ("new-secret", "old-secret")))
        self.assertTrue(secrets_match("new-secret", ("new-secret", "old-secret")))
        self.assertFalse(secrets_match("other", ("new-secret", "old-secret")))


class StartupConfigTests(unittest.TestCase):
    def test_production_fails_when_secret_missing(self):
        env = {
            "APP_ENV": "production",
            "ALLOW_INSECURE_AUTH_START": "0",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("PORTAL_SHARED_SECRET", None)
            with self.assertRaises(ServiceAuthConfigError) as ctx:
                validate_service_auth_config()
        self.assertNotIn("Bearer", str(ctx.exception))
        self.assertNotRegex(str(ctx.exception), r"=.+")

    def test_production_fails_when_secret_is_short(self):
        with patch.dict(
            os.environ,
            {"PORTAL_SHARED_SECRET": "too-short", "APP_ENV": "production"},
            clear=False,
        ):
            with self.assertRaises(ServiceAuthConfigError):
                validate_service_auth_config()

    def test_production_fails_when_secret_is_whitespace(self):
        with patch.dict(
            os.environ,
            {"PORTAL_SHARED_SECRET": "   ", "APP_ENV": "production"},
            clear=False,
        ):
            with self.assertRaises(ServiceAuthConfigError):
                validate_service_auth_config()

    def test_valid_secret_allows_startup(self):
        with patch.dict(os.environ, {"PORTAL_SHARED_SECRET": TEST_SERVICE_TOKEN}):
            validate_service_auth_config()

    def test_development_alone_does_not_bypass_startup(self):
        with patch.dict(os.environ, {"APP_ENV": "development"}, clear=False):
            os.environ.pop("PORTAL_SHARED_SECRET", None)
            os.environ.pop("ALLOW_INSECURE_AUTH_START", None)
            with self.assertRaises(ServiceAuthConfigError):
                validate_service_auth_config()

    def test_insecure_flag_alone_does_not_bypass_startup(self):
        with patch.dict(
            os.environ,
            {"ALLOW_INSECURE_AUTH_START": "1", "APP_ENV": "production"},
            clear=False,
        ):
            os.environ.pop("PORTAL_SHARED_SECRET", None)
            with self.assertRaises(ServiceAuthConfigError):
                validate_service_auth_config()

    def test_explicit_dev_flags_allow_start_without_secret(self):
        with patch.dict(
            os.environ,
            {"APP_ENV": "development", "ALLOW_INSECURE_AUTH_START": "1"},
            clear=False,
        ):
            os.environ.pop("PORTAL_SHARED_SECRET", None)
            validate_service_auth_config()


class ServiceAuthHttpTests(unittest.TestCase):
    def setUp(self):
        from app import app

        reset_auth_failure_log_state_for_tests()
        self.app = app
        self.client = authenticated_client(app)
        self.anon = unauthenticated_client(app)

    def _assert_unauthorized(self, response):
        payload, raw = _auth_body(response)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(payload.get("error"), "Unauthorized")
        self.assertNotIn(TEST_SERVICE_TOKEN, raw)
        self.assertNotIn("PORTAL_SHARED_SECRET", raw)
        self.assertNotIn("Bearer ", raw)

    def test_missing_authorization_header_is_401(self):
        self._assert_unauthorized(self.anon.get("/warmup"))

    def test_wrong_scheme_is_401(self):
        response = self.app.test_client().get(
            "/warmup",
            headers={"Authorization": f"Basic {TEST_SERVICE_TOKEN}"},
        )
        self._assert_unauthorized(response)

    def test_empty_bearer_is_401(self):
        response = self.app.test_client().get(
            "/warmup",
            headers={"Authorization": "Bearer "},
        )
        self._assert_unauthorized(response)

    def test_whitespace_bearer_is_401(self):
        response = self.app.test_client().get(
            "/warmup",
            headers={"Authorization": "Bearer    "},
        )
        self._assert_unauthorized(response)

    def test_invalid_secret_is_401(self):
        response = self.app.test_client().get(
            "/warmup",
            headers={"Authorization": "Bearer definitely-not-the-configured-secret"},
        )
        self._assert_unauthorized(response)

    def test_correct_secret_reaches_endpoint(self):
        with patch("app.get_tie_detector"), patch(
            "app.get_rembg_session", return_value=None
        ):
            response = self.client.get("/warmup")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_legacy_header_still_accepted(self):
        client = self.app.test_client()
        with patch("app.get_tie_detector"), patch(
            "app.get_rembg_session", return_value=None
        ):
            response = client.get(
                "/warmup",
                headers={"X-Service-Token": TEST_SERVICE_TOKEN},
            )
        self.assertEqual(response.status_code, 200)

    def test_broken_authorization_does_not_fall_back_to_legacy(self):
        response = self.app.test_client().get(
            "/warmup",
            headers={
                "Authorization": "Basic not-a-bearer",
                "X-Service-Token": TEST_SERVICE_TOKEN,
            },
        )
        self._assert_unauthorized(response)

    def test_empty_authorization_does_not_fall_back_to_legacy(self):
        response = self.app.test_client().get(
            "/warmup",
            headers={
                "Authorization": "",
                "X-Service-Token": TEST_SERVICE_TOKEN,
            },
        )
        self._assert_unauthorized(response)

    def test_previous_secret_accepted_during_rotation(self):
        previous = "previous-service-token-do-not-use-in-prod"
        with patch.dict(
            os.environ,
            {
                "PORTAL_SHARED_SECRET": TEST_SERVICE_TOKEN,
                "PORTAL_SHARED_SECRET_PREVIOUS": previous,
            },
        ):
            with patch("app.get_tie_detector"), patch(
                "app.get_rembg_session", return_value=None
            ):
                response = self.app.test_client().get(
                    "/warmup",
                    headers={"Authorization": f"Bearer {previous}"},
                )
        self.assertEqual(response.status_code, 200)

    def test_health_is_public(self):
        self.assertEqual(self.anon.get("/health").status_code, 200)

    def test_testing_flag_does_not_bypass_auth(self):
        self.app.config["TESTING"] = True
        self._assert_unauthorized(self.anon.get("/ready"))

    def test_untrusted_headers_are_not_authentication(self):
        response = self.app.test_client().get(
            "/warmup",
            headers={
                "X-Forwarded-User": "portal",
                "X-Authenticated": "true",
                "X-Internal-Request": "1",
            },
        )
        self._assert_unauthorized(response)

    def test_error_body_does_not_echo_credential(self):
        leaked = "leaked-credential-value-must-never-appear"
        response = self.app.test_client().get(
            "/verify",
            headers={"Authorization": f"Bearer {leaked}"},
        )
        payload, raw = _auth_body(response)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(payload.get("error"), "Unauthorized")
        self.assertNotIn(leaked, raw)
        self.assertNotIn("PORTAL_SHARED_SECRET", raw)

    def test_secret_is_never_logged_on_failure(self):
        leaked = "unique-auth-failure-secret-for-log-test"
        reset_auth_failure_log_state_for_tests()
        with self.assertLogs("service_auth", level="WARNING") as captured:
            response = self.app.test_client().get(
                "/warmup",
                headers={"Authorization": f"Bearer {leaked}"},
            )
        self.assertEqual(response.status_code, 401)
        joined = "\n".join(captured.output)
        self.assertIn("Unauthorized request rejected", joined)
        self.assertNotIn(leaked, joined)
        self.assertNotIn(TEST_SERVICE_TOKEN, joined)
        self.assertNotIn("Authorization", joined)
        self.assertNotIn("Bearer", joined)

    def test_first_auth_failure_is_always_logged(self):
        """Rate limiting must not drop the first failure after a reset."""
        reset_auth_failure_log_state_for_tests()
        with self.assertLogs("service_auth", level="WARNING") as captured:
            _log_auth_failure("/warmup")
        self.assertIn("Unauthorized request rejected for /warmup", captured.output[0])

    def test_verify_does_not_process_image_when_unauthenticated(self):
        with patch("app.run_checks") as run_checks, patch(
            "app.load_validated_image"
        ) as load_image:
            response = self.anon.post(
                "/verify",
                data={"photo": _photo_field()},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 401)
        run_checks.assert_not_called()
        load_image.assert_not_called()

    def test_edit_photo_does_not_process_image_when_unauthenticated(self):
        with patch("app.process_photo_edits") as process_edits, patch(
            "app.get_subject_cutout"
        ) as cutout, patch("app.load_validated_image") as load_image:
            response = self.anon.post(
                "/edit-photo",
                data={"photo": _photo_field()},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 401)
        process_edits.assert_not_called()
        cutout.assert_not_called()
        load_image.assert_not_called()

    def test_missing_runtime_secret_does_not_disable_auth(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PORTAL_SHARED_SECRET", None)
            os.environ.pop("PORTAL_SHARED_SECRET_PREVIOUS", None)
            with patch("app.run_checks") as run_checks:
                response = self.client.get("/warmup")
        self.assertEqual(response.status_code, 401)
        run_checks.assert_not_called()
        os.environ["PORTAL_SHARED_SECRET"] = TEST_SERVICE_TOKEN

    def test_short_runtime_secret_is_rejected(self):
        with patch.dict(os.environ, {"PORTAL_SHARED_SECRET": "short-secret"}):
            response = self.app.test_client().get(
                "/warmup",
                headers={"Authorization": "Bearer short-secret"},
            )
        self.assertEqual(response.status_code, 401)
        self.assertLess(len("short-secret"), MIN_SECRET_LENGTH)

    def test_gunicorn_starts_with_auth_hook(self):
        import importlib.util

        # The project file is gunicorn.conf.py. Importing "gunicorn.conf"
        # would resolve to the third-party gunicorn package instead.
        path = pathlib.Path(__file__).resolve().parents[1] / "gunicorn.conf.py"
        spec = importlib.util.spec_from_file_location("passport_gunicorn_conf", path)
        gconf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gconf)

        self.assertTrue(callable(gconf.on_starting))
        self.assertTrue(callable(gconf.post_worker_init))
        self.assertNotIn("%({", gconf.access_log_format)

        with patch.dict(os.environ, {"APP_ENV": "production"}, clear=False):
            os.environ.pop("PORTAL_SHARED_SECRET", None)
            os.environ.pop("ALLOW_INSECURE_AUTH_START", None)
            with self.assertRaises(ServiceAuthConfigError):
                gconf.on_starting(object())
        os.environ["PORTAL_SHARED_SECRET"] = TEST_SERVICE_TOKEN


class AuthLoggerHygieneTests(unittest.TestCase):
    def test_root_logger_does_not_receive_authorization_header(self):
        from app import app

        leaked = "another-unique-secret-must-not-be-logged"
        records = []

        class _Store(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Store()
        handler.setLevel(logging.DEBUG)
        root = logging.getLogger()
        root.addHandler(handler)
        try:
            app.test_client().post(
                "/verify",
                data={"photo": _photo_field()},
                headers={"Authorization": f"Bearer {leaked}"},
                content_type="multipart/form-data",
            )
        finally:
            root.removeHandler(handler)

        joined = "\n".join(records)
        self.assertNotIn(leaked, joined)
        self.assertNotIn(f"Bearer {leaked}", joined)


if __name__ == "__main__":
    unittest.main()
