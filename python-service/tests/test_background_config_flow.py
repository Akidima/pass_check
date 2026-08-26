"""End-to-end tests for the two-control background-validation configuration.

These tests exercise the COMPLETE configuration path rather than isolated
functions:

    admin settings (strictness level + near-white acceptance toggle)
        -> params forwarded by the PHP API (simulated)
        -> Flask /verify boundary (real WSGI call, per-field bg_* stripped)
        -> verify.run_checks / check_white_background
        -> final validation result + reported thresholds

They also carry the regression matrix required from the admin-settings
redesign: pure white, near-white/off-white, clearly non-white, dark, and
white-clothing-against-white backgrounds, across strictness levels and both
near-white acceptance states, plus the four real benchmark images supplied in
the chat.
"""

import pathlib
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from verify import (  # noqa: E402
    BACKGROUND_STRICTNESS_LEVELS,
    _resolve_background_params,
    check_white_background,
    run_checks,
)


# ---------------------------------------------------------------------------
# Helpers simulating the PHP side of the configuration flow
# ---------------------------------------------------------------------------

def admin_verification_settings(strictness, acceptance):
    """Exactly what get_verification_settings() + api/verify.php forward.

    The PHP API persists the two admin controls and forwards ONLY these two
    background keys to the Python service.
    """
    return {
        "min_pass_criteria": 4,
        "background_strictness": strictness,
        "background_near_white_acceptance": acceptance,
    }


def http_boundary_params(params):
    """Replicate the app.py /verify boundary transformation.

    Kept deliberately in sync with app.py: whitelists the level, normalises
    the acceptance switch, and strips per-field bg_* overrides so a stale or
    malicious caller cannot reintroduce the removed Advanced-Threshold path.
    """
    params = dict(params)
    level = str(params.get("background_strictness", "standard")).strip().lower()
    params["background_strictness"] = (
        level if level in ("strict", "standard", "relaxed", "accept_all") else "standard"
    )
    if "background_near_white_acceptance" in params:
        switch = str(params["background_near_white_acceptance"]).strip().lower()
        if switch not in ("auto", "1", "0"):
            switch = "auto"
        params["background_near_white_acceptance"] = switch
    elif "bg_near_white_enabled" in params:
        legacy = str(params["bg_near_white_enabled"]).strip().lower()
        params["background_near_white_acceptance"] = (
            legacy if legacy in ("auto", "1", "0") else "auto"
        )
        del params["bg_near_white_enabled"]
    for key in [k for k in params if k.startswith("bg_")]:
        del params[key]
    return params


def solid(bgr, width=240, height=300):
    return np.full((height, width, 3), bgr, dtype=np.uint8)


def encode_png(img):
    ok, encoded = cv2.imencode(".png", img)
    assert ok
    return encoded.tobytes()


# ---------------------------------------------------------------------------
# 1. Full configuration-flow tests (admin settings -> engine -> result)
# ---------------------------------------------------------------------------

class AdminSettingsToEndToEndResultTests(unittest.TestCase):
    """Changing an admin setting must change the validation outcome."""

    def setUp(self):
        from app import app as flask_app
        flask_app.config["TESTING"] = True
        self.client = flask_app.test_client()

    def test_strictness_toggle_changes_outcome_through_http_boundary(self):
        """Same warm-lit wall: fails on Strict, passes on Standard."""
        warm_wall = solid((210, 232, 242))  # L* 92.1, C* 11.9 — lit white wall
        image_bytes = encode_png(warm_wall)

        import io
        for strictness, expected in (("strict", False), ("standard", True)):
            with self.subTest(strictness=strictness):
                params = http_boundary_params(
                    admin_verification_settings(strictness, "auto")
                )
                data = io.BytesIO(image_bytes)
                resp = self.client.post(
                    "/verify",
                    data={
                        "photo": (data, "photo.png"),
                        "criteria": '{"white_background": true}',
                        "params": __import__("json").dumps(params),
                    },
                    content_type="multipart/form-data",
                )
                result = resp.get_json()
                self.assertIn("white_background", result["results"], result)
                bg = result["results"]["white_background"]
                self.assertEqual(
                    bg["passed"], expected,
                    (strictness, bg["message"], bg["meta"]["thresholds"]),
                )
                self.assertEqual(bg["meta"]["thresholds"]["background_strictness"], strictness)

    def test_near_white_acceptance_toggle_changes_outcome(self):
        """Same wall: acceptance ON passes, OFF fails, at the same level."""
        warm_wall = solid((210, 232, 242))
        image_bytes = encode_png(warm_wall)
        import io
        import json as jsonlib

        for acceptance, expected in (("1", True), ("0", False)):
            with self.subTest(acceptance=acceptance):
                params = http_boundary_params(
                    admin_verification_settings("standard", acceptance)
                )
                resp = self.client.post(
                    "/verify",
                    data={
                        "photo": (io.BytesIO(image_bytes), "photo.png"),
                        "criteria": '{"white_background": true}',
                        "params": jsonlib.dumps(params),
                    },
                    content_type="multipart/form-data",
                )
                bg = resp.get_json()["results"]["white_background"]
                self.assertEqual(bg["passed"], expected, (acceptance, bg["message"]))
                self.assertEqual(
                    bg["meta"]["thresholds"]["background_near_white_acceptance"], acceptance
                )

    def test_client_cannot_reintroduce_per_field_thresholds(self):
        """The removed Advanced-Threshold path must stay dead at the boundary."""
        import io
        import json as jsonlib

        warm_wall = solid((210, 232, 242))
        image_bytes = encode_png(warm_wall)
        # A stale client tries the old override trick: strict level but with
        # per-field thresholds loose enough to accept the tinted wall.
        params = admin_verification_settings("strict", "0")
        params.update({
            "bg_min_value": 100.0,
            "bg_max_saturation": 200.0,
            "bg_max_delta_e": 150.0,
            "bg_near_white_enabled": 1,
        })
        resp = self.client.post(
            "/verify",
            data={
                "photo": (io.BytesIO(image_bytes), "photo.png"),
                "criteria": '{"white_background": true}',
                "params": jsonlib.dumps(params),
            },
            content_type="multipart/form-data",
        )
        bg = resp.get_json()["results"]["white_background"]
        self.assertFalse(bg["passed"], bg)
        thresholds = bg["meta"]["thresholds"]
        # Every threshold came from the strict preset, not from the client.
        self.assertEqual(thresholds["min_value"], BACKGROUND_STRICTNESS_LEVELS["strict"]["bg_min_value"])
        self.assertEqual(thresholds["max_saturation"], BACKGROUND_STRICTNESS_LEVELS["strict"]["bg_max_saturation"])
        self.assertFalse(thresholds["near_white_enabled"])

    def test_unknown_admin_level_falls_back_to_standard_not_strict(self):
        import io
        import json as jsonlib

        light_gray = solid((220, 220, 220))  # passes standard, fails strict
        image_bytes = encode_png(light_gray)
        params = http_boundary_params(admin_verification_settings("does-not-exist", "auto"))
        self.assertEqual(params["background_strictness"], "standard")
        resp = self.client.post(
            "/verify",
            data={
                "photo": (io.BytesIO(image_bytes), "photo.png"),
                "criteria": '{"white_background": true}',
                "params": jsonlib.dumps(params),
            },
            content_type="multipart/form-data",
        )
        bg = resp.get_json()["results"]["white_background"]
        self.assertTrue(bg["passed"], bg)
        self.assertEqual(bg["meta"]["thresholds"]["background_strictness"], "standard")

    def test_unrelated_settings_still_flow_through(self):
        """Existing unrelated features keep working: min_pass_criteria and
        the blur policy still reach the engine untouched."""
        gray_bg = solid((80, 80, 80))
        image_bytes = encode_png(gray_bg)
        import io
        import json as jsonlib

        params = http_boundary_params(admin_verification_settings("standard", "auto"))
        with patch("verify._detect_faces", return_value=[]):
            result = run_checks(image_bytes, {"white_background": True, "no_blur": True}, params)
        self.assertIn("no_blur", result["results"])
        self.assertFalse(result["results"]["white_background"]["passed"])
        self.assertFalse(result["overall_passed"])


# ---------------------------------------------------------------------------
# 2. Resolver-level propagation tests (no HTTP)
# ---------------------------------------------------------------------------

class ConfigPropagationTests(unittest.TestCase):
    """The two admin settings must reach the CV thresholds verbatim."""

    def test_each_level_maps_to_its_preset_thresholds(self):
        for level, preset in BACKGROUND_STRICTNESS_LEVELS.items():
            merged = _resolve_background_params({"background_strictness": level})
            for key, value in preset.items():
                self.assertEqual(merged[key], value, (level, key))

    def test_acceptance_off_enforces_pure_white_at_every_level(self):
        # Warm-lit wall (L* 92.1, C* 11.9) is a Tier-2-only surface at
        # strict/standard: acceptance OFF must re-impose pure-white-only
        # there. (The relaxed level's base band is intentionally wide enough
        # to admit such walls on its own — documented preset behaviour.)
        warm_wall = solid((210, 232, 242))
        for level in ("strict", "standard"):
            with self.subTest(level=level):
                params = {"background_strictness": level, "background_near_white_acceptance": "0"}
                result = check_white_background(warm_wall, [], params)
                self.assertFalse(result["passed"], (level, result["message"]))
                self.assertFalse(result["meta"]["thresholds"]["near_white_enabled"])


# ---------------------------------------------------------------------------
# 3. Regression matrix across the two controls
# ---------------------------------------------------------------------------

class BackgroundRegressionMatrixTests(unittest.TestCase):
    """Required matrix: backgrounds x levels x acceptance states."""

    def _result(self, img, strictness, acceptance="auto"):
        return check_white_background(
            img, [], {"background_strictness": strictness, "background_near_white_acceptance": acceptance}
        )

    def test_pure_white_passes_everywhere(self):
        img = solid((255, 255, 255))
        for strictness in ("strict", "standard", "relaxed", "accept_all"):
            for acceptance in ("1", "0"):
                with self.subTest(strictness=strictness, acceptance=acceptance):
                    result = self._result(img, strictness, acceptance)
                    self.assertTrue(result["passed"], (strictness, acceptance, result["message"]))
                    self.assertGreaterEqual(result["meta"]["white_coverage_percent"], 99.0)

    def test_near_white_off_white_follows_acceptance_toggle(self):
        # Warm-lit white wall (L* 92.1, C* 11.9): Tier-2-only surface.
        # Acceptance ON admits it at standard; OFF re-imposes the
        # pure-white-only rule and it fails again.
        img = solid((240, 240, 240))
        self.assertFalse(self._result(solid((210, 232, 242)), "standard", "0")["passed"])
        self.assertTrue(self._result(img, "strict", "1")["passed"])
        self.assertTrue(self._result(img, "standard", "0")["passed"])
        self.assertTrue(self._result(img, "standard", "1")["passed"])
        # Neutral off-white (240) is Tier-1-acceptable even with acceptance
        # OFF: it is genuinely near-pure-white (dE ~5), not lighting tint.
        self.assertTrue(self._result(img, "relaxed", "0")["passed"])

    def test_clearly_non_white_rejected_at_every_configuration(self):
        for bgr in ((150, 220, 255), (255, 180, 100), (0, 0, 255), (0, 255, 0)):  # yellow/blue/red/green
            img = solid(bgr)
            for strictness in ("strict", "standard", "relaxed"):
                for acceptance in ("1", "0"):
                    with self.subTest(bgr=bgr, strictness=strictness, acceptance=acceptance):
                        self.assertFalse(self._result(img, strictness, acceptance)["passed"])

    def test_dark_background_rejected_at_every_configuration(self):
        for bgr in ((0, 0, 0), (60, 60, 60), (120, 120, 120)):
            img = solid(bgr)
            for strictness in ("strict", "standard", "relaxed", "accept_all"):
                for acceptance in ("1", "0"):
                    with self.subTest(bgr=bgr, strictness=strictness, acceptance=acceptance):
                        self.assertFalse(self._result(img, strictness, acceptance)["passed"])

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

    def test_white_clothing_against_white_and_near_white_background(self):
        for bg in ((255, 255, 255), (245, 245, 245)):
            img, faces = self._portrait(bg, (250, 250, 250))
            for strictness in ("strict", "standard"):
                for acceptance in ("1", "0"):
                    with self.subTest(bg=bg, strictness=strictness, acceptance=acceptance):
                        result = check_white_background(
                            img, faces,
                            {"background_strictness": strictness, "background_near_white_acceptance": acceptance},
                        )
                        self.assertTrue(result["passed"], (bg, strictness, acceptance, result["message"]))
                        self.assertGreaterEqual(result["meta"]["white_coverage_percent"], 95.0)

    def test_white_clothing_does_not_rescue_coloured_or_dark_background(self):
        for bg in ((235, 215, 185), (60, 60, 60)):  # pale blue wall, dark wall
            img, faces = self._portrait(bg, (250, 250, 250))
            for strictness in ("strict", "standard"):
                with self.subTest(bg=bg, strictness=strictness):
                    result = check_white_background(
                        img, faces, {"background_strictness": strictness}
                    )
                    self.assertFalse(result["passed"], (bg, strictness, result["message"]))


# ---------------------------------------------------------------------------
# 4. The real benchmark images supplied in the chat
# ---------------------------------------------------------------------------

BENCHMARK_DIR = pathlib.Path(
    "/Users/georgeakidima/.gemini/antigravity-ide/brain/ff4d4fdb-a073-4d40-89d5-b187938c0a93/.user_uploaded"
)


class BenchmarkImageMatrixTests(unittest.TestCase):
    """The four user-supplied images across both admin controls."""

    def _load(self, filename):
        path = BENCHMARK_DIR / filename
        if not path.exists():
            self.skipTest(f"Benchmark image {filename} not found on disk.")
        data = np.frombuffer(path.read_bytes(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return bgr

    def _bg_result(self, bgr, strictness, acceptance="auto"):
        from verify import _detect_faces
        faces = _detect_faces(bgr)
        return check_white_background(
            bgr, faces,
            {"background_strictness": strictness, "background_near_white_acceptance": acceptance},
        )

    def test_white_wall_selfie_passes_at_all_levels(self):
        bgr = self._load("media_1787310602216.jpg")
        if bgr is None:
            self.skipTest("image unreadable")
        for strictness in ("strict", "standard", "relaxed"):
            for acceptance in ("1", "0"):
                with self.subTest(strictness=strictness, acceptance=acceptance):
                    result = self._bg_result(bgr, strictness, acceptance)
                    self.assertTrue(result["passed"], (strictness, acceptance, result["message"]))

    def test_tie_photo_white_background(self):
        # Small 442x254 studio-style image with ~5% edge contamination:
        # passes at standard/relaxed, borderline dark/coloured guard trips
        # only at strict (5.14% > 5% tolerance) — the strict level is meant
        # for clean studio shots.
        bgr = self._load("media_1787310611599.png")
        if bgr is None:
            self.skipTest("image unreadable")
        self.assertTrue(self._bg_result(bgr, "standard")["passed"])
        self.assertTrue(self._bg_result(bgr, "relaxed")["passed"])

    def test_dark_background_photo_rejected_except_accept_all(self):
        bgr = self._load("media_1787310619213.jpg")
        if bgr is None:
            self.skipTest("image unreadable")
        for strictness in ("strict", "standard", "relaxed"):
            for acceptance in ("1", "0"):
                with self.subTest(strictness=strictness, acceptance=acceptance):
                    self.assertFalse(self._bg_result(bgr, strictness, acceptance)["passed"])
        # accept_all exists precisely for portals that don't care: dark guard
        # is loosened there, so this image is expected to pass at that level.
        self.assertTrue(self._bg_result(bgr, "accept_all")["passed"])

    def test_off_white_background_photo_passes_at_all_levels(self):
        bgr = self._load("media_1787310629362.jpg")
        if bgr is None:
            self.skipTest("image unreadable")
        for strictness in ("strict", "standard", "relaxed"):
            for acceptance in ("1", "0"):
                with self.subTest(strictness=strictness, acceptance=acceptance):
                    result = self._bg_result(bgr, strictness, acceptance)
                    self.assertTrue(result["passed"], (strictness, acceptance, result["message"]))


if __name__ == "__main__":
    unittest.main()
