"""Official-tie staged decision and the four known require_tie failure modes."""

import pathlib
import sys
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from official_tie import (
    OfficialTiePolicy,
    classify_neckwear_type,
    decide_require_tie,
    load_official_tie_policy,
    match_official_appearance,
)
from tie_detector import TieDetection
from verify import _analyze_tie_cv, check_tie


def _faces(x=170, y=75, w=160, h=200):
    return [{"x": x, "y": y, "w": w, "h": h, "score": 1.0, "keypoints": None}]


def _black_shirt_with_black_tie():
    """Extremely dark shirt + slightly structured black necktie blade."""
    bgr = np.full((700, 500, 3), 12, dtype=np.uint8)
    cv2.ellipse(bgr, (250, 170), (80, 95), 0, 0, 360, (28, 32, 36), -1)
    bgr[220:700, 90:410] = (10, 10, 10)
    for y in range(270, 640):
        half = 16 if y < 320 else 13
        bgr[y, 250 - half:250 + half] = (18, 18, 18)
        bgr[y, 250 - half] = (4, 4, 4)
        bgr[y, 250 + half] = (4, 4, 4)
    return bgr, _faces()


def _patterned_necktie():
    bgr = np.full((700, 500, 3), 240, dtype=np.uint8)
    cv2.ellipse(bgr, (250, 170), (80, 95), 0, 0, 360, (90, 120, 160), -1)
    bgr[220:700, 90:410] = (45, 20, 85)
    bgr[250:310, 170:330] = (10, 10, 10)
    for y in range(270, 640):
        for x in range(228, 278):
            bgr[y, x] = (20, 25, 200) if (x + 2 * y) % 8 < 4 else (15, 15, 20)
    return bgr, _faces()


def _bow_tie_image():
    bgr = np.full((700, 500, 3), 240, dtype=np.uint8)
    cv2.ellipse(bgr, (250, 170), (80, 95), 0, 0, 360, (90, 120, 160), -1)
    bgr[220:700, 90:410] = (40, 40, 160)
    # Horizontal lobes at the collar, no descending blade.
    cv2.ellipse(bgr, (210, 300), (42, 16), 0, 0, 360, (10, 10, 10), -1)
    cv2.ellipse(bgr, (290, 300), (42, 16), 0, 0, 360, (10, 10, 10), -1)
    cv2.circle(bgr, (250, 300), 10, (20, 20, 20), -1)
    return bgr, _faces()


def _scarf_image():
    bgr = np.full((700, 500, 3), 240, dtype=np.uint8)
    cv2.ellipse(bgr, (250, 170), (80, 95), 0, 0, 360, (90, 120, 160), -1)
    bgr[220:700, 90:410] = (200, 200, 200)
    bgr[250:430, 110:390] = (30, 80, 180)
    return bgr, _faces()


class TestOfficialTiePolicy(unittest.TestCase):
    def test_default_is_type_only_necktie(self):
        policy = load_official_tie_policy({})
        self.assertEqual(policy.enforcement, "type_only")
        self.assertTrue(policy.allows_type("necktie"))
        self.assertFalse(policy.appearance_configured)

    def test_params_override_appearance(self):
        policy = load_official_tie_policy({
            "official_tie_enforcement": "appearance",
            "official_tie_appearance": {"hue_ranges_hsv": [[0, 15]]},
        })
        self.assertEqual(policy.enforcement, "appearance")
        self.assertTrue(policy.appearance_configured)


class TestNeckwearTypeClassification(unittest.TestCase):
    def test_tall_box_is_necktie(self):
        face = _faces()[0]
        kind = classify_neckwear_type(
            cv_res={"has_tie": False},
            bbox=(200, 350, 280, 550),
            face=face,
            image_hw=(700, 500),
            detection_valid=True,
        )
        self.assertEqual(kind, "necktie")

    def test_wide_short_box_is_bow_tie(self):
        face = _faces()[0]
        kind = classify_neckwear_type(
            cv_res={"has_tie": False, "horizontal_bimodality": 0.6},
            bbox=(150, 290, 350, 340),
            face=face,
            image_hw=(700, 500),
            detection_valid=True,
        )
        self.assertEqual(kind, "bow_tie")

    def test_knot_only_square_box_is_necktie_not_bow(self):
        face = _faces()[0]
        kind = classify_neckwear_type(
            cv_res={"has_tie": True, "knot_only_crop": True},
            bbox=(220, 280, 280, 330),
            face=face,
            image_hw=(360, 400),
            visibility_reason="Collar/knot region is visible (tie blade may be cropped).",
            detection_valid=True,
        )
        self.assertEqual(kind, "necktie")


class TestAppearanceMatch(unittest.TestCase):
    def test_dark_achromatic_sample_is_inconclusive(self):
        policy = OfficialTiePolicy(
            enforcement="appearance",
            hue_ranges_hsv=((0, 15),),
        )
        result = match_official_appearance(
            {"sample_pixels": 400, "mean_l": 20.0, "mean_chroma": 4.0, "mean_hue": 90.0, "hue_std": 4.0},
            policy,
        )
        self.assertEqual(result, "inconclusive")

    def test_chromatic_mismatch_rejects(self):
        policy = OfficialTiePolicy(
            enforcement="appearance",
            hue_ranges_hsv=((100, 130),),
        )
        result = match_official_appearance(
            {"sample_pixels": 400, "mean_l": 80.0, "mean_chroma": 40.0, "mean_hue": 5.0, "hue_std": 4.0},
            policy,
        )
        self.assertEqual(result, "mismatch")


class TestDecisionBoundaries(unittest.TestCase):
    def test_bow_tie_never_auto_passes(self):
        result = decide_require_tie(
            policy=OfficialTiePolicy(),
            cv_res={"has_tie": True, "reason": "tie_detected"},
            detection_valid=True,
            detection_confidence=0.97,
            threshold=0.50,
            localization_reason=None,
            visibility_reason="Upper-body region is sufficiently visible.",
            visibility_sufficient=True,
            neckwear_type="bow_tie",
            official_match="unspecified",
            supports_absence=True,
            model_version="test",
            bbox={"x1": 150, "y1": 290, "x2": 350, "y2": 340},
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["meta"]["decision"], "reject")
        self.assertEqual(result["meta"]["tie_status"], "unofficial_neckwear")
        self.assertEqual(result["meta"]["stage_b_neckwear_type"], "bow_tie")

    def test_cropped_knot_is_not_reported_absent(self):
        result = decide_require_tie(
            policy=OfficialTiePolicy(),
            cv_res={"has_tie": False, "residual_structure": True, "knot_only_crop": True},
            detection_valid=False,
            detection_confidence=0.0,
            threshold=0.50,
            localization_reason=None,
            visibility_reason="Collar/knot region is visible (tie blade may be cropped).",
            visibility_sufficient=True,
            neckwear_type="unknown",
            official_match="unspecified",
            supports_absence=True,
            model_version="test",
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["meta"]["decision"], "manual_review")
        self.assertEqual(result["meta"]["tie_status"], "uncertain")
        self.assertNotEqual(result["meta"]["tie_status"], "tie_absent")


class TestKnownFailureModes(unittest.TestCase):
    @patch("verify.get_tie_detector")
    def test_partial_knot_coco_miss_is_not_no_tie(self, mock_get_detector):
        """Failure 1: a cropped official knot must not become tie_absent."""
        mock_detector = MagicMock()
        mock_detector.supports_absence_decision = True
        mock_detector.detect.return_value = None
        mock_get_detector.return_value = mock_detector

        bgr = np.full((360, 400, 3), 240, dtype=np.uint8)
        cv2.ellipse(bgr, (200, 150), (70, 85), 0, 0, 360, (90, 120, 160), -1)
        bgr[220:360, 70:330] = (45, 20, 85)
        bgr[230:300, 140:260] = (10, 10, 10)
        for y in range(245, 320):
            half = max(6, 22 - (y - 245) // 3)
            bgr[y, 200 - half:200 + half] = (25, 30, 190)
        faces = [{"x": 130, "y": 65, "w": 140, "h": 175, "score": 1.0, "keypoints": None}]
        result = check_tie(bgr, faces, {})
        self.assertFalse(result["passed"] and result["meta"]["tie_status"] == "tie_absent")
        self.assertNotEqual(result["meta"]["tie_status"], "tie_absent")
        self.assertIn(result["meta"]["decision"], ("accept", "manual_review"))

    @patch("verify.get_tie_detector")
    def test_black_on_black_is_not_automatic_absence(self, mock_get_detector):
        """Failure 2: black shirt + black tie must not be reported as no tie."""
        mock_detector = MagicMock()
        mock_detector.supports_absence_decision = True
        mock_detector.detect.return_value = None
        mock_get_detector.return_value = mock_detector

        bgr, faces = _black_shirt_with_black_tie()
        result = check_tie(bgr, faces, {})
        self.assertNotEqual(result["meta"]["tie_status"], "tie_absent")
        self.assertIn(result["meta"]["decision"], ("accept", "manual_review"))
        self.assertNotEqual(result["meta"].get("tie_detected"), False)

    @patch("verify.get_tie_detector")
    def test_appearance_mismatch_rejects_unofficial_necktie(self, mock_get_detector):
        """Failure 3: a necktie that fails the official appearance rule is rejected."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = TieDetection(
            confidence=0.96, bbox=(228, 280, 278, 560)
        )
        mock_get_detector.return_value = mock_detector

        bgr, faces = _patterned_necktie()
        result = check_tie(bgr, faces, {
            "official_tie_enforcement": "appearance",
            "official_tie_appearance": {"hue_ranges_hsv": [[90, 130]]},
        })
        self.assertFalse(result["passed"])
        self.assertEqual(result["meta"]["decision"], "reject")
        self.assertEqual(result["meta"]["tie_status"], "unofficial_neckwear")

    @patch("verify.get_tie_detector")
    def test_bow_tie_box_is_rejected(self, mock_get_detector):
        """Failure 4: bow ties never count as the official necktie."""
        mock_detector = MagicMock()
        mock_detector.detect.return_value = TieDetection(
            confidence=0.96, bbox=(160, 280, 340, 330)
        )
        mock_get_detector.return_value = mock_detector

        bgr, faces = _bow_tie_image()
        result = check_tie(bgr, faces, {})
        self.assertFalse(result["passed"])
        self.assertEqual(result["meta"]["decision"], "reject")
        self.assertEqual(result["meta"]["neckwear_type"], "bow_tie")

    @patch("verify.get_tie_detector")
    def test_scarf_is_rejected(self, mock_get_detector):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = TieDetection(
            confidence=0.90, bbox=(110, 250, 390, 430)
        )
        mock_get_detector.return_value = mock_detector

        bgr, faces = _scarf_image()
        result = check_tie(bgr, faces, {})
        self.assertFalse(result["passed"])
        self.assertEqual(result["meta"]["decision"], "reject")
        self.assertIn(result["meta"]["neckwear_type"], ("scarf", "other_neckwear", "bow_tie"))

    def test_black_on_black_structural_cues_are_not_monochrome_reject_only(self):
        bgr, faces = _black_shirt_with_black_tie()
        cv_res = _analyze_tie_cv(bgr, faces)
        self.assertIn("clahe_vert_edge_score", cv_res)
        self.assertIn("ridge_width_frac", cv_res)
        if not cv_res.get("has_tie"):
            self.assertTrue(
                cv_res.get("residual_structure") or cv_res.get("reason") != "monochrome_garment_feature",
                f"Black-on-black collapsed to a hard no-tie: {cv_res}",
            )


if __name__ == "__main__":
    unittest.main()
