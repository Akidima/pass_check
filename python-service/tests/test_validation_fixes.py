"""Unit and regression tests for recent passport validation fixes:
1. Background border mask exclusion of bottom region when no face is detected.
2. Tie detection CV heuristic false-positive prevention on jackets, zippers, and V-neck collared shirts.
3. Verification across the 4 user-provided benchmark images.
4. Admin background strictness propagation.
"""

import os
import pathlib
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from verify import (
    _analyze_tie_cv,
    _background_border_mask,
    _detect_faces,
    _load_image,
    check_no_tie,
    check_tie,
    check_white_background,
    run_checks,
)

USER_UPLOADED_DIR = pathlib.Path(
    "/Users/georgeakidima/.gemini/antigravity-ide/brain/ff4d4fdb-a073-4d40-89d5-b187938c0a93/.user_uploaded"
)


class NoFaceBackgroundSamplingTests(unittest.TestCase):
    """Test that background border sampling does not include the bottom border when no face is found."""

    def test_bottom_border_is_zero_when_no_face(self):
        """When valid_faces is empty, the bottom 60% of the image must have zero mask."""
        mask = _background_border_mask((1000, 800), [], border_fraction=0.12)
        # Bottom rows should be completely unmasked
        bottom_region = mask[400:, :]
        self.assertEqual(np.sum(bottom_region), 0)

    def test_top_border_is_sampled_when_no_face(self):
        """Top border must be included when no face is detected."""
        mask = _background_border_mask((1000, 800), [], border_fraction=0.12)
        top_border_h = int(1000 * 0.12)
        top_region = mask[:top_border_h, :]
        self.assertGreater(np.sum(top_region), 0)

    def test_clothing_at_bottom_does_not_contaminate_white_bg_when_no_face(self):
        """A photo with white wall at the top and grey clothing at the bottom must pass white bg check."""
        h, w = 600, 400
        img = np.full((h, w, 3), 255, dtype=np.uint8)  # pure white wall
        img[int(h * 0.5):, :] = (120, 120, 120)  # grey hoodie / clothing in lower half

        result = check_white_background(img, [], {"background_strictness": "standard"})
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["meta"]["dark_coverage_percent"], 0.0)
        self.assertEqual(result["meta"]["colored_coverage_percent"], 0.0)


class TieFalsePositivePreventionTests(unittest.TestCase):
    """Test that CV tie heuristics do not trigger on non-tie garment features."""

    def test_monochrome_zipper_is_not_detected_as_tie(self):
        """A jacket zipper with slight texture on same-color fabric must not trigger tie detection."""
        h, w = 200, 300
        chest = np.full((h, w, 3), 50, dtype=np.uint8)
        for y in range(0, h, 4):
            chest[y:y+2, int(w*0.48):int(w*0.52)] = (120, 120, 120)

        face = {"x": 100, "y": 20, "w": 100, "h": 100}
        bgr = np.full((500, 400, 3), 255, dtype=np.uint8)
        bgr[150:350, 50:350] = chest

        result = _analyze_tie_cv(bgr, [face])
        self.assertFalse(result["has_tie"], f"False positive tie detected: {result}")

    def test_vneck_undershirt_gradient_is_not_detected_as_tie(self):
        """A V-neck dark shirt showing a bright white undershirt must not trigger tie detection."""
        h, w = 300, 300
        chest = np.full((h, w, 3), (40, 80, 40), dtype=np.uint8)
        chest[150:, int(w*0.35):int(w*0.65)] = (220, 220, 220)

        bgr = np.full((600, 400, 3), 255, dtype=np.uint8)
        bgr[200:500, 50:350] = chest
        face = {"x": 100, "y": 40, "w": 120, "h": 120}

        result = _analyze_tie_cv(bgr, [face])
        self.assertFalse(result["has_tie"], f"False positive tie detected: {result}")


class BenchmarkUserUploadedImagesTests(unittest.TestCase):
    """Test the 4 actual benchmark images provided by the user."""

    def _load_img_file(self, filename):
        path = USER_UPLOADED_DIR / filename
        if not path.exists():
            self.skipTest(f"Image {filename} not found in {USER_UPLOADED_DIR}")
        with open(path, "rb") as f:
            data = f.read()
            return data, _load_image(data)

    def test_image1_white_bg_hoodie_passes_standard(self):
        """media_1787310602216.jpg: Selfie in grey hoodie against white wall must pass white bg."""
        _, bgr = self._load_img_file("media_1787310602216.jpg")
        faces = _detect_faces(bgr)
        res = check_white_background(bgr, faces, {"background_strictness": "standard"})
        self.assertTrue(res["passed"], res)
        self.assertGreaterEqual(res["meta"]["white_coverage_percent"], 90.0)

    def test_image2_tie_white_shirt_passes_standard(self):
        """media_1787310611599.png: White background with tie/shirt must pass white bg at standard."""
        _, bgr = self._load_img_file("media_1787310611599.png")
        faces = _detect_faces(bgr)
        res = check_white_background(bgr, faces, {"background_strictness": "standard"})
        self.assertTrue(res["passed"], res)

    def test_image3_dark_bg_is_rejected_at_all_levels(self):
        """media_1787310619213.jpg: Dark background with dark jacket must be rejected at all levels."""
        _, bgr = self._load_img_file("media_1787310619213.jpg")
        faces = _detect_faces(bgr)
        for level in ("strict", "standard", "relaxed"):
            res = check_white_background(bgr, faces, {"background_strictness": level})
            self.assertFalse(res["passed"], f"Should have failed at {level}: {res}")

    def test_image3_dark_jacket_no_false_tie(self):
        """media_1787310619213.jpg: Dark jacket zipper must NOT trigger false positive tie detection."""
        _, bgr = self._load_img_file("media_1787310619213.jpg")
        faces = _detect_faces(bgr)
        tie_cv = _analyze_tie_cv(bgr, faces)
        self.assertFalse(tie_cv["has_tie"], f"False tie detected on dark jacket: {tie_cv}")

    def test_image4_off_white_bg_passes(self):
        """media_1787310629362.jpg: Off-white background must pass validation."""
        _, bgr = self._load_img_file("media_1787310629362.jpg")
        faces = _detect_faces(bgr)
        res = check_white_background(bgr, faces, {"background_strictness": "standard"})
        self.assertTrue(res["passed"], res)

    def test_image4_green_shirt_no_false_tie(self):
        """media_1787310629362.jpg: Green shirt with open collar/undershirt must NOT trigger false tie."""
        _, bgr = self._load_img_file("media_1787310629362.jpg")
        faces = _detect_faces(bgr)
        tie_cv = _analyze_tie_cv(bgr, faces)
        self.assertFalse(tie_cv["has_tie"], f"False tie detected on green shirt: {tie_cv}")


class CurrentUserUploadedBackgroundsTests(unittest.TestCase):
    """Test the three real-world white background textures provided by the user."""

    CURRENT_DIR = pathlib.Path(
        "/Users/georgeakidima/.gemini/antigravity-ide/brain/e791e008-27ba-4803-be22-d624d4923541/.user_uploaded"
    )

    def _load_img(self, filename):
        path = self.CURRENT_DIR / filename
        if not path.exists():
            self.skipTest(f"Image {filename} not found in {self.CURRENT_DIR}")
        with open(path, "rb") as f:
            data = f.read()
            return _load_image(data)

    def test_all_three_uploaded_backgrounds_pass_standard_and_relaxed(self):
        """Stucco, foam, and plaster walls must all pass at standard and relaxed levels."""
        files = ["media_1787330689249.jpg", "media_1787330689465.jpg", "media_1787330689598.jpg"]
        for f in files:
            bgr = self._load_img(f)
            for level in ("standard", "relaxed"):
                res = check_white_background(bgr, [], {"background_strictness": level})
                self.assertTrue(
                    res["passed"],
                    f"{f} failed at {level}: {res['message']} (white_cov={res['meta']['white_coverage_percent']}%)",
                )
                self.assertGreaterEqual(res["meta"]["white_coverage_percent"], 80.0)

    def test_all_three_uploaded_backgrounds_fail_strict(self):
        """Strict remains the highest level (pure studio white only); ambient indoor walls fail."""
        files = ["media_1787330689249.jpg", "media_1787330689465.jpg", "media_1787330689598.jpg"]
        for f in files:
            bgr = self._load_img(f)
            res = check_white_background(bgr, [], {"background_strictness": "strict"})
            self.assertFalse(
                res["passed"],
                f"{f} should have failed at strict level: {res}",
            )

    def test_sub_floor_dark_backgrounds_are_rejected(self):
        """Backgrounds below the minimum analyzed white floor (V <= 120 / 100, black) must fail."""
        for val in (120, 100, 80, 50, 0):
            img = np.full((300, 300, 3), val, dtype=np.uint8)
            res_std = check_white_background(img, [], {"background_strictness": "standard"})
            self.assertFalse(res_std["passed"], f"Value {val} should have failed at standard")
            if val <= 105:
                res_rel = check_white_background(img, [], {"background_strictness": "relaxed"})
                self.assertFalse(res_rel["passed"], f"Value {val} should have failed at relaxed")

    def test_color_mixture_and_tints_are_strictly_rejected(self):
        """No mixture of white background with colors (blue, green, red, yellow, split walls) is allowed."""
        test_cases = {
            "vivid_blue": np.full((300, 300, 3), (255, 0, 0), dtype=np.uint8),
            "vivid_red": np.full((300, 300, 3), (0, 0, 255), dtype=np.uint8),
            "vivid_green": np.full((300, 300, 3), (0, 255, 0), dtype=np.uint8),
            "pale_blue": np.full((300, 300, 3), (235, 215, 185), dtype=np.uint8),
            "pale_green": np.full((300, 300, 3), (185, 235, 185), dtype=np.uint8),
            "beige_cream": np.full((300, 300, 3), (180, 215, 225), dtype=np.uint8),
            "split_left_blue_right_white": np.hstack([
                np.full((300, 150, 3), (255, 0, 0), dtype=np.uint8),
                np.full((300, 150, 3), 255, dtype=np.uint8),
            ]),
        }
        for name, img in test_cases.items():
            for level in ("strict", "standard", "relaxed"):
                res = check_white_background(img, [], {"background_strictness": level})
                self.assertFalse(
                    res["passed"],
                    f"{name} should have failed at {level}: {res}",
                )


if __name__ == "__main__":
    unittest.main()
