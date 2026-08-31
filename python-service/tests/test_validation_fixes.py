"""Regression tests for background border masking and tie heuristic edge cases."""

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

# Sample test fixture directories for real-world regression images
BENCHMARK_IMAGES_DIR = pathlib.Path(
    "/Users/georgeakidima/.gemini/antigravity-ide/brain/ff4d4fdb-a073-4d40-89d5-b187938c0a93/.user_uploaded"
)
TEXTURE_IMAGES_DIR = pathlib.Path(
    "/Users/georgeakidima/.gemini/antigravity-ide/brain/e791e008-27ba-4803-be22-d624d4923541/.user_uploaded"
)


class NoFaceBackgroundSamplingTests(unittest.TestCase):
    """Ensure background border sampling ignores the bottom region when no face is found."""

    def test_bottom_border_is_zero_when_no_face(self):
        """Lower region should not be masked when face coordinates aren't available."""
        mask = _background_border_mask((1000, 800), [], border_fraction=0.12)
        # Exclude bottom rows to avoid sampling torso/clothing
        bottom_region = mask[400:, :]
        self.assertEqual(np.sum(bottom_region), 0)

    def test_top_border_is_sampled_when_no_face(self):
        """Upper border should still be sampled even if face detection returns empty."""
        mask = _background_border_mask((1000, 800), [], border_fraction=0.12)
        top_border_h = int(1000 * 0.12)
        top_region = mask[:top_border_h, :]
        self.assertGreater(np.sum(top_region), 0)

    def test_clothing_at_bottom_does_not_contaminate_white_bg_when_no_face(self):
        """Dark clothing in bottom half shouldn't cause false background rejections."""
        h, w = 600, 400
        img = np.full((h, w, 3), 255, dtype=np.uint8)  # white wall
        img[int(h * 0.5):, :] = (120, 120, 120)  # dark hoodie/shirt

        result = check_white_background(img, [], {"background_strictness": "standard"})
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["meta"]["dark_coverage_percent"], 0.0)
        self.assertEqual(result["meta"]["colored_coverage_percent"], 0.0)


class TieFalsePositivePreventionTests(unittest.TestCase):
    """Ensure CV heuristics don't flag non-tie clothing features as ties."""

    def test_monochrome_zipper_is_not_detected_as_tie(self):
        """Jacket zipper track shouldn't trigger a false tie detection."""
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
        """Open collar / V-neck contrast with visible undershirt shouldn't trigger tie detection."""
        h, w = 300, 300
        chest = np.full((h, w, 3), (40, 80, 40), dtype=np.uint8)
        chest[150:, int(w*0.35):int(w*0.65)] = (220, 220, 220)

        bgr = np.full((600, 400, 3), 255, dtype=np.uint8)
        bgr[200:500, 50:350] = chest
        face = {"x": 100, "y": 40, "w": 120, "h": 120}

        result = _analyze_tie_cv(bgr, [face])
        self.assertFalse(result["has_tie"], f"False positive tie detected: {result}")

    def test_patterned_red_tie_on_dark_shirt_is_detected(self):
        """A multi-colored blade must count as a tie even if left/right contrast is uneven."""
        bgr = np.full((700, 500, 3), 240, dtype=np.uint8)
        cv2.ellipse(bgr, (250, 170), (80, 95), 0, 0, 360, (90, 120, 160), -1)
        # Shirt must cover the collar-overlap crop, or the white wall looks
        # like a V-neck luminance jump.
        bgr[220:700, 90:410] = (45, 20, 85)
        bgr[250:310, 170:330] = (10, 10, 10)
        for y in range(270, 640):
            for x in range(228, 278):
                bgr[y, x] = (20, 25, 200) if (x + 2 * y) % 8 < 4 else (15, 15, 20)

        face = {"x": 170, "y": 75, "w": 160, "h": 200, "score": 1.0, "keypoints": None}
        result = _analyze_tie_cv(bgr, [face])
        self.assertTrue(result["has_tie"], f"Patterned tie missed: {result}")

    def test_upper_knot_only_crop_is_detected(self):
        """A short collar crop that shows only the knot is still a tie."""
        bgr = np.full((360, 400, 3), 240, dtype=np.uint8)
        cv2.ellipse(bgr, (200, 150), (70, 85), 0, 0, 360, (90, 120, 160), -1)
        bgr[220:360, 70:330] = (45, 20, 85)
        bgr[230:300, 140:260] = (10, 10, 10)
        # Compact knot in the collar seat; no blade below the frame.
        for y in range(245, 320):
            half = max(6, 22 - (y - 245) // 3)
            bgr[y, 200 - half:200 + half] = (25, 30, 190)
        face = {"x": 130, "y": 65, "w": 140, "h": 175, "score": 1.0, "keypoints": None}
        result = _analyze_tie_cv(bgr, [face])
        self.assertTrue(result["has_tie"], f"Knot-only crop missed: {result}")


class BenchmarkSampleImagesTests(unittest.TestCase):
    """Regression tests against sample real-world benchmark images."""

    def _load_img_file(self, filename):
        path = BENCHMARK_IMAGES_DIR / filename
        if not path.exists():
            self.skipTest(f"Image {filename} not found in {BENCHMARK_IMAGES_DIR}")
        with open(path, "rb") as f:
            data = f.read()
            return data, _load_image(data)

    def test_image1_white_bg_hoodie_passes_standard(self):
        """Sample 1: Light hoodie against off-white wall should pass standard validation."""
        _, bgr = self._load_img_file("media_1787310602216.jpg")
        faces = _detect_faces(bgr)
        res = check_white_background(bgr, faces, {"background_strictness": "standard"})
        self.assertTrue(res["passed"], res)
        self.assertGreaterEqual(res["meta"]["white_coverage_percent"], 90.0)

    def test_image2_tie_white_shirt_passes_standard(self):
        """Sample 2: White shirt with tie against studio background passes standard check."""
        _, bgr = self._load_img_file("media_1787310611599.png")
        faces = _detect_faces(bgr)
        res = check_white_background(bgr, faces, {"background_strictness": "standard"})
        self.assertTrue(res["passed"], res)

    def test_image3_dark_bg_is_rejected_at_all_levels(self):
        """Sample 3: Low-light / dark background should be rejected across all presets."""
        _, bgr = self._load_img_file("media_1787310619213.jpg")
        faces = _detect_faces(bgr)
        for level in ("strict", "standard", "relaxed"):
            res = check_white_background(bgr, faces, {"background_strictness": level})
            self.assertFalse(res["passed"], f"Should have failed at {level}: {res}")

    def test_image3_dark_jacket_no_false_tie(self):
        """Sample 3: Dark jacket zipper line shouldn't trigger false tie detection."""
        _, bgr = self._load_img_file("media_1787310619213.jpg")
        faces = _detect_faces(bgr)
        tie_cv = _analyze_tie_cv(bgr, faces)
        self.assertFalse(tie_cv["has_tie"], f"False tie detected on dark jacket: {tie_cv}")

    def test_image4_off_white_bg_passes(self):
        """Sample 4: Typical indoor off-white wall passes under standard strictness."""
        _, bgr = self._load_img_file("media_1787310629362.jpg")
        faces = _detect_faces(bgr)
        res = check_white_background(bgr, faces, {"background_strictness": "standard"})
        self.assertTrue(res["passed"], res)

    def test_image4_green_shirt_no_false_tie(self):
        """Sample 4: Open collar showing undershirt shouldn't trigger false tie detection."""
        _, bgr = self._load_img_file("media_1787310629362.jpg")
        faces = _detect_faces(bgr)
        tie_cv = _analyze_tie_cv(bgr, faces)
        self.assertFalse(tie_cv["has_tie"], f"False tie detected on green shirt: {tie_cv}")


class RealWorldBackgroundTextureTests(unittest.TestCase):
    """Tests for various wall textures (stucco, foam board, plaster) against strictness levels."""

    def _load_img(self, filename):
        path = TEXTURE_IMAGES_DIR / filename
        if not path.exists():
            self.skipTest(f"Image {filename} not found in {TEXTURE_IMAGES_DIR}")
        with open(path, "rb") as f:
            data = f.read()
            return _load_image(data)

    def test_all_three_uploaded_backgrounds_pass_standard_and_relaxed(self):
        """Textured white walls should pass under standard and relaxed modes."""
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
        """Strict mode should only accept pure/studio white, rejecting textured indoor walls."""
        files = ["media_1787330689249.jpg", "media_1787330689465.jpg", "media_1787330689598.jpg"]
        for f in files:
            bgr = self._load_img(f)
            res = check_white_background(bgr, [], {"background_strictness": "strict"})
            self.assertFalse(
                res["passed"],
                f"{f} should have failed at strict level: {res}",
            )

    def test_sub_floor_dark_backgrounds_are_rejected(self):
        """Underexposed / dark backgrounds below brightness thresholds should fail."""
        for val in (120, 100, 80, 50, 0):
            img = np.full((300, 300, 3), val, dtype=np.uint8)
            res_std = check_white_background(img, [], {"background_strictness": "standard"})
            self.assertFalse(res_std["passed"], f"Value {val} should have failed at standard")
            if val <= 105:
                res_rel = check_white_background(img, [], {"background_strictness": "relaxed"})
                self.assertFalse(res_rel["passed"], f"Value {val} should have failed at relaxed")

    def test_color_mixture_and_tints_are_strictly_rejected(self):
        """Tinted or split-color backgrounds should fail across all strictness modes."""
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

