"""
Upper-Body Visibility Estimator
-------------------------------
Determines whether enough neck/chest/upper-body area is visible in a
passport-style photograph to make a reliable tie-presence decision.

Uses a face-relative ROI and pure geometric heuristics — no ML inference.
All numerical defaults are initial engineering estimates and must be
calibrated on representative university-specific data before production.

Environment overrides
~~~~~~~~~~~~~~~~~~~~~
    TIE_MIN_FACE_HEIGHT        minimum face bbox height in pixels (default 80)
    TIE_MIN_VISIBLE_BELOW_FACE_RATIO
                               required visible-below-face / face-height ratio
                               (default 0.35)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VisibilityResult:
    """Structured output of the visibility check."""
    sufficient: bool
    reason: str
    roi: Optional[tuple[int, int, int, int]]  # (x1, y1, x2, y2) or None


class UpperBodyVisibilityEstimator:
    """Checks whether the neck/chest area below the detected face is visible
    enough for a downstream tie detector to operate reliably.

    Parameters
    ----------
    min_face_height_px : int
        Faces shorter than this (in pixels) are considered too small for
        reliable upper-body analysis.
    min_visible_below_face_ratio : float
        The image must extend at least this many face-heights below the face
        bottom for sufficient upper-body visibility.
    horizontal_face_padding : float
        The ROI extends horizontally by this fraction of the face width on
        each side of the face center.
    """

    def __init__(
        self,
        min_face_height_px: int | None = None,
        min_visible_below_face_ratio: float | None = None,
        horizontal_face_padding: float = 0.85,
    ) -> None:
        self.min_face_height_px = min_face_height_px or int(
            os.environ.get("TIE_MIN_FACE_HEIGHT", "80")
        )
        self.min_visible_below_face_ratio = (
            min_visible_below_face_ratio
            if min_visible_below_face_ratio is not None
            else float(os.environ.get("TIE_MIN_VISIBLE_BELOW_FACE_RATIO", "0.35"))
        )
        self.horizontal_face_padding = horizontal_face_padding

    def estimate(
        self,
        face: dict,
        image_width: int,
        image_height: int,
    ) -> VisibilityResult:
        """Evaluate upper-body visibility for a single detected face.

        Parameters
        ----------
        face : dict
            Must contain ``x``, ``y``, ``w``, ``h`` keys (absolute pixels).
        image_width, image_height : int
            Full image dimensions.

        Returns
        -------
        VisibilityResult
            Contains ``sufficient`` bool, human-readable ``reason``, and the
            computed ``roi`` (x1, y1, x2, y2) if one could be determined.
        """
        if image_width <= 0 or image_height <= 0:
            return VisibilityResult(
                sufficient=False,
                reason="Invalid image dimensions.",
                roi=None,
            )

        try:
            fx = int(face["x"])
            fy = int(face["y"])
            fw = int(face["w"])
            fh = int(face["h"])
        except (KeyError, TypeError, ValueError):
            return VisibilityResult(
                sufficient=False,
                reason="Invalid or missing face bounding box.",
                roi=None,
            )

        if fh < self.min_face_height_px:
            return VisibilityResult(
                sufficient=False,
                reason=(
                    f"Face height ({fh}px) is below the minimum "
                    f"({self.min_face_height_px}px) for reliable upper-body analysis."
                ),
                roi=None,
            )

        # The upper-body ROI starts just below the face and extends downward.
        face_bottom = fy + fh
        visible_below = image_height - face_bottom

        required_visible = self.min_visible_below_face_ratio * fh
        if visible_below < required_visible:
            return VisibilityResult(
                sufficient=False,
                reason=(
                    f"Insufficient visible area below the face "
                    f"({visible_below}px available, {int(required_visible)}px required)."
                ),
                roi=None,
            )

        # Build face-relative ROI for the upper-body / chest area.
        face_cx = fx + fw // 2
        half_roi_w = int(fw * self.horizontal_face_padding)

        roi_x1 = max(0, face_cx - half_roi_w)
        roi_x2 = min(image_width, face_cx + half_roi_w)
        roi_y1 = max(0, face_bottom)
        # Use full available visible chest depth up to 1.5 * fh
        roi_depth = min(visible_below, int(fh * 1.5))
        roi_y2 = min(image_height, face_bottom + max(int(required_visible), roi_depth))

        # Sanity: ROI must have meaningful area.
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1
        if roi_w < 20 or roi_h < 20:
            return VisibilityResult(
                sufficient=False,
                reason="Computed upper-body ROI is too small for reliable analysis.",
                roi=(roi_x1, roi_y1, roi_x2, roi_y2),
            )

        return VisibilityResult(
            sufficient=True,
            reason="Upper-body region is sufficiently visible.",
            roi=(roi_x1, roi_y1, roi_x2, roi_y2),
        )
