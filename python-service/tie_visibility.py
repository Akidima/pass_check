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
                               preferred visible-below-face / face-height ratio
                               for a full blade (default 0.35). A smaller
                               collar/knot band is still analyzable.
    TIE_MIN_COLLAR_VISIBLE_RATIO
                               minimum visible-below-face / face-height ratio
                               that still includes the upper knot (default 0.08)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


def _bounded_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    """Read a bounded numeric environment override, falling back on bad input.

    A malformed override here would otherwise raise at construction time and
    take down every tie-related check, so an invalid value degrades to the
    documented default and is logged instead.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default %s.", name, raw, default)
        return default
    if not math.isfinite(value) or not minimum <= value <= maximum:
        logger.warning(
            "%s=%s outside [%s, %s]; using default %s.",
            name, value, minimum, maximum, default,
        )
        return default
    return value


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
        Preferred chest depth, in face-heights, for a full tie blade.
        A shorter crop is still accepted when the collar/knot band is visible.
    horizontal_face_padding : float
        The ROI extends horizontally by this fraction of the face width on
        each side of the face center.
    """

    def __init__(
        self,
        min_face_height_px: int | None = None,
        min_visible_below_face_ratio: float | None = None,
        min_collar_visible_ratio: float | None = None,
        horizontal_face_padding: float = 0.85,
    ) -> None:
        self.min_face_height_px = int(
            min_face_height_px
            if min_face_height_px
            else _bounded_env("TIE_MIN_FACE_HEIGHT", 80, minimum=1, maximum=10_000)
        )
        self.min_visible_below_face_ratio = (
            float(min_visible_below_face_ratio)
            if min_visible_below_face_ratio is not None
            else _bounded_env(
                "TIE_MIN_VISIBLE_BELOW_FACE_RATIO", 0.35, minimum=0.0, maximum=10.0
            )
        )
        self.min_collar_visible_ratio = (
            float(min_collar_visible_ratio)
            if min_collar_visible_ratio is not None
            else _bounded_env(
                "TIE_MIN_COLLAR_VISIBLE_RATIO", 0.08, minimum=0.0, maximum=2.0
            )
        )
        if not math.isfinite(self.min_visible_below_face_ratio) or self.min_visible_below_face_ratio < 0:
            raise ValueError("min_visible_below_face_ratio must be a non-negative number.")
        if not math.isfinite(self.min_collar_visible_ratio) or self.min_collar_visible_ratio < 0:
            raise ValueError("min_collar_visible_ratio must be a non-negative number.")
        if self.min_face_height_px < 1:
            raise ValueError("min_face_height_px must be a positive integer.")
        if not math.isfinite(horizontal_face_padding) or not 0 < horizontal_face_padding <= 5:
            raise ValueError("horizontal_face_padding must be between 0 and 5.")
        self.horizontal_face_padding = float(horizontal_face_padding)

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

        # The knot sits in the collar seat, overlapping the lower face box.
        # A usable upload only needs that band — not a full chest / blade.
        face_bottom = fy + fh
        visible_below = image_height - face_bottom
        preferred_visible = self.min_visible_below_face_ratio * fh
        min_collar_px = max(12.0, self.min_collar_visible_ratio * fh)
        if visible_below < min_collar_px:
            return VisibilityResult(
                sufficient=False,
                reason=(
                    f"Insufficient visible area below the face "
                    f"({visible_below}px available, {int(min_collar_px)}px required "
                    f"to see the collar/knot)."
                ),
                roi=None,
            )

        # Build face-relative ROI for the collar + upper-body / chest area.
        # The knot sits in the collar seat, which usually overlaps the lower
        # face box. Starting at face_bottom cuts that cue out of the crop and
        # leaves only a short, easy-to-miss blade.
        face_cx = fx + fw // 2
        half_roi_w = int(fw * self.horizontal_face_padding)
        collar_overlap = int(fh * 0.35)

        roi_x1 = max(0, face_cx - half_roi_w)
        roi_x2 = min(image_width, face_cx + half_roi_w)
        roi_y1 = max(0, face_bottom - collar_overlap)
        roi_depth = min(visible_below + collar_overlap, int(fh * 1.85))
        roi_y2 = min(
            image_height,
            roi_y1 + max(int(max(preferred_visible, min_collar_px)) + collar_overlap, roi_depth),
        )

        # Sanity: ROI must have meaningful area.
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1
        if roi_w < 20 or roi_h < 20:
            return VisibilityResult(
                sufficient=False,
                reason="Computed upper-body ROI is too small for reliable analysis.",
                roi=(roi_x1, roi_y1, roi_x2, roi_y2),
            )

        reason = (
            "Upper-body region is sufficiently visible."
            if visible_below >= preferred_visible
            else "Collar/knot region is visible (tie blade may be cropped)."
        )
        return VisibilityResult(
            sufficient=True,
            reason=reason,
            roi=(roi_x1, roi_y1, roi_x2, roi_y2),
        )
