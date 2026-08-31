"""
Official-tie compliance — staged decision
-----------------------------------------
Separates four questions that the COCO “tie” class collapses into one:

    A. Is there evidence of neckwear in the expected neck/chest region?
    B. What type of neckwear is present?
    C. Does it match the approved official tie (only when a spec exists)?
    D. Is the visual evidence sufficient for an automatic decision?

The repository does not contain an official-tie reference image, colour, or
pattern. Until the university supplies one, Stage C cannot identify a specific
design. Default enforcement is therefore ``type_only``: accept a traditional
necktie in the collar slot, and never auto-accept bow ties, scarves, or other
non-necktie neckwear.

Appearance matching is opt-in via ``OFFICIAL_TIE_POLICY_PATH`` / request
params. A colour/pattern mismatch is rejected only when the measurement is
actually available. Low-contrast black-on-black evidence is treated as
inconclusive for colour, not as a failed official match.

No decision uses race, ethnicity, skin tone, gender, religion, or other
sensitive attributes. Skin-distance in the structural analyser is used only to
detect a bare throat, against the same subject's face, not as a demographic
signal.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger(__name__)

ALLOWED_ENFORCEMENT = frozenset({"type_only", "appearance", "off"})
NECKWEAR_TYPES = frozenset({
    "necktie",
    "bow_tie",
    "scarf",
    "other_neckwear",
    "unofficial_tie",
    "no_neckwear",
    "unknown",
})
NEGATIVE_TYPES = frozenset({
    "bow_tie",
    "scarf",
    "other_neckwear",
    "unofficial_tie",
})
DEFAULT_ALLOWED_TYPES = ("necktie",)


@dataclass(frozen=True)
class OfficialTiePolicy:
    """University official-tie rule. Missing appearance fields stay unused."""

    enforcement: str = "type_only"
    allowed_types: tuple[str, ...] = DEFAULT_ALLOWED_TYPES
    hue_ranges_hsv: tuple[tuple[int, int], ...] = ()
    lab_l_range: Optional[tuple[float, float]] = None
    min_chroma: Optional[float] = None
    max_chroma: Optional[float] = None
    source: str = "default"

    def allows_type(self, neckwear_type: str) -> bool:
        return neckwear_type in self.allowed_types

    @property
    def appearance_configured(self) -> bool:
        return bool(
            self.hue_ranges_hsv
            or self.lab_l_range is not None
            or self.min_chroma is not None
            or self.max_chroma is not None
        )


@dataclass
class OfficialTieStages:
    neckwear_evidence: str = "none"
    neckwear_type: str = "unknown"
    official_match: str = "unspecified"
    evidence_sufficient: bool = False
    reasons: list[str] = field(default_factory=list)

    def as_meta(self) -> dict[str, Any]:
        return {
            "stage_a_neckwear_evidence": self.neckwear_evidence,
            "stage_b_neckwear_type": self.neckwear_type,
            "stage_c_official_match": self.official_match,
            "stage_d_evidence_sufficient": self.evidence_sufficient,
            "neckwear_type": self.neckwear_type,
            "official_match": self.official_match,
        }


def _bounded_float_pair(raw, *, lo: float, hi: float) -> Optional[tuple[float, float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        a, b = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(a) and math.isfinite(b)):
        return None
    low, high = (a, b) if a <= b else (b, a)
    return (max(lo, low), min(hi, high))


def _parse_hue_ranges(raw) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    if not isinstance(raw, (list, tuple)):
        return ()
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            a, b = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            continue
        a = max(0, min(179, a))
        b = max(0, min(179, b))
        ranges.append((a, b) if a <= b else (b, a))
    return tuple(ranges)


def _policy_from_mapping(data: dict, *, source: str) -> OfficialTiePolicy:
    enforcement = str(data.get("enforcement", "type_only")).strip().lower()
    if enforcement not in ALLOWED_ENFORCEMENT:
        logger.warning("Invalid official-tie enforcement %r; using type_only.", enforcement)
        enforcement = "type_only"
    raw_types = data.get("allowed_types", list(DEFAULT_ALLOWED_TYPES))
    allowed = []
    if isinstance(raw_types, (list, tuple)):
        for item in raw_types:
            name = str(item).strip().lower()
            if name in NECKWEAR_TYPES and name != "no_neckwear":
                allowed.append(name)
    if not allowed:
        allowed = list(DEFAULT_ALLOWED_TYPES)
    appearance = data.get("appearance") if isinstance(data.get("appearance"), dict) else {}
    return OfficialTiePolicy(
        enforcement=enforcement,
        allowed_types=tuple(dict.fromkeys(allowed)),
        hue_ranges_hsv=_parse_hue_ranges(appearance.get("hue_ranges_hsv")),
        lab_l_range=_bounded_float_pair(
            appearance.get("lab_l_range"), lo=0.0, hi=255.0
        ),
        min_chroma=_safe_float(appearance.get("min_chroma"), 0.0, 200.0),
        max_chroma=_safe_float(appearance.get("max_chroma"), 0.0, 200.0),
        source=source,
    )


def _safe_float(raw, lo: float, hi: float) -> Optional[float]:
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or not lo <= value <= hi:
        return None
    return value


@lru_cache(maxsize=4)
def _load_policy_file(path: str) -> OfficialTiePolicy:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("official-tie policy must be a JSON object")
    return _policy_from_mapping(data, source=path)


def load_official_tie_policy(params: Optional[dict] = None) -> OfficialTiePolicy:
    """Load official-tie rules from request params, then env, then defaults."""
    params = params if isinstance(params, dict) else {}
    if isinstance(params.get("official_tie"), dict):
        return _policy_from_mapping(params["official_tie"], source="params")

    enforcement = params.get("official_tie_enforcement")
    if enforcement is not None:
        mapping = {"enforcement": enforcement}
        if isinstance(params.get("official_tie_allowed_types"), (list, tuple)):
            mapping["allowed_types"] = params["official_tie_allowed_types"]
        if isinstance(params.get("official_tie_appearance"), dict):
            mapping["appearance"] = params["official_tie_appearance"]
        return _policy_from_mapping(mapping, source="params")

    path = os.environ.get("OFFICIAL_TIE_POLICY_PATH", "").strip()
    if path:
        try:
            return _load_policy_file(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Official-tie policy at %s is unusable (%s); using defaults.", path, exc)
    return OfficialTiePolicy()


def crop_is_knot_only(visibility_reason: str, cv_res: dict) -> bool:
    reason = (visibility_reason or "").lower()
    if "blade may be cropped" in reason or "collar/knot" in reason:
        return True
    return bool(cv_res.get("knot_only_crop"))


def classify_neckwear_type(
    *,
    cv_res: dict,
    bbox: Optional[tuple[float, float, float, float]],
    face: dict,
    image_hw: tuple[int, int],
    visibility_reason: str = "",
    detection_valid: bool = False,
) -> str:
    """Assign a neckwear type from geometry and structural cues.

    A square knot-only crop is classified as a necktie candidate, not a bow
    tie, unless the upper crop is clearly bimodal (left/right lobes).
    """
    has_tie = bool(cv_res.get("has_tie"))
    residual = bool(cv_res.get("residual_structure"))
    shape_hint = str(cv_res.get("neckwear_shape") or "")
    knot_only = crop_is_knot_only(visibility_reason, cv_res)
    bimodality = float(cv_res.get("horizontal_bimodality") or 0.0)

    fh = max(1, int(face.get("h") or 1))
    fw = max(1, int(face.get("w") or 1))
    box_metrics = None
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        bw = max(1.0, float(x2) - float(x1))
        bh = max(1.0, float(y2) - float(y1))
        box_metrics = {
            "aspect_hw": bh / bw,
            "aspect_wh": bw / bh,
            "width_face": bw / fw,
            "height_face": bh / fh,
            "area_face": (bw * bh) / float(fw * fh),
        }

    if box_metrics is not None:
        compact_knot = box_metrics["aspect_wh"] < 1.45 and box_metrics["height_face"] <= 0.55
        if knot_only or compact_knot:
            if bimodality >= 0.62 and box_metrics["aspect_wh"] >= 1.60:
                return "bow_tie"
            if has_tie or detection_valid or residual:
                return "necktie"
            return "unknown"
        if box_metrics["aspect_wh"] >= 1.55 and box_metrics["height_face"] <= 0.50:
            return "bow_tie"
        if (
            box_metrics["width_face"] >= 0.85
            and box_metrics["area_face"] >= 0.55
            and box_metrics["aspect_hw"] < 0.85
        ):
            return "scarf"
        if box_metrics["aspect_hw"] >= 1.05 or (has_tie and box_metrics["height_face"] >= 0.18):
            return "necktie"

    if shape_hint == "necktie" and (has_tie or residual):
        return "necktie"
    if shape_hint == "bow_tie":
        return "bow_tie"
    if has_tie or (detection_valid and not knot_only):
        return "necktie"
    if residual:
        return "unknown"
    return "no_neckwear" if not detection_valid else "unknown"


def match_official_appearance(
    appearance_stats: Optional[dict],
    policy: OfficialTiePolicy,
) -> str:
    """Return match / mismatch / inconclusive / unspecified.

    Colour is not used as a reject signal when the sample is too dark or too
    small to measure. That is the black-on-black case: structure can still
    support a necktie, but hue/chroma cannot prove or disprove the official
    design.
    """
    if policy.enforcement != "appearance":
        return "unspecified"
    if not policy.appearance_configured:
        return "unspecified"
    if not appearance_stats:
        return "inconclusive"

    pixels = int(appearance_stats.get("sample_pixels") or 0)
    if pixels < 40:
        return "inconclusive"

    mean_l = appearance_stats.get("mean_l")
    chroma = appearance_stats.get("mean_chroma")
    hue = appearance_stats.get("mean_hue")
    hue_std = float(appearance_stats.get("hue_std") or 0.0)

    # Near-achromatic dark samples cannot support a hue decision.
    if chroma is not None and float(chroma) < 8.0 and mean_l is not None and float(mean_l) < 45.0:
        return "inconclusive"
    if hue_std > 28.0 and (chroma is None or float(chroma) < 18.0):
        return "inconclusive"

    checks = 0
    mismatches = 0
    if policy.lab_l_range is not None and mean_l is not None:
        checks += 1
        low, high = policy.lab_l_range
        if not low <= float(mean_l) <= high:
            mismatches += 1
    if policy.hue_ranges_hsv and hue is not None and chroma is not None and float(chroma) >= 10.0:
        checks += 1
        hue_v = float(hue)
        if not any(lo <= hue_v <= hi for lo, hi in policy.hue_ranges_hsv):
            mismatches += 1
    if policy.min_chroma is not None and chroma is not None:
        checks += 1
        if float(chroma) < policy.min_chroma:
            mismatches += 1
    if policy.max_chroma is not None and chroma is not None:
        checks += 1
        if float(chroma) > policy.max_chroma:
            mismatches += 1

    if checks == 0:
        return "inconclusive"
    if mismatches == 0:
        return "match"
    if mismatches >= max(1, (checks + 1) // 2):
        return "mismatch"
    return "inconclusive"


def measure_appearance(bgr, bbox: Optional[tuple[int, int, int, int]]) -> Optional[dict]:
    """Measure LAB/HSV statistics on a detection box or return None."""
    if bgr is None or bbox is None:
        return None
    import cv2
    import numpy as np

    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    crop = bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    chroma = np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2)
    return {
        "sample_pixels": int(crop.shape[0] * crop.shape[1]),
        "mean_l": float(np.mean(lab[:, :, 0])),
        "mean_chroma": float(np.mean(chroma)),
        "mean_hue": float(np.mean(hsv[:, :, 0])),
        "hue_std": float(np.std(hsv[:, :, 0])),
    }


def _result(passed: bool, message: str, decision: str, status: str, **meta) -> dict:
    payload = {
        "decision": decision,
        "tie_status": status,
        **meta,
    }
    return {"passed": passed, "message": message, "meta": payload}


def decide_require_tie(
    *,
    policy: OfficialTiePolicy,
    cv_res: dict,
    detection_valid: bool,
    detection_confidence: float,
    threshold: float,
    localization_reason: Optional[str],
    visibility_reason: str,
    visibility_sufficient: bool,
    neckwear_type: str,
    official_match: str,
    supports_absence: bool,
    model_version: str,
    bbox: Optional[dict] = None,
    detector_available: bool = True,
) -> dict:
    """Map staged evidence onto accept / reject / manual_review."""
    stages = OfficialTieStages(neckwear_type=neckwear_type, official_match=official_match)
    knot_only = crop_is_knot_only(visibility_reason, cv_res)
    residual = bool(cv_res.get("residual_structure"))
    has_structure = bool(cv_res.get("has_tie"))
    high_conf = detection_valid and detection_confidence >= threshold
    present_evidence = high_conf or has_structure
    ambiguous_negative = str(cv_res.get("reason") or "") in {
        "monochrome_garment_feature",
        "insufficient_tie_profile",
        "solid_shirt_no_tie",
    }
    if present_evidence:
        stages.neckwear_evidence = "present"
    elif residual or (detection_valid and detection_confidence > 0.0) or knot_only or ambiguous_negative:
        stages.neckwear_evidence = "weak"
    else:
        stages.neckwear_evidence = "none"

    extra = {
        "model_version": model_version,
        "upper_body_visible": visibility_sufficient,
        "confidence": round(float(detection_confidence or 0.0), 4),
        **stages.as_meta(),
        "official_tie_enforcement": policy.enforcement,
        "official_tie_policy_source": policy.source,
        "cv_reason": cv_res.get("reason"),
    }
    if bbox is not None:
        extra["bbox"] = bbox
    if localization_reason:
        extra.setdefault("reason", localization_reason)

    if not detector_available and not (has_structure and neckwear_type == "necktie"):
        stages.evidence_sufficient = False
        extra.update(stages.as_meta())
        extra["error"] = "model_unavailable"
        extra["tie_detected"] = None
        return _result(
            False,
            "Tie detection is temporarily unavailable. Manual review is required.",
            "manual_review",
            "uncertain",
            **extra,
        )

    if neckwear_type in NEGATIVE_TYPES:
        stages.official_match = "mismatch"
        stages.evidence_sufficient = True
        extra.update(stages.as_meta())
        extra.update({
            "reason": "unapproved_neckwear_type",
            "tie_detected": True,
            "tie_status": "unofficial_neckwear",
        })
        labels = {
            "bow_tie": "Bow tie is not an approved official necktie.",
            "scarf": "Scarf or wrap is not an approved official necktie.",
            "other_neckwear": "Detected neckwear is not an approved official necktie.",
            "unofficial_tie": "Tie design does not match the approved official necktie.",
        }
        return _result(
            False,
            labels.get(neckwear_type, "Detected neckwear is not an approved official necktie."),
            "reject",
            "unofficial_neckwear",
            **extra,
        )

    if official_match == "mismatch" and policy.enforcement == "appearance":
        stages.official_match = "mismatch"
        stages.evidence_sufficient = True
        extra.update(stages.as_meta())
        extra.update({
            "reason": "official_appearance_mismatch",
            "tie_detected": True,
        })
        return _result(
            False,
            "Tie design does not match the approved official necktie.",
            "reject",
            "unofficial_neckwear",
            **extra,
        )

    if present_evidence and neckwear_type == "necktie":
        if policy.enforcement == "appearance" and official_match == "inconclusive":
            stages.evidence_sufficient = False
            extra.update(stages.as_meta())
            extra.update({
                "reason": "official_appearance_inconclusive",
                "tie_detected": True,
                "corroborated_by_cv": has_structure,
            })
            return _result(
                False,
                "A necktie is visible but the official design cannot be confirmed from this photo. Manual review is required.",
                "manual_review",
                "uncertain",
                **extra,
            )
        stages.official_match = (
            "match" if official_match == "match"
            else ("type_only" if policy.enforcement == "type_only" else official_match)
        )
        stages.evidence_sufficient = True
        extra.update(stages.as_meta())
        extra.update({
            "tie_detected": True,
            "reason": "structural_cv" if (has_structure and not high_conf) else extra.get("reason") or "validated_detection",
            "corroborated_by_cv": has_structure,
        })
        if has_structure and not high_conf:
            extra["reason"] = "structural_cv"
        return _result(
            True,
            "Approved necktie detected.",
            "accept",
            "tie_present",
            **extra,
        )

    if localization_reason and not present_evidence and detection_valid is False and bbox is not None:
        stages.evidence_sufficient = False
        extra.update(stages.as_meta())
        extra.update({
            "reason": localization_reason,
            "tie_detected": None,
        })
        return _result(
            False,
            "Tie-like object is outside the expected neck/chest region. Manual review is required.",
            "manual_review",
            "uncertain",
            **extra,
        )

    # Missing or weak box. A cropped knot or residual dark-on-dark structure
    # is not proof of absence — that was the false “no tie” path.
    if stages.neckwear_evidence in {"weak", "present"} or knot_only or residual:
        stages.evidence_sufficient = False
        extra.update(stages.as_meta())
        extra.update({
            "reason": extra.get("reason") or (
                "below_calibrated_positive_threshold"
                if detection_valid and detection_confidence < threshold
                else "insufficient_official_evidence"
            ),
            "tie_detected": None,
            "tie_present_threshold": threshold,
        })
        return _result(
            False,
            "Tie detection confidence is below the calibrated auto-approval level. Manual review is required."
            if detection_valid and detection_confidence < threshold
            else "The visible knot/neckwear evidence is not sufficient for an automatic official-tie decision. Manual review is required.",
            "manual_review",
            "uncertain",
            **extra,
        )

    if supports_absence and stages.neckwear_evidence == "none" and not knot_only:
        stages.neckwear_type = "no_neckwear"
        stages.official_match = "mismatch"
        stages.evidence_sufficient = True
        extra.update(stages.as_meta())
        extra.update({
            "reason": "no_valid_detection",
            "tie_detected": False,
        })
        return _result(
            False,
            "Traditional necktie not detected in the visible neck/chest region.",
            "reject",
            "tie_absent",
            **extra,
        )

    stages.evidence_sufficient = False
    extra.update(stages.as_meta())
    extra.update({
        "reason": extra.get("reason") or "no_valid_detection",
        "tie_detected": None,
    })
    return _result(
        False,
        "No reliable tie detection was produced. Manual review is required.",
        "manual_review",
        "uncertain",
        **extra,
    )
