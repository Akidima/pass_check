"""
Tie Detector — Learned Object Detection Adapter
------------------------------------------------
Provides a pluggable detector contract (``TieDetector`` Protocol) and a
concrete ``TorchTieDetector`` implementation backed by a fine-tuned
Faster R-CNN (ResNet-50-FPN) from torchvision.

The detector is loaded **once per worker** via ``get_tie_detector()``
(``@lru_cache``).  It never runs its own face detector — the caller
supplies a pre-cropped upper-body ROI.

Environment variables
~~~~~~~~~~~~~~~~~~~~~
    TIE_MODEL_PATH            path to the ``.pt`` checkpoint
    TIE_MODEL_POLICY_PATH     path to the immutable calibrated ``.policy.json``
                              file (defaults beside the checkpoint)
    TIE_MODEL_DEVICE          ``cpu`` | ``cuda`` | ``cuda:0`` …
    TIE_MODEL_RAW_THRESHOLD   minimum raw detector score to keep a
                              detection (default 0.05)
    TIE_MODEL_VERSION         version string included in every result
    TIE_DETECTOR_BACKEND      ``auto`` (default), ``custom``, or ``coco``
    TIE_COCO_THRESHOLD        calibrated operating point for the COCO fallback
                              (default 0.50)

Production readiness
~~~~~~~~~~~~~~~~~~~~
This adapter requires a **trained model artifact** and its checksum-bound,
held-out-set-calibrated policy file before live inference is possible.  Until
then, ``get_tie_detector()`` raises and callers must route the decision to
manual review instead of using image heuristics as a production substitute.
"""

from __future__ import annotations

import logging
import hashlib
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Protocol, runtime_checkable

from PIL import Image

logger = logging.getLogger(__name__)


# ---------- Data Structures ----------

@dataclass(frozen=True)
class TieDetection:
    """A single tie detection result."""
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in ROI coords


class TieModelPolicyError(RuntimeError):
    """Raised when a model has not passed the deployment safety contract."""


@dataclass(frozen=True)
class TieModelPolicy:
    """Calibrated deployment contract for one immutable model artifact.

    Faster R-CNN scores are not probabilities and cannot safely be shared
    across checkpoints.  This policy ties the calibrated operating point and
    anatomical constraints to the exact checkpoint that was evaluated.
    """

    model_version: str
    positive_threshold: float
    geometry: dict[str, float]


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tie_model_policy(model_path: str, policy_path: str) -> TieModelPolicy:
    """Load and validate the policy associated with ``model_path``.

    The policy is deliberately mandatory for live model loading.  It prevents
    an uncalibrated checkpoint, a stale threshold, or an accidentally swapped
    model from silently changing production decisions.
    """
    try:
        with open(policy_path, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise TieModelPolicyError(
            f"Tie model policy is missing or invalid at '{policy_path}'."
        ) from exc

    if not isinstance(document, dict):
        raise TieModelPolicyError("Tie model policy must be a JSON object.")
    required = {
        "schema_version", "model_version", "model_sha256", "input_contract",
        "decision", "geometry", "held_out_metrics",
    }
    missing = required.difference(document)
    if document.get("schema_version") != 1 or missing:
        raise TieModelPolicyError(
            "Tie model policy must use schema_version 1 and include "
            f"{', '.join(sorted(missing))}."
        )

    if document["input_contract"] != "face_relative_upper_body_v1":
        raise TieModelPolicyError(
            "Tie model was not trained for the face-relative upper-body ROI input contract."
        )
    if not isinstance(document["model_sha256"], str) or document["model_sha256"].lower() != _sha256_file(model_path):
        raise TieModelPolicyError("Tie model checksum does not match its calibrated policy.")

    try:
        threshold = float(document["decision"]["tie_present_threshold"])
        geometry = {key: float(value) for key, value in document["geometry"].items()}
        metrics = document["held_out_metrics"]
        min_metrics = document.get("minimum_metrics", {})
        checks = {
            "image_precision": (float(metrics["image_precision"]), float(min_metrics.get("image_precision", 0.0)), "minimum"),
            "image_recall": (float(metrics["image_recall"]), float(min_metrics.get("image_recall", 0.0)), "minimum"),
            "false_positive_rate": (float(metrics["false_positive_rate"]), float(min_metrics.get("false_positive_rate", 1.0)), "maximum"),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise TieModelPolicyError("Tie model policy has an invalid decision, geometry, or metric section.") from exc

    if not 0.0 < threshold < 1.0:
        raise TieModelPolicyError("tie_present_threshold must be between 0 and 1.")
    for name, (actual, target, direction) in checks.items():
        if not 0.0 <= actual <= 1.0 or not 0.0 <= target <= 1.0:
            raise TieModelPolicyError(f"{name} must be between 0 and 1.")
        if (direction == "minimum" and actual < target) or (direction == "maximum" and actual > target):
            raise TieModelPolicyError(f"Tie model does not meet the policy requirement for {name}.")

    required_geometry = {
        "min_width_face_ratio", "max_width_face_ratio",
        "min_height_face_ratio", "max_height_face_ratio",
        "min_top_offset_face_ratio", "max_top_offset_face_ratio",
        "max_center_offset_face_ratio",
    }
    if required_geometry.difference(geometry):
        raise TieModelPolicyError("Tie model policy is missing required geometry constraints.")
    if (
        geometry["min_width_face_ratio"] <= 0
        or geometry["min_height_face_ratio"] <= 0
        or geometry["min_width_face_ratio"] > geometry["max_width_face_ratio"]
        or geometry["min_height_face_ratio"] > geometry["max_height_face_ratio"]
        or geometry["min_top_offset_face_ratio"] > geometry["max_top_offset_face_ratio"]
        or geometry["max_center_offset_face_ratio"] < 0
    ):
        raise TieModelPolicyError("Tie model policy has inconsistent geometry constraints.")

    return TieModelPolicy(
        model_version=str(document["model_version"]),
        positive_threshold=threshold,
        geometry=geometry,
    )


def validate_tie_detection(
    detection: TieDetection,
    face: dict[str, Any],
    image_width: int,
    image_height: int,
    geometry: dict[str, float],
) -> tuple[bool, str]:
    """Validate a predicted tie box against the subject's face/chest frame.

    This is a second, independent guard after the detector's ROI-local box
    filter.  It rejects a plausible-looking object on a lapel, background, or
    another garment because it is not positioned or scaled like a tie below
    the detected face.  The values are model-policy data, calibrated on the
    validation set, rather than hidden inference constants.
    """
    try:
        fx, fy, fw, fh = (float(face[key]) for key in ("x", "y", "w", "h"))
        x1, y1, x2, y2 = detection.bbox
    except (KeyError, TypeError, ValueError):
        return False, "invalid_face_or_bbox"

    values = (fx, fy, fw, fh, x1, y1, x2, y2)
    if not all(math.isfinite(value) for value in values) or fw <= 0 or fh <= 0:
        return False, "invalid_face_or_bbox"
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height or x2 <= x1 or y2 <= y1:
        return False, "bbox_out_of_image"

    box_w, box_h = x2 - x1, y2 - y1
    face_bottom = fy + fh
    center_offset = abs(((x1 + x2) / 2.0) - (fx + fw / 2.0)) / fw
    top_offset = (y1 - face_bottom) / fh
    width_ratio = box_w / fw
    height_ratio = box_h / fh

    ranges = (
        ("width", width_ratio, "min_width_face_ratio", "max_width_face_ratio"),
        ("height", height_ratio, "min_height_face_ratio", "max_height_face_ratio"),
        ("top", top_offset, "min_top_offset_face_ratio", "max_top_offset_face_ratio"),
    )
    for name, value, min_key, max_key in ranges:
        if value < geometry[min_key] or value > geometry[max_key]:
            return False, f"implausible_{name}_relative_to_face"
    if center_offset > geometry["max_center_offset_face_ratio"]:
        return False, "implausible_horizontal_position"
    return True, "valid"


# ---------- Detector Protocol ----------

@runtime_checkable
class TieDetector(Protocol):
    """Minimal contract that any tie detector implementation must satisfy."""

    def detect(
        self,
        image: Image.Image,
        roi_offset: tuple[int, int] = (0, 0),
    ) -> Optional[TieDetection]:
        """Run tie detection on a cropped upper-body ROI image.

        Parameters
        ----------
        image : PIL.Image.Image
            The cropped upper-body region (RGB).
        roi_offset : tuple[int, int]
            (offset_x, offset_y) — the position of the ROI's top-left corner
            in the original full image.  Used so the returned bbox can be
            translated back to full-image coordinates if needed.

        Returns
        -------
        TieDetection or None
            The highest-confidence tie detection, or ``None`` if no tie was
            found above the raw threshold.
        """
        ...


# ---------- TorchVision Faster R-CNN Implementation ----------

class TorchTieDetector:
    """Tie detector backed by a fine-tuned torchvision Faster R-CNN.

    The model must have been trained with **1 foreground class** (``tie``,
    class index 1; background is 0).

    Parameters
    ----------
    model_path : str
        Filesystem path to the ``.pt`` checkpoint.
    device : str
        PyTorch device string (``cpu``, ``cuda``, …).
    raw_threshold : float
        Minimum raw detector confidence to retain a prediction.
    version : str
        Human-readable version tag included in results.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        raw_threshold: float = 0.05,
        version: str = "tie-detector-dev",
        policy: TieModelPolicy | None = None,
    ) -> None:
        import torch
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        self.version = policy.model_version if policy is not None else version
        self.device = torch.device(device)
        self.raw_threshold = raw_threshold
        self.policy = policy

        # Build architecture: 2 classes (background + tie)
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

        # Load trained weights
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self._model = model

        # torchvision transforms
        self._to_tensor = torchvision.transforms.ToTensor()
        logger.info(
            "TorchTieDetector loaded: model=%s device=%s version=%s",
            model_path, device, version,
        )

    @staticmethod
    def _is_plausible_tie_bbox(
        bx1: float, by1: float, bx2: float, by2: float,
        roi_w: int, roi_h: int,
        *,
        min_aspect_ratio: float = 1.2,
        max_aspect_ratio: float = 12.0,
        min_area_frac: float = 0.005,
        max_area_frac: float = 0.50,
        max_horizontal_offset: float = 0.35,
    ) -> bool:
        """Validate that a detected bounding box is geometrically plausible
        for a necktie within an upper-body ROI.

        A necktie is a tall, narrow object centered horizontally in the
        chest region.  This filter rejects micro-detections on noise,
        giant detections that span the entire clothing area, and boxes
        positioned far from the anatomical midline.

        Parameters
        ----------
        bx1, by1, bx2, by2 : float
            Detection bbox in ROI-local coordinates.
        roi_w, roi_h : int
            Dimensions of the ROI image.
        min_aspect_ratio : float
            Minimum height/width ratio (ties are taller than wide).
        max_aspect_ratio : float
            Maximum height/width ratio (rejects single-pixel-wide slivers).
        min_area_frac : float
            Minimum bbox area as fraction of ROI area.
        max_area_frac : float
            Maximum bbox area as fraction of ROI area.
        max_horizontal_offset : float
            Maximum horizontal offset of bbox center from ROI center,
            as fraction of ROI width.
        """
        box_w = bx2 - bx1
        box_h = by2 - by1

        if box_w <= 0 or box_h <= 0:
            return False

        roi_area = max(1, roi_w * roi_h)
        box_area = box_w * box_h

        # Aspect ratio: ties are taller than wide
        aspect = box_h / box_w
        if aspect < min_aspect_ratio or aspect > max_aspect_ratio:
            return False

        # Area fraction: reject micro-detections and giant detections
        area_frac = box_area / roi_area
        if area_frac < min_area_frac or area_frac > max_area_frac:
            return False

        # Horizontal centering: tie should be near the chest midline
        box_cx = (bx1 + bx2) / 2.0
        roi_cx = roi_w / 2.0
        offset_frac = abs(box_cx - roi_cx) / max(1, roi_w)
        if offset_frac > max_horizontal_offset:
            return False

        return True

    def detect(
        self,
        image: Image.Image,
        roi_offset: tuple[int, int] = (0, 0),
    ) -> Optional[TieDetection]:
        import torch

        rgb_image = image.convert("RGB")
        roi_w, roi_h = rgb_image.size
        tensor = self._to_tensor(rgb_image).to(self.device)

        with torch.no_grad():
            outputs = self._model([tensor])[0]

        boxes = outputs["boxes"]
        scores = outputs["scores"]
        labels = outputs["labels"]

        # Filter: class 1 (tie) only, above raw threshold
        mask = (labels == 1) & (scores >= self.raw_threshold)
        if not mask.any():
            return None

        filtered_scores = scores[mask]
        filtered_boxes = boxes[mask]

        # Sort by confidence (descending) and pick the best plausible
        # detection rather than blindly taking the argmax.
        sorted_indices = filtered_scores.argsort(descending=True)

        for idx in sorted_indices:
            score = float(filtered_scores[idx])
            bx1, by1, bx2, by2 = filtered_boxes[idx].tolist()

            if not self._is_plausible_tie_bbox(
                bx1, by1, bx2, by2, roi_w, roi_h,
            ):
                logger.debug(
                    "Rejected implausible tie bbox (%.1f,%.1f,%.1f,%.1f) "
                    "score=%.3f in ROI %dx%d",
                    bx1, by1, bx2, by2, score, roi_w, roi_h,
                )
                continue

            # Translate to full-image coordinates
            ox, oy = roi_offset
            return TieDetection(
                confidence=score,
                bbox=(bx1 + ox, by1 + oy, bx2 + ox, by2 + oy),
            )

        # All detections were geometrically implausible
        return None


class CocoTieDetector:
    """Ready-to-run detector backed by torchvision's COCO tie class.

    COCO label 32 is ``tie`` and covers conventional neckties.  This backend
    is intentionally limited to that class; it is a practical, maintained
    fallback for installations that have not yet trained a university-specific
    checkpoint.  It uses the same ROI and geometry checks as the custom model.
    """

    COCO_TIE_LABEL = 32

    def __init__(self, device: str = "cpu", raw_threshold: float = 0.05) -> None:
        import torch
        import torchvision
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )

        self.version = "torchvision-fasterrcnn-resnet50-fpn-v2-coco"
        self.device = torch.device(device)
        self.raw_threshold = raw_threshold
        self.positive_threshold = float(os.environ.get("TIE_COCO_THRESHOLD", "0.50"))
        if not 0.0 < self.positive_threshold < 1.0:
            raise ValueError("TIE_COCO_THRESHOLD must be between 0 and 1.")
        # Unlike an unvalidated one-class checkpoint, this COCO model was
        # trained with background examples.  A sufficient-visibility ROI with
        # no valid tie box is therefore an operational no-tie decision.
        self.supports_absence_decision = True
        self.geometry = {
            "min_width_face_ratio": 0.04,
            "max_width_face_ratio": 0.85,
            "min_height_face_ratio": 0.12,
            "max_height_face_ratio": 1.60,
            "min_top_offset_face_ratio": -0.30,
            "max_top_offset_face_ratio": 1.35,
            "max_center_offset_face_ratio": 0.35,
        }

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self._model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(self.device).eval()
        self._to_tensor = torchvision.transforms.ToTensor()
        logger.info("CocoTieDetector loaded on %s", device)

    def detect(
        self,
        image: Image.Image,
        roi_offset: tuple[int, int] = (0, 0),
    ) -> Optional[TieDetection]:
        import torch

        rgb_image = image.convert("RGB")
        roi_w, roi_h = rgb_image.size
        tensor = self._to_tensor(rgb_image).to(self.device)
        with torch.no_grad():
            outputs = self._model([tensor])[0]

        mask = (
            (outputs["labels"] == self.COCO_TIE_LABEL)
            & (outputs["scores"] >= self.raw_threshold)
        )
        if not mask.any():
            return None

        scores = outputs["scores"][mask]
        boxes = outputs["boxes"][mask]
        for idx in scores.argsort(descending=True):
            score = float(scores[idx])
            bx1, by1, bx2, by2 = boxes[idx].tolist()
            if not TorchTieDetector._is_plausible_tie_bbox(bx1, by1, bx2, by2, roi_w, roi_h):
                continue
            ox, oy = roi_offset
            return TieDetection(score, (bx1 + ox, by1 + oy, bx2 + ox, by2 + oy))
        return None


# ---------- Cached Loader ----------

@lru_cache(maxsize=1)
def get_tie_detector() -> TieDetector:
    """Return a cached ``TorchTieDetector`` instance.

    Reads configuration from environment variables.  Raises
    ``FileNotFoundError`` if the model checkpoint does not exist.
    """
    backend = os.environ.get("TIE_DETECTOR_BACKEND", "auto").strip().lower()
    if backend not in {"auto", "custom", "coco"}:
        raise ValueError("TIE_DETECTOR_BACKEND must be 'auto', 'custom', or 'coco'.")
    model_path = os.environ.get("TIE_MODEL_PATH", "models/tie_detector_v1.pt")
    policy_path = os.environ.get(
        "TIE_MODEL_POLICY_PATH", f"{os.path.splitext(model_path)[0]}.policy.json"
    )
    device = os.environ.get("TIE_MODEL_DEVICE", "cpu")
    raw_threshold = float(os.environ.get("TIE_MODEL_RAW_THRESHOLD", "0.05"))
    version = os.environ.get("TIE_MODEL_VERSION", "tie-detector-dev")

    if backend == "coco" or (backend == "auto" and not os.path.isfile(model_path)):
        return CocoTieDetector(device=device, raw_threshold=raw_threshold)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Tie detector model not found at '{model_path}'. "
            f"Train one with training/train_tie_detector.py first."
        )

    policy = load_tie_model_policy(model_path, policy_path)
    return TorchTieDetector(
        model_path=model_path,
        device=device,
        raw_threshold=raw_threshold,
        version=version,
        policy=policy,
    )
