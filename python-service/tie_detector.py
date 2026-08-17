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
    TIE_MODEL_DEVICE          ``cpu`` | ``cuda`` | ``cuda:0`` …
    TIE_MODEL_RAW_THRESHOLD   minimum raw detector score to keep a
                              detection (default 0.05)
    TIE_MODEL_VERSION         version string included in every result

Production readiness
~~~~~~~~~~~~~~~~~~~~
This adapter is fully implemented but requires a **trained model artifact**
(``models/tie_detector_v1.pt``) produced by ``training/train_tie_detector.py``
before live inference is possible.  Until then, ``get_tie_detector()`` will
raise ``FileNotFoundError`` when the model path does not exist, and all
callers must handle that gracefully.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Protocol, runtime_checkable

from PIL import Image

logger = logging.getLogger(__name__)


# ---------- Data Structures ----------

@dataclass(frozen=True)
class TieDetection:
    """A single tie detection result."""
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) in ROI coords


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
    ) -> None:
        import torch
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        self.version = version
        self.device = torch.device(device)
        self.raw_threshold = raw_threshold

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

    def detect(
        self,
        image: Image.Image,
        roi_offset: tuple[int, int] = (0, 0),
    ) -> Optional[TieDetection]:
        import torch

        tensor = self._to_tensor(image.convert("RGB")).to(self.device)

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

        # Take highest confidence
        best_idx = filtered_scores.argmax()
        best_score = float(filtered_scores[best_idx])
        bx1, by1, bx2, by2 = filtered_boxes[best_idx].tolist()

        # Translate to full-image coordinates
        ox, oy = roi_offset
        return TieDetection(
            confidence=best_score,
            bbox=(bx1 + ox, by1 + oy, bx2 + ox, by2 + oy),
        )


# ---------- Cached Loader ----------

@lru_cache(maxsize=1)
def get_tie_detector() -> TorchTieDetector:
    """Return a cached ``TorchTieDetector`` instance.

    Reads configuration from environment variables.  Raises
    ``FileNotFoundError`` if the model checkpoint does not exist.
    """
    model_path = os.environ.get("TIE_MODEL_PATH", "models/tie_detector_v1.pt")
    device = os.environ.get("TIE_MODEL_DEVICE", "cpu")
    raw_threshold = float(os.environ.get("TIE_MODEL_RAW_THRESHOLD", "0.05"))
    version = os.environ.get("TIE_MODEL_VERSION", "tie-detector-dev")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Tie detector model not found at '{model_path}'. "
            f"Train one with training/train_tie_detector.py first."
        )

    return TorchTieDetector(
        model_path=model_path,
        device=device,
        raw_threshold=raw_threshold,
        version=version,
    )
