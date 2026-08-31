"""Measure tie-detection latency against decision equivalence.

Tie detection is the dominant cost of /verify (measured ~15s of a ~17s
request), and it is the only thing preventing the service from handling
concurrent uploads within the caller's timeout.

The only speed-ups that are safe to ship without a labelled university
dataset are the ones that do not change what the detector decides. This script
therefore reports latency AND per-image decisions for several input-resolution
budgets, so a configuration can be chosen on evidence:

    python benchmarks/benchmark_tie_detector.py --images php-app/uploads

Baseline is torchvision's default transform (min_size=800, max_size=1333).
Because the upper-body ROI is wide and shallow (roughly 1.7x face width by
1.5x face height), max_size is normally the binding constraint, so it is the
knob that actually changes the pixel count.

Decision equivalence, not accuracy, is what this measures. It shows whether a
cheaper configuration reproduces the current model's answers on real
submissions. It does NOT establish that those answers are correct; that still
requires a labelled, identity-disjoint test set.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# Candidate (min_size, max_size) budgets. The first entry is the shipped
# default and is treated as the reference decision.
CONFIGURATIONS = [
    ("baseline-800/1333", 800, 1333),
    ("1067", 640, 1067),
    ("896", 540, 896),
    ("768", 460, 768),
    ("640", 384, 640),
]


def build_detector(min_size: int, max_size: int):
    """Build a COCO tie detector with an explicit input-resolution budget."""
    import torch
    import torchvision
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_resnet50_fpn_v2,
    )

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights, min_size=min_size, max_size=max_size
    ).eval()
    to_tensor = torchvision.transforms.ToTensor()

    def detect(pil_image):
        tensor = to_tensor(pil_image.convert("RGB"))
        with torch.inference_mode():
            outputs = model([tensor])[0]
        mask = (outputs["labels"] == 32) & (outputs["scores"] >= 0.05)
        if not mask.any():
            return None
        scores = outputs["scores"][mask]
        boxes = outputs["boxes"][mask]
        best = int(scores.argmax())
        return float(scores[best]), [round(v, 1) for v in boxes[best].tolist()]

    return detect


def extract_rois(image_paths):
    """Compute the production upper-body ROI for each image, once."""
    import verify
    from tie_visibility import UpperBodyVisibilityEstimator

    estimator = UpperBodyVisibilityEstimator()
    rois = []
    for path in image_paths:
        try:
            bgr = verify._load_image(path.read_bytes())
        except Exception as exc:
            print(f"  skip {path.name}: cannot decode ({type(exc).__name__})")
            continue
        faces = verify._detect_faces(bgr)
        if len(faces) != 1:
            print(f"  skip {path.name}: {len(faces)} face(s) detected")
            continue
        height, width = bgr.shape[:2]
        result = estimator.estimate(faces[0], width, height)
        if not result.sufficient or result.roi is None:
            print(f"  skip {path.name}: upper body not sufficiently visible")
            continue
        x1, y1, x2, y2 = result.roi
        roi = bgr[y1:y2, x1:x2]
        rois.append(
            (path.name, Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)),
             (x2 - x1, y2 - y1))
        )
    return rois


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="../php-app/uploads")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"Torch threads: {torch.get_num_threads()}")

    directory = Path(args.images)
    if not directory.is_absolute():
        directory = (Path(__file__).resolve().parent / directory).resolve()
    paths = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not paths:
        print(f"No images found in {directory}")
        return 1

    print(f"Computing production ROIs for {len(paths)} image(s) in {directory}")
    rois = extract_rois(paths)
    if not rois:
        print("No usable ROIs; cannot benchmark.")
        return 1
    sizes = [f"{w}x{h}" for _, _, (w, h) in rois]
    print(f"Usable ROIs: {len(rois)}  (e.g. {', '.join(sizes[:4])})\n")

    reference = {}
    for label, min_size, max_size in CONFIGURATIONS:
        detect = build_detector(min_size, max_size)
        latencies = []
        decisions = {}
        for name, roi_image, _ in rois:
            for _ in range(args.repeats):
                started = time.perf_counter()
                outcome = detect(roi_image)
                latencies.append((time.perf_counter() - started) * 1000.0)
            decisions[name] = outcome

        if not reference:
            reference = decisions
            agreement = "reference"
        else:
            same = sum(
                1 for name in decisions
                if (decisions[name] is None) == (reference[name] is None)
            )
            agreement = f"{same}/{len(decisions)} same tie/no-tie verdict"

        detected = sum(1 for outcome in decisions.values() if outcome is not None)
        print(
            f"{label:<18} "
            f"median {statistics.median(latencies):>8.0f} ms  "
            f"p95 {sorted(latencies)[int(len(latencies) * 0.95) - 1]:>8.0f} ms  "
            f"ties found {detected:>2}/{len(rois)}  {agreement}"
        )

        # Report score drift on images where both configurations saw a tie, so
        # a threshold-crossing risk is visible rather than hidden by agreement.
        if reference is not decisions:
            drifts = [
                abs(decisions[n][0] - reference[n][0])
                for n in decisions
                if decisions[n] and reference[n]
            ]
            if drifts:
                print(
                    f"{'':<18} max score drift vs baseline on shared "
                    f"detections: {max(drifts):.3f}"
                )
            disagreements = [
                n for n in decisions
                if (decisions[n] is None) != (reference[n] is None)
            ]
            for name in disagreements[:5]:
                was = "tie" if reference[name] else "no tie"
                now = "tie" if decisions[name] else "no tie"
                print(f"{'':<18} CHANGED {name}: {was} -> {now}")

    print(
        "\nA configuration is only safe to ship if it reproduces every baseline "
        "verdict AND its score drift cannot cross the positive threshold "
        f"(TIE_COCO_THRESHOLD, currently {os.environ.get('TIE_COCO_THRESHOLD', '0.50')})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
