"""Confirm a cheaper tie-detection resolution keeps the PRODUCTION verdict.

The latency benchmark compares raw candidate detections. Production instead
applies TIE_COCO_THRESHOLD (default 0.50) and the face-relative geometry gate,
so the question that actually decides shippability is narrower:

    does any image's thresholded tie/no-tie verdict change?

A score can drift and still be safe if it stays clear of the threshold; a small
drift is dangerous only near the boundary. This script prints per-image scores
for the baseline and the candidate, flags every verdict change, and reports how
close the closest score sits to the threshold.

    python benchmarks/verify_tie_resolution_equivalence.py \
        --images /path/to/photos --candidate-max-size 1067
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
from PIL import Image  # noqa: E402

from benchmark_tie_detector import build_detector, extract_rois  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--candidate-min-size", type=int, default=640)
    parser.add_argument("--candidate-max-size", type=int, default=1067)
    args = parser.parse_args()

    paths = sorted(
        p for p in Path(args.images).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    print(f"Extracting production ROIs from {len(paths)} image(s)")
    rois = extract_rois(paths)
    print(f"Usable ROIs: {len(rois)}\n")

    baseline = build_detector(800, 1333)
    candidate = build_detector(args.candidate_min_size, args.candidate_max_size)

    print(f"threshold = {args.threshold}")
    print(f"{'image':<44} {'base':>7} {'cand':>7} {'verdict':>18}  {'base ms':>8} {'cand ms':>8}")

    changes = []
    margins = []
    base_total = cand_total = 0.0
    for name, roi_image, _ in rois:
        started = time.perf_counter()
        base_out = baseline(roi_image)
        base_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        cand_out = candidate(roi_image)
        cand_ms = (time.perf_counter() - started) * 1000.0
        base_total += base_ms
        cand_total += cand_ms

        base_score = base_out[0] if base_out else 0.0
        cand_score = cand_out[0] if cand_out else 0.0
        base_tie = base_score >= args.threshold
        cand_tie = cand_score >= args.threshold

        if base_tie == cand_tie:
            verdict = "same (tie)" if base_tie else "same (no tie)"
        else:
            verdict = f"CHANGED {base_tie} -> {cand_tie}"
            changes.append(name)

        # How much headroom does the candidate have before it would flip?
        margins.append((abs(cand_score - args.threshold), name, cand_score))

        print(
            f"{name[:44]:<44} {base_score:>7.3f} {cand_score:>7.3f} "
            f"{verdict:>18}  {base_ms:>8.0f} {cand_ms:>8.0f}"
        )

    count = max(1, len(rois))
    print(f"\nmean latency: baseline {base_total / count:.0f} ms, "
          f"candidate {cand_total / count:.0f} ms "
          f"({base_total / max(1e-9, cand_total):.1f}x faster)")

    if changes:
        print(f"\nNOT EQUIVALENT: {len(changes)} verdict change(s): {', '.join(changes[:5])}")
        return 1

    margins.sort()
    closest_margin, closest_name, closest_score = margins[0]
    print(f"\nEQUIVALENT on {len(rois)} real submission(s): every thresholded "
          f"verdict matches the baseline.")
    print(f"Closest candidate score to the threshold: {closest_score:.3f} "
          f"on {closest_name} (margin {closest_margin:.3f}).")
    if closest_margin < 0.10:
        print("WARNING: that margin is thin. A borderline photograph could still "
              "decide differently than the baseline would.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
