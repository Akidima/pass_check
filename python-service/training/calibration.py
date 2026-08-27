#!/usr/bin/env python3
"""Find optimal accept/reject confidence thresholds for the tie detector
by evaluating cost-weighted error rates across a validation set.

Example:
    python training/calibration.py \
        --model models/tie_detector_v1.pt \
        --images data/val/images \
        --json data/val/annotations.json \
        --output models/calibration_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import TieDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(model_path: str, device: torch.device):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_scores(model, data_loader, device):
    """Run model over dataset and collect highest confidence tie score per image."""
    results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            img_id = int(target["image_id"][0])
            has_tie = len(target["boxes"]) > 0

            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            # Filter to class 1 (tie)
            tie_mask = labels == 1
            tie_scores = scores[tie_mask]

            max_score = float(tie_scores.max()) if len(tie_scores) > 0 else 0.0

            results.append({
                "image_id": img_id,
                "has_tie": has_tie,
                "max_score": max_score,
                "n_detections": int(tie_mask.sum()),
            })

    return results


def evaluate_thresholds(score_data, accept_threshold, reject_threshold,
                        false_reject_cost=1.0, false_accept_cost=1.0,
                        manual_review_cost=0.1):
    """Compute total cost and breakdown for a given accept/reject threshold pair."""
    correct_rejections = 0
    false_acceptances = 0
    correct_acceptances = 0
    false_rejections = 0
    manual_reviews = 0
    total = len(score_data)

    for entry in score_data:
        score = entry["max_score"]
        has_tie = entry["has_tie"]

        if score >= reject_threshold:
            decision = "reject"
        elif score <= accept_threshold:
            decision = "accept"
        else:
            decision = "manual_review"

        if decision == "reject":
            if has_tie:
                correct_rejections += 1
            else:
                false_rejections += 1
        elif decision == "accept":
            if has_tie:
                false_acceptances += 1
            else:
                correct_acceptances += 1
        else:
            manual_reviews += 1

    total_cost = (
        false_rejections * false_reject_cost
        + false_acceptances * false_accept_cost
        + manual_reviews * manual_review_cost
    )

    return {
        "accept_threshold": accept_threshold,
        "reject_threshold": reject_threshold,
        "correct_rejections": correct_rejections,
        "false_acceptances": false_acceptances,
        "correct_acceptances": correct_acceptances,
        "false_rejections": false_rejections,
        "manual_reviews": manual_reviews,
        "manual_review_rate": round(manual_reviews / total, 4) if total > 0 else 0.0,
        "total_cost": round(total_cost, 4),
        "total_images": total,
    }


def find_optimal_thresholds(score_data,
                            false_reject_cost=1.0,
                            false_accept_cost=1.0,
                            manual_review_cost=0.1):
    """Grid search accept/reject candidates to find the threshold pair with lowest total cost."""
    candidates_accept = np.arange(0.05, 0.50, 0.05)
    candidates_reject = np.arange(0.50, 0.96, 0.05)

    best = None
    best_cost = float("inf")

    for at in candidates_accept:
        for rt in candidates_reject:
            if at >= rt:
                continue
            result = evaluate_thresholds(
                score_data, float(at), float(rt),
                false_reject_cost, false_accept_cost, manual_review_cost,
            )
            if result["total_cost"] < best_cost:
                best_cost = result["total_cost"]
                best = result

    return best


def main():
    parser = argparse.ArgumentParser(description="Calibrate tie detector thresholds")
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--images", required=True, help="Validation images directory")
    parser.add_argument("--json", required=True, help="Validation annotations JSON")
    parser.add_argument("--output", default="models/calibration_config.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--false-reject-cost", type=float, default=1.0,
                        help="Operational cost of false rejection [FALSE_REJECT_COST]")
    parser.add_argument("--false-accept-cost", type=float, default=1.0,
                        help="Operational cost of false acceptance [FALSE_ACCEPT_COST]")
    parser.add_argument("--manual-review-cost", type=float, default=0.1,
                        help="Operational cost of manual review [MANUAL_REVIEW_COST]")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_model(args.model, device)

    dataset = TieDataset(args.images, args.json)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)

    print(f"Collecting scores for {len(dataset)} validation images...")
    score_data = collect_scores(model, loader, device)

    # Score distribution summary
    tie_scores = [e["max_score"] for e in score_data if e["has_tie"]]
    no_tie_scores = [e["max_score"] for e in score_data if not e["has_tie"]]
    print(f"\nScore distribution:")
    if tie_scores:
        print(f"  Tie images ({len(tie_scores)}): "
              f"mean={np.mean(tie_scores):.3f} "
              f"median={np.median(tie_scores):.3f} "
              f"min={np.min(tie_scores):.3f} "
              f"max={np.max(tie_scores):.3f}")
    if no_tie_scores:
        print(f"  No-tie images ({len(no_tie_scores)}): "
              f"mean={np.mean(no_tie_scores):.3f} "
              f"median={np.median(no_tie_scores):.3f} "
              f"min={np.min(no_tie_scores):.3f} "
              f"max={np.max(no_tie_scores):.3f}")

    # Search for optimal thresholds
    print(f"\nSearching thresholds with costs: "
          f"false_reject={args.false_reject_cost}, "
          f"false_accept={args.false_accept_cost}, "
          f"manual_review={args.manual_review_cost}")

    best = find_optimal_thresholds(
        score_data,
        false_reject_cost=args.false_reject_cost,
        false_accept_cost=args.false_accept_cost,
        manual_review_cost=args.manual_review_cost,
    )

    if best is None:
        print("ERROR: Could not find valid threshold configuration.")
        sys.exit(1)

    print(f"\n=== Optimal Thresholds ===")
    for k, v in best.items():
        print(f"  {k}: {v}")

    # Save output config
    config = {
        "accept_threshold": best["accept_threshold"],
        "reject_threshold": best["reject_threshold"],
        "calibration_results": best,
        "operational_costs": {
            "false_reject_cost": args.false_reject_cost,
            "false_accept_cost": args.false_accept_cost,
            "manual_review_cost": args.manual_review_cost,
        },
        "note": (
            "Thresholds calibrated on validation dataset. "
            "Re-run calibration after model retraining."
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nCalibration config saved to {output_path}")


if __name__ == "__main__":
    main()

