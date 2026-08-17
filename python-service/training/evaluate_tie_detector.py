#!/usr/bin/env python3
"""
Evaluate Tie Detector
---------------------
Computes detector metrics (precision, recall, F1, mAP) and business metrics
(FPR, FNR, auto-rejection precision/recall, manual-review rate) on a held-out
dataset.

Usage::

    python training/evaluate_tie_detector.py \\
        --model    models/tie_detector_v1.pt \\
        --images   data/test/images \\
        --json     data/test/annotations.json \\
        --device   cpu

Production inference must **never** import this module.
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


def compute_iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@torch.no_grad()
def run_evaluation(model, data_loader, device, iou_threshold: float = 0.5):
    """Run model on dataset and collect predictions + ground truth."""
    all_predictions = []
    all_ground_truth = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            img_id = int(target["image_id"][0])
            gt_boxes = target["boxes"].cpu().numpy()

            pred_boxes = output["boxes"].cpu().numpy()
            pred_scores = output["scores"].cpu().numpy()
            pred_labels = output["labels"].cpu().numpy()

            # Only class 1 (tie)
            tie_mask = pred_labels == 1
            pred_boxes = pred_boxes[tie_mask]
            pred_scores = pred_scores[tie_mask]

            all_ground_truth.append({
                "image_id": img_id,
                "boxes": gt_boxes,
                "has_tie": len(gt_boxes) > 0,
            })

            all_predictions.append({
                "image_id": img_id,
                "boxes": pred_boxes,
                "scores": pred_scores,
            })

    return all_predictions, all_ground_truth


def compute_metrics(predictions, ground_truth, score_threshold: float = 0.5,
                    iou_threshold: float = 0.5):
    """Compute detection and business metrics."""
    tp = 0
    fp = 0
    fn = 0
    total_images = len(ground_truth)
    images_with_tie = sum(1 for gt in ground_truth if gt["has_tie"])
    images_without_tie = total_images - images_with_tie

    # Image-level decision metrics
    correct_rejections = 0  # tie present, detected
    false_acceptances = 0   # tie present, not detected
    correct_acceptances = 0  # no tie, not detected
    false_rejections = 0    # no tie, detected

    for pred, gt in zip(predictions, ground_truth):
        gt_boxes = gt["boxes"]
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]

        # Filter by score threshold
        mask = pred_scores >= score_threshold
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]

        has_gt = len(gt_boxes) > 0
        has_pred = len(pred_boxes) > 0

        # Image-level
        if has_gt and has_pred:
            correct_rejections += 1
        elif has_gt and not has_pred:
            false_acceptances += 1
        elif not has_gt and has_pred:
            false_rejections += 1
        else:
            correct_acceptances += 1

        # Box-level
        matched_gt = set()
        for pb, ps in zip(pred_boxes, pred_scores):
            best_iou = 0
            best_gt_idx = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    fpr = false_rejections / images_without_tie if images_without_tie > 0 else 0.0
    fnr = false_acceptances / images_with_tie if images_with_tie > 0 else 0.0

    return {
        "detector_metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        },
        "business_metrics": {
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "correct_rejections": correct_rejections,
            "false_acceptances": false_acceptances,
            "correct_acceptances": correct_acceptances,
            "false_rejections": false_rejections,
            "total_images": total_images,
            "images_with_tie": images_with_tie,
            "images_without_tie": images_without_tie,
        },
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate tie detector")
    parser.add_argument("--model", required=True, help="Model checkpoint path")
    parser.add_argument("--images", required=True, help="Test images directory")
    parser.add_argument("--json", required=True, help="Test annotations JSON")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None, help="Output JSON path for metrics")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_model(args.model, device)

    dataset = TieDataset(args.images, args.json)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, collate_fn=collate_fn)

    print(f"Evaluating {len(dataset)} images...")
    predictions, ground_truth = run_evaluation(model, loader, device,
                                                iou_threshold=args.iou_threshold)

    metrics = compute_metrics(predictions, ground_truth,
                              score_threshold=args.score_threshold,
                              iou_threshold=args.iou_threshold)

    print("\n=== Detector Metrics ===")
    for k, v in metrics["detector_metrics"].items():
        print(f"  {k}: {v}")

    print("\n=== Business Metrics ===")
    for k, v in metrics["business_metrics"].items():
        print(f"  {k}: {v}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
