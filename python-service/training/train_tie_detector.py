#!/usr/bin/env python3
"""
Train Tie Detector — Faster R-CNN Transfer Learning
----------------------------------------------------
Fine-tunes a pretrained Faster R-CNN (ResNet-50-FPN) from torchvision
for single-class (``tie``) object detection on university passport photos.

Usage::

    python training/train_tie_detector.py \\
        --train-images data/train/images \\
        --train-json   data/train/annotations.json \\
        --val-images   data/val/images \\
        --val-json     data/val/annotations.json \\
        --output       models/ \\
        --epochs       30 \\
        --batch-size   4 \\
        --learning-rate 0.005 \\
        --device       cuda

Production inference must **never** import this module.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Allow importing dataset from the training package
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import TieDataset


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes: int = 2, pretrained_backbone: bool = True):
    """Build Faster R-CNN with a replaced prediction head.

    Parameters
    ----------
    num_classes : int
        Number of classes including background (2 = background + tie).
    pretrained_backbone : bool
        Whether to load COCO-pretrained weights for transfer learning.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch: int):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += float(losses)
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    print(f"  Epoch {epoch:3d} | avg_loss={avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def validate(model, data_loader, device):
    """Run validation and compute simple detection metrics."""
    model.eval()
    all_detections = 0
    all_gt = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            gt_boxes = target["boxes"]
            all_gt += len(gt_boxes)
            pred_scores = output["scores"]
            all_detections += int((pred_scores >= 0.5).sum())

    print(f"  Validation: {all_detections} detections, {all_gt} ground-truth boxes")
    return {"detections": all_detections, "ground_truth": all_gt}


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN tie detector")
    parser.add_argument("--train-images", required=True, help="Training images directory")
    parser.add_argument("--train-json", required=True, help="Training COCO annotations JSON")
    parser.add_argument("--val-images", required=True, help="Validation images directory")
    parser.add_argument("--val-json", required=True, help="Validation COCO annotations JSON")
    parser.add_argument("--output", default="models/", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not use COCO-pretrained backbone (not recommended)")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # Datasets
    train_dataset = TieDataset(args.train_images, args.train_json)
    val_dataset = TieDataset(args.val_images, args.val_json)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # Model
    model = build_model(num_classes=2, pretrained_backbone=not args.no_pretrained)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    best_loss = float("inf")
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        lr_scheduler.step()

        val_metrics = validate(model, val_loader, device)

        # Save checkpoint
        checkpoint_path = output_dir / f"tie_detector_epoch{epoch:03d}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = output_dir / "tie_detector_v1.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  ★ New best model saved: {best_path}")

    elapsed = time.time() - started
    print(f"\nTraining complete in {elapsed:.1f}s")

    # Save training config
    config = {
        "architecture": "fasterrcnn_resnet50_fpn",
        "num_classes": 2,
        "class_names": ["background", "tie"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "pretrained_backbone": not args.no_pretrained,
        "best_loss": best_loss,
        "training_time_s": round(elapsed, 1),
        "device": args.device,
    }
    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")


if __name__ == "__main__":
    main()
