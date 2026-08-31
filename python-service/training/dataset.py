"""
Tie Detection Dataset Utilities
-------------------------------
COCO-format dataset loader and identity-disjoint split helpers for
training a single-class (``tie``) object detector.

This module is used by the training / evaluation scripts only and is
**never imported by the production inference path**.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


class TieDataset(Dataset):
    """COCO-format dataset for tie detection.

    Expected annotation structure::

        {
          "images": [
            {"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480, "identity_id": "application-2026-0001"},
            ...
          ],
          "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h]},
            ...
          ],
          "categories": [
            {"id": 1, "name": "tie"}
          ]
        }

    The ``identity_id`` field is the stable applicant/application reference
    used for identity-disjoint splitting. It does not need to be a matric
    number and should not contain a person's name or raw email address.

    Parameters
    ----------
    images_dir : str | Path
        Directory containing the image files.
    annotations_path : str | Path
        Path to the COCO-format JSON annotation file.
    transforms : callable, optional
        Optional image transform (e.g., augmentation pipeline).
    """

    def __init__(
        self,
        images_dir: str | Path,
        annotations_path: str | Path,
        transforms: Any = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.transforms = transforms

        with open(annotations_path) as f:
            coco = json.load(f)

        self._images = {img["id"]: img for img in coco["images"]}
        self._image_ids = list(self._images.keys())

        # Group annotations by image_id
        self._anns_by_image: dict[int, list[dict]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            self._anns_by_image[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, idx: int):
        img_id = self._image_ids[idx]
        img_info = self._images[img_id]
        img_path = self.images_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        anns = self._anns_by_image.get(img_id, [])

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        if boxes:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_id]),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            import torchvision.transforms.functional as F
            image = F.to_tensor(image)

        return image, target


def identity_disjoint_split(
    annotations_path: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    identity_field: str = "identity_id",
) -> dict[str, list[int]]:
    """Split images into train/val/test by applicant identity.

    A stable identity field is mandatory. Image-level fallback is unsafe for
    model evaluation because the same person's photos can cross splits.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to lists of
    image IDs.
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    images = coco["images"]
    rng = random.Random(seed)

    by_student: dict[str, list[int]] = defaultdict(list)
    for img in images:
        key = img.get(identity_field)
        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                f"Every image must contain a non-empty {identity_field!r}; "
                "image-level splitting is not allowed for production evaluation."
            )
        key = key.strip()
        by_student[key].append(img["id"])

    students = list(by_student.keys())
    rng.shuffle(students)

    n = len(students)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_students = students[:n_train]
    val_students = students[n_train : n_train + n_val]
    test_students = students[n_train + n_val :]

    def collect(student_list):
        ids = []
        for s in student_list:
            ids.extend(by_student[s])
        return ids

    return {
        "train": collect(train_students),
        "val": collect(val_students),
        "test": collect(test_students),
    }
