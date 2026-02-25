import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

VALID_CLASSES = [
    "__background__",
    "car",
    "bus",
    "truck",
    "person",
    "rider",
    "bike",
    "motor",
    "traffic light",
    "traffic sign",
    "train",
]

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(VALID_CLASSES)}


class BDDDetectionDataset(Dataset):
    """
    PyTorch Dataset for BDD100K object detection.
    """

    def __init__(
        self, images_dir: Path, labels_json: Path, transforms=None, max_samples=None
    ):
        self.images_dir = images_dir
        self.transforms = transforms

        with open(labels_json, "r") as f:
            self.annotations = json.load(f)

        # Filter images that have valid detection objects
        self.annotations = [
            img
            for img in self.annotations
            if "labels" in img
            and any(obj["category"] in VALID_CLASSES for obj in img["labels"])
        ]

        all_labels = []

        for annotation in self.annotations:
            for obj in annotation["labels"]:
                label_name = obj["category"]
                if label_name in VALID_CLASSES:
                    label_idx = VALID_CLASSES.index(label_name)
                    all_labels.append(label_idx)

        print("Unique labels in dataset:", sorted(set(all_labels)))
        print("Max label in dataset:", max(all_labels))

        if max_samples:
            self.annotations = self.annotations[:max_samples]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int):
        img_info = self.annotations[idx]
        image_path = self.images_dir / img_info["name"]

        image = Image.open(image_path).convert("RGB")
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

        boxes = []
        labels = []
        areas = []

        for obj in img_info["labels"]:
            category = obj["category"]

            if category not in VALID_CLASSES:
                continue

            x1 = obj["box2d"]["x1"]
            y1 = obj["box2d"]["y1"]
            x2 = obj["box2d"]["x2"]
            y2 = obj["box2d"]["y2"]

            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_IDX[category])
            areas.append((x2 - x1) * (y2 - y1))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": areas,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }

        if len(boxes) != len(labels):
            print("Mismatch boxes/labels!")

        if len(boxes) == 0:
            return None

        return image, target
