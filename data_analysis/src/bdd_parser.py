BDD_DETECTION_CLASSES = {
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motor",
    "bike",
    "traffic light",
    "traffic sign",
}

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class BoundingBox:
    """
    Represents a 2D bounding box in image coordinates.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        """Compute bounding box area."""
        return max(0.0, (self.x2 - self.x1)) * max(0.0, (self.y2 - self.y1))

    def aspect_ratio(self) -> float:
        h = self.height()
        return self.width() / h if h > 0 else 0.0


@dataclass
class ObjectAnnotation:
    """
    Represents a single object annotation.
    """

    category: str
    bbox: BoundingBox


@dataclass
class ImageAnnotation:
    """
    Represents annotations for a single image.
    """

    image_name: str
    objects: List[ObjectAnnotation]


class BDDParser:
    def __init__(self, labels_path: Path, images_path: Path):
        self.labels_path = labels_path
        self.images_path = images_path

    def load(self) -> List[ImageAnnotation]:
        with open(self.labels_path, "r") as f:
            raw_data = json.load(f)

        annotations: List[ImageAnnotation] = []

        for entry in raw_data:
            image_name = entry.get("name")
            labels = entry.get("labels", [])

            objects = []

            for label in labels:
                category = label.get("category")

                if category not in BDD_DETECTION_CLASSES:
                    continue

                box2d = label.get("box2d")
                if box2d is None:
                    continue

                try:
                    bbox = BoundingBox(
                        x1=float(box2d["x1"]),
                        y1=float(box2d["y1"]),
                        x2=float(box2d["x2"]),
                        y2=float(box2d["y2"]),
                    )

                    if bbox.area() <= 0:
                        continue

                    objects.append(ObjectAnnotation(category, bbox))

                except (KeyError, ValueError):
                    continue

            annotations.append(
                ImageAnnotation(
                    image_name=image_name,
                    objects=objects,
                )
            )

        return annotations


def compute_basic_stats(annotations: List[ImageAnnotation]) -> None:
    total_images = len(annotations)
    total_objects = sum(len(img.objects) for img in annotations)

    class_counter = Counter()

    for img in annotations:
        for obj in img.objects:
            class_counter[obj.category] += 1

    print("==== Dataset Statistics ====")
    print(f"Total images: {total_images}")
    print(f"Total objects: {total_objects}")
    print(f"Average objects per image: {total_objects / total_images:.2f}")
    print("\nClass distribution:")
    for cls in sorted(BDD_DETECTION_CLASSES):
        print(f"{cls}: {class_counter.get(cls, 0)}")
