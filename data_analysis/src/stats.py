import numpy as np
import os
from collections import defaultdict
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def extract_bbox_metrics(annotations):
    areas = []
    aspect_ratios = []
    objects_per_image = []
    per_class_area = defaultdict(list)

    for img in annotations:
        objects_per_image.append(len(img.objects))

        for obj in img.objects:
            area = obj.bbox.area()
            ar = obj.bbox.aspect_ratio()

            areas.append(area)
            aspect_ratios.append(ar)
            per_class_area[obj.category].append(area)

    return {
        "areas": np.array(areas),
        "aspect_ratios": np.array(aspect_ratios),
        "objects_per_image": np.array(objects_per_image),
        "per_class_area": per_class_area,
    }

def categorize_object_sizes(areas):
    small = np.sum(areas < 32**2)
    medium = np.sum((areas >= 32**2) & (areas < 96**2))
    large = np.sum(areas >= 96**2)

    total = len(areas)

    return {
        "small": small,
        "medium": medium,
        "large": large,
        "small_pct": small / total,
        "medium_pct": medium / total,
        "large_pct": large / total,
    }

def find_most_crowded_images(annotations, top_k=20):
    image_object_counts = [
        (img.image_name, len(img.objects))
        for img in annotations
    ]

    image_object_counts.sort(key=lambda x: x[1], reverse=True)

    return image_object_counts[:top_k]

def find_images_by_class(annotations, target_class):
    images = []

    for img in annotations:
        count = sum(1 for obj in img.objects if obj.category == target_class)
        if count > 0:
            images.append((img.image_name, count))

    return images

def compute_class_spatial_distribution(
    annotations,
    images_path: Path,
    valid_classes: List[str],
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Computes normalized bounding box center coordinates per class
    using BoundingBox(x1, y1, x2, y2) structure.

    Parameters
    ----------
    annotations : list
        List of parsed ImageAnnotation objects.
    images_path : Path
        Path to directory containing images.
    valid_classes : List[str]
        Classes to include in spatial distribution.

    Returns
    -------
    Dict[str, List[Tuple[float, float]]]
        Mapping of class → list of (normalized_x, normalized_y).
    """

    class_centers = defaultdict(list)

    for img in annotations:
        image_file = images_path / img.image_name

        if not image_file.exists():
            continue

        # Read image dimensions
        with Image.open(image_file) as im:
            width, height = im.size

        for obj in img.objects:
            if obj.category not in valid_classes:
                continue

            box = obj.bbox

            center_x = (box.x1 + box.x2) / 2.0
            center_y = (box.y1 + box.y2) / 2.0

            norm_x = center_x / width
            norm_y = center_y / height

            class_centers[obj.category].append((norm_x, norm_y))

    return class_centers

def plot_spatial_heatmaps(
    class_centers: Dict[str, List[Tuple[float, float]]],
    output_dir: str,
    bins: int = 50,
) -> None:
    """
    Generates and saves 2D spatial heatmaps per class.

    Parameters
    ----------
    class_centers : Dict[str, List[Tuple[float, float]]]
        Dictionary of normalized centers per class.
    output_dir : str
        Directory where heatmaps will be saved.
    bins : int, optional
        Number of bins for 2D histogram.
    """
    os.makedirs(output_dir, exist_ok=True)

    for class_name, centers in class_centers.items():
        if len(centers) == 0:
            continue

        xs, ys = zip(*centers)
        xs = np.array(xs)
        ys = np.array(ys)

        plt.figure(figsize=(6, 5))
        plt.hist2d(xs, ys, bins=bins)
        plt.colorbar(label="Object Density")

        plt.title(f"Spatial Distribution - {class_name}")
        plt.xlabel("Normalized X (Left → Right)")
        plt.ylabel("Normalized Y (Top → Bottom)")

        # Invert Y-axis (image coordinate system)
        plt.gca().invert_yaxis()

        save_path = os.path.join(
            output_dir,
            f"spatial_heatmap_{class_name.replace(' ', '_')}.png",
        )
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def compute_class_cooccurrence(
    annotations,
    valid_classes: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Computes class co-occurrence matrix based on image-level presence.

    Parameters
    ----------
    annotations : list
        List of ImageAnnotation objects.
    valid_classes : List[str]
        Classes to include.

    Returns
    -------
    np.ndarray
        Co-occurrence matrix (NxN).
    List[str]
        Ordered class list corresponding to matrix.
    """

    class_to_index = {cls: idx for idx, cls in enumerate(valid_classes)}
    n = len(valid_classes)

    co_matrix = np.zeros((n, n), dtype=int)

    for img in annotations:
        present_classes = set(
            obj.category
            for obj in img.objects
            if obj.category in valid_classes
        )

        for cls1 in present_classes:
            for cls2 in present_classes:
                i = class_to_index[cls1]
                j = class_to_index[cls2]
                co_matrix[i, j] += 1

    return co_matrix, valid_classes

def compute_per_class_size_distribution(
    annotations,
    size_thresholds: Dict[str, float],
):
    """
    Computes small/medium/large distribution per class.

    Parameters
    ----------
    annotations : list
        List of ImageAnnotation objects.
    size_thresholds : dict
        Dictionary containing area thresholds:
        {
            "small": threshold_small,
            "medium": threshold_medium
        }

    Returns
    -------
    dict
        {
            class_name: {
                "small": count,
                "medium": count,
                "large": count,
                "total": total_count
            }
        }
    """

    class_size_stats = defaultdict(
        lambda: {"small": 0, "medium": 0, "large": 0, "total": 0}
    )

    small_thr = size_thresholds["small"]
    medium_thr = size_thresholds["medium"]

    for img in annotations:
        for obj in img.objects:
            bbox = obj.bbox
            area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)

            class_size_stats[obj.category]["total"] += 1

            if area < small_thr:
                class_size_stats[obj.category]["small"] += 1
            elif area < medium_thr:
                class_size_stats[obj.category]["medium"] += 1
            else:
                class_size_stats[obj.category]["large"] += 1

    return class_size_stats