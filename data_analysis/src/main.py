from collections import Counter
from pathlib import Path

import numpy as np

from data_analysis.src.bdd_parser import BDDParser, compute_basic_stats
from data_analysis.src.stats import (categorize_object_sizes,
                                     compute_class_cooccurrence,
                                     compute_class_spatial_distribution,
                                     compute_per_class_size_distribution,
                                     extract_bbox_metrics,
                                     find_images_by_class,
                                     find_most_crowded_images,
                                     plot_spatial_heatmaps)
from data_analysis.src.visualization import (plot_area_histogram,
                                             plot_class_distribution,
                                             plot_cooccurrence_matrix,
                                             plot_objects_per_image,
                                             visualize_image_with_boxes)


def main():
    # Get project root dynamically
    project_root = Path("/app")
    data_analysis_root = Path("/app")
    outputs_dir = data_analysis_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    labels_path = (
        project_root / "data/raw/bdd100k/labels/bdd100k_labels_images_train.json"
    )
    images_path = project_root / "data/raw/bdd100k/images/100k/train"

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    parser = BDDParser(labels_path, images_path)
    annotations = parser.load()

    compute_basic_stats(annotations)
    metrics = extract_bbox_metrics(annotations)

    size_stats = categorize_object_sizes(metrics["areas"])

    print("\n==== Object Size Distribution ====")
    print(f"Small: {size_stats['small']} ({size_stats['small_pct']*100:.2f}%)")
    print(f"Medium: {size_stats['medium']} ({size_stats['medium_pct']*100:.2f}%)")
    print(f"Large: {size_stats['large']} ({size_stats['large_pct']*100:.2f}%)")

    print("\n==== Per-Class Object Size Distribution ====")

    SIZE_THRESHOLDS = {
        "small": 32 * 32,
        "medium": 96 * 96,
    }

    per_class_sizes = compute_per_class_size_distribution(
        annotations,
        SIZE_THRESHOLDS,
    )

    for cls, stats in per_class_sizes.items():
        total = stats["total"]
        if total == 0:
            continue

        small_pct = 100 * stats["small"] / total
        medium_pct = 100 * stats["medium"] / total
        large_pct = 100 * stats["large"] / total

        print(f"\n{cls}:")
        print(f"  Small: {small_pct:.2f}%")
        print(f"  Medium: {medium_pct:.2f}%")
        print(f"  Large: {large_pct:.2f}%")

    print("\n==== Objects Per Image Stats ====")
    print(f"Mean: {metrics['objects_per_image'].mean():.2f}")
    print(f"Median: {np.median(metrics['objects_per_image']):.2f}")
    print(f"Max: {metrics['objects_per_image'].max()}")

    class_counter = Counter()
    for img in annotations:
        for obj in img.objects:
            class_counter[obj.category] += 1

    plot_class_distribution(class_counter, outputs_dir)
    plot_area_histogram(metrics["areas"], outputs_dir)
    plot_objects_per_image(metrics["objects_per_image"], outputs_dir)

    val_labels_path = (
        project_root / "data/raw/bdd100k/labels/bdd100k_labels_images_val.json"
    )
    val_images_path = project_root / "data/raw/bdd100k/images/100k/val"

    val_parser = BDDParser(val_labels_path, val_images_path)
    val_annotations = val_parser.load()

    print("\n==== Validation Dataset Statistics ====")
    compute_basic_stats(val_annotations)

    val_metrics = extract_bbox_metrics(val_annotations)
    val_size_stats = categorize_object_sizes(val_metrics["areas"])

    print("\n==== Validation Object Size Distribution ====")
    print(f"Small: {val_size_stats['small']} ({val_size_stats['small_pct']*100:.2f}%)")
    print(
        f"Medium: {val_size_stats['medium']} ({val_size_stats['medium_pct']*100:.2f}%)"
    )
    print(f"Large: {val_size_stats['large']} ({val_size_stats['large_pct']*100:.2f}%)")

    print("\n==== Validation Objects Per Image Stats ====")
    print(f"Mean: {val_metrics['objects_per_image'].mean():.2f}")
    print(f"Median: {np.median(val_metrics['objects_per_image']):.2f}")
    print(f"Max: {val_metrics['objects_per_image'].max()}")

    print("\n==== Top 20 Most Crowded Training Images ====")
    top_crowded = find_most_crowded_images(annotations, top_k=20)
    for rank, (image_name, count) in enumerate(top_crowded, start=1):
        print(f"{rank:02d}. {image_name} → {count} objects")

    # Take the most crowded image
    most_crowded_name = top_crowded[0][0]

    # Find its annotation object
    most_crowded_annotation = next(
        img for img in annotations if img.image_name == most_crowded_name
    )

    image_path = images_path / most_crowded_name

    visualize_image_with_boxes(
        image_path,
        most_crowded_annotation,
        save_path=outputs_dir / "most_crowded_image.png",
    )

    train_images = find_images_by_class(annotations, "train")

    print("\n==== Images Containing 'train' Class ====")
    print(f"Total images containing train: {len(train_images)}")

    for name, count in train_images[:5]:
        print(f"{name} → {count} trains")

    train_image_name = train_images[0][0]

    train_image_annotation = next(
        img for img in annotations if img.image_name == train_image_name
    )

    image_path = images_path / train_image_name

    visualize_image_with_boxes(
        image_path, train_image_annotation, save_path=outputs_dir / "train_example.png"
    )

    VALID_CLASSES = [
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

    print("\n==== Computing Spatial Distribution ====")

    class_centers = compute_class_spatial_distribution(
        annotations=annotations,
        images_path=images_path,
        valid_classes=VALID_CLASSES,
    )

    plot_spatial_heatmaps(
        class_centers=class_centers,
        output_dir=outputs_dir / "spatial_heatmaps" / "train",
    )

    print("Spatial heatmaps saved to outputs/spatial_heatmaps/train")

    print("\n==== Computing Class Co-occurrence Matrix ====")

    co_matrix, class_names = compute_class_cooccurrence(
        annotations,
        VALID_CLASSES,
    )

    plot_cooccurrence_matrix(
        co_matrix,
        class_names,
        outputs_dir,
    )

    print("Class co-occurrence matrix saved.")


if __name__ == "__main__":
    main()
