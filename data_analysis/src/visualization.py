from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def plot_class_distribution(class_counter, output_dir: Path) -> None:
    """
    Plots class distribution bar chart.

    Parameters
    ----------
    class_counter : Counter
        Counter object containing class frequencies.
    output_dir : Path
        Directory where plot will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = list(class_counter.keys())
    counts = list(class_counter.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts)
    plt.xticks(rotation=45)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    save_path = output_dir / "class_distribution.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_area_histogram(areas, output_dir: Path) -> None:
    """
    Plots bounding box area distribution histogram.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.hist(areas, bins=100)
    plt.title("Bounding Box Area Distribution")
    plt.xlabel("Area")
    plt.ylabel("Frequency")

    save_path = output_dir / "area_distribution.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_objects_per_image(objects_per_image, output_dir: Path) -> None:
    """
    Plots histogram of number of objects per image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.hist(objects_per_image, bins=50)
    plt.title("Objects Per Image Distribution")
    plt.xlabel("Number of Objects")
    plt.ylabel("Frequency")

    save_path = output_dir / "objects_per_image_distribution.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_image_with_boxes(
    image_path: Path,
    annotation,
    save_path: Path,
) -> None:
    """
    Visualizes image with bounding boxes and saves to disk.

    Parameters
    ----------
    image_path : Path
        Path to the image file.
    annotation : ImageAnnotation
        Parsed annotation object containing bounding boxes.
    save_path : Path
        Path where the visualized image will be saved.
    """

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Ensure output directory exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)

    plt.figure(figsize=(10, 8))
    plt.imshow(image)

    for obj in annotation.objects:
        bbox = obj.bbox
        width = bbox.x2 - bbox.x1
        height = bbox.y2 - bbox.y1

        rect = plt.Rectangle(
            (bbox.x1, bbox.y1),
            width,
            height,
            fill=False,
            linewidth=1,
        )
        plt.gca().add_patch(rect)

    plt.title(f"{annotation.image_name} ({len(annotation.objects)} objects)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_cooccurrence_matrix(
    matrix,
    class_names,
    output_dir: Path,
) -> None:
    """
    Plots heatmap of class co-occurrence matrix.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="viridis",
        annot=False,
    )

    plt.title("Class Co-occurrence Matrix")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    save_path = output_dir / "class_cooccurrence_matrix.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
