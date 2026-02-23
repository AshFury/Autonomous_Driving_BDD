from pathlib import Path
from parser import BDDParser, compute_basic_stats


def main():
    # Get project root dynamically
    project_root = Path(__file__).resolve().parents[2]

    labels_path = project_root / "data/raw/bdd100k/labels/bdd100k_labels_images_train.json"
    images_path = project_root / "data/raw/bdd100k/images/100k/train"

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    parser = BDDParser(labels_path, images_path)
    annotations = parser.load()

    compute_basic_stats(annotations)


if __name__ == "__main__":
    main()