from pathlib import Path
from parser import load_bdd_annotations, validate_annotations


if __name__ == "__main__":
    json_path = Path("data/labels/bdd100k_labels_images_train.json")

    annotations = load_bdd_annotations(json_path)

    validate_annotations(annotations)