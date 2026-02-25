import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from model.src.dataset import VALID_CLASSES, BDDDetectionDataset

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(num_classes, checkpoint_path):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def draw_boxes(image, boxes, labels, scores=None, color=(0, 255, 0)):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label_name = VALID_CLASSES[labels[i]]
        text = label_name

        if scores is not None:
            text += f" {scores[i]:.2f}"

        cv2.putText(
            image,
            text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return image


def main():
    project_root = Path(__file__).resolve().parents[2]

    val_images = project_root / "data/raw/bdd100k/images/100k/val"
    val_labels = project_root / "data/raw/bdd100k/labels/bdd100k_labels_images_val.json"
    checkpoint_path = project_root / "model/training/checkpoints/fasterrcnn_epoch5.pth"

    base_output = project_root / "evaluation/visualizations/sample_predictions"
    good_dir = base_output / "good_examples"
    fail_dir = base_output / "failure_cases"

    good_dir.mkdir(parents=True, exist_ok=True)
    fail_dir.mkdir(parents=True, exist_ok=True)

    dataset = BDDDetectionDataset(val_images, val_labels, max_samples=50)
    model = load_model(len(VALID_CLASSES), checkpoint_path)

    print("Generating qualitative visualizations...")

    with torch.no_grad():
        for idx in range(20):
            image_tensor, target = dataset[idx]
            prediction = model([image_tensor.to(DEVICE)])[0]

            image = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            gt_count = len(target["boxes"])
            pred_keep = prediction["scores"].cpu() > 0.6
            pred_count = pred_keep.sum().item()

            # draw GT (green)
            image = draw_boxes(
                image,
                target["boxes"],
                target["labels"],
                color=(0, 255, 0),
            )

            # draw predictions (red)
            image = draw_boxes(
                image,
                prediction["boxes"][pred_keep].cpu(),
                prediction["labels"][pred_keep].cpu(),
                prediction["scores"][pred_keep].cpu(),
                color=(0, 0, 255),
            )

            # classify example
            if abs(gt_count - pred_count) <= 2:
                save_path = good_dir / f"sample_{idx}.jpg"
            else:
                save_path = fail_dir / f"sample_{idx}.jpg"

            cv2.imwrite(str(save_path), image)

    print("Saved qualitative results to:", base_output)


if __name__ == "__main__":
    main()
