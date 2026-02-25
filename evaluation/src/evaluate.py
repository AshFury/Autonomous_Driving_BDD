import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.src.dataset import VALID_CLASSES, BDDDetectionDataset
from model.src.model import get_model

# Add project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


def collate_fn(batch):
    return tuple(zip(*batch))


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.
    box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - inter_area

    if union == 0:
        return 0.0

    return inter_area / union


def compute_ap(recall, precision):
    """
    Compute Average Precision (AP) using 11-point interpolation.
    """
    ap = 0.0
    for t in [i / 10 for i in range(11)]:
        if (recall >= t).any():
            ap += precision[recall >= t].max()
    return ap / 11.0


def main():
    project_root = Path(__file__).resolve().parents[2]

    val_images = project_root / "data/raw/bdd100k/images/100k/val"
    val_labels = project_root / "data/raw/bdd100k/labels/bdd100k_labels_images_val.json"

    checkpoint_path = project_root / "model/training/checkpoints/fasterrcnn_epoch5.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BDDDetectionDataset(
        images_dir=val_images,
        labels_json=val_labels,
        max_samples=200,  # small subset for evaluation demo
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    print("Running evaluation...")

    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]

            outputs = model(images)

            all_predictions.append(outputs[0])
            all_ground_truths.append(targets[0])

    print("Inference complete.")

    print("Computing mAP@0.5...")

    iou_threshold = 0.5
    num_classes = len(VALID_CLASSES)

    ap_per_class = []

    for class_id in range(1, num_classes):  # skip background (0)

        predictions = []
        total_gt = 0

        for pred, gt in zip(all_predictions, all_ground_truths):

            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]

            pred_boxes = pred["boxes"].cpu()
            pred_labels = pred["labels"].cpu()
            pred_scores = pred["scores"].cpu()

            # Filter by class
            gt_mask = gt_labels == class_id
            pred_mask = pred_labels == class_id

            gt_boxes = gt_boxes[gt_mask]
            pred_boxes = pred_boxes[pred_mask]
            pred_scores = pred_scores[pred_mask]

            total_gt += len(gt_boxes)

            # Each GT box should only be matched once per image
            matched = torch.zeros(len(gt_boxes))

            for box, score in zip(pred_boxes, pred_scores):
                predictions.append((box, score, gt_boxes, matched.clone()))

        if total_gt == 0:
            ap_per_class.append(0)
            continue

        # Sort predictions globally by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)

        tp = torch.zeros(len(predictions))
        fp = torch.zeros(len(predictions))

        for i, (pred_box, score, gt_boxes, matched) in enumerate(predictions):

            best_iou = 0
            best_gt_idx = -1

            for j, gt_box in enumerate(gt_boxes):
                iou = compute_iou(pred_box.tolist(), gt_box.tolist())
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if (
                best_iou >= iou_threshold
                and best_gt_idx >= 0
                and matched[best_gt_idx] == 0
            ):
                tp[i] = 1
                matched[best_gt_idx] = 1
            else:
                fp[i] = 1

        tp_cumsum = torch.cumsum(tp, dim=0)
        fp_cumsum = torch.cumsum(fp, dim=0)

        recall = tp_cumsum / total_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        ap = compute_ap(recall, precision)
        ap_per_class.append(float(ap))

    mAP = sum(ap_per_class) / len(ap_per_class)

    print("\nPer-class AP:")
    for cls, ap in zip(VALID_CLASSES[1:], ap_per_class):
        print(f"{cls}: {ap:.4f}")

    print(f"\nmAP@0.5: {mAP:.4f}")

    print(f"Collected {len(all_predictions)} predictions.")

    results = {
        "classes": VALID_CLASSES[1:],  # exclude background
        "ap_per_class": ap_per_class,
        "mAP": mAP,
    }

    output_path = project_root / "evaluation/visualizations/metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("Saved metrics to:", output_path)
    # For now just print sample output
    print("Sample prediction keys:", all_predictions[0].keys())


if __name__ == "__main__":
    main()
