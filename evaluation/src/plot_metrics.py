import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    project_root = Path(__file__).resolve().parents[2]
    metrics_path = project_root / "evaluation/visualizations/metrics.json"

    with open(metrics_path, "r") as f:
        results = json.load(f)

    classes = results["classes"]
    ap_values = results["ap_per_class"]
    mAP = results["mAP"]

    output_dir = project_root / "evaluation/visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- AP Bar Chart ----
    plt.figure(figsize=(10, 6))
    plt.bar(classes, ap_values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Precision")
    plt.title(f"Per-Class AP (mAP@0.5 = {mAP:.4f})")
    plt.tight_layout()

    save_path = output_dir / "ap_per_class.png"
    plt.savefig(save_path)
    plt.close()

    print("Saved AP bar chart to:", save_path)


if __name__ == "__main__":
    main()
