# Autonomous Driving Object Detection Pipeline

This repository contains a structured end-to-end object detection pipeline for autonomous driving scenes using the BDD100K dataset.

The project is divided into three major phases:

1. Data Analysis
2. Model Development
3. Evaluation

---

# Repository Structure
```
.
├── data/ # Dataset (not committed)
├── data_analysis/ # Part 1: Dataset exploration
│ ├── Dockerfile
│ ├── requirements.txt
│ ├── src/
│ └── outputs/
├── model/ # Part 2: Model implementation
│ ├── configs/
│ ├── src/
│ └── training/
├── evaluation/ # Part 3: Evaluation scripts
│ ├── src/
│ └── visualizations/
├── docs/ # Reports
│ ├── data_analysis_report.md
│ ├── model_explanation.md
│ └── evaluation_report.md
├── docker-compose.yml
└── README.md
```

---

# Part 1 — Data Analysis

The dataset was analyzed to understand:

- Class distribution and imbalance
- Object size distribution (small / medium / large)
- Spatial distribution patterns
- Scene density
- Class co-occurrence

## Key Findings

- Severe class imbalance (`car` dominant, `train` rare)
- Heavy small-object dominance (especially traffic lights & signs)
- Dense urban scenes (~18 objects per image)
- Strong spatial priors (road structure bias)

Detailed analysis is available in:
`docs/data_analysis_report.md`


---

# Running Data Analysis

## Option 1 — Run Locally

From project root:
```
cd data_analysis
pip install -r requirements.txt
python src/main.py
```

Outputs will be generated inside:
`data_analysis/outputs/`


---

## Option 2 — Run with Docker (Recommended)

From project root:
`docker compose up --build`


This will:

- Build the container
- Mount dataset from `./data`
- Generate outputs in `data_analysis/outputs`

---

# Dataset

This project uses the BDD100K dataset.

Expected structure:
```
data/raw/bdd100k/
labels/
images/
```

Dataset is not included in the repository.

---

# Next Phases

## Part 2 — Model Development
- Architecture selection based on data insights
- Multi-scale detection
- Handling class imbalance

## Part 3 — Evaluation
- mAP evaluation
- Per-class AP
- Error analysis

---

# Design Philosophy

This project emphasizes:

- Structured engineering
- Reproducibility (Docker support)
- Data-driven modeling decisions
- Clear documentation