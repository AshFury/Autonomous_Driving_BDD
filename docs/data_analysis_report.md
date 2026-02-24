# Data Analysis Report

## 1. Dataset Overview

The BDD100K object detection dataset was analyzed to understand class distribution, object scale variation, spatial priors, and scene density characteristics.

### Training Set
- Total Images: 69,863
- Total Objects: 1,286,871
- Average Objects per Image: 18.42

The dataset contains dense urban driving scenes with high object counts per frame.

---

## 2. Class Distribution

| Class | Count |
|-------|-------|
| car | 713,211 |
| traffic sign | 239,686 |
| traffic light | 186,117 |
| person | 91,349 |
| truck | 29,971 |
| bus | 11,672 |
| bike | 7,210 |
| rider | 4,517 |
| motor | 3,002 |
| train | 136 |

### Observations

- The dataset is heavily imbalanced.
- `car` dominates the distribution.
- `train` is extremely rare.
- Traffic infrastructure classes (traffic light & sign) are highly represented.

**Implication:** Long-tail imbalance must be handled during training.

---

## 3. Object Size Distribution

Objects were categorized into small, medium, and large based on bounding box area.

### Overall Distribution
- Majority of objects are small.
- Area histogram shows heavy skew toward small-scale objects.

### Per-Class Size Distribution Highlights

- Traffic Light: 88% small
- Traffic Sign: 75% small
- Person: 48% small
- Bus/Truck: predominantly medium to large

### Observations

Traffic infrastructure is primarily small-scale, indicating detection at long distances.

**Implication:** Multi-scale detection (FPN) is necessary. Small-object detection is critical.

---

## 4. Spatial Distribution

Class-wise heatmaps show:

- Strong center-lane bias for vehicles.
- Traffic lights positioned in upper image regions.
- Persons concentrated in lower-middle regions.

### Observations

The dataset exhibits strong spatial priors consistent with structured road environments.

**Implication:** Models can benefit from multi-scale spatial awareness.

---

## 5. Scene Density Analysis

Top 20 most crowded images contain significantly higher object counts.

Dense scenes include:
- Multiple distant traffic lights
- High vehicle clustering
- Urban congestion

**Implication:** Model must handle crowded scenes and overlapping objects.

---

## 6. Co-occurrence Analysis

Frequent class pairings:
- car ↔ traffic sign
- car ↔ traffic light
- person ↔ rider
- bus ↔ truck

This reflects realistic road structure relationships.

---

## 7. Key Challenges Identified

1. Heavy small-object dominance
2. Severe class imbalance (long-tail problem)
3. Dense scenes with overlapping objects
4. Strong spatial priors

---

## 8. Design Implications for Model Selection

Based on the above analysis, the chosen detection architecture must:

- Support multi-scale feature extraction
- Handle small-object detection effectively
- Address class imbalance
- Be robust in crowded urban scenes

This analysis directly informs model architecture choice in the next phase.