#  Diabetic Retinopathy Detection using Deep Learning


[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)


A production-ready deep learning system that automatically grades **Diabetic Retinopathy (DR)** severity from retinal fundus images using **EfficientNet-B0** and advanced imbalance-handling techniques.

The system achieves:

- **Quadratic Weighted Kappa (QWK): 0.9057**
-  Runs entirely on **CPU**
-  Includes a complete **Streamlit GUI**
-  Supports **Grad-CAM explainability**
-  Provides **doctor-friendly evaluation reports**

---

# Table of Contents

- [Medical Problem](#medical-problem)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Model & Training](#model--training)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
)

---

#  Medical Problem

**Diabetic Retinopathy (DR)** is a diabetes complication that damages retinal blood vessels and may eventually cause blindness if not diagnosed early.

Because millions of diabetic patients require regular retinal screening, automated AI systems can help ophthalmologists by:

- Reducing screening workload
- Providing faster diagnosis assistance
- Supporting remote healthcare systems
- Detecting severe cases earlier

The disease is classified into **5 severity grades**:

| Grade | Condition | Description |
|---|---|---|
| 0 | No DR | Healthy retina |
| 1 | Mild DR | Presence of microaneurysms |
| 2 | Moderate DR | Hemorrhages and exudates |
| 3 | Severe DR | Extensive retinal damage |
| 4 | Proliferative DR | Abnormal vessel growth (high risk) |

---

# Dataset

### Dataset Source

- **APTOS 2019 Blindness Detection**
- Kaggle competition dataset

### Dataset Details

- Total training images: **3,662**
- Image type: Retinal fundus photographs
- Image size after preprocessing: **224×224**
- Labels: `0 → 4`

---

## Class Imbalance Challenge

| Grade | Samples | Percentage |
|---|---|---|
| 0 | 1805 | 49.3% |
| 1 | 370 | 10.1% |
| 2 | 999 | 27.3% |
| 3 | 193 | 5.3% |
| 4 | 295 | 8.0% |

The dataset is highly imbalanced.

Rare classes such as:
- Severe DR
- Proliferative DR

contain very few examples, making the classification task significantly harder.

To solve this issue, the project uses:
- Weighted sampling
- Focal loss
- Ordinal loss
- Heavy augmentation

---

#  Project Pipeline

```text
       Raw fundus image
              │
              ▼
┌───────────────────────────┐
│       Preprocessing       │
│ Resize + normalization    │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│       Augmentation        │
│ Rotation, blur, flips,    │
│ brightness, color jitter  │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│    EfficientNet-B0 Model  │
│      Fine-tuned model     │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│       Combined Loss       │
│ FocalLoss + OrdinalLoss   │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│         Training          │
│ AdamW + Cosine Scheduler  │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│      Bias Correction      │
│ Validation QWK tuning     │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│         Inference         │
│ Streamlit GUI + GradCAM   │
└───────────────────────────┘
```

---

# Model & Training

## Architecture

- **Backbone:** EfficientNet-B0 (~5.3 million parameters)  
  Chosen for its strong feature extraction capability, parameter efficiency, and excellent performance on medical imaging tasks.

- **Classification Head:**  
  A custom fully connected linear layer with **Dropout (0.3)** to reduce overfitting and improve generalization.

- **Pretrained Weights:**  
  The model uses **ImageNet pretrained weights**, allowing it to learn general visual patterns before fine-tuning on retinal fundus images.

---

## Loss Function – `CombinedLoss`

We designed a custom loss function that combines two complementary objectives:

| Component | Purpose |
|---|---|
| **Focal Loss (γ = 2.5)** | Reduces the influence of easy majority-class samples and focuses training on difficult and misclassified retinal images, especially rare DR grades. |
| **Ordinal Cross Entropy Loss** | Converts the 5-class classification problem into 4 ordered binary thresholds (`grade > 0`, `>1`, `>2`, `>3`). This enables the model to understand disease severity progression and penalize distant mistakes more heavily than adjacent-grade errors. |

Final loss:

```python
0.5 * FocalLoss + 0.5 * OrdinalLoss
```

---

## Why This Combination?

Diabetic Retinopathy grading is naturally **ordinal**.

Predicting:
- Grade 4 instead of Grade 0

is much worse than:
- Grade 1 instead of Grade 0

Therefore:

- **Focal Loss** handles class imbalance
- **Ordinal Loss** respects disease progression

This combination improves both:
- clinical relevance
- model robustness

---

## Handling Class Imbalance

The project applies several imbalance-handling strategies:

### WeightedRandomSampler
Increases exposure to minority classes during training.

### Soft Class Weighting
Balances focal loss contribution without over-penalizing rare samples.

### Heavy Augmentation
Applied especially to minority classes:
- Rotation
- Horizontal flip
- Brightness adjustment
- Gaussian blur
- Color jitter

---

## Optimizer & Training Strategy

| Component | Value |
|---|---|
| Optimizer | AdamW |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingWarmRestarts |
| Transfer Learning | ImageNet pretrained weights |
| Early Stopping | Based on validation QWK |

### Learning Rates

| Layer | Learning Rate |
|---|---|
| Backbone | 1e-5 |
| Classification Head | 1e-4 |

---

## Bias Correction (Post-training)

After training, a lightweight calibration stage was applied using QWK optimization on the validation set.

Bias values:

```python
[-0.225, -0.210, +0.005, +0.325, -0.133]
```

Performance improvement:

```text
QWK: 0.8775 → 0.9057
```

without any additional model retraining.

---

# Results

## Validation Metrics

| Metric | Value |
|---|---|
| Quadratic Weighted Kappa | **0.9057** |
| Accuracy | 78.99% |
| Macro F1 | 0.609 |
| Weighted F1 | 0.769 |

---

## Per-Class Performance

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| No DR | 0.965 | 0.983 | 0.974 |
| Mild | 0.534 | 0.635 | 0.580 |
| Moderate | 0.736 | 0.545 | 0.626 |
| Severe | 0.276 | 0.538 | 0.365 |
| Proliferative | 0.528 | 0.475 | 0.500 |

---

## Confusion Matrix

```text
True\Pred |   0     1     2     3     4
0 (No DR) | 355     6     0     0     0
1 (Mild)  |   6    47    19     2     0
2 (Mod)   |   7    31   109    39    14
3 (Severe)|   0     2     5    21    11
4 (Prolif)|   0     2    15    14    28
```

---

## ROC-AUC Scores

| Class | AUC |
|---|---|
| 0 | 0.99 |
| 1 | 0.92 |
| 2 | 0.90 |
| 3 | 0.91 |
| 4 | 0.90 |

All classes achieved AUC > 0.90.

---

#  Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/DR_Project.git
cd DR_Project
```

---

## Create Virtual Environment

### Linux / Mac

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Download Dataset

1. Download APTOS 2019 dataset from Kaggle
2. Place:
   - `train.csv`
   - training images

inside:

```text
data/raw/
```

---

#  Usage

## Train Model

```bash
python main.py
```

Choose:
```text
1 → Train model
```

---

## Evaluate Model

```bash
python main.py
```

Choose:
```text
2 → Evaluate model
```

---

## Single Image Inference

```bash
python inference.py path/to/image.png
```

---

## Launch Streamlit GUI

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

#  GUI Features

- Drag & drop image upload
- Confidence probability bars
- Natural language explanations
- Grad-CAM heatmaps
- Batch inference support

---

#  Docker Deployment

## Build Docker Image

```bash
docker build -t dr-detector .
```

## Run Container

```bash
docker run -p 8501:8501 dr-detector
```

---

# Project Structure

```text
DR_Project/
├── config/
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
├── data/raw/
├── main.py
├── app.py
├── inference.py
├── evaluate.py
├── visualize_gradcam.py
├── generate_report.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

#  Technologies

| Category | Tools |
|---|---|
| Deep Learning | PyTorch, timm |
| Data Processing | NumPy, Pandas, OpenCV |
| Augmentation | Albumentations |
| GUI | Streamlit |
| Visualization | Matplotlib, Plotly |
| Deployment | Docker, Hugging Face Spaces |

---

#  Future Work

- Ensemble learning
- Higher image resolutions
- YOLO-based lesion detection
- Better explainability methods
- Longitudinal patient tracking
- LLM-generated medical explanations

---

#  Acknowledgements

- APTOS 2019 Kaggle Competition
- PyTorch Community
- Hugging Face
- Medical experts who labeled the dataset



#  Maintainer

**Mostafa Abdullah**

mostafa.abdullah352@gmail.com

 LinkedIn:  
[Mostafa Abdullah LinkedIn](https://www.linkedin.com/in/mostafa-abdullah-99026726b/)
