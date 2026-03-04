# GCP Detection Assignment

# Aerial GCP Detection

## Overview

This project detects Ground Control Point (GCP) markers in aerial drone imagery.
For each image, the model predicts:

1. The pixel coordinates of the marker center `(x, y)`
2. The marker shape (`Cross`, `Square`, or `L-Shaped`)

The goal is to automate GCP localization, which is a key step in aerial surveying and photogrammetry pipelines.

---

## Dataset

The dataset consists of high-resolution aerial images (~4096×2730) containing GCP markers.

Each training image has annotations in `curated_gcp_marks.json`:

```
{
  "image_path.jpg": {
    "mark": { "x": ..., "y": ... },
    "verified_shape": "Cross"
  }
}
```

Observations from EDA:

* The marker occupies a very small region of the image
* Marker centers are distributed across the entire frame
* Dataset is imbalanced (Square markers dominate)
* A few samples are missing shape labels

---

## Approach

### 1. Data Processing

* Images resized to **256×256** for faster training
* Coordinates normalized to `[0,1]`
* Missing shape labels filtered out
* Dataset split into **80% training / 20% validation**

### 2. Model Architecture

The model uses a **MobileNetV2 backbone** with two output heads:

```
Image
  ↓
MobileNetV2 Backbone
  ↓
Shared Feature Vector
  ↓
├── Coordinate Head → (x, y)
└── Classification Head → shape
```

* Coordinate prediction: regression
* Shape prediction: multi-class classification

### 3. Loss Function

Total loss is the sum of:

```
Loss = MSELoss(coords) + CrossEntropyLoss(shape)
```

Class weights were used to address class imbalance.

---

## Training

Training configuration:

* Image size: 256×256
* Batch size: 16
* Optimizer: Adam
* Learning rate: 1e-4
* Epochs: 8

Validation loss was used to save the **best model checkpoint**.

---

## Inference

After training, the model predicts coordinates and shape for each test image.

Normalized coordinates are converted back to pixel coordinates using the original image dimensions.

Running inference generates:

```
outputs/predictions.json
```

---

## How to Run

### Train the Model

```
python src/train.py
```

### Run Inference

```
python src/inference.py
```

This generates the required `predictions.json` file.

---

## Project Structure

```
gcp_detection/
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
│
├── notebooks/
│   └── eda.ipynb
│
├── outputs/
│   ├── model.pth
│   └── predictions.json
│
├── README.md
└── requirements.txt
```

---

## Dependencies

Install requirements with:

```
pip install -r requirements.txt
```

---

## Results

The model successfully learns to:

* Predict marker center coordinates
* Classify marker shape

Validation loss decreases consistently during training, indicating good model convergence.

---
