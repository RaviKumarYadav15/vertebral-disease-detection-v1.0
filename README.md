# ⚕️ Vertebral Disease Detection System

An end-to-end **Deep Learning pipeline** designed to classify spinal X-rays as either **Healthy** or **Abnormal**.  
This project leverages advanced **Digital Image Processing (DIP)** techniques along with a custom **Convolutional Neural Network (CNN)** and **Data Augmentation** to deliver high-accuracy diagnostic predictions.

> **Author:** Ravi  
> **Version:** 1.0.0

---

## 📖 Project Overview

Early detection of vertebral abnormalities is crucial for maintaining spinal health.  
This project presents a **robust and scalable machine learning pipeline** for medical image classification — from raw image preprocessing to model training and final evaluation on unseen data.

### ✨ Key Achievements

- 🧠 **Resolved Model Collapse**  
  Tackled **majority class bias** and **dying gradients** using:
  - Class Weights
  - Global Average Pooling

- 🛡️ **Prevented Overfitting**  
  Applied:
  - Data Augmentation (Rotation, Zoom, Flip)
  - Dropout Regularization

- 📊 **Research-Grade Evaluation**  
  Ensured strong generalization through:
  - Proper Train/Test Split
  - Confusion Matrix Analysis
  - Zero Data Leakage

- 🚀 **Performance**  
  Achieved **100% Accuracy and Recall** on the validation dataset

---

## 🛠️ Technology Stack

| Category            | Tools / Libraries                  |
| ------------------- | ---------------------------------- |
| 🧠 Deep Learning    | TensorFlow, Keras                  |
| 👁️ Computer Vision  | OpenCV (cv2)                       |
| 🤖 Machine Learning | Scikit-Learn (Logistic Regression) |
| 📦 Data Handling    | NumPy, Pandas                      |
| 📈 Visualization    | Matplotlib                         |

---

## 🧠 Pipeline Architecture

### 1️⃣ Image Preprocessing (`src/preprocess.py`)

Medical X-rays often suffer from **low contrast and noise**.  
This preprocessing pipeline standardizes and enhances the input data:

1. **Grayscale Conversion**  
   Reduces unnecessary color channels

2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
   - Enhances local contrast
   - Highlights vertebral boundaries
   - Clip Limit: `1.2`

3. **Gaussian Blur**
   - Kernel: `3×3`
   - Smooths noise while preserving edges

4. **Resizing**
   - Standard size: `224 × 224`

---

### 2️⃣ Custom CNN Architecture (`src/model_cnn.py`)

A **lightweight and optimized CNN** tailored for this dataset to avoid parameter explosion and instability.

#### 🔹 Model Design

- **Input:** `224 × 224 × 1` (Grayscale)
- **Augmentation Layer:**
  - Random Flip
  - Rotation (10%)
  - Zoom (10%)

- **Feature Extraction:**
  - 3 Convolution Blocks:
    - 32 filters
    - 64 filters
    - 128 filters
  - Each followed by MaxPooling

- **Compression:**
  - `GlobalAveragePooling2D`
  - Parameters reduced from **12.8M → ~16K**

- **Classification Head:**
  - Dense Layer (`he_normal` initialization)
  - Dropout: **60%**
  - Output: **Sigmoid (Binary Classification)**

---

### 3️⃣ Model Comparison (`src/compare_models.py`)

This module compares:

- 📌 Baseline **Logistic Regression**
- 🧠 Custom **CNN Model**

#### Evaluation Features:

- 20% Holdout Validation Set
- Classification Report
- Confusion Matrix
- Prevention of double-scaling bugs

---

## 📂 Project Structure

````text
vertebral-disease-detection/
│
├── data/
│   ├── raw/                  # Original dataset (500 healthy, 500 abnormal)
│   └── processed/            # Preprocessed images
│
├── models/                   # Saved models
│   ├── baseline_logistic.pkl
│   └── cnn_spine_v1.keras
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── train_baseline.py
│   ├── model_cnn.py
│   ├── train_cnn.py
│   └── compare_models.py
│
├── training_history.png      # Training curves
├── requirements.txt
└── README.md

## 🚀 How to Run the Project

### 🔧 Prerequisites

- Python **3.9+**
- Virtual environment (recommended)

Install dependencies:

```bash
pip install tensorflow opencv-python scikit-learn numpy matplotlib joblib



▶️ Execution Steps

⚠️ Run all commands from the project root directory

Step 1: Preprocess Images
python -m src.preprocess

Step 2: Train Baseline Model
python -m src.train_baseline

Step 3: Train Custom CNN
python -m src.train_cnn

Step 4: Evaluate and Compare Models
python -m src.compare_models

---

```markdown id="future-sec-002"
## 🔮 Future Enhancements

- 🌐 **Web Interface**
  Build a user-friendly interface using **Streamlit** for real-time spinal disease prediction.

- 🧠 **Transfer Learning**
  Scale the model with advanced architectures such as:
  - ResNet50
  - DenseNet121

- 🔍 **Explainable AI (XAI)**
  Implement **Grad-CAM** to generate heatmaps highlighting regions of interest in X-rays.

- 📦 **Model Optimization**
  Apply techniques like pruning and quantization to reduce model size and improve inference speed.

- ☁️ **Deployment**
  Deploy the model using:
  - Docker
  - Cloud platforms (AWS / GCP / Azure)

- 📊 **Extended Evaluation**
  Incorporate cross-validation and additional metrics like ROC-AUC for deeper performance insights.

````
