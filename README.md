<div align="center">

# 🏥 Vertebral Disease Detection System

### AI-Powered Spinal X-Ray Analysis for Early Diagnosis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io)

</div>

---

## 🔬 Overview

The **Vertebral Disease Detection System** is an end-to-end deep learning pipeline designed to classify spinal X-rays as **Healthy** or **Abnormal** with exceptional accuracy. Leveraging advanced digital image processing techniques and a custom CNN architecture, this system demonstrates the potential of AI in medical diagnostics.

> **⚠️ Medical Disclaimer:** This is an educational proof-of-concept project. It should **NOT** be used for actual medical diagnosis without proper clinical validation and physician oversight.

### 🎯 Project Objectives

- ✅ Build a robust binary classifier for vertebral abnormality detection
- ✅ Implement production-grade preprocessing pipelines using OpenCV
- ✅ Overcome common ML challenges (class imbalance, overfitting, gradient issues)
- ✅ Deploy an interactive web interface for real-time predictions
- ✅ Achieve research-level performance metrics

---

## ✨ Features

### 🧠 **Advanced Deep Learning**
- Custom CNN architecture optimized for medical imaging
- Global Average Pooling to prevent parameter explosion
- Class-weighted training to handle dataset imbalance
- Strategic dropout (60%) for robust generalization

### 👁️ **Medical Image Processing**
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for enhanced contrast
- Gaussian blur filtering for noise reduction
- Standardized preprocessing pipeline
- Automated image quality normalization

### 🎨 **Data Augmentation**
- Dynamic rotation (±20°)
- Random zoom (80%-120%)
- Horizontal flipping
- Width/height shifts for translation invariance

### 🌐 **Production-Ready Deployment**
- Interactive Streamlit web interface
- Real-time image upload and prediction
- Confidence score visualization
- Responsive design for all devices

---

## 🎬 Demo

### Web Interface
```bash
streamlit run app.py
```

**Upload an X-ray → Get Instant Diagnosis**

<div align="center">

| Input X-Ray | Preprocessing | Prediction |
|------------|---------------|------------|
| Raw Medical Image | CLAHE Enhancement | Healthy / Abnormal |

</div>

---

## 🛠️ Technology Stack

<div align="center">

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white) ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white) |
| **Computer Vision** | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white) |
| **ML Tools** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) |
| **Data Science** | ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) |
| **Visualization** | ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white) |
| **Web Framework** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white) |

</div>

---

## 🏗️ Architecture

### System Pipeline

```
Raw X-Ray Image
      ↓
Preprocessing (Grayscale Conversion)
      ↓
CLAHE Enhancement (Clip Limit: 1.2)
      ↓
Gaussian Blur (3×3 Kernel)
      ↓
Resize to 224×224
      ↓
CNN Model
      ↓
Prediction (Healthy/Abnormal)
```

### CNN Architecture

```
Input (224×224×1)
    ↓
Conv2D(32) → ReLU → MaxPool
    ↓
Conv2D(64) → ReLU → MaxPool
    ↓
Conv2D(128) → ReLU → MaxPool
    ↓
GlobalAveragePooling2D
    ↓
Dense(64, he_normal) → ReLU
    ↓
Dropout(0.6)
    ↓
Dense(1, sigmoid) → Output
```

**Key Innovation:** GlobalAveragePooling reduced parameters from **12.8M → 16K** while maintaining accuracy!

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/RaviKumarYadav15/vertebral-disease-detection-v1.0
cd vertebral-disease-detection

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
opencv-python==4.8.0.76
tensorflow==2.16.1
scikit-learn==1.3.2
numpy==1.26.2
matplotlib==3.8.2
streamlit==1.29.0
pillow>=9.3.0
pandas>=1.5.0
```

---

## 🚀 Usage

### Training Pipeline

```bash
# Step 1: Preprocess raw X-ray images
python -m src.preprocess

# Step 2: Train baseline logistic regression model
python -m src.train_baseline

# Step 3: Train custom CNN model
python -m src.train_cnn

# Step 4: Compare model performance
python -m src.compare_models
```

### Web Application

```bash
# Launch Streamlit interface
streamlit run app.py
```

Then navigate to `http://localhost:8501` in your browser.

### Programmatic Inference

```python
from src.predict import predict_image

# Predict on a single image
result = predict_image('path/to/xray.jpg')
print(f"Prediction: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 📁 Project Structure

```
vertebral-disease-detection/
│
├── 📂 data/
│   ├── raw/                    # Original X-ray images (500 healthy, 500 abnormal)
│   └── processed/              # Enhanced images after DIP pipeline
│
├── 📂 models/                  # Saved model checkpoints
│   ├── baseline_logistic.pkl   # Baseline ML model
│   └── cnn_spine_v1.keras      # Trained CNN model
│
├── 📂 src/                     # Source code
│   ├── config.py               # Configuration & hyperparameters
│   ├── preprocess.py           # Image preprocessing pipeline
│   ├── model_cnn.py            # CNN architecture definition
│   ├── train_cnn.py            # Training script with callbacks
│   ├── predict.py              # Inference engine
│   └── compare_models.py       # Model evaluation & comparison
│
├── 📄 app.py                   # Streamlit web application
├── 📊 training_history.png     # Training/validation curves
├── 📋 requirements.txt         # Python dependencies
├── 📖 README.md                # Project documentation
└── 📜 LICENSE                  # MIT License
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset:** Vertebral X-ray images from [source/repository]
- **Inspiration:** Medical AI research community
- **Libraries:** TensorFlow, OpenCV, Streamlit teams
- **Guidance:** Deep learning and computer vision best practices

---

## 📬 Contact

**Ravi** - Project Author

- GitHub: [@RaviKumarYadav15](https://github.com/RaviKumarYadav15)
---

<div align="center">

### ⭐ If you found this project helpful, please consider giving it a star!

**Made with ❤️ and 🧠 for advancing medical AI**

[⬆ Back to Top](#-vertebral-disease-detection-system)

</div>
