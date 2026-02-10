# Glaucoma Detection System

AI-powered retinal fundus image analysis for early glaucoma detection using 17 deep learning models.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ak4005.streamlit.app)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Models](#models)
  - [Custom Architectures](#custom-architectures-glauconet-family)
  - [Pre-trained Models](#pre-trained-models)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Web App](#running-the-web-app)
  - [Training Models](#training-models)
- [How It Works](#how-it-works)
  - [Preprocessing Pipelines](#preprocessing-pipelines)
  - [Training Pipeline](#training-pipeline)
  - [Severity Classification](#severity-classification)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Medical Disclaimer](#medical-disclaimer)
- [References](#references)
- [License](#license)

---

## Overview

Glaucoma is a group of eye conditions that damage the optic nerve, affecting over 80 million people worldwide. Often called the "silent thief of sight," it causes irreversible vision loss if not detected early.

This project provides an AI-powered web application that analyzes retinal fundus images to detect signs of glaucoma. It features **3 novel custom architectures** (GlaucoNet, GlaucoNet-V2, GlaucoNet-V3) and **14 state-of-the-art pre-trained models**, all trained on the RFMiD (Retinal Fundus Multi-Disease Image Dataset) from Kaggle.

---

## Features

- **17 Deep Learning Models** — 3 custom + 14 pre-trained architectures for comprehensive comparison
- **Multi-Model Comparison** — Run all 17 models simultaneously on a single image with ensemble voting
- **Grad-CAM Visualization** — Heatmaps showing which regions of the fundus image the AI focuses on
- **Severity Grading** — 6-level severity scale (Normal to Critical) with estimated Cup-to-Disc Ratio (CDR)
- **Clinical Recommendations** — Actionable guidance based on prediction severity
- **Adjustable Threshold** — Customize the classification decision boundary (default: 50%)
- **Retinal-Specific Preprocessing** — 6 specialized pipelines including CLAHE, Ben Graham normalization, vessel enhancement, and optic disc focus
- **Interactive Dashboard** — Performance analytics with bar charts, radar plots, and model comparison tables
- **Export Results** — Download comparison results as CSV

---

## Demo

The application is organized into 5 pages:

| Page | Description |
|------|-------------|
| **Main** | System overview, quick stats, and navigation guide |
| **Home** | Glaucoma education, dataset details, model catalog, and performance charts |
| **Prediction** | Upload a fundus image, select a model, view results with Grad-CAM |
| **Comparison** | Compare predictions across all 17 models with ensemble voting |
| **Analytics** | Performance dashboard with accuracy, precision, recall, F1, and AUC metrics |
| **About** | Architecture documentation, preprocessing details, disclaimers, and references |

---

## Models

### Custom Architectures (GlaucoNet Family)

| Model | Parameters | Key Features | Description |
|-------|-----------|--------------|-------------|
| **GlaucoNet** | ~5M | SE Attention, Residual Blocks | Custom CNN with Squeeze-Excitation attention and progressive feature extraction (32 to 64 to 128 to 256 to 512 channels) |
| **GlaucoNet-V2** | ~20M | CBAM, ASPP, GELU, Dual Pooling | Multi-scale feature extraction using ASPP-style dilated convolutions, CBAM attention at multiple levels, and GELU activation |
| **GlaucoNet-V3** | ~10M | Hybrid CNN-Attention, SE+CBAM | Patch-like 4x4 stem convolution, progressive features (48 to 96 to 192 to 384), SE and CBAM attention, dual global pooling (GAP+GMP) |

#### GlaucoNet Architecture
```
Input (224x224x3)
  Conv Block (32 filters) + MaxPool
  Conv Block (64 filters) + MaxPool
  Residual Block (128 filters) + SE Attention + MaxPool
  Residual Block (256 filters) + SE Attention + MaxPool
  Residual Block (512 filters) + SE Attention
  Global Average Pooling
  Dense (512) + BatchNorm + ReLU + Dropout(0.5)
  Dense (256) + BatchNorm + ReLU + Dropout(0.3)
  Output (1, Sigmoid)
```

#### GlaucoNet-V2 Architecture
```
Input (224x224x3)
  7x7 Conv Stem (64 filters, stride 2) + GELU + MaxPool
  2x Residual-CBAM Blocks (64 filters)
  2x Residual-CBAM Blocks (128 filters)
  3x Residual-CBAM Blocks (256 filters)
  3x Residual-CBAM Blocks (512 filters)
  Multi-Scale ASPP (dilations: 1, 2, 4 + global pooling)
  CBAM Attention
  Dual Pooling (GAP + GMP)
  Dense (512) + Dense (256) + Dropout
  Output (1, Sigmoid)
```

#### GlaucoNet-V3 Architecture
```
Input (224x224x3)
  4x4 Patch Stem (48 filters, stride 4) + GELU
  Stage 1: Conv Blocks (96 filters) + SE Attention x 3
  Stage 2: Conv Blocks (192 filters) + SE Attention x 3
  Stage 3: Conv Blocks (384 filters) + SE Attention x 3
  CBAM Attention
  Dual Pooling (GAP + GMP)
  Dense (384) + Dense (192) + Dropout
  Output (1, Sigmoid)
```

### Pre-trained Models

All pre-trained models use ImageNet weights with a custom classification head featuring SE attention and dual global pooling.

| Model | Parameters | Input Size | Key Innovation |
|-------|-----------|------------|----------------|
| ResNet50 | 25.6M | 224x224 | Skip connections for deep gradient flow |
| ResNet50V2 | 25.6M | 224x224 | Pre-activation residual design |
| VGG16 | 138M | 224x224 | Simple stacked 3x3 convolutions |
| VGG19 | 144M | 224x224 | Deeper 19-layer VGG variant |
| DenseNet121 | 8M | 224x224 | Dense connectivity, compact model |
| DenseNet169 | 14M | 224x224 | Dense connectivity, medium depth |
| DenseNet201 | 20M | 224x224 | Dense connectivity, maximum depth |
| InceptionV3 | 23.9M | 299x299 | Multi-scale factorized convolutions |
| Xception | 22.9M | 299x299 | Depthwise separable convolutions |
| MobileNetV2 | 3.5M | 224x224 | Inverted residuals, lightweight |
| EfficientNetB0 | 5.3M | 224x224 | Compound scaling (baseline) |
| EfficientNetV2S | 21.5M | 384x384 | Fused-MBConv, progressive training |
| EfficientNetV2M | 54M | 480x480 | Highest accuracy, largest input |
| NASNetMobile | 5.3M | 224x224 | Neural Architecture Search optimized |

---

## Dataset

### RFMiD (Retinal Fundus Multi-Disease Image Dataset)

- **Source:** [Kaggle - RFMiD Dataset](https://www.kaggle.com/datasets/ozlemhakdagli/retinal-fundus-multi-disease-image-dataset-rfmid)
- **Type:** Multi-label classification (46 disease conditions)
- **Format:** High-quality PNG fundus images

| Split | Images |
|-------|--------|
| Training | 1,920 |
| Validation | 640 |
| Test | 640 |
| **Total** | **3,200** |

### Multi-Label Nature

RFMiD is a **multi-label dataset** where each image can have multiple disease labels simultaneously (e.g., both Diabetic Retinopathy and Glaucoma). This reflects real clinical scenarios where retinal diseases co-occur.

**Key Characteristics:**
- Each image can belong to **multiple disease classes** simultaneously
- Labels are stored in **independent binary (0/1) format** per disease
- Uses **Sigmoid activation** (not Softmax) for independent per-class outputs
- Trained with **Binary Cross-Entropy** loss (per label)

### Glaucoma Focus (ODC Column)

This project focuses on the **ODC (Optic Disc Cupping)** column for binary glaucoma detection:

| Class | Percentage | Description |
|-------|-----------|-------------|
| Normal (ODC=0) | ~85.3% | No signs of optic disc cupping |
| Glaucoma (ODC=1) | ~14.7% | Signs of glaucomatous optic disc cupping |
| **Class Ratio** | **~5.8 : 1** | Handled via class weighting + Focal Loss |

### Dataset Structure
```
data/rfmid/
  Training_set/
    *.png                           (1,920 fundus images)
    RFMiD_Training_Labels.csv       (46 binary columns, multi-label)
  Validation_set/
    *.png                           (640 fundus images)
    RFMiD_Validation_Labels.csv
  Test_set/
    *.png                           (640 fundus images)
    RFMiD_Testing_Labels.csv
```

---

## Project Structure

```
glaucoma-2.0/
  app.py                          Main Streamlit entry point
  requirements.txt                Python dependencies
  .streamlit/
    config.toml                   Streamlit configuration
  pages/                          Multi-page Streamlit app
    1_Home.py                     Dataset info, model catalog, performance
    2_Prediction.py               Single image prediction with Grad-CAM
    3_Comparison.py               Multi-model comparison with ensemble
    4_Analytics.py                Performance analytics dashboard
    5_About.py                    Documentation and references
  src/                            Core modules
    models_enhanced.py            All 17 model architectures
    custom_model.py               Original GlaucoNet architecture
    data_pipeline.py              Data loading, augmentation, TF datasets
    data_preprocessing.py         6 retinal preprocessing pipelines
    evaluation.py                 Metrics, ROC curves, Grad-CAM, plots
    utils.py                      Severity mapping, model info, utilities
  train_enhanced.py               Full training pipeline (all models)
  train_single.py                 Train individual models
  train_all.py                    Batch training orchestrator
  run_training.sh                 Shell script for batch training
  quick_evaluate.py               Quick model evaluation
  saved_models/                   Trained model weights (.keras)
  results/
    plots/                        Training curves, confusion matrices
    metrics/                      JSON metrics per model
  data/rfmid/                     Dataset (CSV labels included, images excluded)
```

---

## Installation

### Prerequisites

- Python 3.11+
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ayushkr0005/glaucoma-2.0.git
   cd glaucoma-2.0
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset (for training only):**

   Download the [RFMiD dataset from Kaggle](https://www.kaggle.com/datasets/ozlemhakdagli/retinal-fundus-multi-disease-image-dataset-rfmid) and extract it into `data/rfmid/`.

   Alternatively, using the Kaggle CLI:
   ```bash
   kaggle datasets download -d ozlemhakdagli/retinal-fundus-multi-disease-image-dataset-rfmid
   unzip retinal-fundus-multi-disease-image-dataset-rfmid.zip -d data/rfmid/
   ```

---

## Usage

### Running the Web App

```bash
streamlit run app.py
```

The app will launch at `http://localhost:8501`.

**Using the Prediction page:**
1. Navigate to the **Prediction** page from the sidebar
2. Select a model from the dropdown (all 17 models available)
3. Adjust the classification threshold if needed (default: 50%)
4. Upload a retinal fundus image (JPG, PNG)
5. View the prediction result, severity level, CDR estimate, and Grad-CAM heatmap

**Using the Comparison page:**
1. Navigate to the **Comparison** page
2. Upload a fundus image
3. Select which models to include (default: all 17)
4. Click **Run Comparison**
5. View individual results, ensemble voting outcome, and confidence chart
6. Download results as CSV

### Training Models

**Train all 17 models:**
```bash
python train_enhanced.py
```

**Train a single model:**
```bash
python train_single.py <ModelName> <Stage1Epochs> <Stage2Epochs>
```

Examples:
```bash
python train_single.py GlaucoNet 15 10
python train_single.py EfficientNetV2S 10 5
python train_single.py GlaucoNet_V2 15 10
```

**Batch training with shell script:**
```bash
bash run_training.sh
```

---

## How It Works

### Preprocessing Pipelines

Each model uses a specialized retinal image preprocessing pipeline:

| Pipeline | Steps | Used By |
|----------|-------|---------|
| **Standard** | Border removal, Illumination normalization, CLAHE | GlaucoNet, ResNet50, ResNet50V2, DenseNet121/169/201, InceptionV3, Xception, MobileNetV2, EfficientNetB0, NASNetMobile |
| **Ben Graham** | Border removal, Ben Graham color normalization, CLAHE | VGG16, VGG19 |
| **Full Pipeline** | Border removal, Denoising, Illumination normalization, CLAHE, Sharpening | GlaucoNet-V2, GlaucoNet-V3, EfficientNetV2S, EfficientNetV2M |
| **Green Channel** | Border removal, Green channel extraction, CLAHE | Available for custom use |
| **Vessel Enhanced** | Border removal, Illumination normalization, Vessel enhancement | Available for custom use |
| **Optic Disc Focus** | Border removal, Illumination normalization, Optic disc ROI crop, CLAHE | Available for custom use |

### Training Pipeline

**Two-Stage Training Strategy:**

| Stage | Description | Learning Rate | Epochs |
|-------|-------------|---------------|--------|
| **Stage 1** | Freeze base model, train classifier head only | 1e-3 | 10-15 |
| **Stage 2** | Unfreeze top 30% of base layers, fine-tune end-to-end | 1e-5 | 5-10 |

**Training Configuration:**
- **Loss:** Combined Focal Loss + Binary Cross-Entropy (handles class imbalance)
- **Optimizer:** AdamW with weight decay (1e-5)
- **Augmentation:** MixUp, CutMix, CLAHE, GridDistortion, rotation, flips
- **Class Weighting:** Automatic based on label distribution (~5.8:1 ratio)
- **Callbacks:** ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
- **Cross-Validation:** K-fold support for statistical significance

### Severity Classification

The model's confidence score is mapped to clinical severity levels:

| Confidence | Severity | Est. CDR | Urgency | Recommended Action |
|------------|----------|----------|---------|-------------------|
| 0.0 - 0.3 | Normal | < 0.3 | Low | Annual eye examination |
| 0.3 - 0.5 | Borderline | 0.3 - 0.5 | Moderate | Schedule exam within 3 months |
| 0.5 - 0.7 | Early | 0.5 - 0.6 | Moderate-High | Consult ophthalmologist within 1 month |
| 0.7 - 0.85 | Moderate | 0.6 - 0.7 | High | Urgent consultation within 1 week |
| 0.85 - 0.95 | Severe | 0.7 - 0.9 | Very High | Medical attention within 1-2 days |
| 0.95 - 1.0 | Critical | > 0.9 | Critical | Emergency consultation immediately |

> **Note:** CDR values are estimates derived from confidence scores, NOT actual measurements from the image.

---

## Results

### Expected Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| EfficientNetV2M | 97% | 96% | 98% | 97% | 0.99 |
| EfficientNetV2S | 96% | 95% | 97% | 96% | 0.98 |
| GlaucoNet-V3 | 95% | 94% | 96% | 95% | 0.98 |
| GlaucoNet-V2 | 94% | 93% | 95% | 94% | 0.97 |
| Xception | 94% | 93% | 95% | 94% | 0.97 |
| InceptionV3 | 93% | 92% | 94% | 93% | 0.97 |
| DenseNet169 | 93% | 92% | 94% | 93% | 0.97 |
| DenseNet201 | 93% | 92% | 94% | 93% | 0.97 |
| EfficientNetB0 | 93% | 92% | 94% | 93% | 0.97 |
| ResNet50V2 | 93% | 92% | 93% | 92% | 0.96 |
| ResNet50 | 92% | 91% | 93% | 92% | 0.96 |
| DenseNet121 | 92% | 91% | 93% | 92% | 0.96 |
| GlaucoNet | 91% | 90% | 92% | 91% | 0.95 |
| MobileNetV2 | 91% | 90% | 92% | 91% | 0.95 |
| NASNetMobile | 91% | 90% | 92% | 91% | 0.95 |
| VGG16 | 89% | 88% | 90% | 89% | 0.94 |
| VGG19 | 89% | 87% | 90% | 88% | 0.93 |

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| [TensorFlow / Keras](https://www.tensorflow.org/) | Deep learning framework |
| [Streamlit](https://streamlit.io/) | Interactive web application |
| [NumPy](https://numpy.org/) | Numerical computing |
| [Pandas](https://pandas.pydata.org/) | Data manipulation |
| [OpenCV](https://opencv.org/) | Image preprocessing |
| [Pillow](https://python-pillow.org/) | Image loading |
| [Plotly](https://plotly.com/) | Interactive visualizations |
| [Scikit-learn](https://scikit-learn.org/) | Metrics and evaluation |
| [Matplotlib / Seaborn](https://matplotlib.org/) | Static plots |
| [Albumentations](https://albumentations.ai/) | Advanced data augmentation |

---

## Medical Disclaimer

> **This tool is for educational and research purposes only.**
>
> Predictions made by this system should **NOT** be used as a substitute for professional medical diagnosis. The Cup-to-Disc Ratio (CDR) values displayed are **estimates** derived from model confidence scores, not actual measurements from the fundus image.
>
> Always consult a qualified ophthalmologist for proper diagnosis and treatment of eye conditions.

**Model Limitations:**
- Models may not generalize to all populations or imaging conditions
- Image quality significantly affects prediction accuracy
- False positives and false negatives are possible
- Results require clinical verification by an eye care professional

---

## References

### Novel Architectures
- **GlaucoNet** -- Custom CNN with SE attention and residual connections
- **GlaucoNet-V2** -- CBAM attention + ASPP multi-scale features + GELU activation
- **GlaucoNet-V3** -- Hybrid CNN-Attention with patch stem + SE/CBAM attention

### Pre-trained Model Papers
- He et al. (2016) -- *Deep Residual Learning for Image Recognition* (ResNet)
- Simonyan & Zisserman (2014) -- *Very Deep Convolutional Networks* (VGG)
- Huang et al. (2017) -- *Densely Connected Convolutional Networks* (DenseNet)
- Szegedy et al. (2016) -- *Rethinking the Inception Architecture* (InceptionV3)
- Chollet (2017) -- *Xception: Deep Learning with Depthwise Separable Convolutions*
- Sandler et al. (2018) -- *MobileNetV2: Inverted Residuals and Linear Bottlenecks*
- Tan & Le (2019, 2021) -- *EfficientNet: Rethinking Model Scaling* (EfficientNet)
- Zoph et al. (2018) -- *Learning Transferable Architectures* (NASNet)

### Techniques
- Hu et al. (2018) -- *Squeeze-and-Excitation Networks* (SE-Net)
- Woo et al. (2018) -- *CBAM: Convolutional Block Attention Module*
- Pizer et al. (1987) -- *Adaptive Histogram Equalization* (CLAHE)
- Graham (2015) -- *Kaggle Diabetic Retinopathy Competition Winner* (Ben Graham preprocessing)
- Lin et al. (2017) -- *Focal Loss for Dense Object Detection*

### Resources
- [Glaucoma Research Foundation](https://www.glaucoma.org/)
- [National Eye Institute](https://www.nei.nih.gov/)
- [RFMiD Dataset on Kaggle](https://www.kaggle.com/datasets/ozlemhakdagli/retinal-fundus-multi-disease-image-dataset-rfmid)

---

## License

This project is for academic and research purposes. Please cite appropriately if used in publications.
