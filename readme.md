# Glaucoma Detection System - Enhanced Edition

## Overview
An AI-powered Streamlit application for detecting glaucoma from retinal fundus images using deep learning. Features 17 models including 3 novel custom architectures (GlaucoNet, V2, V3) and 14 state-of-the-art pre-trained models, trained on the RFMID (Retinal Fundus Multi-Disease Image Dataset) from Kaggle.

## Recent Changes
- **2026-02-09**: Expanded to 17 models for comprehensive comparison
  - Added original GlaucoNet custom architecture
  - Added ResNet50, VGG16, VGG19, DenseNet121, DenseNet169
  - Added InceptionV3, Xception, MobileNetV2, EfficientNetB0, NASNetMobile
  - Created train_single.py for per-model training with memory management
  - Created run_training.sh for batch training all models
- **2026-02-05**: Major enhancement for publication-quality results
  - Switched to RFMID dataset from Kaggle for glaucoma detection
  - Created novel GlaucoNet-V2 architecture with CBAM attention and multi-scale ASPP features
  - Created GlaucoNet-V3 with hybrid CNN-Attention design
  - Implemented advanced data augmentation with MixUp and CutMix
  - Added Focal Loss and combined loss functions for class imbalance
  - Created comprehensive training pipeline with two-stage fine-tuning

## Project Architecture

### Directory Structure
```
glaucoma_detection/
├── app.py                      # Main Streamlit entry point
├── train_enhanced.py           # Enhanced training pipeline (all models)
├── train_single.py             # Single model training script
├── train_all.py                # Batch training orchestrator
├── run_training.sh             # Shell script to train all models
├── pages/                      # Streamlit multi-page structure
│   ├── 1_🏠_Home.py           # Overview with RFMID dataset info
│   ├── 2_🔬_Prediction.py     # Image upload and prediction
│   ├── 3_📊_Comparison.py     # Multi-model comparison
│   ├── 4_📈_Analytics.py      # Performance dashboard
│   └── 5_ℹ️_About.py          # Documentation
├── src/                        # Source modules
│   ├── models_enhanced.py      # All architectures (custom + pretrained)
│   ├── data_pipeline.py        # Advanced data loading with augmentation
│   ├── custom_model.py         # Original GlaucoNet architecture
│   ├── evaluation.py           # Metrics and Grad-CAM visualization
│   └── utils.py                # Shared utilities (17 models)
├── saved_models/               # Trained model files (.keras)
├── results/                    # Training results
│   ├── plots/                  # Training curves, confusion matrices
│   └── metrics/                # JSON metrics files
└── data/rfmid/                 # RFMID dataset (ODC column for glaucoma)
    ├── Training_set/ (1920 images)
    ├── Validation_set/ (640 images)
    └── Test_set/ (640 images)
```

### Novel Architectures

#### GlaucoNet (Original)
- Custom CNN with residual connections
- Squeeze-Excitation attention blocks
- Progressive feature extraction (32→64→128→256→512)

#### GlaucoNet-V2
- CBAM (Convolutional Block Attention Module) at multiple levels
- Multi-scale feature extraction using ASPP-style dilated convolutions
- Residual connections with GELU activation
- Dual global pooling (GAP + GMP)

#### GlaucoNet-V3
- Patch-like stem convolution (4x4 stride)
- Progressive feature extraction (48→96→192→384 channels)
- Squeeze-Excitation channel attention throughout
- CBAM spatial-channel attention
- Hybrid CNN-attention design with dual global pooling

### Models (17 Total)
| Model | Type | Input Size | Key Features |
|-------|------|------------|--------------|
| GlaucoNet | Custom | 224x224 | SE Attention + Residual |
| GlaucoNet_V2 | Custom | 224x224 | CBAM + ASPP + GELU |
| GlaucoNet_V3 | Custom | 224x224 | Hybrid + SE + CBAM |
| ResNet50 | Pretrained | 224x224 | Skip connections |
| ResNet50V2 | Pretrained | 224x224 | Pre-activation design |
| VGG16 | Pretrained | 224x224 | Classic 3x3 convolutions |
| VGG19 | Pretrained | 224x224 | Deeper VGG variant |
| DenseNet121 | Pretrained | 224x224 | Dense connections (compact) |
| DenseNet169 | Pretrained | 224x224 | Dense connections (medium) |
| DenseNet201 | Pretrained | 224x224 | Dense connections (deep) |
| InceptionV3 | Pretrained | 299x299 | Multi-scale features |
| Xception | Pretrained | 299x299 | Depthwise separable conv |
| MobileNetV2 | Pretrained | 224x224 | Inverted residuals |
| EfficientNetB0 | Pretrained | 224x224 | Compound scaling |
| EfficientNetV2S | Pretrained | 384x384 | Fused-MBConv |
| EfficientNetV2M | Pretrained | 480x480 | Best accuracy |
| NASNetMobile | Pretrained | 224x224 | NAS optimized |

### Dataset: RFMID
- Source: Kaggle (ozlemhakdagli/retinal-fundus-multi-disease-image-dataset-rfmid)
- Task: Binary classification (ODC - Optic Disc Cupping for Glaucoma)
- Training: 1920 images (1638 normal, 282 glaucoma)
- Validation: 640 images
- Test: 640 images
- Format: High-quality PNG fundus images

### Training Features
- Two-stage training: classifier head → fine-tuning with unfreezing
- Advanced augmentation: MixUp, CutMix, CLAHE, Equalize
- Combined loss: Focal Loss + Binary Cross-Entropy
- AdamW optimizer with cosine annealing LR
- Class weighting for imbalanced data (3.4:1 ratio)
- K-fold cross-validation support

### Running the Application
```bash
streamlit run app.py --server.port 5000
```

### Training Models
```bash
# Train a single model
python train_single.py GlaucoNet 10 5

# Train all models
bash run_training.sh

# Train with original pipeline
python train_enhanced.py
```

### Severity Mapping
| Confidence | Severity | Est. CDR |
|------------|----------|----------|
| 0.0-0.3 | Normal | < 0.3 |
| 0.3-0.5 | Borderline | 0.3-0.5 |
| 0.5-0.7 | Early | 0.5-0.6 |
| 0.7-0.85 | Moderate | 0.6-0.7 |
| 0.85-0.95 | Severe | 0.7-0.9 |
| 0.95-1.0 | Critical | > 0.9 |

## Publication Notes
- 17 models for comprehensive comparison study
- 3 novel custom architectures designed for originality
- Comprehensive ablation studies supported
- Cross-validation for statistical significance
- Grad-CAM for interpretability
- All metrics logged for reproducibility

## User Preferences
- Clinical color scheme (white/blue background)
- Medical disclaimers on all predictions
- CDR values marked as estimates
