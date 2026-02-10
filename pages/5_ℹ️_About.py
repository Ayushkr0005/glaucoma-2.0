import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import MODEL_INFO, get_disclaimer, get_clinical_notes
from src.custom_model import get_model_architecture_info

st.set_page_config(page_title="About - Glaucoma Detection", page_icon="ℹ️", layout="wide")

if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def get_css():
    is_dark = st.session_state.theme == 'dark'
    bg_main = '#0f172a' if is_dark else '#ffffff'
    bg_card = '#1e293b' if is_dark else '#f8fafc'
    text_primary = '#f1f5f9' if is_dark else '#1e293b'
    border_color = '#475569' if is_dark else '#e2e8f0'
    accent = '#3b82f6'
    
    return f"""
    <style>
        .stApp {{ background-color: {bg_main}; }}
        .page-title {{ color: {accent}; font-size: 2.2rem; font-weight: 700; text-align: center; padding: 1rem 0; border-bottom: 2px solid {border_color}; margin-bottom: 2rem; }}
        .card {{ background: {bg_card}; border: 1px solid {border_color}; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }}
        .warning-box {{ background: #fef3c7; border: 1px solid #f59e0b; border-radius: 10px; padding: 1.5rem; margin: 1rem 0; color: #92400e; }}
        .danger-box {{ background: #fee2e2; border: 1px solid #ef4444; border-radius: 10px; padding: 1.5rem; margin: 1rem 0; color: #991b1b; }}
        .info-box {{ background: #dbeafe; border: 1px solid #3b82f6; border-radius: 10px; padding: 1.5rem; margin: 1rem 0; color: #1e40af; }}
    </style>
    """

st.markdown(get_css(), unsafe_allow_html=True)

st.markdown('<div class="page-title">ℹ️ About This System</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📖 Overview", "🏗️ Architecture", "📊 Models", "⚠️ Disclaimers", "📚 References"])

with tab1:
    st.markdown("## Project Overview")
    
    st.markdown("""
    This Glaucoma Detection System is an AI-powered tool for analyzing fundus images 
    and detecting signs of glaucoma using deep learning.
    
    ### Key Features
    
    - **Binary Classification:** Normal vs Glaucoma detection
    - **17 Models:** 3 custom GlaucoNet architectures + 14 pre-trained models
    - **Advanced Preprocessing:** Retinal-specific image processing (CLAHE, Ben Graham, vessel enhancement)
    - **Adjustable Threshold:** Set your own classification threshold
    - **Severity Mapping:** Confidence-based severity estimation
    - **Grad-CAM:** Visual explanations of model focus
    - **Clinical Guidance:** Recommended actions based on results
    - **Test-Time Augmentation:** Enhanced prediction reliability
    
    ### How It Works
    
    1. User uploads a fundus image
    2. Selects a model and threshold
    3. Image is preprocessed with retinal-specific pipeline
    4. Model outputs confidence score
    5. Score is compared to threshold for classification
    6. Severity level derived from confidence
    7. Grad-CAM shows attention regions
    
    ### Severity Mapping
    
    | Confidence | Severity | Est. CDR |
    |------------|----------|----------|
    | 0.0 - 0.3 | Normal | < 0.3 |
    | 0.3 - 0.5 | Borderline | 0.3 - 0.5 |
    | 0.5 - 0.7 | Early | 0.5 - 0.6 |
    | 0.7 - 0.85 | Moderate | 0.6 - 0.7 |
    | 0.85 - 0.95 | Severe | 0.7 - 0.9 |
    | 0.95 - 1.0 | Critical | > 0.9 |
    """)

with tab2:
    st.markdown("## System Architecture")
    
    st.code("""
User Upload → Preprocessing → Model Selection → Inference → Post-processing → Results
                    ↓                ↓
         [Retinal-Specific]  [17 Models Available]
         - Black border      Custom:
           removal            - GlaucoNet (SE + Residual)
         - Illumination       - GlaucoNet-V2 (CBAM + ASPP)
           normalization      - GlaucoNet-V3 (Hybrid CNN-Attention)
         - CLAHE             Pre-trained:
         - Ben Graham         - ResNet50, ResNet50V2
         - Denoising          - VGG16, VGG19
         - Sharpening         - DenseNet121, DenseNet169, DenseNet201
                              - InceptionV3, Xception
                              - MobileNetV2, EfficientNetB0
                              - EfficientNetV2S, EfficientNetV2M
                              - NASNetMobile
    """)
    
    st.markdown("### GlaucoNet Architecture")
    
    arch = get_model_architecture_info()
    st.markdown(f"**{arch['name']}:** {arch['description']}")
    
    st.markdown("**Layers:**")
    st.code('\n'.join(arch['architecture']))
    
    st.markdown("### Image Preprocessing Pipelines")
    st.markdown("""
    **Standard:** Border removal → Illumination normalization → CLAHE
    
    **Ben Graham:** Border removal → Ben Graham color normalization → CLAHE
    
    **Full Pipeline:** Border removal → Denoising → Illumination normalization → CLAHE → Sharpening
    
    **Green Channel:** Border removal → Green channel extraction → CLAHE
    
    **Vessel Enhanced:** Border removal → Illumination normalization → Vessel enhancement
    
    **Optic Disc Focus:** Border removal → Illumination normalization → Optic disc ROI crop → CLAHE
    """)
    
    st.markdown("### Training Pipeline")
    st.markdown("""
    **Two-Stage Training:**
    - Stage 1: Freeze base, train classifier head (10-15 epochs, lr=0.001)
    - Stage 2: Unfreeze top 30% layers, fine-tune (5-10 epochs, lr=0.00001)
    
    **Configuration:**
    - Loss: Combined Focal Loss + Binary Cross-Entropy
    - Optimizer: AdamW with weight decay
    - Advanced augmentation: MixUp, CutMix, CLAHE, GridDistortion
    - Class weighting for imbalanced data (3.4:1 ratio)
    - Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    """)

with tab3:
    st.markdown("## Model Details")
    
    for name, info in MODEL_INFO.items():
        with st.expander(f"📦 {name}"):
            st.markdown(f"**Description:** {info['description']}")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Parameters", info['params'])
            with col2:
                st.metric("Input Size", info['input_size'])
            if 'highlights' in info:
                st.markdown("**Key Features:** " + ", ".join(info['highlights']))

with tab4:
    st.markdown("## Important Disclaimers")
    
    st.markdown("""
    <div class="danger-box">
    <h4>⚠️ Medical Disclaimer</h4>
    <p><b>This tool is for educational and research purposes only.</b></p>
    <p>Predictions should NOT replace professional medical diagnosis. 
    Always consult a qualified ophthalmologist for proper evaluation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <h4>📊 CDR Estimation Disclaimer</h4>
    <p>The Cup-to-Disc Ratio (CDR) values are <b>estimates derived from confidence scores</b>, 
    NOT actual measurements from the image. Accurate CDR requires clinical examination.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>🔬 Model Limitations</h4>
    <ul>
        <li>Models may not generalize to all populations</li>
        <li>Image quality affects accuracy</li>
        <li>False positives/negatives are possible</li>
        <li>Results require clinical verification</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📚 Clinical Notes"):
        st.markdown(get_clinical_notes())

with tab5:
    st.markdown("## References")
    
    st.markdown("""
    ### Novel Architectures
    - **GlaucoNet:** Custom CNN with SE attention and residual connections
    - **GlaucoNet-V2:** CBAM attention + ASPP multi-scale features + GELU activation
    - **GlaucoNet-V3:** Hybrid CNN-Attention with patch stem + SE/CBAM attention
    
    ### Pre-trained Models
    - **ResNet:** He et al. (2016) - Deep Residual Learning
    - **VGG:** Simonyan & Zisserman (2014) - Very Deep CNNs
    - **DenseNet:** Huang et al. (2017) - Densely Connected CNNs
    - **InceptionV3:** Szegedy et al. (2016) - Rethinking Inception
    - **Xception:** Chollet (2017) - Depthwise Separable Convolutions
    - **MobileNetV2:** Sandler et al. (2018) - Inverted Residuals
    - **EfficientNet:** Tan & Le (2019, 2021) - Compound Scaling
    - **NASNet:** Zoph et al. (2018) - Neural Architecture Search
    
    ### Image Processing
    - **CLAHE:** Pizer et al. (1987) - Adaptive Histogram Equalization
    - **Ben Graham:** Graham (2015) - Kaggle Diabetic Retinopathy Winner
    - **CBAM:** Woo et al. (2018) - Convolutional Block Attention Module
    - **SE-Net:** Hu et al. (2018) - Squeeze-and-Excitation Networks
    
    ### Resources
    - [Glaucoma Research Foundation](https://www.glaucoma.org/)
    - [National Eye Institute](https://www.nei.nih.gov/)
    - [TensorFlow](https://www.tensorflow.org/)
    - [Streamlit](https://docs.streamlit.io/)
    """)
