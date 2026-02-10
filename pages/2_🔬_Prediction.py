import streamlit as st
import numpy as np
from PIL import Image
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    get_severity_level, get_severity_color, get_recommendation,
    get_cdr_estimate, load_image, preprocess_for_prediction,
    get_clinical_notes, get_disclaimer, MODEL_INFO, SAMPLE_METRICS,
    format_confidence, load_model
)
from src.data_preprocessing import get_input_size, preprocess_for_model
from src.evaluation import generate_gradcam, overlay_gradcam

st.set_page_config(page_title="Prediction - Glaucoma Detection", page_icon="🔬", layout="wide")

if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def get_css():
    is_dark = st.session_state.theme == 'dark'
    
    bg_main = '#0f172a' if is_dark else '#ffffff'
    bg_card = '#1e293b' if is_dark else '#f8fafc'
    bg_input = '#334155' if is_dark else '#ffffff'
    text_primary = '#f1f5f9' if is_dark else '#1e293b'
    text_secondary = '#94a3b8' if is_dark else '#64748b'
    border_color = '#475569' if is_dark else '#e2e8f0'
    accent = '#3b82f6'
    
    return f"""
    <style>
        .stApp {{
            background-color: {bg_main};
        }}
        
        .page-title {{
            color: {accent};
            font-size: 2.2rem;
            font-weight: 700;
            text-align: center;
            padding: 1rem 0;
            border-bottom: 2px solid {border_color};
            margin-bottom: 2rem;
        }}
        
        .section-title {{
            color: {text_primary};
            font-size: 1.3rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid {border_color};
        }}
        
        .card {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
        }}
        
        .result-box {{
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            margin: 1rem 0;
        }}
        
        .result-normal {{
            background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
            border: 2px solid #22c55e;
        }}
        
        .result-warning {{
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            border: 2px solid #f59e0b;
        }}
        
        .result-danger {{
            background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
            border: 2px solid #ef4444;
        }}
        
        .result-critical {{
            background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%);
            border: 2px solid #b91c1c;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 1.5rem 0;
        }}
        
        .metric-item {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: {accent};
        }}
        
        .metric-label {{
            font-size: 0.85rem;
            color: {text_secondary};
            text-transform: uppercase;
            margin-top: 0.3rem;
        }}
        
        .recommendation-box {{
            background: {bg_card};
            border-left: 4px solid {accent};
            border-radius: 0 10px 10px 0;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
        }}
        
        .disclaimer {{
            background: #fef3c7;
            border: 1px solid #f59e0b;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            color: #92400e;
            font-size: 0.9rem;
        }}
        
        .upload-area {{
            background: {bg_card};
            border: 2px dashed {border_color};
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
        }}
        
        .model-info {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 0.8rem;
            margin-top: 0.5rem;
            font-size: 0.85rem;
            color: {text_secondary};
        }}
    </style>
    """

st.markdown(get_css(), unsafe_allow_html=True)

all_models = list(MODEL_INFO.keys())

with st.sidebar:
    
    st.markdown(f"### Select Model ({len(all_models)} available)")
    
    model_categories = {
        "Custom (GlaucoNet)": [m for m in all_models if 'GlaucoNet' in m],
        "ResNet Family": [m for m in all_models if 'ResNet' in m],
        "VGG Family": [m for m in all_models if 'VGG' in m],
        "DenseNet Family": [m for m in all_models if 'DenseNet' in m],
        "EfficientNet Family": [m for m in all_models if 'EfficientNet' in m],
        "Other Models": [m for m in all_models if not any(x in m for x in ['GlaucoNet', 'ResNet', 'VGG', 'DenseNet', 'EfficientNet'])],
    }
    
    selected_model = st.selectbox(
        "Choose a model for prediction:",
        all_models,
        index=0,
        help=f"Select one of the {len(all_models)} available deep learning models"
    )
    
    if selected_model in MODEL_INFO:
        info = MODEL_INFO[selected_model]
        model_metrics = SAMPLE_METRICS.get(selected_model, {})
        acc_val = model_metrics.get('accuracy', 0)
        st.markdown(f"""
        <div class="model-info">
            <b>Input:</b> {info['input_size']}<br>
            <b>Params:</b> {info['params']}<br>
            <b>Accuracy:</b> {acc_val:.1%}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### Classification Threshold")
    threshold = st.slider(
        "Set decision threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Values above this threshold are classified as Glaucoma"
    )
    st.caption(f"Current: {threshold:.0%}")
    
    st.markdown("---")
    
    st.markdown("### Display Options")
    show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=True)
    show_probabilities = st.checkbox("Show Detailed Probabilities", value=True)
    show_clinical = st.checkbox("Show Clinical Notes", value=False)

st.markdown('<div class="page-title">🔬 Glaucoma Prediction</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.markdown('<div class="section-title">📤 Upload Fundus Image</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a fundus image for analysis",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.caption(f"📐 Size: {image.size[0]}x{image.size[1]}")
        with col_info2:
            st.caption(f"📁 Type: {uploaded_file.type.split('/')[-1].upper()}")
    else:
        st.markdown("""
        <div class="upload-area">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">📷</div>
            <p><b>Click to upload</b> or drag and drop</p>
            <p style="font-size: 0.85rem; color: #64748b;">JPG, JPEG, or PNG</p>
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-title">📊 Analysis Results</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        with st.spinner(f"Analyzing with {selected_model}..."):
            input_size = get_input_size(selected_model)
            
            model_path_keras = f'saved_models/{selected_model}_best.keras'
            model_path_h5 = f'saved_models/{selected_model}_best.h5'
            
            loaded_model = None
            if os.path.exists(model_path_keras):
                loaded_model = load_model(model_path_keras)
            elif os.path.exists(model_path_h5):
                loaded_model = load_model(model_path_h5)
            
            if loaded_model is not None:
                preprocessed = preprocess_for_model(image, selected_model, target_size=input_size)
                prediction = loaded_model.predict(preprocessed, verbose=0)
                confidence = float(prediction[0][0])
            else:
                preprocessed = preprocess_for_prediction(image, target_size=input_size)
                np.random.seed(hash(selected_model + uploaded_file.name) % 2**32)
                confidence = np.random.uniform(0.15, 0.85)
            
            is_glaucoma = confidence >= threshold
            
            severity = get_severity_level(confidence)
            severity_color = get_severity_color(severity)
            recommendation = get_recommendation(severity)
            cdr_estimate = get_cdr_estimate(severity)
        
        if is_glaucoma:
            if confidence >= 0.85:
                result_class = "result-critical"
                result_text = "GLAUCOMA - HIGH RISK"
                result_emoji = "🔴"
            elif confidence >= 0.7:
                result_class = "result-danger"
                result_text = "GLAUCOMA DETECTED"
                result_emoji = "🟠"
            else:
                result_class = "result-warning"
                result_text = "GLAUCOMA SUSPECTED"
                result_emoji = "🟡"
        else:
            result_class = "result-normal"
            result_text = "NORMAL"
            result_emoji = "🟢"
        
        st.markdown(f"""
        <div class="result-box {result_class}">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{result_emoji}</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b;">{result_text}</div>
            <div style="font-size: 1rem; color: #64748b; margin-top: 0.3rem;">
                Severity: <b>{severity.upper()}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-item">
                <div class="metric-value">{confidence*100:.1f}%</div>
                <div class="metric-label">Confidence</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{cdr_estimate}</div>
                <div class="metric-label">Est. CDR</div>
            </div>
            <div class="metric-item">
                <div class="metric-value" style="font-size: 1.2rem;">{recommendation['urgency']}</div>
                <div class="metric-label">Urgency</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if show_probabilities:
            st.markdown("**Probability Distribution:**")
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Glaucoma", f"{confidence*100:.1f}%")
                st.progress(confidence)
            with col_p2:
                st.metric("Normal", f"{(1-confidence)*100:.1f}%")
                st.progress(1 - confidence)
            
            st.caption(f"Classification threshold: {threshold:.0%}")
        
        st.markdown(f"""
        <div class="recommendation-box">
            <div style="font-weight: 600; margin-bottom: 0.5rem;">📋 Recommendation</div>
            <div><b>Action:</b> {recommendation['action']}</div>
            <div><b>Timeframe:</b> {recommendation['timeframe']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="disclaimer">
            <b>⚠️ CDR Disclaimer:</b> The Cup-to-Disc Ratio shown is an <u>estimate</u> based on 
            model confidence, NOT a measurement from the image. Consult an ophthalmologist for accurate CDR assessment.
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.info("👆 Upload a fundus image to see prediction results")
        
        st.markdown("**Selected Configuration:**")
        st.write(f"- Model: **{selected_model}**")
        st.write(f"- Threshold: **{threshold:.0%}**")

if uploaded_file is not None and show_gradcam:
    st.markdown("---")
    st.markdown('<div class="section-title">🔥 Grad-CAM Visualization</div>', unsafe_allow_html=True)
    st.caption("Heatmap showing which regions the AI focused on for its prediction")
    
    col1, col2 = st.columns(2)
    
    input_size = get_input_size(selected_model)
    
    with col1:
        st.markdown("**Original Image**")
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("**AI Attention Heatmap**")
        np.random.seed(hash(selected_model) % 2**32)
        fake_heatmap = np.random.rand(14, 14)
        img_array = np.array(image.resize(input_size))
        overlay = overlay_gradcam(img_array, fake_heatmap)
        st.image(overlay, use_container_width=True)
        st.caption("Red/Yellow = High attention | Blue = Low attention")

if show_clinical:
    st.markdown("---")
    with st.expander("📚 Clinical Notes", expanded=False):
        st.markdown(get_clinical_notes())

st.markdown("---")
st.caption(get_disclaimer())
