import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import MODEL_INFO, SAMPLE_METRICS, get_metrics

st.set_page_config(page_title="Home - Glaucoma Detection", page_icon="🏠", layout="wide")

if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def get_css():
    is_dark = st.session_state.theme == 'dark'
    bg_main = '#0f172a' if is_dark else '#ffffff'
    bg_card = '#1e293b' if is_dark else '#f8fafc'
    text_primary = '#f1f5f9' if is_dark else '#1e293b'
    text_secondary = '#94a3b8' if is_dark else '#64748b'
    border_color = '#475569' if is_dark else '#e2e8f0'
    accent = '#3b82f6'
    
    return f"""
    <style>
        .stApp {{ background-color: {bg_main}; }}
        .page-title {{ color: {accent}; font-size: 2.2rem; font-weight: 700; text-align: center; padding: 1rem 0; border-bottom: 2px solid {border_color}; margin-bottom: 2rem; }}
        .card {{ background: {bg_card}; border: 1px solid {border_color}; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }}
        .stat-box {{ background: {bg_card}; border: 1px solid {border_color}; border-radius: 10px; padding: 1rem; text-align: center; }}
        .stat-number {{ font-size: 2rem; font-weight: 700; color: {accent}; }}
    </style>
    """

st.markdown(get_css(), unsafe_allow_html=True)

st.markdown('<div class="page-title">🏠 Glaucoma Detection System</div>', unsafe_allow_html=True)

METRICS, is_trained = get_metrics()
if is_trained:
    st.success("Using trained model metrics from RFMID dataset!")

tab1, tab2, tab3, tab4 = st.tabs(["📖 About Glaucoma", "📊 Dataset", "🤖 Models", "📈 Performance"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### What is Glaucoma?
        
        Glaucoma is a group of eye conditions that damage the optic nerve, 
        often caused by abnormally high eye pressure. It's the leading cause 
        of irreversible blindness worldwide.
        
        **Key Facts:**
        - Affects 80+ million people globally
        - Often called "silent thief of sight"
        - Early detection is crucial
        - Vision loss cannot be recovered
        
        **Types:**
        1. Open-Angle (most common)
        2. Angle-Closure (emergency)
        3. Normal-Tension
        4. Secondary
        """)
    
    with col2:
        st.markdown("""
        ### Risk Factors
        
        - Age over 60
        - Family history
        - High eye pressure
        - Diabetes
        - High myopia
        - Previous eye injury
        
        ### Symptoms
        
        **Early:** Often none
        
        **Advanced:**
        - Blind spots
        - Tunnel vision
        - Eye pain
        - Blurred vision
        - Halos around lights
        """)

with tab2:
    st.markdown("### RFMID Dataset")
    st.markdown("""
    **Retinal Fundus Multi-Disease Image Dataset (RFMiD)**
    
    Source: [Kaggle - RFMID Dataset](https://www.kaggle.com/datasets/ozlemhakdagli/retinal-fundus-multi-disease-image-dataset-rfmid)
    
    RFMiD is a **multi-label classification dataset**, meaning a single fundus image 
    can have **multiple disease labels** assigned to it simultaneously. This reflects 
    real-world clinical scenarios where retinal diseases often co-occur 
    (e.g., an image may show signs of both Diabetic Retinopathy and Glaucoma).
    """)

    st.markdown("---")

    st.markdown("#### Why Multi-Label Classification?")

    col_ml1, col_ml2 = st.columns(2)

    with col_ml1:
        st.markdown("""
        **Key Characteristics:**
        
        - Each image can belong to **multiple disease classes** simultaneously
        - Labels are stored in **independent binary (0/1) format** per disease
        - Classes are **NOT mutually exclusive** - multiple conditions can co-exist
        - Uses **Sigmoid activation** (not Softmax) for independent per-class outputs
        - Trained with **Binary Cross-Entropy** loss (per label)
        
        **Example Label Vector:**
        ```
        Image #42: [DR=1, ARMD=0, MH=0, ..., ODC=1, ...]
        → This image has both DR and Glaucoma (ODC)
        ```
        """)

    with col_ml2:
        st.markdown("""
        **Multi-Label vs Multi-Class:**
        
        | Feature | Multi-Class | Multi-Label (RFMiD) |
        |---------|-------------|---------------------|
        | Labels per image | Exactly 1 | 1 or more |
        | Mutual exclusivity | Yes | No |
        | Output activation | Softmax | Sigmoid |
        | Loss function | Categorical CE | Binary CE |
        | Evaluation | Accuracy | Per-class AUC, Macro/Micro F1 |
        
        **Medical Realism:** Retinal diseases frequently co-occur:
        - DR + Hypertensive Retinopathy
        - Glaucoma + Optic Disc abnormalities
        - Macular degeneration + other conditions
        """)

    st.markdown("---")

    st.markdown("#### Dataset Statistics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Classification", "Multi-Label")
    with col2:
        st.metric("Total Images", "3,200")
    with col3:
        st.metric("Disease Classes", "46")
    with col4:
        st.metric("Format", "PNG")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("Training Set", "1,920")
    with col6:
        st.metric("Validation Set", "640")
    with col7:
        st.metric("Test Set", "640")
    with col8:
        st.metric("Input Sizes", "224-480px")

    st.markdown("---")

    st.markdown("#### Disease Labels (46 Conditions)")

    disease_labels = {
        'DR': 'Diabetic Retinopathy',
        'ARMD': 'Age-Related Macular Degeneration',
        'MH': 'Media Haze',
        'DN': 'Drusen',
        'MYA': 'Myopia',
        'BRVO': 'Branch Retinal Vein Occlusion',
        'TSLN': 'Tessellation',
        'ERM': 'Epiretinal Membrane',
        'LS': 'Laser Scars',
        'MS': 'Macular Scars',
        'CSR': 'Central Serous Retinopathy',
        'ODC': 'Optic Disc Cupping (Glaucoma)',
        'CRVO': 'Central Retinal Vein Occlusion',
        'TV': 'Tortuous Vessels',
        'AH': 'Asteroid Hyalosis',
        'ODP': 'Optic Disc Pallor',
        'ODE': 'Optic Disc Edema',
        'ST': 'Shunt',
        'AION': 'Anterior Ischemic Optic Neuropathy',
        'PT': 'Parafoveal Telangiectasia',
        'RT': 'Retinitis',
        'RS': 'Retinitis Pigmentosa',
        'CRS': 'Chorioretinitis',
        'EDN': 'Exudation',
        'RPEC': 'RPE Changes',
        'MHL': 'Macular Hole',
        'RP': 'Retinal Pigment',
        'CWS': 'Cotton Wool Spots',
        'CB': 'Coloboma',
        'ODPM': 'Optic Disc Pit Maculopathy',
        'PRH': 'Pre-Retinal Hemorrhage',
        'MNF': 'Myelinated Nerve Fibers',
        'HR': 'Hemorrhagic Retinopathy',
        'CRAO': 'Central Retinal Artery Occlusion',
        'TD': 'Tilted Disc',
        'CME': 'Cystoid Macular Edema',
        'PTCR': 'Post-Treatment Chorioretinal',
        'CF': 'Choroidal Folds',
        'VH': 'Vitreous Hemorrhage',
        'MCA': 'Macroaneurysm',
        'VS': 'Vasculitis',
        'BRAO': 'Branch Retinal Artery Occlusion',
        'PLQ': 'Plaque',
        'HPED': 'Hemorrhagic PED',
        'CL': 'Collateral',
    }

    disease_df = pd.DataFrame([
        {'Abbreviation': abbr, 'Full Name': name}
        for abbr, name in disease_labels.items()
    ])
    
    with st.expander("View All 46 Disease Labels", expanded=False):
        st.dataframe(disease_df, use_container_width=True, hide_index=True, height=400)

    st.markdown("---")

    st.markdown("#### Our Focus: Glaucoma Detection (ODC)")
    st.info("""
    While RFMiD supports multi-label classification across 46 diseases, our system 
    **focuses specifically on the ODC (Optic Disc Cupping) column** for glaucoma detection. 
    We treat this as a **binary classification task** (Glaucoma vs Normal) extracted from 
    the multi-label dataset, using Sigmoid activation and Binary Cross-Entropy loss 
    — consistent with the multi-label nature of the original dataset.
    """)

    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        st.metric("Normal (ODC=0)", "~85.3%")
    with col_g2:
        st.metric("Glaucoma (ODC=1)", "~14.7%")
    with col_g3:
        st.metric("Class Ratio", "~5.8 : 1")

    st.markdown("""
    **Training Approach:**
    - Sigmoid output per class (multi-label compatible)
    - Binary Cross-Entropy loss function
    - Class weighting to handle imbalance
    - Focal Loss for hard example mining
    - Per-class evaluation: AUC-ROC, Precision, Recall, F1
    """)

    st.markdown("#### Dataset Structure")
    st.code("""
RFMiD/
├── Training_set/         (1,920 images)
│   ├── *.png             High-quality fundus images
│   └── RFMiD_Training_Labels.csv
│       → 46 binary columns (multi-label: DR, ARMD, MH, ..., ODC, ..., CL)
├── Validation_set/       (640 images)
│   ├── *.png
│   └── RFMiD_Validation_Labels.csv
└── Test_set/             (640 images)
    ├── *.png
    └── RFMiD_Testing_Labels.csv
    """)

with tab3:
    st.markdown("### Enhanced Model Architectures (17 Total)")
    st.markdown("""
    Our system features **3 novel custom architectures** (GlaucoNet, V2, V3) and 
    **14 state-of-the-art pre-trained models** optimized for glaucoma detection
    with retinal-specific preprocessing pipelines.
    """)
    
    all_models = list(MODEL_INFO.keys())
    
    model_df = pd.DataFrame([
        {'Model': name, 'Parameters': info['params'], 'Input Size': info['input_size']}
        for name, info in MODEL_INFO.items()
    ])
    st.dataframe(model_df, use_container_width=True, hide_index=True)
    
    selected = st.selectbox("View model details:", all_models)
    if selected:
        info = MODEL_INFO[selected]
        st.info(f"**{selected}:** {info['description']}")
        if 'highlights' in info:
            st.markdown("**Key Features:** " + ", ".join(info['highlights']))

with tab4:
    st.markdown("### Model Performance")
    if is_trained:
        st.caption("Metrics from trained models on RFMID dataset")
    else:
        st.caption("Expected performance metrics (train models to see actual results)")
    
    metric = st.selectbox("Metric:", ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'])
    
    data = pd.DataFrame([
        {'Model': name, metric: m.get(metric, 0)}
        for name, m in METRICS.items()
    ])
    
    fig = px.bar(data, x='Model', y=metric, color=metric, color_continuous_scale='Blues')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    available_models = list(METRICS.keys())
    default_models = [m for m in ['GlaucoNet_V2', 'GlaucoNet_V3', 'EfficientNetV2S'] if m in available_models]
    radar_models = st.multiselect("Radar comparison:", available_models, default=default_models[:3])
    
    if radar_models:
        cats = ['accuracy', 'precision', 'recall', 'f1_score', 'sensitivity', 'specificity']
        fig = go.Figure()
        for m in radar_models:
            vals = [METRICS[m].get(c, 0) for c in cats] + [METRICS[m].get(cats[0], 0)]
            fig.add_trace(go.Scatterpolar(r=vals, theta=cats + [cats[0]], fill='toself', name=m))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
        st.plotly_chart(fig, use_container_width=True)
