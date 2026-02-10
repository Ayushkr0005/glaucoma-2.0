import streamlit as st

st.set_page_config(
    page_title="Glaucoma Detection System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #3b82f6;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
    }
    .info-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 👁️ Glaucoma AI")
    st.markdown("---")
    st.markdown("### Quick Stats")
    st.metric("Models", "17")
    st.metric("Accuracy", "89-97%")
    st.metric("Speed", "< 3 sec")
    st.markdown("---")
    st.info("**Version 2.0.0**\nBuilt with TensorFlow & Streamlit")

st.markdown('<p class="main-title">👁️ Glaucoma Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Fundus Image Analysis for Early Glaucoma Detection</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("AI Models", "17")
with col2:
    st.metric("Top Accuracy", "97%")
with col3:
    st.metric("Severity Levels", "6")
with col4:
    st.metric("Analysis Time", "<3s")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔬 17 AI Models")
    st.markdown("""
    Choose from 3 custom GlaucoNet architectures and 14 pre-trained models 
    including ResNet, VGG, DenseNet, Inception, Xception, MobileNet, 
    EfficientNet, and NASNet families.
    """)

with col2:
    st.markdown("### 📊 Detailed Analysis")
    st.markdown("""
    Get confidence scores, severity levels (Normal to Critical), 
    estimated CDR values, and personalized clinical recommendations 
    with customizable threshold settings.
    """)

with col3:
    st.markdown("### 🎯 Visual Explanations")
    st.markdown("""
    Grad-CAM heatmaps show exactly where the AI focuses on the fundus 
    image, making predictions transparent and interpretable for 
    clinical review.
    """)

st.markdown("---")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 🚀 How to Use")
    st.markdown("""
    1. Go to the **Prediction** page
    2. Select a model from the dropdown (all 17 models available)
    3. Set your classification threshold (default: 50%)
    4. Upload a fundus image (JPG, PNG)
    5. View results, severity, and Grad-CAM visualization
    """)

with col_b:
    st.markdown("### 📋 Available Pages")
    st.markdown("""
    - **🏠 Home** - Overview and glaucoma information
    - **🔬 Prediction** - Upload and analyze images
    - **📊 Comparison** - Compare all models at once
    - **📈 Analytics** - Model performance dashboard
    - **ℹ️ About** - Documentation and disclaimers
    """)

st.warning("""
⚠️ **Medical Disclaimer**: This tool is for educational and research purposes only. 
Results should NOT replace professional medical diagnosis. Always consult a qualified 
ophthalmologist for proper evaluation.
""")
