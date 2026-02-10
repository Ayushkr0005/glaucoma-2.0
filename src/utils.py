import os
import numpy as np
from PIL import Image
import streamlit as st

SEVERITY_THRESHOLDS = {
    'normal': (0.0, 0.3),
    'borderline': (0.3, 0.5),
    'early': (0.5, 0.7),
    'moderate': (0.7, 0.85),
    'severe': (0.85, 0.95),
    'critical': (0.95, 1.0)
}

CDR_ESTIMATES = {
    'normal': '< 0.3',
    'borderline': '0.3 - 0.5',
    'early': '0.5 - 0.6',
    'moderate': '0.6 - 0.7',
    'severe': '0.7 - 0.9',
    'critical': '> 0.9'
}

RECOMMENDATIONS = {
    'normal': {
        'action': 'Regular eye checkup recommended',
        'urgency': 'Low',
        'color': '#10B981',
        'timeframe': 'Annual eye examination'
    },
    'borderline': {
        'action': 'Schedule an eye examination',
        'urgency': 'Moderate',
        'color': '#F59E0B',
        'timeframe': 'Within 3 months'
    },
    'early': {
        'action': 'Consult an ophthalmologist',
        'urgency': 'Moderate-High',
        'color': '#F97316',
        'timeframe': 'Within 1 month'
    },
    'moderate': {
        'action': 'Urgent ophthalmologist consultation',
        'urgency': 'High',
        'color': '#EF4444',
        'timeframe': 'Within 1 week'
    },
    'severe': {
        'action': 'Immediate medical attention required',
        'urgency': 'Very High',
        'color': '#DC2626',
        'timeframe': 'Within 1-2 days'
    },
    'critical': {
        'action': 'Emergency consultation required',
        'urgency': 'Critical',
        'color': '#991B1B',
        'timeframe': 'Immediately'
    }
}

MODEL_INFO = {
    'GlaucoNet': {
        'description': 'Original custom CNN with residual connections and squeeze-excitation attention blocks.',
        'params': '~5M',
        'input_size': '224x224',
        'highlights': ['SE Attention', 'Residual Blocks', 'Custom Architecture']
    },
    'GlaucoNet_V2': {
        'description': 'Novel architecture with CBAM attention, multi-scale ASPP features, and residual connections.',
        'params': '~20M',
        'input_size': '224x224',
        'highlights': ['CBAM Attention', 'Multi-Scale ASPP', 'GELU Activation', 'Dual Pooling']
    },
    'GlaucoNet_V3': {
        'description': 'Hybrid CNN-Attention architecture with patch-like stem, SE/CBAM attention, and progressive features.',
        'params': '~10M',
        'input_size': '224x224',
        'highlights': ['Hybrid Architecture', 'SE Blocks', 'CBAM Attention', 'Dual Pooling']
    },
    'ResNet50': {
        'description': 'Deep residual network with 50 layers and skip connections for effective gradient flow.',
        'params': '25.6M',
        'input_size': '224x224',
        'highlights': ['Skip Connections', 'Batch Normalization', 'ImageNet Pretrained']
    },
    'ResNet50V2': {
        'description': 'Improved ResNet with pre-activation design and better gradient flow.',
        'params': '25.6M',
        'input_size': '224x224',
        'highlights': ['Pre-Activation', 'Skip Connections', 'ImageNet Pretrained']
    },
    'VGG16': {
        'description': 'Classic deep CNN with 16 layers using small 3x3 convolution filters throughout.',
        'params': '138M',
        'input_size': '224x224',
        'highlights': ['Simple Architecture', '3x3 Convolutions', 'ImageNet Pretrained']
    },
    'VGG19': {
        'description': 'Extended VGG architecture with 19 layers for deeper feature extraction.',
        'params': '144M',
        'input_size': '224x224',
        'highlights': ['Deeper Network', '3x3 Convolutions', 'ImageNet Pretrained']
    },
    'DenseNet121': {
        'description': 'DenseNet with 121 layers featuring dense connectivity for maximum feature reuse.',
        'params': '8M',
        'input_size': '224x224',
        'highlights': ['Dense Connections', 'Feature Reuse', 'Compact Model']
    },
    'DenseNet169': {
        'description': 'DenseNet with 169 layers balancing depth and computational efficiency.',
        'params': '14M',
        'input_size': '224x224',
        'highlights': ['Dense Connections', 'Feature Reuse', 'Gradient Highway']
    },
    'DenseNet201': {
        'description': 'Deep DenseNet with 201 layers and maximum feature reuse through dense connections.',
        'params': '20M',
        'input_size': '224x224',
        'highlights': ['Dense Connections', 'Feature Reuse', 'Gradient Highway']
    },
    'InceptionV3': {
        'description': 'Google Inception with factorized convolutions and auxiliary classifiers.',
        'params': '23.9M',
        'input_size': '299x299',
        'highlights': ['Multi-Scale Features', 'Factorized Convolutions', 'ImageNet Pretrained']
    },
    'Xception': {
        'description': 'Extreme Inception using depthwise separable convolutions for efficiency.',
        'params': '22.9M',
        'input_size': '299x299',
        'highlights': ['Depthwise Separable Conv', 'Efficient Design', 'ImageNet Pretrained']
    },
    'MobileNetV2': {
        'description': 'Lightweight architecture with inverted residuals and linear bottlenecks.',
        'params': '3.5M',
        'input_size': '224x224',
        'highlights': ['Inverted Residuals', 'Lightweight', 'Mobile-Optimized']
    },
    'EfficientNetB0': {
        'description': 'Baseline EfficientNet with compound scaling for balanced width, depth, and resolution.',
        'params': '5.3M',
        'input_size': '224x224',
        'highlights': ['Compound Scaling', 'MBConv Blocks', 'ImageNet Pretrained']
    },
    'EfficientNetV2S': {
        'description': 'Latest EfficientNet with improved training speed and higher accuracy.',
        'params': '21.5M',
        'input_size': '384x384',
        'highlights': ['Fused-MBConv', 'Progressive Training', 'State-of-the-Art']
    },
    'EfficientNetV2M': {
        'description': 'Medium-sized EfficientNetV2 with excellent accuracy-speed trade-off.',
        'params': '54M',
        'input_size': '480x480',
        'highlights': ['Higher Resolution', 'More Capacity', 'Best Accuracy']
    },
    'NASNetMobile': {
        'description': 'Neural architecture search optimized model for mobile deployment.',
        'params': '5.3M',
        'input_size': '224x224',
        'highlights': ['NAS Optimized', 'Mobile-Friendly', 'ImageNet Pretrained']
    }
}

def get_severity_level(confidence):
    for level, (low, high) in SEVERITY_THRESHOLDS.items():
        if low <= confidence < high:
            return level
    return 'critical' if confidence >= 0.95 else 'normal'

def get_severity_color(severity):
    return RECOMMENDATIONS.get(severity, {}).get('color', '#6B7280')

def get_recommendation(severity):
    return RECOMMENDATIONS.get(severity, RECOMMENDATIONS['normal'])

def get_cdr_estimate(severity):
    return CDR_ESTIMATES.get(severity, 'Unknown')

def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def preprocess_for_prediction(image, target_size=(224, 224)):
    img = image.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img)
    img_array = img_array.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@st.cache_resource
def load_model(model_path):
    import tensorflow as tf
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_available_models(models_dir='saved_models'):
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for file in os.listdir(models_dir):
        if file.endswith('.h5') or file.endswith('.keras'):
            model_name = file.replace('.h5', '').replace('.keras', '').replace('_best', '')
            models.append(model_name)
    
    return models

def format_confidence(confidence):
    return f"{confidence * 100:.1f}%"

def create_progress_bar_html(confidence, severity):
    color = get_severity_color(severity)
    percentage = confidence * 100
    
    html = f"""
    <div style="width: 100%; background-color: #e0e0e0; border-radius: 10px; overflow: hidden;">
        <div style="width: {percentage}%; background-color: {color}; padding: 10px 0; text-align: center; color: white; font-weight: bold; border-radius: 10px;">
            {percentage:.1f}%
        </div>
    </div>
    """
    return html

def get_clinical_notes():
    return """
    **Important Clinical Considerations:**
    
    - **Intraocular Pressure (IOP):** Normal range is 10-21 mmHg. Elevated IOP is a major risk factor for glaucoma.
    
    - **Visual Field Testing:** Recommended for comprehensive glaucoma assessment to detect any peripheral vision loss.
    
    - **OCT Imaging:** Optical Coherence Tomography provides detailed imaging of the optic nerve and retinal nerve fiber layer.
    
    - **Family History:** Glaucoma has a genetic component. Patients with family history are at higher risk.
    
    - **Treatment Options:**
      - Eye drops to reduce IOP
      - Laser therapy
      - Surgical procedures (trabeculectomy, drainage implants)
    """

def get_disclaimer():
    return """
    **Medical Disclaimer:**
    
    This tool is for educational and research purposes only. The predictions made by this system 
    should NOT be used as a substitute for professional medical diagnosis. 
    
    The Cup-to-Disc Ratio (CDR) values displayed are **estimates** derived from the model's 
    confidence scores and are NOT actual measurements from the fundus image.
    
    Always consult a qualified ophthalmologist for proper diagnosis and treatment of eye conditions.
    """

SAMPLE_METRICS = {
    'GlaucoNet': {'accuracy': 0.91, 'precision': 0.90, 'recall': 0.92, 'f1_score': 0.91, 'roc_auc': 0.95, 'sensitivity': 0.92, 'specificity': 0.90},
    'GlaucoNet_V2': {'accuracy': 0.94, 'precision': 0.93, 'recall': 0.95, 'f1_score': 0.94, 'roc_auc': 0.97, 'sensitivity': 0.95, 'specificity': 0.93},
    'GlaucoNet_V3': {'accuracy': 0.95, 'precision': 0.94, 'recall': 0.96, 'f1_score': 0.95, 'roc_auc': 0.98, 'sensitivity': 0.96, 'specificity': 0.94},
    'ResNet50': {'accuracy': 0.92, 'precision': 0.91, 'recall': 0.93, 'f1_score': 0.92, 'roc_auc': 0.96, 'sensitivity': 0.93, 'specificity': 0.91},
    'ResNet50V2': {'accuracy': 0.93, 'precision': 0.92, 'recall': 0.93, 'f1_score': 0.92, 'roc_auc': 0.96, 'sensitivity': 0.93, 'specificity': 0.92},
    'VGG16': {'accuracy': 0.89, 'precision': 0.88, 'recall': 0.90, 'f1_score': 0.89, 'roc_auc': 0.94, 'sensitivity': 0.90, 'specificity': 0.88},
    'VGG19': {'accuracy': 0.89, 'precision': 0.87, 'recall': 0.90, 'f1_score': 0.88, 'roc_auc': 0.93, 'sensitivity': 0.90, 'specificity': 0.87},
    'DenseNet121': {'accuracy': 0.92, 'precision': 0.91, 'recall': 0.93, 'f1_score': 0.92, 'roc_auc': 0.96, 'sensitivity': 0.93, 'specificity': 0.91},
    'DenseNet169': {'accuracy': 0.93, 'precision': 0.92, 'recall': 0.94, 'f1_score': 0.93, 'roc_auc': 0.97, 'sensitivity': 0.94, 'specificity': 0.92},
    'DenseNet201': {'accuracy': 0.93, 'precision': 0.92, 'recall': 0.94, 'f1_score': 0.93, 'roc_auc': 0.97, 'sensitivity': 0.94, 'specificity': 0.92},
    'InceptionV3': {'accuracy': 0.93, 'precision': 0.92, 'recall': 0.94, 'f1_score': 0.93, 'roc_auc': 0.97, 'sensitivity': 0.94, 'specificity': 0.92},
    'Xception': {'accuracy': 0.94, 'precision': 0.93, 'recall': 0.95, 'f1_score': 0.94, 'roc_auc': 0.97, 'sensitivity': 0.95, 'specificity': 0.93},
    'MobileNetV2': {'accuracy': 0.91, 'precision': 0.90, 'recall': 0.92, 'f1_score': 0.91, 'roc_auc': 0.95, 'sensitivity': 0.92, 'specificity': 0.90},
    'EfficientNetB0': {'accuracy': 0.93, 'precision': 0.92, 'recall': 0.94, 'f1_score': 0.93, 'roc_auc': 0.97, 'sensitivity': 0.94, 'specificity': 0.92},
    'EfficientNetV2S': {'accuracy': 0.96, 'precision': 0.95, 'recall': 0.97, 'f1_score': 0.96, 'roc_auc': 0.98, 'sensitivity': 0.97, 'specificity': 0.95},
    'EfficientNetV2M': {'accuracy': 0.97, 'precision': 0.96, 'recall': 0.98, 'f1_score': 0.97, 'roc_auc': 0.99, 'sensitivity': 0.98, 'specificity': 0.96},
    'NASNetMobile': {'accuracy': 0.91, 'precision': 0.90, 'recall': 0.92, 'f1_score': 0.91, 'roc_auc': 0.95, 'sensitivity': 0.92, 'specificity': 0.90},
}

def load_trained_metrics():
    """Load metrics from trained models if available"""
    import json
    
    metrics_file = 'results/metrics/all_models_metrics.json'
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    metrics_dir = 'results/metrics'
    if os.path.exists(metrics_dir):
        all_metrics = {}
        for f in os.listdir(metrics_dir):
            if f.endswith('_metrics.json') and f != 'all_models_metrics.json':
                model_name = f.replace('_metrics.json', '')
                filepath = os.path.join(metrics_dir, f)
                try:
                    with open(filepath, 'r') as fh:
                        m = json.load(fh)
                        if 'error' not in m:
                            all_metrics[model_name] = m
                except Exception:
                    pass
        if all_metrics:
            return all_metrics
    
    return None

def get_metrics():
    """Get metrics - trained if available, otherwise sample"""
    trained = load_trained_metrics()
    if trained:
        merged = dict(SAMPLE_METRICS)
        merged.update(trained)
        return merged, True
    return SAMPLE_METRICS, False

