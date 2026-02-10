import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def calculate_metrics(y_true, y_pred, y_prob=None):
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int) if y_prob is not None else y_pred
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1_score': f1_score(y_true, y_pred_binary, zero_division=0),
        'sensitivity': recall_score(y_true, y_pred_binary, zero_division=0),
        'specificity': recall_score(1 - np.array(y_true), 1 - np.array(y_pred_binary), zero_division=0)
    }
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel() if len(set(y_true)) > 1 else (0, 0, 0, 0)
    
    metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics['roc_auc'] = auc(fpr, tpr)
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names=['Normal', 'Glaucoma']):
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names,
        y=class_names,
        color_continuous_scale='Blues',
        text_auto=True
    )
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label'
    )
    
    return fig

def plot_roc_curve(y_true, y_prob, model_name='Model'):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'{model_name} (AUC = {roc_auc:.3f})',
        mode='lines',
        line=dict(width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Classifier',
        mode='lines',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    
    return fig

def plot_multi_roc_curves(results_dict):
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for i, (model_name, data) in enumerate(results_dict.items()):
        y_true = data['y_true']
        y_prob = data['y_prob']
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'{model_name} (AUC = {roc_auc:.3f})',
            mode='lines',
            line=dict(width=2, color=colors[i % len(colors)])
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Random Classifier',
        mode='lines',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True
    )
    
    return fig

def plot_precision_recall_curve(y_true, y_prob, model_name='Model'):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        name=model_name,
        mode='lines',
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        showlegend=True
    )
    
    return fig

def plot_training_history(history):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Model Loss', 'Model Accuracy'))
    
    fig.add_trace(
        go.Scatter(y=history['loss'], name='Train Loss', mode='lines'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=history['val_loss'], name='Val Loss', mode='lines'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(y=history['accuracy'], name='Train Accuracy', mode='lines'),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(y=history['val_accuracy'], name='Val Accuracy', mode='lines'),
        row=1, col=2
    )
    
    fig.update_layout(title='Training History', showlegend=True)
    
    return fig

def plot_metrics_comparison(metrics_dict):
    models = list(metrics_dict.keys())
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    fig = go.Figure()
    
    for metric in metric_names:
        values = [metrics_dict[model].get(metric, 0) for model in models]
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=models,
            y=values
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Model',
        yaxis_title='Score',
        barmode='group',
        xaxis_tickangle=-45
    )
    
    return fig

def plot_radar_chart(metrics_dict):
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'sensitivity', 'specificity']
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Plotly
    
    for i, (model_name, metrics) in enumerate(metrics_dict.items()):
        values = [metrics.get(m, 0) for m in metrics_to_plot]
        values.append(values[0])
        
        categories = [m.replace('_', ' ').title() for m in metrics_to_plot]
        categories.append(categories[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=model_name,
            line=dict(color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='Model Performance Radar Chart'
    )
    
    return fig

def generate_gradcam(model, image, last_conv_layer_name=None):
    import tensorflow as tf
    
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer_name = layer.name
                break
    
    if last_conv_layer_name is None:
        return None
    
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image)
            loss = predictions[:, 0]
        
        grads = tape.gradient(loss, conv_outputs)
        
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

def overlay_gradcam(image, heatmap, alpha=0.4):
    import cv2
    
    if heatmap is None:
        return image
    
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    if image.max() <= 1.0:
        image = np.uint8(255 * image)
    
    superimposed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return superimposed
