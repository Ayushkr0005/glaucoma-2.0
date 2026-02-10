"""
Enhanced Training Pipeline for Glaucoma Detection
Features:
- Multi-model training with progressive unfreezing
- K-fold cross-validation
- Advanced callbacks and learning rate scheduling
- Comprehensive metrics logging
- Ensemble model creation
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.models_enhanced import (
    create_glauconet_v2, create_glauconet_v3,
    create_pretrained_model, unfreeze_model_layers,
    FocalLoss, CombinedLoss, get_optimizer, MODEL_CONFIGS
)
from src.data_pipeline import (
    prepare_rfmid_dataset, create_tf_dataset,
    get_class_weights, create_kfold_splits
)
from src.data_preprocessing import MODEL_PREPROCESSING


def setup_directories():
    """Create necessary directories"""
    dirs = ['saved_models', 'results/plots', 'results/metrics', 'results/history']
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_callbacks(model_name, fold=None, patience=10):
    """Get training callbacks"""
    suffix = f"_fold{fold}" if fold is not None else ""
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f'saved_models/{model_name}{suffix}_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            f'results/history/{model_name}{suffix}_history.csv'
        ),
    ]
    
    return callbacks


def train_single_model(model_name, train_data, val_data, config, class_weights=None):
    """Train a single model with two-stage training"""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    image_size = config.get('input_size', (224, 224))
    
    batch_size = 8 if max(image_size) > 300 else 16
    
    train_gen = create_tf_dataset(
        train_data[0], train_data[1],
        batch_size=batch_size,
        image_size=image_size,
        augment=True,
        model_name=model_name,
        oversample_minority=True,
        aug_strength='medium'
    )
    
    val_gen = create_tf_dataset(
        val_data[0], val_data[1],
        batch_size=batch_size,
        image_size=image_size,
        augment=False,
        shuffle=False,
        model_name=model_name
    )
    
    if 'create_fn' in config:
        model = config['create_fn'](
            input_shape=(*image_size, 3),
            num_classes=1
        )
        base_model = None
    elif config.get('pretrained', False):
        model, base_model = create_pretrained_model(
            model_name,
            input_shape=(*image_size, 3),
            num_classes=1
        )
    else:
        raise ValueError(f"Unknown configuration for {model_name}")
    
    model.compile(
        optimizer=get_optimizer(learning_rate=1e-3),
        loss=CombinedLoss(focal_weight=0.5),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    callbacks = get_callbacks(model_name, patience=7)
    
    stage1_epochs = 15
    print(f"\nStage 1: Training classifier head ({stage1_epochs} epochs)...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=stage1_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    if base_model is not None:
        stage2_epochs = 10
        print(f"\nStage 2: Fine-tuning with unfrozen layers ({stage2_epochs} epochs)...")
        model = unfreeze_model_layers(model, base_model, unfreeze_ratio=0.3)
        
        model.compile(
            optimizer=get_optimizer(learning_rate=1e-5),
            loss=CombinedLoss(focal_weight=0.5),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        callbacks = get_callbacks(model_name + '_finetuned', patience=5)
        
        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=stage2_epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        history = {
            'stage1': history1.history,
            'stage2': history2.history
        }
    else:
        history = {'stage1': history1.history}
    
    model.save(f'saved_models/{model_name}_final.keras')
    
    return model, history


def evaluate_model(model, test_data, image_size=(224, 224)):
    """Comprehensive model evaluation"""
    test_gen = create_tf_dataset(
        test_data[0], test_data[1],
        batch_size=16,
        image_size=image_size,
        augment=False,
        shuffle=False
    )
    
    y_pred_proba = []
    y_true = []
    
    for X_batch, y_batch in test_gen:
        pred = model.predict(X_batch, verbose=0)
        y_pred_proba.extend(pred.flatten())
        y_true.extend(y_batch.flatten())
    
    y_pred_proba = np.array(y_pred_proba)
    y_true = np.array(y_true)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5,
        'sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
    }
    
    cm = confusion_matrix(y_true, y_pred)
    
    return metrics, cm, y_pred_proba, y_true


def plot_training_history(history, model_name):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    all_loss = []
    all_val_loss = []
    all_acc = []
    all_val_acc = []
    
    for stage_name, stage_history in history.items():
        all_loss.extend(stage_history.get('loss', []))
        all_val_loss.extend(stage_history.get('val_loss', []))
        all_acc.extend(stage_history.get('accuracy', []))
        all_val_acc.extend(stage_history.get('val_accuracy', []))
    
    epochs = range(1, len(all_loss) + 1)
    
    axes[0].plot(epochs, all_loss, 'b-', label='Training Loss')
    axes[0].plot(epochs, all_val_loss, 'r-', label='Validation Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, all_acc, 'b-', label='Training Accuracy')
    axes[1].plot(epochs, all_val_acc, 'r-', label='Validation Accuracy')
    axes[1].set_title(f'{model_name} - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/plots/{model_name}_training_history.png', dpi=150)
    plt.close()


def plot_confusion_matrix(cm, model_name, classes=['Normal', 'Glaucoma']):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'results/plots/{model_name}_confusion_matrix.png', dpi=150)
    plt.close()


def save_metrics(all_metrics, filename='results/metrics/all_models_metrics.json'):
    """Save all metrics to JSON"""
    with open(filename, 'w') as f:
        json.dump(all_metrics, f, indent=2)


def train_all_models(data_dir='data/rfmid', models_to_train=None):
    """Train all models and generate comprehensive results"""
    print("="*60)
    print("GLAUCOMA DETECTION - ENHANCED TRAINING PIPELINE")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    setup_directories()
    
    print("\nLoading RFMID dataset...")
    data = prepare_rfmid_dataset(data_dir, label_column='ODC')
    
    train_paths, train_labels = data['train']
    val_paths, val_labels = data['val']
    test_paths, test_labels = data['test']
    
    class_weights = get_class_weights(train_labels)
    print(f"\nClass weights: {class_weights}")
    
    if models_to_train is None:
        models_to_train = list(MODEL_CONFIGS.keys())
    
    all_metrics = {}
    
    for model_name in models_to_train:
        try:
            config = MODEL_CONFIGS[model_name]
            
            model, history = train_single_model(
                model_name,
                (train_paths, train_labels),
                (val_paths, val_labels),
                config,
                class_weights
            )
            
            plot_training_history(history, model_name)
            
            print(f"\nEvaluating {model_name} on test set...")
            image_size = config.get('input_size', (224, 224))
            metrics, cm, y_pred, y_true = evaluate_model(
                model, (test_paths, test_labels), image_size
            )
            
            plot_confusion_matrix(cm, model_name)
            
            all_metrics[model_name] = metrics
            
            print(f"\n{model_name} Test Results:")
            print(f"  Accuracy:    {metrics['accuracy']:.4f}")
            print(f"  Precision:   {metrics['precision']:.4f}")
            print(f"  Recall:      {metrics['recall']:.4f}")
            print(f"  F1-Score:    {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
            print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"  Specificity: {metrics['specificity']:.4f}")
            
            tf.keras.backend.clear_session()
            
        except Exception as e:
            print(f"\nError training {model_name}: {e}")
            all_metrics[model_name] = {'error': str(e)}
    
    save_metrics(all_metrics)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nFinal Results Summary:")
    
    valid_metrics = {k: v for k, v in all_metrics.items() if 'error' not in v}
    if valid_metrics:
        results_df = pd.DataFrame(valid_metrics).T
        results_df = results_df.sort_values('accuracy', ascending=False)
        print(results_df.to_string())
        results_df.to_csv('results/metrics/model_comparison.csv')
    else:
        print("No models completed successfully.")

    failed = {k: v for k, v in all_metrics.items() if 'error' in v}
    if failed:
        print(f"\nFailed models: {list(failed.keys())}")
    
    return all_metrics


def train_with_kfold(data_dir='data/rfmid', model_name='GlaucoNet_V2', n_folds=5):
    """Train model with K-fold cross-validation"""
    print(f"\n{'='*60}")
    print(f"K-FOLD CROSS-VALIDATION: {model_name}")
    print(f"{'='*60}")
    
    setup_directories()
    
    data = prepare_rfmid_dataset(data_dir)
    
    all_paths = data['train'][0] + data['val'][0]
    all_labels = data['train'][1] + data['val'][1]
    
    test_paths, test_labels = data['test']
    
    splits = create_kfold_splits(all_paths, all_labels, n_splits=n_folds)
    
    fold_metrics = []
    
    for fold, split in enumerate(splits):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        config = MODEL_CONFIGS[model_name]
        class_weights = get_class_weights(split['train'][1])
        
        model, history = train_single_model(
            f"{model_name}_fold{fold}",
            split['train'],
            split['val'],
            config,
            class_weights
        )
        
        image_size = config.get('input_size', (224, 224))
        metrics, _, _, _ = evaluate_model(model, (test_paths, test_labels), image_size)
        fold_metrics.append(metrics)
        
        tf.keras.backend.clear_session()
    
    avg_metrics = {}
    std_metrics = {}
    for key in fold_metrics[0].keys():
        values = [m[key] for m in fold_metrics]
        avg_metrics[key] = np.mean(values)
        std_metrics[key] = np.std(values)
    
    print(f"\n{'='*60}")
    print(f"K-FOLD RESULTS: {model_name}")
    print(f"{'='*60}")
    for key in avg_metrics:
        print(f"{key}: {avg_metrics[key]:.4f} (+/- {std_metrics[key]:.4f})")
    
    return avg_metrics, std_metrics, fold_metrics


if __name__ == '__main__':
    all_metrics = train_all_models(
        data_dir='data/rfmid',
        models_to_train=None
    )
