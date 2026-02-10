"""
Practical Training Script for All 17 Models
Trains one model at a time with proper memory management
"""
import os
import sys
import json
import subprocess
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ALL_MODELS = [
    'GlaucoNet', 'GlaucoNet_V2', 'GlaucoNet_V3',
    'ResNet50', 'ResNet50V2', 'VGG16', 'VGG19',
    'DenseNet121', 'DenseNet169', 'DenseNet201',
    'InceptionV3', 'Xception', 'MobileNetV2',
    'EfficientNetB0', 'EfficientNetV2S', 'EfficientNetV2M',
    'NASNetMobile'
]

TRAIN_SCRIPT = '''
import os, gc, json, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from src.models_enhanced import MODEL_CONFIGS, create_pretrained_model, unfreeze_model_layers, CombinedLoss, get_optimizer
from src.data_pipeline import prepare_rfmid_dataset, create_tf_dataset, get_class_weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

MODEL_NAME = sys.argv[1]
print(f"Training: {MODEL_NAME}")

os.makedirs('saved_models', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)

data = prepare_rfmid_dataset('data/rfmid', label_column='ODC')
train_paths, train_labels = data['train']
val_paths, val_labels = data['val']
test_paths, test_labels = data['test']

class_weights = get_class_weights(train_labels)
config = MODEL_CONFIGS[MODEL_NAME]
image_size = config.get('input_size', (224, 224))
batch_size = 8 if max(image_size) > 300 else 16

train_gen = create_tf_dataset(train_paths, train_labels, batch_size=batch_size, image_size=image_size, augment=True)
val_gen = create_tf_dataset(val_paths, val_labels, batch_size=batch_size, image_size=image_size, augment=False, shuffle=False)

if 'create_fn' in config:
    model = config['create_fn'](input_shape=(*image_size, 3), num_classes=1)
    base_model = None
elif config.get('pretrained', False):
    model, base_model = create_pretrained_model(MODEL_NAME, input_shape=(*image_size, 3), num_classes=1)

model.compile(
    optimizer=get_optimizer(learning_rate=1e-3),
    loss=CombinedLoss(focal_weight=0.5),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        f'saved_models/{MODEL_NAME}_best.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
]

print(f"Stage 1: Training classifier head (15 epochs)...")
model.fit(train_gen, validation_data=val_gen, epochs=15, callbacks=callbacks, class_weight=class_weights, verbose=2)

if base_model is not None:
    print(f"Stage 2: Fine-tuning (10 epochs)...")
    total_layers = len(base_model.layers)
    for layer in base_model.layers[-int(total_layers * 0.3):]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    model.compile(
        optimizer=get_optimizer(learning_rate=1e-5),
        loss=CombinedLoss(focal_weight=0.5),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    callbacks2 = [
        tf.keras.callbacks.ModelCheckpoint(
            f'saved_models/{MODEL_NAME}_best.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ]
    model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=callbacks2, class_weight=class_weights, verbose=2)

model.save(f'saved_models/{MODEL_NAME}_final.keras')

test_gen = create_tf_dataset(test_paths, test_labels, batch_size=batch_size, image_size=image_size, augment=False, shuffle=False)
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
    'accuracy': float(accuracy_score(y_true, y_pred)),
    'precision': float(precision_score(y_true, y_pred, zero_division=0)),
    'recall': float(recall_score(y_true, y_pred, zero_division=0)),
    'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
    'roc_auc': float(roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5),
    'sensitivity': float(recall_score(y_true, y_pred, zero_division=0)),
    'specificity': float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
}

print(f"Results for {MODEL_NAME}:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

with open(f'results/metrics/{MODEL_NAME}_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"DONE: {MODEL_NAME}")
del model
gc.collect()
'''

def train_model(model_name):
    print(f"\n{'='*60}")
    print(f"Starting: {model_name}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [sys.executable, '-c', TRAIN_SCRIPT, model_name],
        capture_output=False,
        timeout=3600
    )
    
    return result.returncode == 0

def merge_metrics():
    all_metrics = {}
    metrics_dir = 'results/metrics'
    for f in os.listdir(metrics_dir):
        if f.endswith('_metrics.json') and f != 'all_models_metrics.json':
            model_name = f.replace('_metrics.json', '')
            with open(os.path.join(metrics_dir, f)) as fh:
                all_metrics[model_name] = json.load(fh)
    
    with open(os.path.join(metrics_dir, 'all_models_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\nMerged metrics for {len(all_metrics)} models")
    return all_metrics


if __name__ == '__main__':
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    models = sys.argv[1:] if len(sys.argv) > 1 else ALL_MODELS
    
    results = {}
    for model_name in models:
        start = time.time()
        try:
            success = train_model(model_name)
            elapsed = time.time() - start
            results[model_name] = 'OK' if success else 'FAILED'
            print(f"{model_name}: {'OK' if success else 'FAILED'} ({elapsed/60:.1f} min)")
        except Exception as e:
            results[model_name] = f'ERROR: {e}'
            print(f"{model_name}: ERROR - {e}")
    
    all_metrics = merge_metrics()
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for model, status in results.items():
        print(f"  {model}: {status}")
