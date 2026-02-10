"""Train a single model - called as subprocess"""
import os, gc, json, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from src.models_enhanced import MODEL_CONFIGS, create_pretrained_model, CombinedLoss, get_optimizer
from src.data_pipeline import prepare_rfmid_dataset, create_tf_dataset, get_class_weights
from src.data_preprocessing import MODEL_PREPROCESSING
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

MODEL_NAME = sys.argv[1]
EPOCHS_S1 = int(sys.argv[2]) if len(sys.argv) > 2 else 10
EPOCHS_S2 = int(sys.argv[3]) if len(sys.argv) > 3 else 5

print(f'Training: {MODEL_NAME} (Stage1={EPOCHS_S1}ep, Stage2={EPOCHS_S2}ep)')
sys.stdout.flush()

os.makedirs('saved_models', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

data = prepare_rfmid_dataset('data/rfmid', label_column='ODC')
train_paths, train_labels = data['train']
val_paths, val_labels = data['val']
test_paths, test_labels = data['test']
class_weights = get_class_weights(train_labels)

config = MODEL_CONFIGS[MODEL_NAME]
image_size = config.get('input_size', (224, 224))
batch_size = 8 if max(image_size) > 300 else 16

train_gen = create_tf_dataset(
    train_paths, train_labels, batch_size=batch_size, image_size=image_size,
    augment=True, model_name=MODEL_NAME, oversample_minority=True, aug_strength='medium'
)
val_gen = create_tf_dataset(
    val_paths, val_labels, batch_size=batch_size, image_size=image_size,
    augment=False, shuffle=False, model_name=MODEL_NAME
)

if 'create_fn' in config:
    model = config['create_fn'](input_shape=(*image_size, 3), num_classes=1)
    base_model = None
elif config.get('pretrained', False):
    model, base_model = create_pretrained_model(MODEL_NAME, input_shape=(*image_size, 3), num_classes=1)
else:
    raise ValueError(f"Unknown config for {MODEL_NAME}")

model.compile(
    optimizer=get_optimizer(learning_rate=1e-3),
    loss=CombinedLoss(focal_weight=0.5),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        f'saved_models/{MODEL_NAME}_best.keras', monitor='val_accuracy',
        save_best_only=True, mode='max', verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
    ),
]

print(f'Stage 1: {EPOCHS_S1} epochs...')
sys.stdout.flush()
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_S1,
          callbacks=callbacks, class_weight=class_weights, verbose=2)

if base_model is not None and EPOCHS_S2 > 0:
    print(f'Stage 2: Fine-tuning {EPOCHS_S2} epochs...')
    sys.stdout.flush()
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
            f'saved_models/{MODEL_NAME}_best.keras', monitor='val_accuracy',
            save_best_only=True, mode='max', verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
        ),
    ]
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_S2,
              callbacks=callbacks2, class_weight=class_weights, verbose=2)

model.save(f'saved_models/{MODEL_NAME}_final.keras')

print('Evaluating on test set...')
sys.stdout.flush()
test_gen = create_tf_dataset(test_paths, test_labels, batch_size=batch_size,
                             image_size=image_size, augment=False, shuffle=False,
                             model_name=MODEL_NAME)
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

print(f'Results for {MODEL_NAME}:')
for k, v in metrics.items():
    print(f'  {k}: {v:.4f}')

with open(f'results/metrics/{MODEL_NAME}_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f'COMPLETED: {MODEL_NAME}')
sys.stdout.flush()
del model
gc.collect()
