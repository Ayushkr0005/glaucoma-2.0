import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    ResNet50, VGG16, VGG19, DenseNet121, DenseNet169,
    InceptionV3, Xception, MobileNetV2, EfficientNetB0, NASNetMobile
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
from datetime import datetime

from src.data_preprocessing import (
    create_train_generator, create_validation_generator, create_test_generator,
    get_input_size, get_class_weights, split_validation_folder, MODEL_INPUT_SIZES
)
from src.custom_model import build_glauconet
from src.evaluation import calculate_metrics

PRETRAINED_MODELS = {
    'ResNet50': ResNet50,
    'VGG16': VGG16,
    'VGG19': VGG19,
    'DenseNet121': DenseNet121,
    'DenseNet169': DenseNet169,
    'InceptionV3': InceptionV3,
    'Xception': Xception,
    'MobileNetV2': MobileNetV2,
    'EfficientNetB0': EfficientNetB0,
    'NASNetMobile': NASNetMobile
}

def build_pretrained_model(model_name, input_shape, freeze_base=True):
    base_model_class = PRETRAINED_MODELS.get(model_name)
    if base_model_class is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    base_model = base_model_class(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    if freeze_base:
        for layer in base_model.layers:
            layer.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output, name=model_name)
    
    return model, base_model

def unfreeze_layers(model, base_model, num_layers_to_unfreeze=20):
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True
    return model

def get_callbacks(model_name, log_dir='logs'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    checkpoint = ModelCheckpoint(
        filepath=f'saved_models/{model_name}_best.h5',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    tensorboard = TensorBoard(
        log_dir=f'{log_dir}/{model_name}_{timestamp}',
        histogram_freq=1
    )
    
    return [checkpoint, early_stopping, reduce_lr, tensorboard]

def train_model(model_name, train_dir, val_dir, test_dir=None, epochs_stage1=10, epochs_stage2=20, batch_size=32):
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    input_size = get_input_size(model_name)
    input_shape = (*input_size, 3)
    
    train_gen = create_train_generator(train_dir, target_size=input_size, batch_size=batch_size)
    val_gen = create_validation_generator(val_dir, target_size=input_size, batch_size=batch_size)
    
    class_weights = get_class_weights(train_gen)
    print(f"Class weights: {class_weights}")
    
    if model_name == 'GlaucoNet':
        model = build_glauconet(input_shape=input_shape)
        base_model = None
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = get_callbacks(model_name)
        
        history = model.fit(
            train_gen,
            epochs=epochs_stage1 + epochs_stage2,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks
        )
    else:
        print("\n--- Stage 1: Training top layers ---")
        model, base_model = build_pretrained_model(model_name, input_shape, freeze_base=True)
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        callbacks = get_callbacks(model_name)
        
        history1 = model.fit(
            train_gen,
            epochs=epochs_stage1,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks
        )
        
        print("\n--- Stage 2: Fine-tuning ---")
        model = unfreeze_layers(model, base_model, num_layers_to_unfreeze=20)
        
        model.compile(
            optimizer=Adam(learning_rate=0.00001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        history2 = model.fit(
            train_gen,
            epochs=epochs_stage2,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks
        )
        
        history = {
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy']
        }
    
    model.save(f'saved_models/{model_name}_final.h5')
    
    if test_dir and os.path.exists(test_dir):
        test_gen = create_test_generator(test_dir, target_size=input_size, batch_size=batch_size)
        
        y_pred = model.predict(test_gen)
        y_true = test_gen.classes
        
        metrics = calculate_metrics(y_true, y_pred.flatten(), y_pred.flatten())
        
        with open(f'results/metrics/{model_name}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nTest Results for {model_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    
    if isinstance(history, dict):
        history_dict = history
    else:
        history_dict = history.history
    
    with open(f'results/metrics/{model_name}_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    return model, history_dict

def train_all_models(train_dir, val_dir, test_dir=None, models_to_train=None):
    if models_to_train is None:
        models_to_train = list(PRETRAINED_MODELS.keys()) + ['GlaucoNet']
    
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    results = {}
    
    for model_name in models_to_train:
        try:
            model, history = train_model(
                model_name=model_name,
                train_dir=train_dir,
                val_dir=val_dir,
                test_dir=test_dir
            )
            results[model_name] = {'status': 'success', 'history': history}
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            results[model_name] = {'status': 'failed', 'error': str(e)}
    
    with open('results/training_summary.json', 'w') as f:
        json.dump({k: {'status': v['status']} for k, v in results.items()}, f, indent=2)
    
    return results

if __name__ == '__main__':
    TRAIN_DIR = 'Fundus_Scanes_Sorted/Train'
    VAL_DIR = 'Fundus_Scanes_Sorted/Validation'
    TEST_DIR = 'Fundus_Scanes_Sorted/Test'
    
    if os.path.exists(VAL_DIR) and not os.path.exists(TEST_DIR):
        print("Splitting validation folder into val and test sets...")
        split_validation_folder(VAL_DIR, TEST_DIR, split_ratio=0.5)
    
    results = train_all_models(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        test_dir=TEST_DIR
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    for model_name, result in results.items():
        print(f"{model_name}: {result['status']}")
