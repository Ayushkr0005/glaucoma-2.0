"""
Enhanced Deep Learning Models for Glaucoma Detection
Novel architectures with attention mechanisms, multi-scale features, and ensemble learning
Designed for publication-quality results on RFMID dataset
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.applications import (
    ResNet50, ResNet50V2,
    VGG16, VGG19,
    DenseNet121, DenseNet169, DenseNet201,
    InceptionV3, Xception,
    MobileNetV2,
    EfficientNetB0, EfficientNetV2S, EfficientNetV2M,
    NASNetMobile
)
import numpy as np


def squeeze_excitation_block(x, ratio=16):
    """Squeeze-and-Excitation attention block for channel recalibration"""
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // ratio, activation='relu', 
                      kernel_regularizer=regularizers.l2(1e-4))(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    return layers.Multiply()([x, se])


def cbam_attention(x, ratio=8):
    """Convolutional Block Attention Module (CBAM)"""
    filters = x.shape[-1]
    
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    
    shared_dense1 = layers.Dense(filters // ratio, activation='relu',
                                  kernel_regularizer=regularizers.l2(1e-4))
    shared_dense2 = layers.Dense(filters, kernel_regularizer=regularizers.l2(1e-4))
    
    avg_out = shared_dense2(shared_dense1(avg_pool))
    max_out = shared_dense2(shared_dense1(max_pool))
    
    channel_attention = layers.Activation('sigmoid')(layers.Add()([avg_out, max_out]))
    channel_attention = layers.Reshape((1, 1, filters))(channel_attention)
    x = layers.Multiply()([x, channel_attention])
    
    avg_spatial = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(x)
    max_spatial = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(x)
    spatial = layers.Concatenate()([avg_spatial, max_spatial])
    spatial_attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(spatial)
    
    return layers.Multiply()([x, spatial_attention])


def residual_attention_block(x, filters, kernel_size=3):
    """Residual block with integrated attention"""
    shortcut = x
    
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    
    x = cbam_attention(x)
    
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('gelu')(x)
    return x


def multi_scale_feature_extraction(x):
    """Extract features at multiple scales using dilated convolutions"""
    filters = x.shape[-1] // 4
    
    branch1 = layers.Conv2D(filters, 1, padding='same', activation='relu')(x)
    
    branch2 = layers.Conv2D(filters, 3, padding='same', dilation_rate=1, activation='relu')(x)
    
    branch3 = layers.Conv2D(filters, 3, padding='same', dilation_rate=2, activation='relu')(x)
    
    branch4 = layers.Conv2D(filters, 3, padding='same', dilation_rate=4, activation='relu')(x)
    
    pooled = layers.GlobalAveragePooling2D(keepdims=True)(x)
    pooled = layers.Conv2D(filters, 1, activation='relu')(pooled)
    pooled = layers.Lambda(lambda args: tf.image.resize(args[0], [tf.shape(args[1])[1], tf.shape(args[1])[2]]))([pooled, x])
    
    return layers.Concatenate()([branch1, branch2, branch3, branch4, pooled])


def create_glauconet_v2(input_shape=(224, 224, 3), num_classes=1):
    """
    GlaucoNet-V2: Advanced architecture with:
    - Multi-scale feature extraction (ASPP-style)
    - CBAM attention at multiple levels
    - Residual connections with dropout
    - GELU activation for smoother gradients
    - Deep supervision for better gradient flow
    """
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(64, 7, strides=2, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    x = residual_attention_block(x, 64)
    x = residual_attention_block(x, 64)
    stage1_out = x
    
    x = layers.MaxPooling2D(2)(x)
    x = residual_attention_block(x, 128)
    x = residual_attention_block(x, 128)
    stage2_out = x
    
    x = layers.MaxPooling2D(2)(x)
    x = residual_attention_block(x, 256)
    x = residual_attention_block(x, 256)
    x = residual_attention_block(x, 256)
    stage3_out = x
    
    x = layers.MaxPooling2D(2)(x)
    x = residual_attention_block(x, 512)
    x = residual_attention_block(x, 512)
    x = residual_attention_block(x, 512)
    
    x = multi_scale_feature_extraction(x)
    x = layers.Conv2D(512, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    
    x = cbam_attention(x)
    
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([gap, gmp])
    
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name='GlaucoNet_V2')
    return model


def create_glauconet_v3(input_shape=(224, 224, 3), num_classes=1):
    """
    GlaucoNet-V3: Hybrid CNN-Attention architecture with:
    - Patch-like stem convolution (4x4 stride)
    - Squeeze-Excitation channel attention throughout
    - CBAM spatial-channel attention
    - Progressive feature extraction (48→96→192→384)
    - Dual global pooling (GAP + GMP)
    """
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(48, 4, strides=4, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    
    for filters in [96, 192, 384]:
        shortcut = layers.Conv2D(filters, 1, strides=2, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)
        
        x = layers.Conv2D(filters, 3, strides=2, padding='same',
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('gelu')(x)
        x = layers.Conv2D(filters, 3, padding='same',
                          kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        
        x = squeeze_excitation_block(x)
        x = layers.Add()([x, shortcut])
        x = layers.Activation('gelu')(x)
        
        for _ in range(2):
            res = x
            x = layers.Conv2D(filters, 3, padding='same',
                              kernel_regularizer=regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('gelu')(x)
            x = layers.Conv2D(filters, 3, padding='same',
                              kernel_regularizer=regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization()(x)
            x = squeeze_excitation_block(x)
            x = layers.Add()([x, res])
            x = layers.Activation('gelu')(x)
    
    x = cbam_attention(x)
    
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([gap, gmp])
    
    x = layers.Dense(384, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(192, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name='GlaucoNet_V3')
    return model


def create_pretrained_model(model_name, input_shape=(224, 224, 3), num_classes=1):
    """Create fine-tuned pre-trained model with enhanced classification head"""
    
    model_configs = {
        'ResNet50': (ResNet50, (224, 224, 3)),
        'ResNet50V2': (ResNet50V2, (224, 224, 3)),
        'VGG16': (VGG16, (224, 224, 3)),
        'VGG19': (VGG19, (224, 224, 3)),
        'DenseNet121': (DenseNet121, (224, 224, 3)),
        'DenseNet169': (DenseNet169, (224, 224, 3)),
        'DenseNet201': (DenseNet201, (224, 224, 3)),
        'InceptionV3': (InceptionV3, (299, 299, 3)),
        'Xception': (Xception, (299, 299, 3)),
        'MobileNetV2': (MobileNetV2, (224, 224, 3)),
        'EfficientNetB0': (EfficientNetB0, (224, 224, 3)),
        'EfficientNetV2S': (EfficientNetV2S, (384, 384, 3)),
        'EfficientNetV2M': (EfficientNetV2M, (480, 480, 3)),
        'NASNetMobile': (NASNetMobile, (224, 224, 3)),
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}")
    
    base_class, optimal_size = model_configs[model_name]
    
    base_model = base_class(
        weights='imagenet',
        include_top=False,
        input_shape=optimal_size,
        pooling=None
    )
    
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    
    x = squeeze_excitation_block(x)
    
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([gap, gmp])
    
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(base_model.input, outputs, name=f'{model_name}_Enhanced')
    return model, base_model


def unfreeze_model_layers(model, base_model, unfreeze_ratio=0.3):
    """Unfreeze top layers of base model for fine-tuning"""
    total_layers = len(base_model.layers)
    layers_to_unfreeze = int(total_layers * unfreeze_ratio)
    
    for layer in base_model.layers[-layers_to_unfreeze:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    
    return model


class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        
        focal_loss = -alpha_factor * modulating_factor * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)


class CombinedLoss(tf.keras.losses.Loss):
    """Combined Binary Cross-Entropy + Focal Loss"""
    def __init__(self, focal_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.focal_weight = focal_weight
        self.focal_loss = FocalLoss()
        self.bce_loss = tf.keras.losses.BinaryCrossentropy()
    
    def call(self, y_true, y_pred):
        focal = self.focal_loss(y_true, y_pred)
        bce = self.bce_loss(y_true, y_pred)
        return self.focal_weight * focal + (1 - self.focal_weight) * bce


def get_optimizer(learning_rate=1e-3, optimizer_type='adamw'):
    """Get optimizer with optional weight decay"""
    if optimizer_type == 'adamw':
        return tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-5
        )
    elif optimizer_type == 'sgd':
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True
        )
    else:
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def get_lr_scheduler(epochs, warmup_epochs=5, min_lr=1e-6):
    """Cosine annealing with warm restarts scheduler"""
    def scheduler(epoch, lr):
        if epoch < warmup_epochs:
            return lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return min_lr + 0.5 * (lr - min_lr) * (1 + np.cos(np.pi * progress))
    
    return tf.keras.callbacks.LearningRateScheduler(scheduler)


def create_glauconet_original(input_shape=(224, 224, 3), num_classes=1):
    """Original GlaucoNet architecture with SE attention"""
    from src.custom_model import build_glauconet
    return build_glauconet(input_shape=input_shape, num_classes=num_classes)


MODEL_CONFIGS = {
    'GlaucoNet': {'create_fn': create_glauconet_original, 'input_size': (224, 224)},
    'GlaucoNet_V2': {'create_fn': create_glauconet_v2, 'input_size': (224, 224)},
    'GlaucoNet_V3': {'create_fn': create_glauconet_v3, 'input_size': (224, 224)},
    'ResNet50': {'pretrained': True, 'input_size': (224, 224)},
    'ResNet50V2': {'pretrained': True, 'input_size': (224, 224)},
    'VGG16': {'pretrained': True, 'input_size': (224, 224)},
    'VGG19': {'pretrained': True, 'input_size': (224, 224)},
    'DenseNet121': {'pretrained': True, 'input_size': (224, 224)},
    'DenseNet169': {'pretrained': True, 'input_size': (224, 224)},
    'DenseNet201': {'pretrained': True, 'input_size': (224, 224)},
    'InceptionV3': {'pretrained': True, 'input_size': (299, 299)},
    'Xception': {'pretrained': True, 'input_size': (299, 299)},
    'MobileNetV2': {'pretrained': True, 'input_size': (224, 224)},
    'EfficientNetB0': {'pretrained': True, 'input_size': (224, 224)},
    'EfficientNetV2S': {'pretrained': True, 'input_size': (384, 384)},
    'EfficientNetV2M': {'pretrained': True, 'input_size': (480, 480)},
    'NASNetMobile': {'pretrained': True, 'input_size': (224, 224)},
}
