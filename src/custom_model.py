import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

def squeeze_excitation_block(input_tensor, ratio=16):
    channels = input_tensor.shape[-1]
    
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(channels // ratio, activation='relu', kernel_regularizer=regularizers.l2(0.01))(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, channels))(se)
    
    return layers.Multiply()([input_tensor, se])

def conv_block(x, filters, kernel_size=3, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def residual_block(x, filters, use_se=True):
    shortcut = x
    
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same', kernel_regularizer=regularizers.l2(0.01))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = conv_block(x, filters)
    x = layers.Conv2D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    
    if use_se:
        x = squeeze_excitation_block(x)
    
    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    
    return x

def build_glauconet(input_shape=(224, 224, 3), num_classes=1):
    inputs = layers.Input(shape=input_shape)
    
    x = conv_block(inputs, 32)
    x = layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, 64)
    x = layers.MaxPooling2D(2)(x)
    
    x = residual_block(x, 128, use_se=True)
    x = layers.MaxPooling2D(2)(x)
    
    x = residual_block(x, 256, use_se=True)
    x = layers.MaxPooling2D(2)(x)
    
    x = residual_block(x, 512, use_se=True)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs, outputs, name='GlaucoNet')
    
    return model

def get_glauconet_summary():
    model = build_glauconet()
    
    summary_list = []
    model.summary(print_fn=lambda x: summary_list.append(x))
    
    return '\n'.join(summary_list)

def get_model_architecture_info():
    return {
        'name': 'GlaucoNet',
        'description': 'Custom CNN architecture for glaucoma detection with residual connections and squeeze-excitation attention mechanisms.',
        'input_shape': '224x224x3',
        'total_layers': 'Multiple convolutional, residual, and SE attention blocks',
        'key_features': [
            'Residual connections for better gradient flow',
            'Squeeze-and-Excitation attention blocks',
            'Batch normalization for training stability',
            'Dropout regularization to prevent overfitting',
            'L2 weight regularization',
            'Binary classification with sigmoid activation'
        ],
        'architecture': [
            'Input (224x224x3)',
            'Conv Block 1 (32 filters) + MaxPool',
            'Conv Block 2 (64 filters) + MaxPool',
            'Residual Block 1 (128 filters) + SE + MaxPool',
            'Residual Block 2 (256 filters) + SE + MaxPool',
            'Residual Block 3 (512 filters) + SE',
            'Global Average Pooling',
            'Dense (512) + BN + ReLU + Dropout(0.5)',
            'Dense (256) + BN + ReLU + Dropout(0.3)',
            'Output (1, sigmoid)'
        ]
    }
