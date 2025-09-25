# backend/model_builder.py
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_default_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Rebuild the architecture you trained:
    EfficientNetB4 (include_top=False, pooling='max') -> Dense(256, relu) -> Dropout(0.45) -> Dense(num_classes, softmax)
    """
    base_model = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights=None,  # no pre-trained weights for serving; we're going to load weights
        input_shape=input_shape,
        pooling='max'
    )
    model = Sequential([
        base_model,
        Dense(256, activation='relu'),
        Dropout(rate=0.45, seed=123),
        Dense(num_classes, activation='softmax')
    ])
    # No compile needed for inference
    return model