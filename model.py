import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def AnalogyBasedCNN(input_shape, num_classes):
    """
    A classical Convolutional Neural Network (CNN) architecture inspired by
    hierarchical feature extraction analogies.
    
    This model creates a multi-scale representation of the input image,
    conceptually similar to how quantum states might represent superpositions,
    but implemented entirely using classical deep learning operations (Conv2D).
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model


