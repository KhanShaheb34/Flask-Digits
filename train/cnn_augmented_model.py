"""
This module trains an augmented CNN model on the MNIST dataset with various
data augmentations to make it more robust for real-world digit recognition.
"""

import os
import logging

import tensorflow as tf
import keras
from keras import layers, models
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def create_augmentation_layer():
    """
    Create a data augmentation pipeline to make model more robust
    """
    return keras.Sequential([
        # Random translation (position variations)
        layers.RandomTranslation(height_factor=0.2, width_factor=0.2),

        # Random zoom (size variations)
        layers.RandomZoom(height_factor=(-0.5, 0.2), width_factor=(-0.5, 0.2)),

        # Random rotation (orientation variations)
        layers.RandomRotation(factor=0.2),

        # Random contrast (helps with different stroke thicknesses)
        layers.RandomContrast(factor=0.2)
    ])


def load_mnist_data():
    """
    Load MNIST dataset from OpenML and prepare it for CNN
    """
    logger.info("Loading MNIST dataset...")
    # Load data
    X, y = fetch_openml('mnist_784', version=1,
                        return_X_y=True, as_frame=False)

    # Convert data to float32 and normalize to [0,1]
    X = X.astype('float32') / 255.0

    # Reshape data for CNN (add channel dimension)
    X = X.reshape(-1, 28, 28, 1)

    # Convert labels to integers
    y = y.astype(int)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(
        "Data loaded and split: %d training samples, %d test samples",
        X_train.shape[0], X_test.shape[0])
    return X_train, X_test, y_train, y_test


def create_augmented_cnn_model():
    """
    Create a CNN model with integrated data augmentation
    """
    # Input layer
    inputs = layers.Input(shape=(28, 28, 1))

    # Data augmentation layer (only active during training)
    augmented = create_augmentation_layer()(inputs, training=True)

    # First Convolutional Block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(augmented)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Second Convolutional Block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Third Convolutional Block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Dense Layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)


def train_augmented_cnn(X_train, y_train, X_test, y_test):
    """
    Train the augmented CNN model
    """
    logger.info("Creating and compiling augmented CNN model...")
    model = create_augmented_cnn_model()

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Add callbacks for better training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,  # More patience for augmented training
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5
        )
    ]

    # Train the model
    logger.info("Training augmented CNN model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,  # More epochs for better learning with augmentation
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the augmented CNN model
    """
    logger.info("Evaluating model...")

    # Get overall accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info("Test accuracy: %.4f", test_accuracy)

    # Get per-class accuracy
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    for digit in range(10):
        mask = y_test == digit
        digit_acc = (y_pred[mask] == y_test[mask]).mean()
        logger.info("Accuracy for digit %d: %.4f", digit, digit_acc)

    return test_accuracy


def save_model(model, _accuracy):
    """
    Save the trained model
    """
    models_dir = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'cnn-augmented-classifier.h5')

    logger.info("Saving model to %s", model_path)
    model.save(model_path)
    logger.info("Model saved successfully")


def main():
    """
    Main function to orchestrate the training process
    """
    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_mnist_data()

        # Train model
        model, _history = train_augmented_cnn(X_train, y_train, X_test, y_test)

        # Evaluate model
        accuracy = evaluate_model(model, X_test, y_test)

        # Save model
        save_model(model, accuracy)

    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise


if __name__ == "__main__":
    main()
