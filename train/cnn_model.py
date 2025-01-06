"""
This module trains a CNN model on the MNIST dataset.
It includes functions to load data, create and train the model,
evaluate its performance, and save the trained model.
"""

import os
import logging

import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np

# pylint: disable=import-error
from tensorflow.keras import layers, models

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


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


def create_cnn_model():
    """
    Create a CNN model architecture
    """
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])

    return model


def train_cnn(X_train, y_train, X_test, y_test):
    """
    Train the CNN model
    """
    logger.info("Creating and compiling CNN model...")
    model = create_cnn_model()

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
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3
        )
    ]

    # Train the model
    logger.info("Training CNN model...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=128,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    return model, history


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the CNN model
    """
    logger.info("Evaluating model...")

    # Get overall accuracy
    _test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
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

    model_path = os.path.join(models_dir, 'cnn-digit-classifier.h5')

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
        model, _history = train_cnn(X_train, y_train, X_test, y_test)

        # Evaluate model
        accuracy = evaluate_model(model, X_test, y_test)

        # Save model
        save_model(model, accuracy)

    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        raise


if __name__ == "__main__":
    main()
