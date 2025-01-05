from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def load_mnist_data():
    """
    Load MNIST dataset from OpenML
    Returns normalized training and testing sets
    """
    logger.info("Loading MNIST dataset...")
    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version=1,
                        return_X_y=True, as_frame=False)

    # Convert data to float32 for faster processing
    X = X.astype('float32')

    # Normalize the data
    X = X / 255.0

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(
        f"Data loaded and split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier
    """
    logger.info("Training Random Forest classifier...")
    # Create and train the model
    rf_clf = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=10,      # Maximum depth of trees
        n_jobs=-1,        # Use all available cores
        random_state=42
    )

    rf_clf.fit(X_train, y_train)
    logger.info("Training completed")
    return rf_clf


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print accuracy
    """
    logger.info("Evaluating model...")
    accuracy = model.score(X_test, y_test)
    logger.info(f"Model accuracy on test set: {accuracy:.4f}")
    return accuracy


def save_model(model, accuracy):
    """
    Save the trained model
    """
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Save the model with timestamp and accuracy in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(models_dir, f'rf-digit-classifier.sav')

    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)
    logger.info("Model saved successfully")


def main():
    """
    Main function to orchestrate the training process
    """
    try:
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_mnist_data()

        # Train model
        model = train_random_forest(X_train, y_train)

        # Evaluate model
        accuracy = evaluate_model(model, X_test, y_test)

        # Save model
        save_model(model, accuracy)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
