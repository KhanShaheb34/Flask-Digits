from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
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

    # Normalize pixel values to [0,1] range first
    X = X / 255.0

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        # Stratify to ensure balanced classes
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(
        f"Data loaded and split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier with optimized parameters
    """
    logger.info("Training Random Forest classifier...")

    # Create and train the model with better parameters
    rf_clf = RandomForestClassifier(
        n_estimators=200,      # More trees
        max_depth=20,          # Deeper trees
        min_samples_split=2,   # Minimum samples required to split
        min_samples_leaf=1,    # Minimum samples required at leaf node
        max_features='sqrt',   # Number of features to consider at each split
        bootstrap=True,        # Use bootstrap samples
        n_jobs=-1,            # Use all available cores
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )

    # Perform cross-validation to check model stability
    logger.info("Performing cross-validation...")
    cv_scores = cross_val_score(rf_clf, X_train, y_train, cv=5, n_jobs=-1)
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(
        f"Average CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train the final model on full training data
    rf_clf.fit(X_train, y_train)
    logger.info("Training completed")

    return rf_clf


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print detailed metrics
    """
    logger.info("Evaluating model...")

    # Get overall accuracy
    accuracy = model.score(X_test, y_test)
    logger.info(f"Model accuracy on test set: {accuracy:.4f}")

    # Get predictions
    y_pred = model.predict(X_test)

    # Calculate per-class accuracy
    for digit in sorted(set(y_test)):
        mask = y_test == digit
        digit_acc = (y_pred[mask] == y_test[mask]).mean()
        logger.info(f"Accuracy for digit {digit}: {digit_acc:.4f}")

    return accuracy


def save_model(model, accuracy):
    """
    Save the trained model
    """
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Save the model
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
