"""
Flask web application for digit classification using multiple ML models.
Provides an interface to draw digits and get predictions using either
Random Forest or CNN models.
"""

import io
import base64

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image
import joblib

# Set matplotlib to use non-GUI backend
matplotlib.use('Agg')

# Initialize Flask application
app = Flask(__name__)

# Model paths
RF_MODEL_PATH = 'models/rf-digit-classifier.sav'
CNN_MODEL_PATH = 'models/cnn-digit-classifier.h5'
CNN_AUG_MODEL_PATH = 'models/cnn-augmented-classifier.h5'

# Dictionary to store loaded models
models = {}


def load_models():
    """Load both RF and CNN models"""
    try:
        models['rf'] = joblib.load(RF_MODEL_PATH)
        print("Random Forest model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load Random Forest model: {str(e)}")
        models['rf'] = None

    try:
        models['cnn'] = tf.keras.models.load_model(CNN_MODEL_PATH)
        print("CNN model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load CNN model: {str(e)}")
        models['cnn'] = None

    try:
        models['cnn_aug'] = tf.keras.models.load_model(CNN_AUG_MODEL_PATH)
        print("Augmented CNN model loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load Augmented CNN model: {str(e)}")
        models['cnn_aug'] = None


# Load models at startup
load_models()


def save_debug_image(image_array, prediction):
    """Save the preprocessed image for debugging"""
    try:
        # Create a figure with the image
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(image_array.reshape(28, 28), cmap='gray')
        plt.title(f'Preprocessed Image (Prediction: {prediction})')
        plt.axis('off')

        # Save to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)

        # Convert to base64
        debug_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        return debug_image
    except Exception as e:
        print(f"Error saving debug image: {str(e)}")
        return None


def preprocess_image(image_data, model_type='rf'):
    """
    Preprocess the drawn digit image based on model type
    """
    # Remove the data URL prefix to get the base64 string
    image_data = image_data.split(',')[1]

    # Convert base64 to image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to grayscale and resize
    image = image.convert('L')
    image = image.resize((28, 28), Image.Resampling.BILINEAR)

    # Convert to numpy array and normalize
    image_array = np.array(image, dtype=np.float32)

    # Invert the colors (since MNIST has white digits on black background)
    image_array = 255 - image_array

    # Normalize to [0,1] range
    image_array = image_array / 255.0

    if model_type in ['cnn', 'cnn_aug']:
        # Reshape for CNN (add batch and channel dimensions)
        return image_array.reshape(1, 28, 28, 1)
    else:
        # Flatten for RF
        return image_array.reshape(1, -1)


@app.route('/')
def home():
    """Render the main page with the drawing canvas"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to handle digit prediction
    Receives: JSON with base64 encoded image data and model type
    Returns: Prediction result with debug image
    """
    try:
        # Get the image data and model type from the request
        image_data = request.json['image']
        # Default to RF if not specified
        model_type = request.json.get('model_type', 'rf')

        if model_type not in models or models[model_type] is None:
            return jsonify({'error': f'{model_type.upper()} model not loaded'}), 500

        # Preprocess the image
        processed_array = preprocess_image(image_data, model_type)

        # Make prediction
        if model_type in ['cnn', 'cnn_aug']:
            prediction = int(
                np.argmax(models[model_type].predict(processed_array, verbose=0)[0]))
        else:
            prediction = int(models[model_type].predict(processed_array)[0])

        # Get debug image (use the 2D version for visualization)
        debug_image = save_debug_image(
            processed_array.reshape(28, 28),
            prediction
        )

        return jsonify({
            'prediction': prediction,
            'debug_image': f'data:image/png;base64,{debug_image}' if debug_image else None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
