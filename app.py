from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from PIL import Image
import io
import base64

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
# Note: Make sure the model file exists in the models directory
MODEL_PATH = 'models/rf-digit-classifier.sav'


def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        return None


model = load_model()


def preprocess_image(image_data):
    """
    Preprocess the drawn digit image to match the format expected by the model
    1. Convert base64 to image
    2. Resize to 28x28
    3. Convert to grayscale
    4. Normalize pixel values
    """
    # Remove the data URL prefix to get the base64 string
    image_data = image_data.split(',')[1]

    # Convert base64 to image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))

    # Convert to grayscale and resize
    image = image.convert('L')
    image = image.resize((28, 28))

    # Convert to numpy array and normalize
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize to [0,1]

    # Flatten the image as our model expects a 1D array
    return image_array.reshape(1, -1)


@app.route('/')
def home():
    """Render the main page with the drawing canvas"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to handle digit prediction
    Receives: JSON with base64 encoded image data
    Returns: Prediction result
    """
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get the image data from the request
        image_data = request.json['image']

        # Preprocess the image
        processed_image = preprocess_image(image_data)

        # Make prediction
        prediction = model.predict(processed_image)[0]

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
