import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from PIL import Image
import io
import base64
# Set matplotlib to use non-GUI backend
import matplotlib
matplotlib.use('Agg')

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


def save_debug_image(image_array, prediction):
    """Save the preprocessed image for debugging"""
    try:
        # Create a figure with the image
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(image_array.reshape(28, 28),
                   cmap='gray')  # Show original scale
        plt.title(f'Preprocessed Image (Prediction: {prediction})')
        plt.axis('off')

        # Save to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)  # Explicitly close the figure
        buf.seek(0)

        # Convert to base64
        debug_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        return debug_image
    except Exception as e:
        print(f"Error saving debug image: {str(e)}")
        return None


def preprocess_image(image_data):
    """
    Preprocess the drawn digit image to match the format expected by the model
    1. Convert base64 to image
    2. Resize to 28x28
    3. Convert to grayscale
    4. Normalize pixel values to [0,1] range
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

    return image_array


@app.route('/')
def home():
    """Render the main page with the drawing canvas"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to handle digit prediction
    Receives: JSON with base64 encoded image data
    Returns: Prediction result with debug image
    """
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        # Get the image data from the request
        image_data = request.json['image']

        # Preprocess the image
        processed_array = preprocess_image(image_data)

        # Reshape for prediction
        model_input = processed_array.reshape(1, -1)

        # Make prediction
        prediction = model.predict(model_input)[0]

        # Get debug image
        debug_image = save_debug_image(processed_array, prediction)

        return jsonify({
            'prediction': int(prediction),
            'debug_image': f'data:image/png;base64,{debug_image}' if debug_image else None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
