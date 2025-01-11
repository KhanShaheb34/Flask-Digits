# Digit Classifier Web App

A simple Flask web application that allows users to draw digits and get predictions using a trained Random Forest classifier.

## Project Structure

```
.
├── app.py              # Main Flask application
├── models/             # Directory for ML models
│   └── rf-digit-classifier.sav  # Trained Random Forest model
│   └── cnn-digit-classifier.h5  # Trained CNN model
│   └── cnn-augmented-classifier.h5  # Trained CNN model with data augmentation
├── templates/          # HTML templates
│   └── index.html     # Main page with drawing canvas
├── train/             # Training scripts
│   ├── random_forest.py       # Random Forest training
│   ├── cnn_model.py          # Basic CNN training
│   └── cnn_augmented_model.py # CNN with data augmentation
└── requirements.txt    # Python dependencies
```

## Setup Instructions

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training Models

You have three options for training the digit classifier:

1. Random Forest Classifier (Default):

```bash
python train/random_forest.py
```

This will train a Random Forest model and save it as `models/rf-digit-classifier.sav`.

2. Basic CNN Model:

```bash
python train/cnn_model.py
```

This will train a basic Convolutional Neural Network model.

3. CNN with Data Augmentation:

```bash
python train/cnn_augmented_model.py
```

This version uses data augmentation techniques for improved performance.

Choose one of the above training scripts based on your needs. The web application is configured to use the Random Forest model by default.

## Running Flask Server

4. Run the application:

```bash
python app.py
```

5. Open your browser and go to `http://localhost:5000`

## Usage

1. Draw a digit (0-9) on the canvas using your mouse
2. Click the "Predict" button to get the model's prediction
3. Use the "Clear" button to erase and start over

## Code Structure and Learning Points

- `app.py`:

  - Flask application setup and routing
  - Model loading and error handling
  - Image preprocessing pipeline
  - RESTful API endpoint for predictions

- `templates/index.html`:
  - HTML5 Canvas for drawing
  - JavaScript event handling
  - Frontend-backend communication using Fetch API
  - Simple and clean UI with CSS

## Educational Notes

This project demonstrates several important MLOps concepts:

1. Model Serving:

   - Loading a trained model in a web application
   - Error handling for model loading and predictions
   - Input preprocessing pipeline

2. Web Development:

   - RESTful API design
   - Frontend-backend communication
   - User input handling
   - Real-time predictions

3. Best Practices:
   - Clean code structure
   - Error handling
   - User feedback
   - Code comments and documentation
