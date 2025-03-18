import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
from PIL import Image  # Ensure PIL is imported for Image processing
from flask import Flask, request, jsonify
from flask_cors import CORS  # For enabling CORS
import os  # Import os to handle dynamic port binding

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model (ensure this path is correct and model file is uploaded correctly)
MODEL_PATH = "model/model_weights.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define the class labels (ensure this matches your training setup)
class_labels = ['BAC_PNEUMONIA', 'NORMAL', 'VIR_PNEUMONIA']

# Image preprocessing function to match the specification
def preprocess_image(image):
    # Resize and convert image to grayscale
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((224, 224))  # Resize to match model input size
    img_array = np.array(image)  # Convert image to numpy array
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the image URL from the request
        data = request.get_json()
        image_url = data.get('image_url')
        
        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400
        
        # Fetch the image from the URL
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch image'}), 400
        
        image = Image.open(BytesIO(response.content))
        img_array = preprocess_image(image)

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)  # Get the class with the highest probability

        # Prepare the prediction response
        result = class_labels[predicted_class]
        probabilities = {class_labels[i]: predictions[0][i] * 100 for i in range(len(class_labels))}

        return jsonify({
            'prediction': result,
            'probabilities': probabilities
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Start the Flask app with dynamic port binding
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if no port is found
    app.run(host='0.0.0.0', port=port, debug=False)
