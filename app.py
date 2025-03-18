from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model (ensure this path is correct and model file is uploaded correctly)
MODEL_PATH = "model/model_weights.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define the class labels (ensure this matches your training setup)
class_labels = ['BAC_PNEUMONIA', 'NORMAL', 'VIR_PNEUMONIA']

# Image preprocessing function
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((224, 224))  # Resize to match model input size
    img_array = np.array(image) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension for grayscale
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
        
        # Fetch the image from URL
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

# Start the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)
