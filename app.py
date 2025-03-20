import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import requests
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import gc

app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = "model/model_weights.h5"
model = load_model(MODEL_PATH)

# Define class labels
class_labels = ['BAC_PNEUMONIA', 'NORMAL', 'VIR_PNEUMONIA']

# Image preprocessing function
def preprocess_image(img):
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((224, 224))  # Resize to match model input size
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def home():
    return jsonify({'message': 'Server is running. Use /predict or /upload for API requests.'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        
        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400
        
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch image'}), 400
        
        img = Image.open(BytesIO(response.content))
        img_array = preprocess_image(img)
        
        # Ensure TensorFlow session cleanup to avoid memory issues
        tf.keras.backend.clear_session()
        
        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        result = class_labels[predicted_class]
        probabilities = {class_labels[i]: float(predictions[0][i]) * 100 for i in range(len(class_labels))}
        
        # Force garbage collection to free up memory
        gc.collect()
        
        return jsonify({
            'prediction': result,
            'probabilities': probabilities
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            img = Image.open(file)
            img_array = preprocess_image(img)
            
            # Ensure TensorFlow session cleanup to avoid memory issues
            tf.keras.backend.clear_session()
            
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            result = class_labels[predicted_class]
            probabilities = {class_labels[i]: float(predictions[0][i]) * 100 for i in range(len(class_labels))}
            
            # Force garbage collection to free up memory
            gc.collect()
            
            return jsonify({
                'prediction': result,
                'probabilities': probabilities
            })
        else:
            return jsonify({'error': 'Invalid file format'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
