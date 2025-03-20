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
import gradio as gr

app = Flask(__name__)
CORS(app)

# Load the trained model
MODEL_PATH = "model/model_weights.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
class_labels = ['BAC_PNEUMONIA', 'NORMAL', 'VIR_PNEUMONIA']

# Image preprocessing function
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((224, 224))  # Resize to match model input size
    img_array = np.array(image) / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

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
        
        image = Image.open(BytesIO(response.content))
        img_array = preprocess_image(image)

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        # Prepare the response
        result = class_labels[predicted_class]
        probabilities = {class_labels[i]: predictions[0][i] * 100 for i in range(len(class_labels))}

        return jsonify({
            'prediction': result,
            'probabilities': probabilities
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Gradio Interface
def predict_gradio(image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    result = class_labels[predicted_class]
    probabilities = {class_labels[i]: round(predictions[0][i] * 100, 2) for i in range(len(class_labels))}
    return result, probabilities

iface = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Image(type="pil"),  # Fixed issue with 'shape'
    outputs=["text", "label"]
)

# Start the Flask app with port binding for Render
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if not set
    app.run(host='0.0.0.0', port=port, debug=False)
