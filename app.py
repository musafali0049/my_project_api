import tensorflow as tf
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
import gradio as gr
import threading

app = Flask(__name__)

# Load the TensorFlow model (updated path)
MODEL_PATH = "model/model_weights.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to match model input
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def home():
    return "Welcome to the Prediction API! Use /predict to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract the image URL from the request
        data = request.get_json()
        image_url = data.get('image_url')
        
        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400
        
        # Fetch image from URL
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch image'}), 400
        
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image = preprocess_image(image)

        # Perform prediction
        prediction = model.predict(image)
        prediction = prediction.tolist()  # Convert to JSON serializable format
        
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Gradio interface function
def predict_gradio(image):
    image = Image.fromarray(image)
    image = preprocess_image(image)
    prediction = model.predict(image)
    return prediction.tolist()

# Gradio interface setup
iface = gr.Interface(fn=predict_gradio, inputs=gr.Image(), outputs=gr.Label())

# Function to run Flask app
def run_flask():
    app.run(host='0.0.0.0', port=10000, debug=False, use_reloader=False)

# Function to run Gradio interface
def run_gradio():
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)

# Run Flask and Gradio together
if __name__ == '__main__':
    # Running Flask on a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # Running Gradio interface on a separate thread
    gradio_thread = threading.Thread(target=run_gradio)
    gradio_thread.start()
