import tensorflow as tf
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
import gradio as gr

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

@app.route('/predict', methods=['POST'])
def predict():
    try:
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

iface = gr.Interface(fn=predict_gradio, inputs=gr.Image(), outputs=gr.Label())

if __name__ == '__main__':
    from threading import Thread
    
    # Run Flask in a separate thread
    def run_flask():
        app.run(host='0.0.0.0', port=8000)
    
    flask_thread = Thread(target=run_flask)
    flask_thread.start()
    
    # Run Gradio UI
    iface.launch()