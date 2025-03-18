import tensorflow as tf
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify
import gradio as gr
import threading
import gdown
import os

app = Flask(__name__)

# Google Drive model link
MODEL_URL = "https://drive.google.com/uc?id=1R5s5MekrMvI5-HgPFusSf40OSVdWR8vE"
MODEL_PATH = "model/model_weights.h5"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load the TensorFlow model
model = tf.keras.models.load_model(MODEL_PATH)

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
        
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch image'}), 400
        
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image = preprocess_image(image)

        prediction = model.predict(image)
        prediction = prediction.tolist()
        
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def predict_gradio(image):
    image = Image.fromarray(image)
    image = preprocess_image(image)
    prediction = model.predict(image)
    return prediction.tolist()

iface = gr.Interface(fn=predict_gradio, inputs=gr.Image(), outputs=gr.Label())

if __name__ == '__main__':
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)).start()
    iface.launch()
