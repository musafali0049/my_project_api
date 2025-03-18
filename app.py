import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from flask import Flask, request, jsonify
import gradio as gr
import threading

app = Flask(__name__)

# Load the trained model (update path if needed)
MODEL_PATH = "model_weights.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define the class labels (ensure this matches your training setup)
class_labels = ['BAC_PNEUMONIA', 'NORMAL', 'VIR_PNEUMONIA']

# Image preprocessing function to match the Colab setup
def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((224, 224))  # Resize to match model input size
    img_array = np.array(image) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension for grayscale (1 channel)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

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

# Gradio interface function for image upload
def predict_gradio(image):
    image = Image.fromarray(image)
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Get the class with the highest probability
    return class_labels[predicted_class]

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
