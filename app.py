import tensorflow as tf
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model("model/model_weights.h5")  # Updated path

# Define image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize
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
        prediction = prediction.tolist()  # Convert to Python list for JSON serialization
        
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
