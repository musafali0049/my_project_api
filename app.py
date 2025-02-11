from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import zipfile
import os

app = Flask(__name__)

# Paths for the compressed model and extraction directory
MODEL_ZIP_PATH = "model/model_weights.zip"
MODEL_EXTRACTION_PATH = "model/"
MODEL_FILE_PATH = MODEL_EXTRACTION_PATH + "model_weights.h5"

# Extract the model if it's zipped
if os.path.exists(MODEL_ZIP_PATH) and not os.path.exists(MODEL_FILE_PATH):
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_EXTRACTION_PATH)
        print("Model extracted successfully.")

# Load the model
model = tf.keras.models.load_model(MODEL_FILE_PATH)

# Preprocess the image before passing it to the model
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust size based on model requirements
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_url = data.get("url")

    if not image_url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)
        result = prediction.tolist()

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
