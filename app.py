from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load the model
MODEL_PATH = "model_weights.h5"
model = tf.keras.models.load_model(MODEL_PATH)

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
