from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import zipfile
import os
from fastapi import FastAPI

app = FastAPI(title="Image Classification API", description="Upload an image or provide a URL for classification.", version="1.0")

# Paths for the compressed model and extraction directory
MODEL_ZIP_PATH = "model/model_weights.zip"
MODEL_EXTRACTION_PATH = "model/"
MODEL_FILE_PATH = MODEL_EXTRACTION_PATH + "model_weights.h5"

# Extract the model if it's zipped
if os.path.exists(MODEL_ZIP_PATH) and not os.path.exists(MODEL_FILE_PATH):
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_EXTRACTION_PATH)
        print("✅ Model extracted successfully.")

# Load the model with error handling
try:
    model = tf.keras.models.load_model(MODEL_FILE_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Preprocess the image before passing it to the model
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust size based on model requirements
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(None), url: str = None):
    """
    Predict the class of an uploaded image or an image from a URL.
    """
    if file:
        try:
            image = Image.open(BytesIO(await file.read())).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")
    elif url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"URL processing error: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="No image provided")

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    result = prediction.tolist()

    return {"prediction": result}
