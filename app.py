from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import zipfile
import os

# Initialize FastAPI app with OpenAPI enabled
app = FastAPI(
    title="Image Classification API",
    description="Upload an image or provide a URL for classification.",
    version="1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Paths for the model
MODEL_ZIP_PATH = "model/model_weights.zip"
MODEL_EXTRACTION_PATH = "model/"
MODEL_FILE_PATH = os.path.join(MODEL_EXTRACTION_PATH, "model_weights.h5")

# Extract the model if it's zipped
if os.path.exists(MODEL_ZIP_PATH) and not os.path.exists(MODEL_FILE_PATH):
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_EXTRACTION_PATH)
        print("✅ Model extracted successfully.")

# Load the model with error handling
model = None
if os.path.exists(MODEL_FILE_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_FILE_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
else:
    print("❌ Model file not found. Ensure model_weights.h5 is uploaded.")

# Preprocess the image before passing it to the model
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, H, W, C)
    return image.astype(np.float32)

# ✅ Add a Root Route to Avoid 404 Errors
@app.get("/")
async def home():
    return {
        "message": "Welcome to the Image Classification API!",
        "usage": "Use /predict/ to classify images",
        "docs": "/docs for API documentation"
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(None), url: str = None):
    """
    Predict the class of an uploaded image or an image from a URL.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="❌ Model failed to load, cannot make predictions.")

    if file and url:
        raise HTTPException(status_code=400, detail="Provide either a file or a URL, not both.")

    # Read image from file upload
    if file:
        try:
            image = Image.open(BytesIO(await file.read())).convert("RGB")  # Keep RGB
            print("📂 Received image via file upload.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")

    # Read image from URL
    elif url:
        try:
            response = requests.get(url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Invalid image URL or server error.")
            image = Image.open(BytesIO(response.content)).convert("RGB")  # Keep RGB
            print("🌍 Received image via URL.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"URL processing error: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail="No image provided.")

    # Preprocess and predict
    processed_image = preprocess_image(image)
    try:
        prediction = model.predict(processed_image)
        result = prediction.tolist()
        print("✅ Prediction successful!")
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
