from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import zipfile
import os

# Initialize FastAPI
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

# Extract model if necessary
if os.path.exists(MODEL_ZIP_PATH) and not os.path.exists(MODEL_FILE_PATH):
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(MODEL_EXTRACTION_PATH)
        print("✅ Model extracted successfully.")

# Load model
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

# Correct labels for classification
labels = ["Normal", "Viral Pneumonia", "Bacterial Pneumonia"]

# ✅ Preprocess function (Updated)
def preprocess_image(image: Image.Image):
    try:
        image = image.resize((150, 150))  # Resize to match model input
        image = np.array(image)
        
        # Ensure 3-channel RGB (in case of grayscale images)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB
        
        # Normalize pixel values
        image = image / 255.0
        
        # Expand dimensions to match model input
        image = np.expand_dims(image, axis=0)
        
        return image.astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image preprocessing error: {str(e)}")

# ✅ Root endpoint to check if API is running
@app.get("/")
async def home():
    return {
        "message": "Welcome to the Image Classification API!",
        "usage": "Use /predict to classify images",
        "docs": "/docs for API documentation"
    }

# ✅ `/predict` endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(None), url: str = Form(None)):
    if model is None:
        raise HTTPException(status_code=500, detail="❌ Model failed to load.")

    if file and url:
        raise HTTPException(status_code=400, detail="Provide either a file or a URL, not both.")

    image = None
    
    # Read image from file
    if file:
        try:
            image = Image.open(BytesIO(await file.read()))
            print("📂 Received image via file upload.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")
    
    # Read image from URL
    elif url:
        try:
            response = requests.get(url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Invalid image URL or server error.")
            image = Image.open(BytesIO(response.content))
            print("🌍 Received image via URL.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"URL processing error: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="No image provided.")

    # ✅ Preprocess image
    processed_image = preprocess_image(image)

    # ✅ Make Prediction
    try:
        prediction = model.predict(processed_image)
        print(f"Raw model output: {prediction}")  # Debugging output
        
        # Apply softmax for proper probability distribution
        probabilities = tf.nn.softmax(prediction[0]).numpy()
        predicted_index = np.argmax(probabilities)
        predicted_class = labels[predicted_index]

        print(f"✅ Prediction: {predicted_class} with confidence: {probabilities.tolist()}")
        return {"prediction": predicted_class, "confidence": probabilities.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ✅ Set proper deployment host & port
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
