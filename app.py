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
    title="Pneumonia Classification API",
    description="Upload an X-ray image to classify as Normal, Viral Pneumonia, or Bacterial Pneumonia.",
    version="1.1",
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

# Define class labels
LABELS = ["Normal", "Viral Pneumonia", "Bacterial Pneumonia"]

# ✅ Preprocess function (Updated)
def preprocess_image(image: Image.Image):
    try:
        image = image.resize((150, 150))  # Ensure correct input size
        image = np.array(image) / 255.0   # Normalize pixel values

        # Check if the model expects grayscale or RGB
        if len(image.shape) == 2:  # Grayscale image (H, W)
            image = np.expand_dims(image, axis=-1)  # Add channel dimension
        elif len(image.shape) == 3 and image.shape[-1] == 3:  # RGB image (H, W, 3)
            image = np.mean(image, axis=-1, keepdims=True)  # Convert to grayscale

        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image.astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image preprocessing error: {str(e)}")

# ✅ Root endpoint to check if API is running
@app.get("/")
async def home():
    return {
        "message": "Welcome to the Pneumonia Classification API!",
        "usage": "Use /predict to classify images",
        "docs": "/docs for API documentation"
    }

# ✅ `/predict` endpoint (Supports both file & URL)
@app.post("/predict")
async def predict(file: UploadFile = File(None), url: str = Form(None)):
    """
    Predict the type of pneumonia in an X-ray image.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="❌ Model failed to load.")

    if file and url:
        raise HTTPException(status_code=400, detail="Provide either a file or a URL, not both.")

    image = None  # Initialize image variable

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

    # ✅ Preprocess Image
    processed_image = preprocess_image(image)

    # ✅ Make Prediction
    try:
        prediction = model.predict(processed_image)[0]  # Extract first prediction
        
        # Debugging: Print raw predictions
        print("🔍 Raw model output:", prediction)
        
        # Get the predicted class index & confidence
        predicted_index = np.argmax(prediction)
        predicted_class = LABELS[predicted_index]
        confidence_score = float(prediction[predicted_index])  # Extract confidence score

        print(f"✅ Final Prediction: {predicted_class} (Confidence: {confidence_score:.4f})")
        return {"prediction": predicted_class, "confidence": confidence_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ✅ Set proper deployment host & port
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use Render's assigned port
    uvicorn.run(app, host="0.0.0.0", port=port)
