from fastapi import FastAPI, File, UploadFile, HTTPException, Form
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import os

# Initialize FastAPI
app = FastAPI(
    title="Pneumonia Detection API",
    description="Upload a chest X-ray image to classify it as Normal, Viral Pneumonia, or Bacterial Pneumonia.",
    version="1.0",
    docs_url="/docs"
)

# Load model
MODEL_PATH = "model/model_weights.h5"

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
else:
    print("❌ Model file not found. Ensure model_weights.h5 is uploaded.")

# Define labels as per your trained model
LABELS = ["Normal", "Viral Pneumonia", "Bacterial Pneumonia"]

# Image preprocessing function
def preprocess_image(image: Image.Image):
    try:
        image = image.resize((150, 150))  # Resize to match model input size
        image = image.convert("RGB")  # Ensure 3 channels (if trained on RGB)
        image = np.array(image) / 255.0  # Normalize to 0-1 range
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image.astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image preprocessing error: {str(e)}")

# Root endpoint
@app.get("/")
async def home():
    return {
        "message": "Welcome to the Pneumonia Detection API!",
        "usage": "Use /predict to classify X-ray images.",
        "docs": "/docs for API documentation"
    }

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(None), url: str = Form(None)):
    """
    Predict the class of an uploaded X-ray image.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="❌ Model failed to load.")

    if file and url:
        raise HTTPException(status_code=400, detail="Provide either a file or a URL, not both.")

    image = None

    # Load image from file
    if file:
        try:
            image = Image.open(BytesIO(await file.read()))
            print("📂 Image received from file upload.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")

    # Load image from URL
    elif url:
        try:
            response = requests.get(url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Invalid image URL or server error.")
            image = Image.open(BytesIO(response.content))
            print("🌍 Image received from URL.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"URL processing error: {str(e)}")

    else:
        raise HTTPException(status_code=400, detail="No image provided.")

    # Preprocess image
    processed_image = preprocess_image(image)

    # Predict
    try:
        prediction = model.predict(processed_image)  # Model output
        predicted_index = np.argmax(prediction)  # Highest probability class
        predicted_class = LABELS[predicted_index]  # Class label
        confidence = float(np.max(prediction))  # Confidence score

        print(f"✅ Prediction: {predicted_class} (Confidence: {confidence:.2f})")
        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "raw_output": prediction.tolist()  # For debugging
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Run the API
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use Render's assigned port
    uvicorn.run(app, host="0.0.0.0", port=port)
