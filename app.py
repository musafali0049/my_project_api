import os
import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from PIL import Image
from io import BytesIO
import requests

# ✅ Initialize FastAPI
app = FastAPI(
    title="Pneumonia Classification API",
    version="1.0",
    description="Upload an X-ray image to classify pneumonia type."
)

# ✅ Define Model Path
MODEL_FILE_PATH = os.path.join(os.getcwd(), "model_weights.h5")

# ✅ Check if model file exists
if not os.path.exists(MODEL_FILE_PATH):
    raise RuntimeError(f"❌ Model file not found at {MODEL_FILE_PATH}. Ensure model_weights.h5 is uploaded.")

# ✅ Load the Model
try:
    model = tf.keras.models.load_model(MODEL_FILE_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"❌ Error loading model: {str(e)}")


# ✅ Image Preprocessing Function
def preprocess_image(image: Image.Image):
    try:
        image = image.convert("L")  # Convert to grayscale
        image = image.resize((150, 150))  # Resize to match model input
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=-1)  # Add grayscale channel
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image.astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image preprocessing error: {str(e)}")


# ✅ Root Endpoint
@app.get("/")
async def home():
    return {"message": "Pneumonia Classification API is running!", "docs": "/docs"}


# ✅ Prediction Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(None), url: str = Form(None)):
    if model is None:
        raise HTTPException(status_code=500, detail="❌ Model failed to load.")

    if not file and not url:
        raise HTTPException(status_code=400, detail="Provide either a file or a URL.")

    # ✅ Load image from file
    if file:
        try:
            image = Image.open(BytesIO(await file.read()))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"File processing error: {str(e)}")

    # ✅ Load image from URL
    elif url:
        try:
            response = requests.get(url)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Invalid image URL.")
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"URL processing error: {str(e)}")

    # ✅ Preprocess and Predict
    processed_image = preprocess_image(image)
    try:
        prediction = model.predict(processed_image)
        labels = ["Normal", "Viral Pneumonia", "Bacterial Pneumonia"]
        predicted_index = np.argmax(prediction)
        predicted_class = labels[predicted_index]
        confidence = float(np.max(prediction))
        return {"prediction": predicted_class, "confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ✅ Start FastAPI Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
