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
    description="Classifies chest X-ray images into Normal, Viral Pneumonia, or Bacterial Pneumonia.",
    version="1.0",
)

# Model Path
MODEL_PATH = "model_weights.h5"

# Load Model
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"❌ Model loading error: {e}")
else:
    raise RuntimeError("❌ Model file not found. Ensure model_weights.h5 is uploaded.")

# ✅ Preprocessing function (Fixed grayscale issue)
def preprocess_image(image: Image.Image):
    try:
        image = image.convert("L")  # Convert to grayscale (1 channel)
        image = image.resize((150, 150))  # Resize to match model input
        image = np.array(image) / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=-1)  # Ensure shape (150, 150, 1)
        image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 150, 150, 1)
        return image.astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image preprocessing error: {str(e)}")

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

    # ✅ Convert to grayscale and preprocess
    processed_image = preprocess_image(image)

    # ✅ Make Prediction
    try:
        prediction = model.predict(processed_image)
        
        # Define labels
        labels = ["Normal", "Viral Pneumonia", "Bacterial Pneumonia"]

        # Get the predicted class index
        predicted_index = np.argmax(prediction)

        # Get the corresponding class label
        predicted_class = labels[predicted_index]

        confidence_score = float(np.max(prediction))  # Get confidence level

        print(f"✅ Prediction: {predicted_class}, Confidence: {confidence_score:.4f}")
        return {"prediction": predicted_class, "confidence": confidence_score}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ✅ Run API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
