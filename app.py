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
        print(f"✅ Model loaded successfully. Expected input shape: {model.input_shape}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
else:
    print("❌ Model file not found. Ensure model_weights.h5 is uploaded.")

# ✅ Correct Label Order
LABELS = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"]

# ✅ Preprocess function (Updated: Handles grayscale & RGB cases)
def preprocess_image(image: Image.Image):
    try:
        # Resize image to (150, 150)
        image = image.resize((150, 150))

        # Convert to grayscale if model expects 1 channel
        if model.input_shape[-1] == 1:
            image = image.convert("L")  # Convert to grayscale
        else:
            image = image.convert("RGB")  # Ensure RGB if needed
        
        # Convert to numpy array
        image = np.array(image)

        # Normalize pixel values (0-1 scale)
        image = image / 255.0

        # Expand dimensions to match model input
        if model.input_shape[-1] == 1:
            image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        print(f"✅ Preprocessed Image Shape: {image.shape}")  # Should match model.input_shape

        return image.astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image preprocessing error: {str(e)}")

# ✅ Root endpoint
@app.get("/")
async def home():
    return {
        "message": "Welcome to the Pneumonia Classification API!",
        "usage": "Use /predict to classify images",
        "docs": "/docs for API documentation"
    }

# ✅ `/predict` endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(None), url: str = Form(None)):
    """
    Predict the class of an uploaded image or an image from a URL.
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

    # ✅ Preprocess image
    processed_image = preprocess_image(image)

    # ✅ Make Prediction
    try:
        prediction = model.predict(processed_image)

        # ✅ Debugging: Print raw predictions
        print(f"🔍 Raw Model Output: {prediction}")

        # ✅ Ensure correct output shape
        if prediction.shape[-1] == len(LABELS):
            probabilities = prediction[0]  # Softmax probabilities
        else:
            raise HTTPException(status_code=500, detail="❌ Model output shape mismatch.")

        # ✅ Get the predicted class index
        predicted_index = np.argmax(probabilities)

        # ✅ Get the corresponding class label
        predicted_class = LABELS[predicted_index]

        # ✅ Get confidence score
        confidence_score = probabilities[predicted_index]

        print(f"✅ Prediction: {predicted_class} (Confidence: {confidence_score:.2f})")
        return {
            "prediction": predicted_class,
            "confidence": confidence_score,
            "raw_probabilities": probabilities.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ✅ Run FastAPI on correct host/port
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use Render's assigned port
    uvicorn.run(app, host="0.0.0.0", port=port)
