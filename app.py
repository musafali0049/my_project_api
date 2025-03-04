import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify
import requests
from io import BytesIO

app = Flask(__name__)

# Load the model (adjust model path if needed)
model_path = "model_weights.pth"
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400
        
        # Fetch image from URL
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to fetch image'}), 400
        
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image = transform(image).unsqueeze(0)

        # Perform prediction
        with torch.no_grad():
            output = model(image)

            # Check output shape
            if len(output.shape) == 1:
                prediction = output.item()  # For single-value regression
            elif output.shape[1] == 1:
                prediction = torch.sigmoid(output).item()  # Binary classification
            else:
                prediction = torch.argmax(output, dim=1).item()  # Multi-class classification
        
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
