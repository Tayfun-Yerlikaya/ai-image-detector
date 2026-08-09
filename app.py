import os
import io
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)

# 1. Device Selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. Load ResNet-18 Model
def load_model(model_path):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

MODEL_PATH = 'cifake_resnet18_model.pth'

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
else:
    print(f"❌ WARNING: '{MODEL_PATH}' not found!")

# 3. Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# 4. Class Labels (English)
CLASS_NAMES = {
    0: {'label': 'AI Generated', 'color': '#ef4444'},
    1: {'label': 'Real Image', 'color': '#10b981'},
    'gray': {'label': 'Uncertain / Suspicious', 'color': '#f59e0b'}
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='Please select an image file.')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No file selected.')

        if file:
            try:
                # Read Image
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                # Convert Image to Base64 for HTML Preview
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                uploaded_image_data = f"data:image/jpeg;base64,{img_str}"

                # Model Prediction
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                model.eval()
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    
                    confidence, predicted_class = torch.max(probabilities, 1)
                    pred_idx = predicted_class.item()
                    conf_score = round(confidence.item() * 100, 2)

                # 🎯 THRESHOLD / GRAY AREA LOGIC:
                if conf_score < 60.0:
                    result = {
                        'prediction': CLASS_NAMES['gray']['label'],
                        'confidence': conf_score,
                        'color': CLASS_NAMES['gray']['color'],
                        'image_data': uploaded_image_data,
                        'is_gray': True
                    }
                else:
                    result = {
                        'prediction': CLASS_NAMES[pred_idx]['label'],
                        'confidence': conf_score,
                        'color': CLASS_NAMES[pred_idx]['color'],
                        'image_data': uploaded_image_data,
                        'is_gray': False
                    }

                return render_template('index.html', result=result)

            except Exception as e:
                return render_template('index.html', error=f'An error occurred while processing the image: {str(e)}')

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)