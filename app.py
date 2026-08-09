import os
import io
import base64
import numpy as np
import onnxruntime as ort
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = 'model.onnx'

# ONNX Modelini Yükle (Sadece ~50MB RAM harcar, asla çökmez)
if os.path.exists(MODEL_PATH):
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print("✅ ONNX Model loaded successfully!")
else:
    print(f"❌ WARNING: '{MODEL_PATH}' not found!")

# PyTorch'un torchvision.transforms.Resize() ve Normalize() adımlarını %100 taklit eden fonksiyon
def preprocess_image(image):
    # PyTorch ile aynı Bilinear interpolasyon kullanarak piksel kalitesini koruyoruz
    image = image.resize((224, 224), Image.BILINEAR)
    img_data = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    img_data = img_data.transpose(2, 0, 1) # HWC -> CHW
    img_data = np.expand_dims(img_data, axis=0) # Batch boyutu ekle
    return img_data

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

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
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

                # Görsel kalitesini bozmadan (%95 kalite) önizleme verisi oluşturma
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG", quality=95)
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                uploaded_image_data = f"data:image/jpeg;base64,{img_str}"

                # ONNX Runtime Tahmini
                input_data = preprocess_image(image)
                outputs = session.run(None, {input_name: input_data})[0]
                probabilities = softmax(outputs)[0]

                pred_idx = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_idx])
                conf_score = round(confidence * 100, 2)

                # THRESHOLD / GRAY AREA LOGIC
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