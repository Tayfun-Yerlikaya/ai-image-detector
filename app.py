import os
import io
import base64
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter
from PIL.ExifTags import TAGS
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = 'model.onnx'

# ONNX Modelini Yükle
if os.path.exists(MODEL_PATH):
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print("✅ ONNX Model loaded successfully!")
else:
    print(f"❌ WARNING: '{MODEL_PATH}' not found!")

def preprocess_image(image):
    image = image.resize((224, 224), Image.BILINEAR)
    img_data = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

def temperature_scaled_softmax(logits, temperature=4.5):
    scaled_logits = logits / temperature
    e_x = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# 🔍 EXIF METADATA OKUYUCU
def extract_exif(image):
    exif_details = {}
    try:
        exif_data = image._getexif()
        if exif_data:
            for tag, value in exif_data.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name in ['Make', 'Model', 'DateTime', 'ExposureTime', 'ISOSpeedRatings', 'FNumber']:
                    exif_details[tag_name] = str(value)
    except Exception:
        pass
    return exif_details

# 🔴 ISI HARİTASI (HEATMAP) ÜRETECİ (Occlusion Saliency Map)
def generate_heatmap(image, original_prob, pred_idx):
    base_img = image.resize((224, 224), Image.BILINEAR)
    grid_size = 8
    patch_size = 224 // grid_size
    heatmap_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    # Görseli 8x8 bölgelere ayırıp modelin duyarlılığını test ediyoruz
    for i in range(grid_size):
        for j in range(grid_size):
            masked_img = base_img.copy()
            # İlgili bölgeyi blurluyoruz
            box = (j * patch_size, i * patch_size, (j + 1) * patch_size, (i + 1) * patch_size)
            patch = masked_img.crop(box).filter(ImageFilter.GaussianBlur(radius=10))
            masked_img.paste(patch, box)

            # Test
            inp = preprocess_image(masked_img)
            out = session.run(None, {input_name: inp})[0]
            prob = temperature_scaled_softmax(out, temperature=4.5)[0][pred_idx]
            
            # Karar ne kadar düştüyse o bölge o kadar önemlidir
            heatmap_grid[i, j] = max(0, original_prob - prob)

    # Normalizasyon
    if heatmap_grid.max() > 0:
        heatmap_grid = heatmap_grid / heatmap_grid.max()

    # Heatmap görselini büyüt ve renklendir (Kırmızı/Sarı odak noktaları)
    heat_img = Image.fromarray((heatmap_grid * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
    heat_np = np.array(heat_img)

    # RGB Renk Katmanı Oluşturma (Kırmızı Odak)
    overlay_np = np.array(base_img).astype(np.float32)
    overlay_np[:, :, 0] = np.clip(overlay_np[:, :, 0] + heat_np * 1.2, 0, 255) # Kırmızı kanalı vurgula

    result_heatmap = Image.fromarray(overlay_np.astype(np.uint8))
    
    buffered = io.BytesIO()
    result_heatmap.save(buffered, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

CLASS_NAMES = {
    0: {'label': 'AI Generated', 'color': '#ef4444'},
    1: {'label': 'Real Image', 'color': '#10b981'},
    'gray': {'label': 'Uncertain / Suspicious (Filter Detected)', 'color': '#f59e0b'}
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

                # Base64 Orijinal Görsel
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG", quality=95)
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                uploaded_image_data = f"data:image/jpeg;base64,{img_str}"

                # 1. EXIF Analizi
                exif_data = extract_exif(image)

                # 2. ONNX Tahmini
                input_data = preprocess_image(image)
                raw_outputs = session.run(None, {input_name: input_data})[0]
                probabilities = temperature_scaled_softmax(raw_outputs, temperature=4.5)[0]

                pred_idx = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_idx])
                conf_score = round(confidence * 100, 2)

                # 3. Heatmap Üretimi
                heatmap_data = generate_heatmap(image, probabilities[pred_idx], pred_idx)

                # 4. Eşik Kontrolü
                ai_threshold = 78.0
                real_threshold = 65.0

                is_gray = False
                if pred_idx == 0 and conf_score < ai_threshold:
                    is_gray = True
                elif pred_idx == 1 and conf_score < real_threshold:
                    is_gray = True

                if is_gray:
                    result = {
                        'prediction': CLASS_NAMES['gray']['label'],
                        'confidence': conf_score,
                        'color': CLASS_NAMES['gray']['color'],
                        'image_data': uploaded_image_data,
                        'heatmap_data': heatmap_data,
                        'exif': exif_data,
                        'is_gray': True
                    }
                else:
                    result = {
                        'prediction': CLASS_NAMES[pred_idx]['label'],
                        'confidence': conf_score,
                        'color': CLASS_NAMES[pred_idx]['color'],
                        'image_data': uploaded_image_data,
                        'heatmap_data': heatmap_data,
                        'exif': exif_data,
                        'is_gray': False
                    }

                return render_template('index.html', result=result)

            except Exception as e:
                return render_template('index.html', error=f'An error occurred while processing the image: {str(e)}')

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)