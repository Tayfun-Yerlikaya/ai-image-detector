import os
import io
import base64
import traceback
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter, ImageOps
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
                    exif_details[str(tag_name)] = str(value)
    except Exception as e:
        print(f"EXIF Read Error: {e}")
    return exif_details

# 🔴 ISI HARİTASI (HEATMAP) ÜRETECİ
def generate_heatmap(image, original_prob, pred_idx):
    try:
        base_img = image.resize((224, 224), Image.BILINEAR)
        grid_size = 6
        patch_size = 224 // grid_size
        heatmap_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

        for i in range(grid_size):
            for j in range(grid_size):
                masked_img = base_img.copy()
                box = (j * patch_size, i * patch_size, (j + 1) * patch_size, (i + 1) * patch_size)
                patch = masked_img.crop(box).filter(ImageFilter.GaussianBlur(radius=8))
                masked_img.paste(patch, box)

                inp = preprocess_image(masked_img)
                out = session.run(None, {input_name: inp})[0]
                prob = temperature_scaled_softmax(out, temperature=4.5)[0][pred_idx]
                
                heatmap_grid[i, j] = max(0, original_prob - prob)

        if heatmap_grid.max() > 0:
            heatmap_grid = heatmap_grid / heatmap_grid.max()

        heat_img = Image.fromarray((heatmap_grid * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        heat_np = np.array(heat_img)

        overlay_np = np.array(base_img).astype(np.float32)
        overlay_np[:, :, 0] = np.clip(overlay_np[:, :, 0] + heat_np * 1.5, 0, 255)

        result_heatmap = Image.fromarray(overlay_np.astype(np.uint8))
        
        buffered = io.BytesIO()
        result_heatmap.save(buffered, format="JPEG", quality=90)
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    except Exception as e:
        print(f"Heatmap Error: {e}")
        return None

# 🌊 FREKANS SPEKTRUMU (FFT / Noise Profile) ÜRETECİ
def generate_fft_spectrum(image):
    try:
        # Gri tonlamaya çevirip 224x224 boyutlandırıyoruz
        gray_img = ImageOps.grayscale(image.resize((224, 224), Image.BILINEAR))
        img_np = np.array(gray_img, dtype=np.float32)

        # 2D Fast Fourier Transform (2D FFT)
        f_transform = np.fft.fft2(img_np)
        f_shift = np.fft.fftshift(f_transform)
        
        # Logaritmik Genlik Spektrumu (Magnitude Spectrum)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-5)
        
        # Normalizasyon (0 - 255)
        mag_min, mag_max = magnitude_spectrum.min(), magnitude_spectrum.max()
        if mag_max > mag_min:
            norm_spectrum = (magnitude_spectrum - mag_min) / (mag_max - mag_min) * 255.0
        else:
            norm_spectrum = np.zeros_like(magnitude_spectrum)

        norm_np = norm_spectrum.astype(np.uint8)

        # Renklendirme (Purple / Cyan Nebula efekti)
        colored_spectrum = np.zeros((224, 224, 3), dtype=np.uint8)
        colored_spectrum[:, :, 0] = np.clip(norm_np * 0.8, 0, 255)  # Red
        colored_spectrum[:, :, 1] = np.clip(norm_np * 0.5 + 20, 0, 255)  # Green
        colored_spectrum[:, :, 2] = np.clip(norm_np * 1.2, 0, 255)  # Blue

        result_fft = Image.fromarray(colored_spectrum)
        
        buffered = io.BytesIO()
        result_fft.save(buffered, format="JPEG", quality=90)
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    except Exception as e:
        print(f"FFT Error: {e}")
        return None

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

                # 3. Heatmap ve FFT Üretimi
                heatmap_data = generate_heatmap(image, probabilities[pred_idx], pred_idx)
                fft_data = generate_fft_spectrum(image)

                # 4. Eşik Kontrolü
                ai_threshold = 78.0
                real_threshold = 65.0

                is_gray = False
                if pred_idx == 0 and conf_score < ai_threshold:
                    is_gray = True
                elif pred_idx == 1 and conf_score < real_threshold:
                    is_gray = True

                result = {
                    'prediction': CLASS_NAMES['gray']['label'] if is_gray else CLASS_NAMES[pred_idx]['label'],
                    'confidence': conf_score,
                    'color': CLASS_NAMES['gray']['color'] if is_gray else CLASS_NAMES[pred_idx]['color'],
                    'image_data': uploaded_image_data,
                    'heatmap_data': heatmap_data,
                    'fft_data': fft_data,
                    'exif': exif_data,
                    'is_gray': is_gray
                }

                return render_template('index.html', result=result)

            except Exception as e:
                error_msg = f"Processing Error: {str(e)}"
                print(traceback.format_exc())
                return render_template('index.html', error=error_msg)

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)