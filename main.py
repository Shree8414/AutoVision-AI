"""
CodeOrbit AI Internship - Task 3: Enterprise AI Image Classification & Vision Intelligence
Features:
- MobileNetV2 Deep Learning Inferences (ONNX Runtime)
- Instant Client-side Image Preview
- Worldwide Showcase Gallery (Cars, Wildlife, Food, Travel) with 1-Click Classification
- Interactive 'Start Game' (Human vs AI Vision Challenge)
- Top-5 Probabilities Breakdown + Latency Benchmarks
"""

import os
import time
import json
import random
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '').strip()

gemini_client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("Google Gemini Vision Client initialized successfully!")
    except Exception as e:
        print(f"Warning initializing Gemini Client: {e}")
else:
    print("Notice: GEMINI_API_KEY not set yet in .env - running with local MobileNetV2.")

# Initialize Flask application
app = Flask(__name__)

# Base directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
SAMPLES_FOLDER = os.path.join(BASE_DIR, 'static', 'samples')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLES_FOLDER, exist_ok=True)

# 1. Load Pretrained MobileNetV2 Model
MODEL_PATH = os.path.join(BASE_DIR, 'mobilenetv2.onnx')
LABELS_PATH = os.path.join(BASE_DIR, 'imagenet_classes.txt')

print("Initializing MobileNetV2 Deep Learning Session...")
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# 2. Load 1,000 ImageNet Labels
with open(LABELS_PATH, 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]
print(f"Loaded {len(class_labels)} ImageNet categories.")

# ImageNet normalization parameters
MEAN = np.array([0.485, 0.456, 0.406], dtype='float32')
STD = np.array([0.229, 0.224, 0.225], dtype='float32')

# Curated Worldwide Gallery Items
WORLD_GALLERY = [
    {
        "id": "ferrari",
        "title": "Ferrari F8 Tributo",
        "category": "supercar",
        "category_name": "🏎️ Supercars",
        "file": "ferrari.jpg",
        "expected": "Sports Car",
        "description": "Italian mid-engine supercar with iconic aerodynamics."
    },
    {
        "id": "porsche",
        "title": "Porsche 911 GT3",
        "category": "supercar",
        "category_name": "🏎️ Supercars",
        "file": "porsche.jpg",
        "expected": "Sports Car",
        "description": "Precision German motorsport engineering and agility."
    },
    {
        "id": "lamborghini",
        "title": "Lamborghini Huracán",
        "category": "supercar",
        "category_name": "🏎️ Supercars",
        "file": "lamborghini.jpg",
        "expected": "Racer / Supercar",
        "description": "V10 Italian track-bred sports car with sharp lines."
    },
    {
        "id": "lion",
        "title": "African Lion",
        "category": "wildlife",
        "category_name": "🦁 Wildlife",
        "file": "lion.jpg",
        "expected": "Lion",
        "description": "Apex predator inhabiting the sub-Saharan grasslands."
    },
    {
        "id": "dog",
        "title": "Golden Retriever",
        "category": "wildlife",
        "category_name": "🦁 Wildlife",
        "file": "golden_retriever.jpg",
        "expected": "Golden Retriever",
        "description": "Friendly, intelligent, and loyal companion dog."
    },
    {
        "id": "fox",
        "title": "Siberian Red Fox",
        "category": "wildlife",
        "category_name": "🦁 Wildlife",
        "file": "red_fox.jpg",
        "expected": "Red Fox",
        "description": "Remarkable wild canine known for intelligence and reddish coat."
    },
    {
        "id": "macaw",
        "title": "Rainbow Lorikeet",
        "category": "wildlife",
        "category_name": "🦁 Wildlife",
        "file": "macaw.jpg",
        "expected": "Lorikeet / Parrot",
        "description": "Vibrant plumage native to rainforest canopies."
    },
    {
        "id": "airliner",
        "title": "Commercial Jetliner",
        "category": "aviation",
        "category_name": "✈️ Aviation & Places",
        "file": "airliner.jpg",
        "expected": "Airliner",
        "description": "Long-range twin-engine passenger aircraft cruising at high altitude."
    },
    {
        "id": "lighthouse",
        "title": "Coastal Lighthouse",
        "category": "aviation",
        "category_name": "✈️ Aviation & Places",
        "file": "lighthouse.jpg",
        "expected": "Seashore / Lighthouse",
        "description": "Maritime beacon perched along picturesque rugged coastlines."
    },
    {
        "id": "guitar",
        "title": "Acoustic Guitar",
        "category": "objects",
        "category_name": "🍕 Gourmet & Objects",
        "file": "guitar.jpg",
        "expected": "Acoustic Guitar",
        "description": "Handcrafted string instrument resonant with warm acoustic tones."
    },
    {
        "id": "pizza",
        "title": "Wood-Fired Pizza",
        "category": "objects",
        "category_name": "🍕 Gourmet & Objects",
        "file": "pizza.jpg",
        "expected": "Pizza",
        "description": "Artisanal crust topped with San Marzano tomatoes and mozzarella."
    },
    {
        "id": "coffee",
        "title": "Artisan Espresso",
        "category": "objects",
        "category_name": "🍕 Gourmet & Objects",
        "file": "coffee.jpg",
        "expected": "Espresso / Coffee",
        "description": "Rich roasted coffee beans brewed with thick velvety crema."
    }
]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    """Preprocess image for MobileNetV2 input dimensions (1, 3, 224, 224)."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    
    img_data = np.array(img).astype('float32') / 255.0
    img_data = (img_data - MEAN) / STD
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0)
    return img_data


def classify_with_gemini(image_path):
    """Classify image using Google Gemini Multimodal Vision API."""
    start_time = time.perf_counter()
    pil_img = Image.open(image_path).convert('RGB')

    prompt = (
        "You are an expert Computer Vision Classifier. Analyze this image with precision.\n"
        "Identify the primary subject, object, animal, vehicle, or food item.\n"
        "Return ONLY a pure JSON object (without markdown fences, or with standard json block) matching this schema:\n"
        "{\n"
        '  "top_label": "Precise primary object name capitalized",\n'
        '  "top_confidence": 98.5,\n'
        '  "confidence_grade": "High Confidence",\n'
        '  "ai_insight": "One brief sentence explaining the distinct visual cues and features observed in the image.",\n'
        '  "all_predictions": [\n'
        '    {"label": "Top precise identification", "confidence": 98.5},\n'
        '    {"label": "General category or related candidate", "confidence": 92.0},\n'
        '    {"label": "Related candidate 3", "confidence": 84.0},\n'
        '    {"label": "Related candidate 4", "confidence": 75.0},\n'
        '    {"label": "Related candidate 5", "confidence": 62.0}\n'
        '  ]\n'
        "}"
    )

    response = gemini_client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents=[prompt, pil_img]
    )

    text = response.text.strip()
    if '```json' in text:
        text = text.split('```json')[1].split('```')[0].strip()
    elif '```' in text:
        text = text.split('```')[1].split('```')[0].strip()

    data = json.loads(text)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

    top_conf = float(data.get('top_confidence', 96.0))
    if top_conf >= 75:
        badge_class = "badge-high"
    elif top_conf >= 45:
        badge_class = "badge-mid"
    else:
        badge_class = "badge-low"

    return {
        'top_label': str(data.get('top_label', 'Identified Object')).title(),
        'top_confidence': round(top_conf, 1),
        'confidence_grade': data.get('confidence_grade', 'High Confidence'),
        'confidence_badge_class': badge_class,
        'latency_ms': latency_ms,
        'ai_insight': data.get('ai_insight', 'Multimodal neural perception analysis.'),
        'ai_engine': 'Google Gemini Vision AI',
        'all_predictions': data.get('all_predictions', [])
    }


def classify_with_mobilenet(image_path):
    """Run local MobileNetV2 model inference and return top 5 ranked probabilities."""
    start_time = time.perf_counter()
    input_tensor = preprocess_image(image_path)

    raw_outputs = session.run(None, {input_name: input_tensor})[0][0]

    # Softmax probabilities
    exp_scores = np.exp(raw_outputs - np.max(raw_outputs))
    probabilities = exp_scores / np.sum(exp_scores)

    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

    top_indices = np.argsort(probabilities)[::-1][:5]

    predictions = []
    for idx in top_indices:
        label = class_labels[idx]
        score = float(probabilities[idx] * 100)
        predictions.append({
            'label': label.title(),
            'confidence': round(score, 1)
        })

    top_pred = predictions[0]

    if top_pred['confidence'] >= 75:
        confidence_grade = "High Confidence"
        confidence_badge_class = "badge-high"
    elif top_pred['confidence'] >= 45:
        confidence_grade = "Moderate Confidence"
        confidence_badge_class = "badge-mid"
    else:
        confidence_grade = "Low / Ambiguous"
        confidence_badge_class = "badge-low"

    return {
        'top_label': top_pred['label'],
        'top_confidence': top_pred['confidence'],
        'confidence_grade': confidence_grade,
        'confidence_badge_class': confidence_badge_class,
        'latency_ms': latency_ms,
        'ai_insight': 'Pretrained MobileNetV2 convolutional neural feature extraction.',
        'ai_engine': 'MobileNetV2 CNN (Local)',
        'all_predictions': predictions
    }


def classify_image(image_path):
    """Primary classification router: Uses Gemini Vision if API key set, otherwise MobileNetV2."""
    if gemini_client:
        try:
            return classify_with_gemini(image_path)
        except Exception as e:
            print(f"Gemini API inference error (falling back to MobileNetV2): {e}")

    return classify_with_mobilenet(image_path)


@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        gallery=WORLD_GALLERY,
        result=None
    )


@app.route('/predict', methods=['POST'])
def predict():
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or 'json' in request.headers.get('Accept', '')
    sample_file = request.form.get('sample_file')
    
    # Case 1: User selected a sample from the world gallery or quick samples
    if sample_file:
        sample_path = os.path.join(SAMPLES_FOLDER, secure_filename(sample_file))
        if os.path.exists(sample_path):
            result = classify_image(sample_path)
            image_url = f"/static/samples/{sample_file}"
            source_name = sample_file.replace('_', ' ').replace('.jpg', '').title()
            if is_ajax:
                return jsonify({
                    'success': True,
                    'image_url': image_url,
                    'result': result,
                    'source_name': source_name
                })
            return render_template(
                'index.html',
                gallery=WORLD_GALLERY,
                image_url=image_url,
                result=result,
                source_name=source_name
            )

    # Case 2: User uploaded a file directly
    if 'file' not in request.files:
        if is_ajax:
            return jsonify({'success': False, 'error': "No image file provided."}), 400
        return render_template('index.html', gallery=WORLD_GALLERY, error="No image file provided.")

    file = request.files['file']
    if file.filename == '':
        if is_ajax:
            return jsonify({'success': False, 'error': "Please select an image file to classify."}), 400
        return render_template('index.html', gallery=WORLD_GALLERY, error="Please select an image file to classify.")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(saved_path)

        result = classify_image(saved_path)
        image_url = f"/static/uploads/{filename}"

        if is_ajax:
            return jsonify({
                'success': True,
                'image_url': image_url,
                'result': result,
                'source_name': filename
            })

        return render_template(
            'index.html',
            gallery=WORLD_GALLERY,
            image_url=image_url,
            result=result,
            source_name=filename
        )
    else:
        if is_ajax:
            return jsonify({'success': False, 'error': "Invalid file format. Please upload JPG, PNG, or WEBP."}), 400
        return render_template('index.html', gallery=WORLD_GALLERY, error="Invalid file format. Please upload JPG, PNG, or WEBP.")


@app.route('/game-data', methods=['GET'])
def game_data():
    """Provides a fresh round for the AI Vision Game."""
    item = random.choice(WORLD_GALLERY)
    
    # Create 4 multiple choice options: 1 correct + 3 decoys
    all_options = [g['expected'] for g in WORLD_GALLERY if g['expected'] != item['expected']]
    decoys = random.sample(list(set(all_options)), 3)
    choices = decoys + [item['expected']]
    random.shuffle(choices)
    
    return jsonify({
        'image_url': f"/static/samples/{item['file']}",
        'sample_file': item['file'],
        'correct_label': item['expected'],
        'choices': choices
    })


@app.route('/game-verify', methods=['POST'])
def game_verify():
    """Verify player guess and show AI prediction comparison."""
    data = request.get_json() or {}
    sample_file = data.get('sample_file')
    player_guess = data.get('player_guess')
    
    sample_path = os.path.join(SAMPLES_FOLDER, secure_filename(sample_file))
    if not os.path.exists(sample_path):
        return jsonify({'error': 'Image not found'}), 404
        
    ai_result = classify_image(sample_path)
    
    return jsonify({
        'ai_top_label': ai_result['top_label'],
        'ai_confidence': ai_result['top_confidence'],
        'latency_ms': ai_result['latency_ms'],
        'all_predictions': ai_result['all_predictions']
    })


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("AutoVision AI - Deep Learning & Vision Platform Running!")
    print("Live Server: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host='127.0.0.1', port=5000, debug=False)
