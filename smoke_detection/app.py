"""
Flask Application for Smoke Detection System
Provides web interface for image upload and real-time camera feed processing
"""

from flask import Flask, render_template, request, jsonify, Response
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
import json
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Global variables
model = None
MODEL_PATH = 'model/smoke_detector.h5'
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.75

def load_model():
    """
    Load the trained smoke detection model
    """
    global model
    
    if os.path.exists(MODEL_PATH):
        try:
            model = keras.models.load_model(MODEL_PATH)
            print(f"✓ Model loaded successfully from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print(f"❌ Model not found at {MODEL_PATH}")
        print("Please train the model first using: python scripts/train_model.py")
        return False

def allowed_file(filename):
    """
    Check if file extension is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image):
    """
    Preprocess image for model prediction
    """
    # Resize to model input size
    img = cv2.resize(image, IMG_SIZE)
    
    # Convert BGR to RGB (OpenCV uses BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_smoke(image):
    """
    Predict if image contains smoke
    Returns: (is_smoke, confidence, timestamp)
    """
    if model is None:
        return False, 0.0, None
    
    # Preprocess image
    processed_img = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_img, verbose=0)[0][0]
    
    # Determine if smoke is detected
    is_smoke = prediction >= CONFIDENCE_THRESHOLD
    confidence = float(prediction)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return is_smoke, confidence, timestamp

@app.route('/')
def index():
    """
    Home page
    """
    model_loaded = model is not None
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Handle image upload and smoke detection
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        file.save(filepath)
        
        # Read image
        image = cv2.imread(filepath)
        
        if image is None:
            return jsonify({'error': 'Failed to read image'}), 400
        
        # Predict smoke
        is_smoke, confidence, detection_time = predict_smoke(image)
        
        # Prepare response
        response = {
            'success': True,
            'smoke_detected': is_smoke,
            'confidence': confidence,
            'timestamp': detection_time,
            'image_url': f'/static/uploads/{filename}',
            'threshold': CONFIDENCE_THRESHOLD
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """
    Handle real-time camera frame prediction
    """
    try:
        # Get image data from request
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image data'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Predict smoke
        is_smoke, confidence, detection_time = predict_smoke(image)
        
        # Prepare response
        response = {
            'success': True,
            'smoke_detected': is_smoke,
            'confidence': confidence,
            'timestamp': detection_time,
            'threshold': CONFIDENCE_THRESHOLD
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

@app.route('/model_info')
def model_info():
    """
    Get model information
    """
    info_path = 'model/model_info.json'
    
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        return jsonify(info)
    else:
        return jsonify({
            'error': 'Model info not found',
            'model_loaded': model is not None
        })

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    """
    Update confidence threshold
    """
    global CONFIDENCE_THRESHOLD
    
    data = request.get_json()
    
    if 'threshold' not in data:
        return jsonify({'error': 'No threshold value provided'}), 400
    
    try:
        new_threshold = float(data['threshold'])
        
        if not 0.0 <= new_threshold <= 1.0:
            return jsonify({'error': 'Threshold must be between 0.0 and 1.0'}), 400
        
        CONFIDENCE_THRESHOLD = new_threshold
        
        return jsonify({
            'success': True,
            'threshold': CONFIDENCE_THRESHOLD
        })
    
    except ValueError:
        return jsonify({'error': 'Invalid threshold value'}), 400

@app.route('/health')
def health():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'threshold': CONFIDENCE_THRESHOLD
    })

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("SMOKE DETECTION SYSTEM - FLASK APPLICATION")
    print("=" * 60)
    
    # Load model
    model_loaded = load_model()
    
    if not model_loaded:
        print("\n⚠ WARNING: Model not loaded!")
        print("The application will start, but predictions will not work.")
        print("Please train the model first: python scripts/train_model.py")
    
    print("\n" + "=" * 60)
    print("Starting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("=" * 60)
    print("\nPress CTRL+C to stop the server\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
