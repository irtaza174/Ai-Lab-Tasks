# Smoke Detection System - Complete Setup Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Data Collection](#data-collection)
4. [Model Training](#model-training)
5. [Running the Application](#running-the-application)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Webcam (for live detection)
- Modern web browser (Chrome, Firefox, Safari)

### Required Software
- Python 3.8+
- pip (Python package manager)
- Virtual environment (recommended)

---

## Installation

### Step 1: Clone or Navigate to Project
```bash
cd smoke_detection
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- Flask (web framework)
- TensorFlow (deep learning)
- OpenCV (image processing)
- NumPy (numerical computing)
- Pillow (image handling)

**Note:** TensorFlow installation may take 5-10 minutes.

---

## Data Collection

### Option 1: Download Existing Datasets (Recommended)

#### Using Kaggle CLI
```bash
# Install Kaggle CLI
pip install kaggle

# Setup API credentials
# 1. Go to https://www.kaggle.com/account
# 2. Click "Create New API Token"
# 3. Save kaggle.json to ~/.kaggle/

# Download dataset
kaggle datasets download -d deepcontractor/smoke-detection-dataset
unzip smoke-detection-dataset.zip -d temp_data/

# Organize into project structure
# Move smoke images to: datasets/smoke/
# Move non-smoke images to: datasets/no_smoke/
```

#### Manual Download
1. Visit: https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset
2. Click "Download" (requires Kaggle account)
3. Extract ZIP file
4. Organize images:
   - Place smoke images in `datasets/smoke/`
   - Place non-smoke images in `datasets/no_smoke/`

### Option 2: Collect Your Own Images

#### Google Images
Search for:
- "smoke detection"
- "building fire smoke"
- "industrial smoke"
- "wildfire smoke"

For non-smoke images:
- "building exterior"
- "indoor scenes"
- "industrial facility"

#### Requirements
- Minimum: 200 images per class (400 total)
- Recommended: 500-1000 images per class
- Formats: JPG, PNG, JPEG
- Resolution: Any (will be resized to 224x224)

### View Dataset Information
```bash
python scripts/download_sample_data.py
```

---

## Data Preparation

### Step 1: Verify Dataset Structure
```
datasets/
├── smoke/          # Images containing smoke
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── no_smoke/       # Images without smoke
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Step 2: Run Data Preparation
```bash
python scripts/prepare_data.py
```

This script will:
- Validate your dataset
- Resize images to 224x224
- Apply data augmentation (rotation, flip, brightness, etc.)
- Create augmented versions (5x increase in dataset size)
- Save processed images in `augmented/` subdirectories

**Expected Output:**
```
SMOKE DETECTION - DATA PREPARATION
==================================================
Dataset Statistics:
  Smoke images: 500
  No-smoke images: 500
  Total: 1000

Processing 'smoke' category...
  ✓ Created 2500 augmented images

Processing 'no_smoke' category...
  ✓ Created 2500 augmented images

Total dataset size: 5000 images
```

---

## Model Training

### Step 1: Start Training
```bash
python scripts/train_model.py
```

### Training Process
The script will:
1. Load MobileNetV2 pre-trained weights
2. Create custom classification head
3. Split data (80% training, 20% validation)
4. Train for 20 epochs (with early stopping)
5. Save best model to `model/smoke_detector.h5`

### Training Configuration
- **Architecture:** MobileNetV2 (transfer learning)
- **Input Size:** 224x224x3
- **Batch Size:** 32
- **Epochs:** 20 (with early stopping)
- **Optimizer:** Adam (lr=0.001)
- **Loss:** Binary crossentropy

### Expected Training Time
- **CPU:** 30-60 minutes
- **GPU:** 5-15 minutes

### Training Output
```
TRAINING MODEL
==================================================
Epoch 1/20
125/125 [==============================] - 45s 360ms/step
  loss: 0.3456 - accuracy: 0.8523 - val_loss: 0.2134 - val_accuracy: 0.9123

...

Epoch 15/20
125/125 [==============================] - 42s 336ms/step
  loss: 0.0823 - accuracy: 0.9712 - val_loss: 0.1245 - val_accuracy: 0.9534

✓ Model saved to: model/smoke_detector.h5
```

### Model Files Created
- `model/smoke_detector.h5` - Trained model
- `model/best_model.h5` - Best model checkpoint
- `model/training_history.json` - Training metrics
- `model/model_info.json` - Model metadata

---

## Running the Application

### Step 1: Start Flask Server
```bash
python app.py
```

**Note:** As per your requirements, do NOT run this command during setup. This is for when you're ready to use the application.

### Step 2: Access Web Interface
Open browser and navigate to:
```
http://localhost:5000
```

### Application Features

#### 1. Image Upload
- Click "Choose File" or drag-and-drop
- Supported formats: JPG, PNG, GIF
- Max size: 16MB
- Click "Detect Smoke" to analyze

#### 2. Live Camera Feed
- Click "Start Camera"
- Allow camera permissions
- Real-time detection every 1 second
- Automatic alerts on smoke detection

#### 3. Settings
- Adjust detection threshold (0.0 - 1.0)
- Default: 0.75 (75% confidence)
- Higher = fewer false positives
- Lower = more sensitive detection

#### 4. Alert System
- Popup modal on smoke detection
- Shows timestamp and confidence
- Alert history tracking
- Sound notification

---

## Testing

### Test Trained Model
```bash
# Test all images in dataset
python scripts/test_model.py

# Test specific image
python scripts/test_model.py path/to/image.jpg

# Test with custom threshold
python scripts/test_model.py path/to/image.jpg 0.8

# Test directory
python scripts/test_model.py datasets/smoke/ 0.75
```

### Expected Test Output
```
SMOKE DETECTION MODEL - TEST SCRIPT
==================================================
✓ Model loaded successfully

Testing images in: datasets/smoke
Total images: 100
Threshold: 0.75
==================================================

🚨 smoke_001.jpg | SMOKE      | Confidence:  92.3%
🚨 smoke_002.jpg | SMOKE      | Confidence:  87.5%
✅ smoke_003.jpg | NO SMOKE   | Confidence:  68.2%
...

SUMMARY
==================================================
Total images tested: 100
Smoke detected: 94
No smoke: 6
Average confidence: 85.7%
```

### Performance Metrics
Good model should achieve:
- **Accuracy:** > 90%
- **Precision:** > 85%
- **Recall:** > 85%
- **F1 Score:** > 85%

---

## Troubleshooting

### Issue: Model Not Loading
**Error:** `Model not found at: model/smoke_detector.h5`

**Solution:**
```bash
# Train the model first
python scripts/train_model.py
```

### Issue: Dataset Empty
**Error:** `Dataset is empty! Please run prepare_data.py first.`

**Solution:**
1. Download dataset (see Data Collection section)
2. Place images in `datasets/smoke/` and `datasets/no_smoke/`
3. Run `python scripts/prepare_data.py`

### Issue: Camera Not Working
**Error:** Camera access denied

**Solution:**
- Check browser permissions (allow camera access)
- Use HTTPS or localhost
- Try different browser (Chrome recommended)
- Check if camera is being used by another application

### Issue: Low Accuracy
**Problem:** Model accuracy < 80%

**Solution:**
1. Collect more training data (aim for 1000+ images per class)
2. Ensure dataset quality (remove mislabeled images)
3. Increase training epochs
4. Adjust learning rate
5. Try different augmentation strategies

### Issue: Slow Inference
**Problem:** Predictions take too long

**Solution:**
1. Use GPU if available
2. Reduce image size (already optimized at 224x224)
3. Use model quantization:
```python
# In train_model.py, after saving model:
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### Issue: Too Many False Positives
**Problem:** Detecting smoke when there isn't any

**Solution:**
1. Increase confidence threshold (Settings panel)
2. Add more diverse non-smoke images to training data
3. Retrain model with balanced dataset

### Issue: Missing Smoke Detection
**Problem:** Not detecting actual smoke

**Solution:**
1. Decrease confidence threshold
2. Add more varied smoke images to training data
3. Check if smoke is clearly visible in test images

### Issue: TensorFlow Installation Failed
**Error:** Failed to install TensorFlow

**Solution:**
```bash
# Try specific version
pip install tensorflow==2.15.0

# Or use CPU-only version
pip install tensorflow-cpu==2.15.0

# On Mac M1/M2
pip install tensorflow-macos==2.15.0
pip install tensorflow-metal==1.1.0
```

### Issue: Out of Memory
**Error:** OOM when training

**Solution:**
1. Reduce batch size in `train_model.py`:
```python
BATCH_SIZE = 16  # Instead of 32
```
2. Close other applications
3. Use smaller dataset
4. Enable memory growth:
```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

---

## Advanced Configuration

### Adjust Model Architecture
Edit `scripts/train_model.py`:

```python
# Change input size
IMG_SIZE = (128, 128)  # Smaller = faster, less accurate

# Change base model
from tensorflow.keras.applications import ResNet50
base_model = ResNet50(...)  # More accurate, slower

# Adjust training parameters
EPOCHS = 30  # More epochs
BATCH_SIZE = 16  # Smaller batches
```

### Custom Confidence Threshold
Edit `app.py`:

```python
CONFIDENCE_THRESHOLD = 0.85  # Higher = fewer false positives
```

### Enable Model Quantization
For faster inference on mobile/edge devices:

```python
# Add to train_model.py after saving model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model/smoke_detector.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

## Production Deployment

### Security Considerations
1. Enable HTTPS
2. Add authentication
3. Rate limiting
4. Input validation
5. Secure file uploads

### Performance Optimization
1. Use production WSGI server (Gunicorn, uWSGI)
2. Enable caching
3. Use CDN for static files
4. Implement load balancing
5. Monitor system resources

### Example Production Setup
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## Support

For issues or questions:
1. Check this guide
2. Review README.md
3. Check training logs in `model/training_history.json`
4. Test model with `scripts/test_model.py`

---

## Summary Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset collected (500+ images per class)
- [ ] Data prepared (`python scripts/prepare_data.py`)
- [ ] Model trained (`python scripts/train_model.py`)
- [ ] Model tested (`python scripts/test_model.py`)
- [ ] Application ready to run (`python app.py`)

**Congratulations! Your smoke detection system is ready to use! 🎉**
