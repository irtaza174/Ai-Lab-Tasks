# AI-Based Smoke Detection System

A real-time smoke detection system using deep learning and Flask web application.

## Features
- Binary classification (smoke/no smoke)
- Real-time camera feed processing
- Static image upload support
- Popup alerts with timestamps
- Transfer learning with MobileNetV2
- Optimized for real-time inference

## Project Structure
```
smoke_detection/
├── model/                  # Trained models and weights
├── datasets/              # Training data
│   ├── smoke/            # Smoke images
│   └── no_smoke/         # Non-smoke images
├── scripts/              # Data preparation and training scripts
├── static/               # Static web assets
│   ├── css/
│   ├── js/
│   └── uploads/          # Uploaded images
├── templates/            # HTML templates
├── app.py               # Flask application
└── requirements.txt     # Python dependencies
```

## Installation

1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Collect smoke and non-smoke images
2. Place images in respective folders:
   - `datasets/smoke/` - Images containing smoke
   - `datasets/no_smoke/` - Images without smoke

3. Run data preparation script:
```bash
python scripts/prepare_data.py
```

## Model Training

Train the smoke detection model:
```bash
python scripts/train_model.py
```

This will:
- Load and preprocess images
- Apply data augmentation
- Train MobileNetV2 with transfer learning
- Save the trained model to `model/smoke_detector.h5`

## Running the Application

Start the Flask server:
```bash
python app.py
```

Access the application at: `http://localhost:5000`

## Usage

### Image Upload
1. Navigate to the home page
2. Click "Choose File" and select an image
3. Click "Detect Smoke"
4. View results and alerts

### Live Camera Feed
1. Click "Start Camera" button
2. Allow camera permissions
3. System will analyze frames in real-time
4. Popup alerts appear when smoke is detected

## Model Details

- **Architecture**: MobileNetV2 (transfer learning)
- **Input Size**: 224x224 pixels
- **Output**: Binary classification (smoke/no smoke)
- **Confidence Threshold**: 0.75 (adjustable)

## Alert System

When smoke is detected:
- Popup notification appears on screen
- Timestamp of detection
- Confidence score displayed
- Alert sound (optional)

## Performance Optimization

- Model quantization for faster inference
- Frame skipping for camera feeds
- Confidence threshold tuning
- Batch processing support

## Troubleshooting

**Camera not working:**
- Check browser permissions
- Ensure HTTPS or localhost
- Try different browser

**Low accuracy:**
- Add more training data
- Adjust confidence threshold
- Retrain with data augmentation

**Slow inference:**
- Reduce input image size
- Use model quantization
- Skip frames in video processing

## License
MIT License
