# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                        │
│                     (Web Browser)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Image Upload │  │ Camera Feed  │  │   Settings   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    FLASK WEB SERVER                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  API Endpoints                        │  │
│  │  • /upload        • /predict_frame                   │  │
│  │  • /model_info    • /set_threshold                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  IMAGE PROCESSING                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • Resize (224x224)                                  │  │
│  │  • Normalize (0-1)                                   │  │
│  │  • Color conversion (BGR→RGB)                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    ML MODEL (TensorFlow)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              MobileNetV2 Base Model                  │  │
│  │                      ↓                               │  │
│  │           Global Average Pooling                     │  │
│  │                      ↓                               │  │
│  │              Dense Layer (128)                       │  │
│  │                      ↓                               │  │
│  │            Output Layer (Sigmoid)                    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    ALERT SYSTEM                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • Popup Modal                                       │  │
│  │  • Sound Notification                                │  │
│  │  • Alert History                                     │  │
│  │  • Timestamp Logging                                 │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Image Upload Flow
```
User selects image
       ↓
Frontend validates file
       ↓
POST /upload with FormData
       ↓
Flask receives file
       ↓
Save to uploads folder
       ↓
Read image with OpenCV
       ↓
Preprocess image
       ↓
Model prediction
       ↓
Check threshold
       ↓
Return JSON response
       ↓
Frontend displays result
       ↓
Show alert if smoke detected
```

### Camera Stream Flow
```
User clicks "Start Camera"
       ↓
Request camera permissions
       ↓
Start video stream
       ↓
Capture frame every 1 second
       ↓
Convert to base64
       ↓
POST /predict_frame
       ↓
Decode base64 image
       ↓
Preprocess frame
       ↓
Model prediction
       ↓
Return JSON response
       ↓
Update confidence display
       ↓
Show alert if smoke detected
       ↓
Continue loop
```

## Component Interaction

```
┌──────────────┐
│   Browser    │
│  (Frontend)  │
└──────┬───────┘
       │ HTTP/HTTPS
       │
┌──────▼───────┐
│    Flask     │
│  (Backend)   │
└──────┬───────┘
       │
       ├─────────────┐
       │             │
┌──────▼───────┐ ┌──▼──────────┐
│   OpenCV     │ │ TensorFlow  │
│ (Processing) │ │   (Model)   │
└──────────────┘ └─────────────┘
```

## File System Structure

```
smoke_detection/
│
├── app.py                    # Main Flask application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
│
├── model/                    # ML models
│   ├── smoke_detector.h5    # Trained model
│   └── model_info.json      # Model metadata
│
├── datasets/                 # Training data
│   ├── smoke/               # Smoke images
│   └── no_smoke/            # Non-smoke images
│
├── scripts/                  # Utility scripts
│   ├── prepare_data.py      # Data preparation
│   ├── train_model.py       # Model training
│   └── test_model.py        # Model testing
│
├── static/                   # Web assets
│   ├── css/style.css        # Styles
│   ├── js/main.js           # Frontend logic
│   └── uploads/             # Uploaded images
│
└── templates/                # HTML templates
    └── index.html           # Main interface
```

## Technology Stack

### Backend
- **Framework:** Flask 3.0.0
- **ML Library:** TensorFlow 2.15.0
- **Image Processing:** OpenCV 4.8.1
- **Numerical Computing:** NumPy 1.24.3

### Frontend
- **HTML5:** Structure
- **CSS3:** Styling (Gradients, Flexbox, Grid)
- **JavaScript:** Logic (ES6+)
- **APIs:** MediaDevices, Canvas, Fetch

### Model
- **Architecture:** MobileNetV2
- **Framework:** Keras (TensorFlow)
- **Input:** 224×224×3 RGB images
- **Output:** Binary classification

## Deployment Architecture

### Development
```
Local Machine
    ↓
Flask Development Server
    ↓
http://localhost:5000
```

### Production (Recommended)
```
                    ┌──────────────┐
                    │  Load Balancer│
                    │   (Nginx)     │
                    └───────┬───────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
    ┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
    │  Flask App 1 │ │ Flask App 2│ │ Flask App 3│
    │  (Gunicorn)  │ │ (Gunicorn) │ │ (Gunicorn) │
    └──────────────┘ └────────────┘ └────────────┘
            │               │               │
            └───────────────┼───────────────┘
                            │
                    ┌───────▼───────┐
                    │  File Storage │
                    │   (S3/NFS)    │
                    └───────────────┘
```

## Security Layers

```
┌─────────────────────────────────────┐
│         SSL/TLS (HTTPS)             │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│      Authentication (JWT)           │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│      Rate Limiting                  │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│      Input Validation               │
└─────────────────────────────────────┘
┌─────────────────────────────────────┐
│      File Type Checking             │
└─────────────────────────────────────┘
```

## Model Architecture Details

```
Input: (224, 224, 3)
    ↓
Rescaling (1/127.5, offset=-1)
    ↓
MobileNetV2 Base (frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.3)
    ↓
Dense (128, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense (1, Sigmoid)
    ↓
Output: [0.0 - 1.0]
```

## Performance Metrics

### Latency
- Image preprocessing: ~10ms
- Model inference: ~50-100ms
- Total response time: ~100-200ms

### Throughput
- Single instance: ~10-20 requests/second
- With GPU: ~50-100 requests/second

### Resource Usage
- Memory: ~500MB (model loaded)
- CPU: 10-30% (per request)
- GPU: 20-40% (if available)

## Scalability Strategy

### Horizontal Scaling
```
Add more Flask instances
    ↓
Use load balancer
    ↓
Shared model storage
    ↓
Distributed caching
```

### Vertical Scaling
```
Increase CPU/RAM
    ↓
Add GPU
    ↓
Optimize model
    ↓
Use faster storage
```

## Monitoring Points

1. **Application Level:**
   - Request count
   - Response time
   - Error rate
   - Active users

2. **Model Level:**
   - Prediction accuracy
   - Confidence distribution
   - False positive rate
   - False negative rate

3. **System Level:**
   - CPU usage
   - Memory usage
   - Disk I/O
   - Network traffic

## Future Enhancements

### Phase 1 (Short-term)
- Object detection (bounding boxes)
- Multi-camera support
- Database integration
- User authentication

### Phase 2 (Medium-term)
- Real-time streaming (WebRTC)
- Mobile app
- Cloud deployment
- Advanced analytics

### Phase 3 (Long-term)
- Edge deployment
- Federated learning
- IoT integration
- Predictive maintenance
