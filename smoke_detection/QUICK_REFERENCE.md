# Quick Reference Guide

## 🚀 Quick Start Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Data Preparation
python scripts/download_sample_data.py  # View dataset sources
python scripts/prepare_data.py          # Prepare and augment data

# Training
python scripts/train_model.py           # Train model (30-60 min)

# Testing
python scripts/test_model.py            # Test on all images
python scripts/test_model.py image.jpg  # Test single image

# Run Application (DO NOT RUN during setup as per requirements)
python app.py                           # Start Flask server
# Access: http://localhost:5000
```

## 📁 Project Structure

```
smoke_detection/
├── app.py                    # Main Flask app
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
├── README.md                 # Overview
├── SETUP_GUIDE.md           # Detailed setup
├── PROJECT_SUMMARY.md       # Complete summary
├── API_DOCUMENTATION.md     # API reference
├── ARCHITECTURE.md          # System architecture
├── quick_start.sh           # Automated setup
│
├── model/                   # Trained models
│   ├── smoke_detector.h5   # Main model
│   └── model_info.json     # Metadata
│
├── datasets/               # Training data
│   ├── smoke/             # Smoke images
│   └── no_smoke/          # Non-smoke images
│
├── scripts/               # Utilities
│   ├── prepare_data.py   # Data prep
│   ├── train_model.py    # Training
│   ├── test_model.py     # Testing
│   └── download_sample_data.py
│
├── static/               # Web assets
│   ├── css/style.css    # Styles
│   ├── js/main.js       # Frontend
│   └── uploads/         # Uploads
│
└── templates/           # HTML
    └── index.html      # Main page
```

## 🔧 Configuration

### Model Settings (config.py)
```python
MODEL_CONFIG = {
    'input_size': (224, 224),
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 0.001,
}
```

### Detection Settings (app.py)
```python
CONFIDENCE_THRESHOLD = 0.75  # 75% confidence
IMG_SIZE = (224, 224)        # Input size
```

## 📊 Key Metrics

| Metric | Target |
|--------|--------|
| Accuracy | >90% |
| Precision | >85% |
| Recall | >85% |
| Inference Time | <100ms |

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main interface |
| `/upload` | POST | Upload image |
| `/predict_frame` | POST | Predict frame |
| `/model_info` | GET | Model info |
| `/set_threshold` | POST | Set threshold |
| `/health` | GET | Health check |

## 🎯 Common Tasks

### Change Threshold
```python
# In app.py
CONFIDENCE_THRESHOLD = 0.85  # Higher = fewer false positives
```

### Adjust Training
```python
# In scripts/train_model.py
EPOCHS = 30          # More training
BATCH_SIZE = 16      # Smaller batches
```

### Change Model
```python
# In scripts/train_model.py
from tensorflow.keras.applications import ResNet50
base_model = ResNet50(...)  # Different architecture
```

## 🐛 Troubleshooting

### Model Not Found
```bash
python scripts/train_model.py
```

### Dataset Empty
```bash
# 1. Download dataset
# 2. Place in datasets/smoke/ and datasets/no_smoke/
python scripts/prepare_data.py
```

### Camera Not Working
- Check browser permissions
- Use HTTPS or localhost
- Try different browser

### Low Accuracy
- Collect more data (1000+ per class)
- Increase training epochs
- Check data quality

### Slow Inference
- Use GPU
- Reduce image size
- Optimize model

## 📦 Dependencies

```
flask==3.0.0              # Web framework
tensorflow==2.15.0        # ML library
opencv-python==4.8.1.78   # Image processing
numpy==1.24.3             # Numerical computing
pillow==10.1.0            # Image handling
werkzeug==3.0.1           # WSGI utilities
```

## 🔐 Security Checklist

- [ ] Enable HTTPS
- [ ] Add authentication
- [ ] Implement rate limiting
- [ ] Validate all inputs
- [ ] Sanitize filenames
- [ ] Use secure headers
- [ ] Enable CSRF protection
- [ ] Log security events

## 📈 Performance Tips

1. **Use GPU** for faster training/inference
2. **Batch requests** for multiple images
3. **Cache predictions** for repeated images
4. **Compress images** before upload
5. **Use CDN** for static files
6. **Enable gzip** compression
7. **Implement caching** (Redis)
8. **Use load balancer** for scaling

## 🧪 Testing

### Test Model
```bash
# All images
python scripts/test_model.py

# Single image
python scripts/test_model.py path/to/image.jpg

# Custom threshold
python scripts/test_model.py image.jpg 0.8

# Directory
python scripts/test_model.py datasets/smoke/
```

### Test API
```bash
# Upload image
curl -X POST -F "file=@image.jpg" http://localhost:5000/upload

# Get model info
curl http://localhost:5000/model_info

# Health check
curl http://localhost:5000/health
```

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| README.md | Quick overview |
| SETUP_GUIDE.md | Detailed setup |
| PROJECT_SUMMARY.md | Complete summary |
| API_DOCUMENTATION.md | API reference |
| ARCHITECTURE.md | System design |
| QUICK_REFERENCE.md | This file |

## 🎓 Dataset Sources

1. **Kaggle - Smoke Detection Dataset**
   - https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset
   - ~2000 images

2. **Kaggle - Fire and Smoke Dataset**
   - https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset
   - ~1000 images

3. **Roboflow - Smoke Detection**
   - https://universe.roboflow.com/search?q=smoke%20detection
   - Various datasets

## 🔄 Workflow

```
1. Collect Data → 2. Prepare Data → 3. Train Model
                                          ↓
6. Deploy ← 5. Test Model ← 4. Evaluate Model
```

## 💡 Tips

- **Start small**: Test with 100 images per class
- **Augment data**: Increases dataset 5x
- **Monitor training**: Watch for overfitting
- **Test thoroughly**: Use diverse test images
- **Adjust threshold**: Balance false positives/negatives
- **Document changes**: Keep track of experiments
- **Version models**: Save different versions
- **Backup data**: Don't lose training data

## 🚨 Important Notes

⚠️ **Do NOT run `python app.py` during setup** (as per requirements)

✅ **DO run these during setup:**
- `pip install -r requirements.txt`
- `python scripts/prepare_data.py`
- `python scripts/train_model.py`
- `python scripts/test_model.py`

## 📞 Getting Help

1. Check documentation files
2. Review error messages
3. Test with sample images
4. Verify model is trained
5. Check system requirements

## ✅ Setup Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Dataset collected (500+ per class)
- [ ] Data prepared and augmented
- [ ] Model trained successfully
- [ ] Model tested and validated
- [ ] Application ready to run

## 🎉 Success Indicators

✓ Model accuracy >90%  
✓ Inference time <100ms  
✓ No errors in logs  
✓ Web interface loads  
✓ Predictions working  
✓ Alerts triggering  

---

**Ready to detect smoke! 🔥🚒**

*For detailed information, see other documentation files.*
