"""
Test Script for Smoke Detection Model
Tests the trained model with sample images
"""

import os
import cv2
import numpy as np
from tensorflow import keras
from pathlib import Path
import sys

def load_model(model_path='model/smoke_detector.h5'):
    """Load trained model"""
    base_dir = Path(__file__).parent.parent
    full_path = base_dir / model_path
    
    if not full_path.exists():
        print(f"❌ Model not found at: {full_path}")
        print("Please train the model first: python scripts/train_model.py")
        return None
    
    try:
        model = keras.models.load_model(str(full_path))
        print(f"✓ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def preprocess_image(image_path, img_size=(224, 224)):
    """Preprocess image for prediction"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    img = cv2.resize(img, img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    return img

def predict_image(model, image_path, threshold=0.75):
    """Predict smoke in image"""
    img = preprocess_image(image_path)
    if img is None:
        return None, 0.0
    
    prediction = model.predict(img, verbose=0)[0][0]
    is_smoke = prediction >= threshold
    
    return is_smoke, float(prediction)

def test_directory(model, directory, threshold=0.75):
    """Test all images in a directory"""
    base_dir = Path(__file__).parent.parent
    test_dir = base_dir / directory
    
    if not test_dir.exists():
        print(f"❌ Directory not found: {test_dir}")
        return
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(test_dir.glob(ext))
    
    if not image_files:
        print(f"No images found in {test_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Testing images in: {directory}")
    print(f"Total images: {len(image_files)}")
    print(f"Threshold: {threshold}")
    print(f"{'='*60}\n")
    
    results = []
    
    for img_path in image_files:
        is_smoke, confidence = predict_image(model, img_path, threshold)
        
        if is_smoke is not None:
            result_icon = "🚨" if is_smoke else "✅"
            result_text = "SMOKE" if is_smoke else "NO SMOKE"
            
            print(f"{result_icon} {img_path.name:40s} | {result_text:10s} | Confidence: {confidence*100:5.1f}%")
            
            results.append({
                'file': img_path.name,
                'smoke': is_smoke,
                'confidence': confidence
            })
    
    # Summary
    smoke_count = sum(1 for r in results if r['smoke'])
    no_smoke_count = len(results) - smoke_count
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total images tested: {len(results)}")
    print(f"Smoke detected: {smoke_count}")
    print(f"No smoke: {no_smoke_count}")
    
    if results:
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"Average confidence: {avg_confidence*100:.1f}%")

def test_single_image(model, image_path, threshold=0.75):
    """Test a single image"""
    print(f"\n{'='*60}")
    print(f"Testing image: {image_path}")
    print(f"{'='*60}\n")
    
    is_smoke, confidence = predict_image(model, image_path, threshold)
    
    if is_smoke is None:
        print("❌ Failed to process image")
        return
    
    result_icon = "🚨" if is_smoke else "✅"
    result_text = "SMOKE DETECTED!" if is_smoke else "No smoke detected"
    
    print(f"{result_icon} {result_text}")
    print(f"Confidence: {confidence*100:.1f}%")
    print(f"Threshold: {threshold*100:.0f}%")
    
    if is_smoke:
        print("\n⚠️  ALERT: Smoke has been detected in this image!")

def main():
    """Main test function"""
    print("\n" + "="*60)
    print("SMOKE DETECTION MODEL - TEST SCRIPT")
    print("="*60)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # Test specific image or directory
        path = sys.argv[1]
        threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.75
        
        if os.path.isfile(path):
            test_single_image(model, path, threshold)
        elif os.path.isdir(path):
            test_directory(model, path, threshold)
        else:
            print(f"❌ Path not found: {path}")
    else:
        # Test default directories
        print("\nNo path specified. Testing default directories...\n")
        
        # Test smoke images
        test_directory(model, 'datasets/smoke', threshold=0.75)
        
        # Test no-smoke images
        test_directory(model, 'datasets/no_smoke', threshold=0.75)
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)
    print("\nUsage:")
    print("  Test single image: python scripts/test_model.py path/to/image.jpg [threshold]")
    print("  Test directory:    python scripts/test_model.py path/to/directory [threshold]")
    print("  Test defaults:     python scripts/test_model.py")

if __name__ == "__main__":
    main()
