"""
Configuration file for Smoke Detection System
Centralized settings for easy customization
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Model Configuration
MODEL_CONFIG = {
    'architecture': 'MobileNetV2',
    'input_size': (224, 224),
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 0.001,
    'model_path': BASE_DIR / 'model' / 'smoke_detector.h5',
    'best_model_path': BASE_DIR / 'model' / 'best_model.h5',
}

# Dataset Configuration
DATASET_CONFIG = {
    'base_dir': BASE_DIR / 'datasets',
    'smoke_dir': BASE_DIR / 'datasets' / 'smoke',
    'no_smoke_dir': BASE_DIR / 'datasets' / 'no_smoke',
    'validation_split': 0.2,
    'augmentation': True,
}

# Data Augmentation Parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'brightness_range': (0.8, 1.2),
}

# Flask Application Configuration
FLASK_CONFIG = {
    'upload_folder': BASE_DIR / 'static' / 'uploads',
    'max_content_length': 16 * 1024 * 1024,  # 16MB
    'allowed_extensions': {'png', 'jpg', 'jpeg', 'gif'},
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
}

# Detection Configuration
DETECTION_CONFIG = {
    'confidence_threshold': 0.75,
    'frame_interval': 1.0,  # seconds between frame processing
    'alert_cooldown': 5.0,  # seconds between alerts
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': BASE_DIR / 'logs' / 'smoke_detection.log',
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'use_gpu': True,
    'gpu_memory_growth': True,
    'mixed_precision': False,
    'model_quantization': False,
}

# Alert Configuration
ALERT_CONFIG = {
    'enable_sound': True,
    'enable_popup': True,
    'enable_history': True,
    'max_history_items': 100,
}

# Create necessary directories
def create_directories():
    """Create required directories if they don't exist"""
    directories = [
        FLASK_CONFIG['upload_folder'],
        MODEL_CONFIG['model_path'].parent,
        DATASET_CONFIG['smoke_dir'],
        DATASET_CONFIG['no_smoke_dir'],
        LOGGING_CONFIG['log_file'].parent,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories on import
create_directories()
