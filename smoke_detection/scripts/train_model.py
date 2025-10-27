"""
Smoke Detection Model Training Script
Uses transfer learning with MobileNetV2 for efficient real-time inference
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import json
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SmokeDetectionModel:
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
    def create_model(self):
        """
        Create smoke detection model using MobileNetV2 transfer learning
        """
        print("\n" + "=" * 60)
        print("CREATING MODEL ARCHITECTURE")
        print("=" * 60)
        
        # Load pre-trained MobileNetV2 (without top classification layer)
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        print(f"✓ Loaded MobileNetV2 base model")
        print(f"  Input shape: {self.img_size}")
        print(f"  Base model layers: {len(base_model.layers)}")
        
        # Build classification head
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Preprocessing
        x = layers.Rescaling(1./127.5, offset=-1)(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print(f"✓ Model compiled successfully")
        print(f"\nModel Summary:")
        self.model.summary()
        
        return self.model
    
    def prepare_data(self, dataset_dir):
        """
        Prepare training and validation datasets with augmentation
        """
        print("\n" + "=" * 60)
        print("PREPARING DATASETS")
        print("=" * 60)
        
        # Check for augmented data first, fall back to original
        smoke_dir = Path(dataset_dir) / 'smoke'
        no_smoke_dir = Path(dataset_dir) / 'no_smoke'
        
        # Use augmented data if available
        if (smoke_dir / 'augmented').exists():
            print("✓ Using augmented smoke images")
            smoke_source = smoke_dir / 'augmented'
        else:
            print("✓ Using original smoke images")
            smoke_source = smoke_dir
            
        if (no_smoke_dir / 'augmented').exists():
            print("✓ Using augmented no-smoke images")
            no_smoke_source = no_smoke_dir / 'augmented'
        else:
            print("✓ Using original no-smoke images")
            no_smoke_source = no_smoke_dir
        
        # Count images
        smoke_count = len(list(smoke_source.glob('*.jpg'))) + len(list(smoke_source.glob('*.png')))
        no_smoke_count = len(list(no_smoke_source.glob('*.jpg'))) + len(list(no_smoke_source.glob('*.png')))
        
        print(f"\nDataset Statistics:")
        print(f"  Smoke images: {smoke_count}")
        print(f"  No-smoke images: {no_smoke_count}")
        print(f"  Total: {smoke_count + no_smoke_count}")
        
        if smoke_count == 0 or no_smoke_count == 0:
            raise ValueError("Dataset is empty! Please run prepare_data.py first.")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            validation_split=0.2
        )
        
        # Validation data (no augmentation)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Create data generators
        train_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            dataset_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        print(f"\n✓ Data generators created")
        print(f"  Training samples: {train_generator.samples}")
        print(f"  Validation samples: {val_generator.samples}")
        print(f"  Class indices: {train_generator.class_indices}")
        
        return train_generator, val_generator
    
    def train(self, train_gen, val_gen, epochs=20):
        """
        Train the smoke detection model
        """
        print("\n" + "=" * 60)
        print("TRAINING MODEL")
        print("=" * 60)
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Image size: {self.img_size}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'model/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        print("\n" + "-" * 60)
        print("Starting training...")
        print("-" * 60)
        
        # Train model
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        
        return self.history
    
    def evaluate(self, val_gen):
        """
        Evaluate model performance
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        results = self.model.evaluate(val_gen, verbose=1)
        
        print(f"\nValidation Results:")
        print(f"  Loss: {results[0]:.4f}")
        print(f"  Accuracy: {results[1]:.4f}")
        print(f"  Precision: {results[2]:.4f}")
        print(f"  Recall: {results[3]:.4f}")
        
        # Calculate F1 score
        precision = results[2]
        recall = results[3]
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  F1 Score: {f1_score:.4f}")
        
        return results
    
    def save_model(self, filepath='model/smoke_detector.h5'):
        """
        Save trained model
        """
        base_dir = Path(__file__).parent.parent
        model_path = base_dir / filepath
        model_path.parent.mkdir(exist_ok=True)
        
        self.model.save(str(model_path))
        print(f"\n✓ Model saved to: {model_path}")
        
        # Save training history
        history_path = model_path.parent / 'training_history.json'
        history_dict = {
            'loss': [float(x) for x in self.history.history['loss']],
            'accuracy': [float(x) for x in self.history.history['accuracy']],
            'val_loss': [float(x) for x in self.history.history['val_loss']],
            'val_accuracy': [float(x) for x in self.history.history['val_accuracy']],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"✓ Training history saved to: {history_path}")
        
        # Save model info
        info_path = model_path.parent / 'model_info.json'
        model_info = {
            'architecture': 'MobileNetV2',
            'input_shape': list(self.img_size) + [3],
            'classes': ['no_smoke', 'smoke'],
            'training_date': datetime.now().isoformat(),
            'total_params': self.model.count_params()
        }
        
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✓ Model info saved to: {info_path}")

def main():
    """
    Main training pipeline
    """
    print("\n" + "=" * 60)
    print("SMOKE DETECTION MODEL TRAINING")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Configuration
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 20
    
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / 'datasets'
    
    # Check if dataset exists
    if not dataset_dir.exists():
        print("\n❌ ERROR: Dataset directory not found!")
        print(f"Expected: {dataset_dir}")
        print("\nPlease run 'python scripts/prepare_data.py' first.")
        return
    
    try:
        # Initialize model
        smoke_model = SmokeDetectionModel(img_size=IMG_SIZE, batch_size=BATCH_SIZE)
        
        # Create model architecture
        smoke_model.create_model()
        
        # Prepare datasets
        train_gen, val_gen = smoke_model.prepare_data(dataset_dir)
        
        # Train model
        smoke_model.train(train_gen, val_gen, epochs=EPOCHS)
        
        # Evaluate model
        smoke_model.evaluate(val_gen)
        
        # Save model
        smoke_model.save_model()
        
        print("\n" + "=" * 60)
        print("✓ TRAINING PIPELINE COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review training metrics in model/training_history.json")
        print("2. Test the model with: python app.py")
        print("3. Access web interface at: http://localhost:5000")
        
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
