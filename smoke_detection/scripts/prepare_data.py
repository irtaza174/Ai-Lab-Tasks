"""
Data Preparation Script for Smoke Detection
Prepares and augments dataset for training
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil

def create_sample_dataset():
    """
    Creates a sample dataset structure and provides guidance for data collection.
    In production, replace this with actual smoke/no-smoke images.
    """
    base_dir = Path(__file__).parent.parent
    smoke_dir = base_dir / 'datasets' / 'smoke'
    no_smoke_dir = base_dir / 'datasets' / 'no_smoke'
    
    print("=" * 60)
    print("SMOKE DETECTION - DATA PREPARATION")
    print("=" * 60)
    print("\nDataset Directory Structure:")
    print(f"  Smoke images: {smoke_dir}")
    print(f"  No-smoke images: {no_smoke_dir}")
    
    # Count existing images
    smoke_count = len(list(smoke_dir.glob('*.jpg'))) + len(list(smoke_dir.glob('*.png')))
    no_smoke_count = len(list(no_smoke_dir.glob('*.jpg'))) + len(list(no_smoke_dir.glob('*.png')))
    
    print(f"\nCurrent Dataset:")
    print(f"  Smoke images: {smoke_count}")
    print(f"  No-smoke images: {no_smoke_count}")
    
    if smoke_count == 0 or no_smoke_count == 0:
        print("\n" + "!" * 60)
        print("WARNING: Dataset is empty or incomplete!")
        print("!" * 60)
        print("\nTo prepare your dataset:")
        print("\n1. COLLECT SMOKE IMAGES:")
        print("   - Download from Kaggle datasets:")
        print("     * 'Smoke Detection Dataset'")
        print("     * 'Fire and Smoke Dataset'")
        print("   - Or use Google Images with search terms:")
        print("     * 'smoke detection'")
        print("     * 'building fire smoke'")
        print("     * 'industrial smoke'")
        print(f"   - Place in: {smoke_dir}")
        print("   - Recommended: 500-1000 images")
        
        print("\n2. COLLECT NO-SMOKE IMAGES:")
        print("   - Similar scenes without smoke:")
        print("     * 'building exterior'")
        print("     * 'industrial facility'")
        print("     * 'indoor scenes'")
        print(f"   - Place in: {no_smoke_dir}")
        print("   - Recommended: 500-1000 images")
        
        print("\n3. KAGGLE DATASET RECOMMENDATIONS:")
        print("   - Search 'smoke detection' on Kaggle")
        print("   - Download and extract to datasets folder")
        print("   - Organize into smoke/no_smoke folders")
        
        print("\n4. DATA AUGMENTATION:")
        print("   - This script will automatically augment your data")
        print("   - Creates variations: rotation, flip, brightness, etc.")
        print("   - Increases dataset size 5x")
        
        return False
    
    return True

def augment_image(image):
    """
    Apply data augmentation to increase dataset diversity
    """
    augmented_images = [image]
    
    # Horizontal flip
    augmented_images.append(cv2.flip(image, 1))
    
    # Rotation
    height, width = image.shape[:2]
    for angle in [15, -15]:
        matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (width, height))
        augmented_images.append(rotated)
    
    # Brightness adjustment
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-30)
    augmented_images.append(bright)
    augmented_images.append(dark)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    augmented_images.append(blurred)
    
    return augmented_images

def prepare_dataset(augment=True):
    """
    Prepare and optionally augment the dataset
    """
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / 'datasets'
    
    categories = ['smoke', 'no_smoke']
    stats = {}
    
    print("\n" + "=" * 60)
    print("PROCESSING DATASET")
    print("=" * 60)
    
    for category in categories:
        category_dir = dataset_dir / category
        images = list(category_dir.glob('*.jpg')) + list(category_dir.glob('*.png')) + \
                 list(category_dir.glob('*.jpeg')) + list(category_dir.glob('*.JPG'))
        
        original_count = len(images)
        processed_count = 0
        
        print(f"\nProcessing '{category}' category...")
        print(f"  Found {original_count} images")
        
        if augment and original_count > 0:
            augmented_dir = category_dir / 'augmented'
            augmented_dir.mkdir(exist_ok=True)
            
            for idx, img_path in enumerate(images):
                try:
                    # Read image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # Resize to standard size
                    img = cv2.resize(img, (224, 224))
                    
                    # Save original (resized)
                    cv2.imwrite(str(augmented_dir / f"orig_{idx}.jpg"), img)
                    
                    # Generate augmented versions
                    augmented = augment_image(img)
                    for aug_idx, aug_img in enumerate(augmented[1:], 1):
                        cv2.imwrite(str(augmented_dir / f"aug_{idx}_{aug_idx}.jpg"), aug_img)
                    
                    processed_count += 1
                    
                    if (idx + 1) % 50 == 0:
                        print(f"  Processed {idx + 1}/{original_count} images...")
                
                except Exception as e:
                    print(f"  Error processing {img_path.name}: {e}")
            
            total_augmented = len(list(augmented_dir.glob('*.jpg')))
            print(f"  ✓ Created {total_augmented} augmented images")
            stats[category] = {
                'original': original_count,
                'processed': processed_count,
                'augmented': total_augmented
            }
        else:
            stats[category] = {
                'original': original_count,
                'processed': 0,
                'augmented': 0
            }
    
    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print("\nFinal Statistics:")
    for category, stat in stats.items():
        print(f"\n{category.upper()}:")
        print(f"  Original images: {stat['original']}")
        print(f"  Augmented images: {stat['augmented']}")
        print(f"  Total: {stat['original'] + stat['augmented']}")
    
    total_images = sum(s['original'] + s['augmented'] for s in stats.values())
    print(f"\nTotal dataset size: {total_images} images")
    
    if total_images < 100:
        print("\n⚠ WARNING: Dataset is small. Recommend at least 1000 images total.")
        print("  Consider collecting more data for better accuracy.")
    
    return stats

if __name__ == "__main__":
    # Check if dataset exists
    has_data = create_sample_dataset()
    
    if has_data:
        # Prepare and augment dataset
        print("\nStarting data augmentation...")
        response = input("\nProceed with data augmentation? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            stats = prepare_dataset(augment=True)
            print("\n✓ Dataset preparation complete!")
            print("\nNext step: Run 'python scripts/train_model.py' to train the model")
        else:
            print("\nData preparation cancelled.")
    else:
        print("\n" + "=" * 60)
        print("Please collect and organize your dataset first.")
        print("=" * 60)
