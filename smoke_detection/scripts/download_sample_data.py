"""
Sample Data Download Script
Provides guidance and utilities for downloading smoke detection datasets
"""

import os
from pathlib import Path

def print_dataset_sources():
    """
    Print information about available smoke detection datasets
    """
    print("\n" + "="*70)
    print("SMOKE DETECTION DATASETS - DOWNLOAD GUIDE")
    print("="*70)
    
    print("\n📊 RECOMMENDED DATASETS:\n")
    
    print("1. KAGGLE - Smoke Detection Dataset")
    print("   URL: https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset")
    print("   Size: ~2000 images")
    print("   Classes: Smoke, No Smoke")
    print("   Quality: High quality, diverse scenes")
    print("   Download: Requires Kaggle account")
    print()
    
    print("2. KAGGLE - Fire and Smoke Dataset")
    print("   URL: https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset")
    print("   Size: ~1000 images")
    print("   Classes: Fire, Smoke, Neutral")
    print("   Quality: Good for training")
    print("   Download: Requires Kaggle account")
    print()
    
    print("3. ROBOFLOW - Smoke Detection")
    print("   URL: https://universe.roboflow.com/search?q=smoke%20detection")
    print("   Size: Various datasets available")
    print("   Classes: Multiple smoke detection datasets")
    print("   Quality: Pre-processed and annotated")
    print("   Download: Free with account")
    print()
    
    print("4. GOOGLE IMAGES - Manual Collection")
    print("   Search terms:")
    print("     - 'smoke detection'")
    print("     - 'building fire smoke'")
    print("     - 'industrial smoke'")
    print("     - 'wildfire smoke'")
    print("   Quality: Variable, requires manual curation")
    print("   Download: Manual download")
    print()
    
    print("="*70)
    print("\n📥 DOWNLOAD INSTRUCTIONS:\n")
    
    print("METHOD 1: Kaggle CLI (Recommended)")
    print("-" * 70)
    print("1. Install Kaggle CLI:")
    print("   pip install kaggle")
    print()
    print("2. Setup Kaggle API credentials:")
    print("   - Go to https://www.kaggle.com/account")
    print("   - Click 'Create New API Token'")
    print("   - Save kaggle.json to ~/.kaggle/")
    print()
    print("3. Download dataset:")
    print("   kaggle datasets download -d deepcontractor/smoke-detection-dataset")
    print("   unzip smoke-detection-dataset.zip -d datasets/")
    print()
    
    print("METHOD 2: Manual Download")
    print("-" * 70)
    print("1. Visit dataset URL in browser")
    print("2. Click 'Download' button")
    print("3. Extract ZIP file")
    print("4. Organize images into folders:")
    print("   - datasets/smoke/       (images with smoke)")
    print("   - datasets/no_smoke/    (images without smoke)")
    print()
    
    print("METHOD 3: Web Scraping (Advanced)")
    print("-" * 70)
    print("1. Use tools like:")
    print("   - google-images-download")
    print("   - bing-image-downloader")
    print("2. Manually review and filter images")
    print("3. Organize into appropriate folders")
    print()
    
    print("="*70)
    print("\n📁 DATASET ORGANIZATION:\n")
    
    base_dir = Path(__file__).parent.parent
    smoke_dir = base_dir / 'datasets' / 'smoke'
    no_smoke_dir = base_dir / 'datasets' / 'no_smoke'
    
    print(f"Place your images in these directories:")
    print(f"  Smoke images:     {smoke_dir}")
    print(f"  No-smoke images:  {no_smoke_dir}")
    print()
    print("Supported formats: JPG, PNG, JPEG")
    print("Recommended: 500-1000 images per class")
    print()
    
    # Check current status
    smoke_count = len(list(smoke_dir.glob('*.jpg'))) + len(list(smoke_dir.glob('*.png')))
    no_smoke_count = len(list(no_smoke_dir.glob('*.jpg'))) + len(list(no_smoke_dir.glob('*.png')))
    
    print("Current dataset status:")
    print(f"  Smoke images: {smoke_count}")
    print(f"  No-smoke images: {no_smoke_count}")
    
    if smoke_count == 0 or no_smoke_count == 0:
        print("\n⚠️  Dataset is empty or incomplete!")
    else:
        print("\n✓ Dataset found!")
    
    print("\n" + "="*70)
    print("\n🚀 NEXT STEPS:\n")
    print("1. Download and organize your dataset")
    print("2. Run: python scripts/prepare_data.py")
    print("3. Run: python scripts/train_model.py")
    print("4. Run: python app.py")
    print()
    print("="*70)

def create_sample_images():
    """
    Create placeholder images for testing (optional)
    """
    try:
        import cv2
        import numpy as np
        
        base_dir = Path(__file__).parent.parent
        smoke_dir = base_dir / 'datasets' / 'smoke'
        no_smoke_dir = base_dir / 'datasets' / 'no_smoke'
        
        print("\n" + "="*70)
        print("CREATING SAMPLE PLACEHOLDER IMAGES")
        print("="*70)
        print("\n⚠️  These are placeholder images for testing only!")
        print("Replace with real smoke detection images for actual training.\n")
        
        # Create sample smoke images (gray with random noise)
        for i in range(5):
            img = np.random.randint(100, 150, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(smoke_dir / f'sample_smoke_{i}.jpg'), img)
        
        # Create sample no-smoke images (lighter with random noise)
        for i in range(5):
            img = np.random.randint(150, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(str(no_smoke_dir / f'sample_no_smoke_{i}.jpg'), img)
        
        print("✓ Created 10 placeholder images (5 smoke, 5 no-smoke)")
        print("\nThese are NOT suitable for real training!")
        print("Please download actual smoke detection datasets.\n")
        
    except ImportError:
        print("\n❌ OpenCV not installed. Cannot create sample images.")
        print("Install with: pip install opencv-python")

def main():
    """Main function"""
    print_dataset_sources()
    
    # Ask if user wants to create sample images
    print("\n" + "="*70)
    response = input("\nCreate placeholder images for testing? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        create_sample_images()
    
    print("\n" + "="*70)
    print("For questions or issues, refer to README.md")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
