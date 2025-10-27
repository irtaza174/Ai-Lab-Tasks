#!/bin/bash

# Smoke Detection System - Quick Start Script
# This script automates the setup process

echo "=========================================="
echo "SMOKE DETECTION SYSTEM - QUICK START"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "❌ Python 3 is not installed!"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✓ Python found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
echo "This may take 5-10 minutes..."
pip install -r requirements.txt --quiet

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies installed"
echo ""

# Check dataset
echo "Checking dataset..."
SMOKE_COUNT=$(find datasets/smoke -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)
NO_SMOKE_COUNT=$(find datasets/no_smoke -maxdepth 1 -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) 2>/dev/null | wc -l)

echo "Smoke images: $SMOKE_COUNT"
echo "No-smoke images: $NO_SMOKE_COUNT"
echo ""

if [ $SMOKE_COUNT -eq 0 ] || [ $NO_SMOKE_COUNT -eq 0 ]; then
    echo "⚠️  WARNING: Dataset is empty or incomplete!"
    echo ""
    echo "Next steps:"
    echo "1. Download smoke detection dataset"
    echo "2. Place images in datasets/smoke/ and datasets/no_smoke/"
    echo "3. Run: python scripts/prepare_data.py"
    echo "4. Run: python scripts/train_model.py"
    echo ""
    echo "For dataset sources, run:"
    echo "  python scripts/download_sample_data.py"
    echo ""
else
    echo "✓ Dataset found"
    echo ""
    
    # Ask to prepare data
    read -p "Prepare and augment dataset? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Running data preparation..."
        python scripts/prepare_data.py
        echo ""
    fi
    
    # Check if model exists
    if [ -f "model/smoke_detector.h5" ]; then
        echo "✓ Trained model found"
        echo ""
    else
        echo "⚠️  No trained model found"
        echo ""
        read -p "Train model now? This may take 30-60 minutes. (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Starting model training..."
            python scripts/train_model.py
            echo ""
        fi
    fi
fi

echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "To start the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run: python app.py"
echo "  3. Open browser: http://localhost:5000"
echo ""
echo "For detailed instructions, see SETUP_GUIDE.md"
echo ""
