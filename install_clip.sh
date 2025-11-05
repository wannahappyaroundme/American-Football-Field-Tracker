#!/bin/bash

# CLIP Integration Installation Script
# This script installs all dependencies for the CLIP-enhanced football analysis system

echo "=========================================="
echo "CLIP Integration Installation"
echo "=========================================="

# Check Python version
echo ""
echo "[1/4] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install base requirements
echo ""
echo "[2/4] Installing base dependencies..."
pip install ultralytics opencv-python numpy scipy scikit-learn pillow

# Install PyTorch (with CUDA support if available)
echo ""
echo "[3/4] Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "No NVIDIA GPU detected. Installing CPU-only PyTorch..."
    pip install torch torchvision
fi

# Install OpenAI CLIP
echo ""
echo "[4/4] Installing OpenAI CLIP..."
pip install git+https://github.com/openai/CLIP.git

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

python3 << 'EOF'
import sys

try:
    import torch
    print(f"✓ PyTorch installed (version {torch.__version__})")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("✗ PyTorch not installed")
    sys.exit(1)

try:
    import clip
    print("✓ CLIP installed")
    models = clip.available_models()
    print(f"  - Available models: {models}")
except ImportError:
    print("✗ CLIP not installed")
    sys.exit(1)

try:
    import cv2
    print(f"✓ OpenCV installed (version {cv2.__version__})")
except ImportError:
    print("✗ OpenCV not installed")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy installed (version {np.__version__})")
except ImportError:
    print("✗ NumPy not installed")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print("✓ Ultralytics YOLO installed")
except ImportError:
    print("✗ Ultralytics YOLO not installed")
    sys.exit(1)

print("")
print("=========================================="")
print("Installation Complete!")
print("=========================================="")
print("")
print("Next steps:")
print("1. Run: python calibrate_homography.py")
print("2. Place your video in: input/video.mp4")
print("3. Run: python main.py")
print("")
print("For more information, see CLIP_INTEGRATION_README.md")
EOF

echo ""
