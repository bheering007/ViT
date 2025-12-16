#!/bin/bash
# Raspberry Pi Deployment Setup Script
# Tested on Raspberry Pi 4 (4GB+) and Raspberry Pi 5

set -e

echo "=========================================="
echo "Radar Gesture Classifier - Pi Setup"
echo "=========================================="

# Detect architecture
ARCH=$(uname -m)
echo "Architecture: $ARCH"

# Check if we're on a Pi
if [[ ! "$ARCH" =~ ^(aarch64|armv7l)$ ]]; then
    echo "Warning: This script is optimized for Raspberry Pi (ARM)"
    echo "Detected: $ARCH"
fi

# Update system
echo ""
echo "Step 1: Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install system dependencies
echo ""
echo "Step 2: Installing system dependencies..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libopenblas-dev \
    libjpeg-dev \
    zlib1g-dev \
    libffi-dev \
    libssl-dev

# Create virtual environment
echo ""
echo "Step 3: Creating Python virtual environment..."
VENV_DIR="$HOME/radar_classifier_venv"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "Step 4: Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch for ARM
echo ""
echo "Step 5: Installing PyTorch..."

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

if [[ "$ARCH" == "aarch64" ]]; then
    # Raspberry Pi 4/5 (64-bit)
    # Use official PyTorch wheel for ARM
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    # 32-bit (older Pi)
    echo "Warning: 32-bit ARM has limited PyTorch support"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
echo ""
echo "Step 6: Installing Python packages..."
pip install \
    timm \
    pillow \
    numpy

# Verify installation
echo ""
echo "Step 7: Verifying installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'Device: cpu (CUDA not available on Pi)')

import timm
print(f'timm version: {timm.__version__}')

from PIL import Image
print('PIL: OK')

print()
print('All dependencies installed successfully!')
"

# Copy model files
echo ""
echo "Step 8: Setup complete!"
echo ""
echo "=========================================="
echo "NEXT STEPS:"
echo "=========================================="
echo ""
echo "1. Copy your model file to the Pi:"
echo "   scp cnn_vit_radar.pt pi@<PI_IP>:~/deploy/"
echo ""
echo "2. Copy the inference script:"
echo "   scp deploy/infer_edge.py pi@<PI_IP>:~/deploy/"
echo ""
echo "3. Activate the virtual environment:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo "4. Run inference:"
echo "   python3 infer_edge.py image.png"
echo ""
echo "5. Run benchmark:"
echo "   python3 infer_edge.py --benchmark"
echo ""
echo "=========================================="
