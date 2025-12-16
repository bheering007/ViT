# Raspberry Pi Deployment Guide

## Hardware Requirements

| Raspberry Pi | RAM | Performance | Recommended |
|--------------|-----|-------------|-------------|
| Pi 3B+ | 1GB | ~500ms/inference | ❌ Too slow |
| Pi 4 (2GB) | 2GB | ~150ms/inference | ⚠️ Minimal |
| **Pi 4 (4GB+)** | 4GB+ | ~100ms/inference | ✅ Good |
| **Pi 5** | 4GB+ | ~50ms/inference | ✅ Best |

## Quick Start

### 1. On your development machine (Windows)

```powershell
# Train the model first
py -3.12 train_vit.py

# The model will be saved as cnn_vit_radar.pt
```

### 2. Transfer files to Raspberry Pi

```bash
# From Windows PowerShell
scp cnn_vit_radar.pt pi@<PI_IP>:~/deploy/
scp deploy/infer_edge.py pi@<PI_IP>:~/deploy/
scp deploy/setup_pi.sh pi@<PI_IP>:~/deploy/
```

### 3. On the Raspberry Pi

```bash
# Run setup script
chmod +x ~/deploy/setup_pi.sh
~/deploy/setup_pi.sh

# Activate environment
source ~/radar_classifier_venv/bin/activate

# Run inference
cd ~/deploy
python3 infer_edge.py image.png
```

## Python Integration

```python
from infer_edge import RadarClassifier

# Initialize (loads model once)
classifier = RadarClassifier("cnn_vit_radar.pt")

# Predict from file
result = classifier.predict("range_doppler.png")
print(f"{result.prediction}: {result.confidence:.2%}")

# Predict from numpy array (from radar driver)
import numpy as np
radar_data = np.random.rand(64, 128)  # Your Range-Doppler map
result = classifier.predict(radar_data)

# Check reliability
if result.is_reliable:
    trigger_action(result.prediction)
else:
    print("Low confidence, waiting for clearer signal...")
```

## Continuous Inference Loop

```python
import time
from infer_edge import RadarClassifier

classifier = RadarClassifier("cnn_vit_radar.pt")

while True:
    # Get image from your radar driver
    image_path = get_latest_radar_frame()
    
    result = classifier.predict(image_path)
    
    if result.is_reliable:
        print(f"[{time.strftime('%H:%M:%S')}] {result.prediction} ({result.confidence:.1%})")
    
    time.sleep(0.1)  # 10 FPS max
```

## Performance Optimization

### 1. Use ONNX for faster inference (optional)

```python
# On your Windows machine, export to ONNX
import torch
from train_vit import CNNViTHybrid

model = CNNViTHybrid(11, backbone="resnet18", embed_dim=256, num_heads=4, num_layers=4)
ckpt = torch.load("cnn_vit_radar.pt", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, "cnn_vit_radar.onnx", opset_version=14)
```

Then on Pi, use ONNX Runtime:
```bash
pip install onnxruntime
```

### 2. Use smaller backbone (mobilenetv3)

In `train_vit.py`, change:
```python
"backbone": "mobilenetv3",  # Faster than resnet18
```

### 3. Reduce input resolution

For faster inference, modify the transform in `infer_edge.py`:
```python
transforms.Resize((160, 160)),  # Instead of 224x224
```

## Systemd Service (Auto-start)

Create `/etc/systemd/system/radar-classifier.service`:

```ini
[Unit]
Description=Radar Gesture Classifier
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/deploy
Environment="PATH=/home/pi/radar_classifier_venv/bin"
ExecStart=/home/pi/radar_classifier_venv/bin/python3 your_app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable with:
```bash
sudo systemctl enable radar-classifier
sudo systemctl start radar-classifier
```

## Troubleshooting

### Out of memory
- Use a Pi with 4GB+ RAM
- Close other applications
- Use `mobilenetv3` backbone instead of `resnet18`

### Slow inference
- Check no throttling: `vcgencmd get_throttled`
- Ensure adequate cooling (heatsink + fan)
- Use Pi 5 instead of Pi 4

### Module not found
- Ensure virtual environment is activated
- Run `pip install torch timm pillow numpy`

## Expected Performance

| Device | Time/inference | FPS |
|--------|---------------|-----|
| Pi 4 (4GB) | ~100ms | 10 |
| Pi 5 | ~50ms | 20 |
| Jetson Orin Nano | ~5ms | 200 |

For real-time (>30 FPS), consider NVIDIA Jetson instead.
