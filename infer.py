"""
CNN-ViT Hybrid Inference Script for Radar Gesture Classification
Optimized for TI IWR6843ISK Range-Doppler maps

Usage:
    python infer.py <image_path>              # Single image
    python infer.py <folder_path>             # All images in folder
    python infer.py <image_path> --top 5      # Show top 5 predictions
    python infer.py --benchmark               # Run inference speed benchmark
"""
import sys
import os
import time
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms
from pathlib import Path


class CNNViTHybrid(nn.Module):
    """
    Hybrid CNN-ViT model optimized for Range-Doppler radar classification.
    Supports multiple backbones for speed/accuracy tradeoff.
    """
    
    BACKBONES = {
        'mobilenetv3': ('mobilenetv3_small_100', 576, 7),
        'resnet18': ('resnet18', 512, 7),
        'resnet34': ('resnet34', 512, 7),
        'efficientnet_b0': ('efficientnet_b0', 1280, 7),
        'efficientnet_b2': ('efficientnet_b2', 1408, 7),
    }
    
    def __init__(self, num_classes, embed_dim=384, num_heads=6, num_layers=6, 
                 dropout=0.1, backbone='resnet18'):
        super().__init__()
        
        self.backbone_name = backbone
        if backbone not in self.BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(self.BACKBONES.keys())}")
        
        model_name, cnn_out_channels, feat_size = self.BACKBONES[backbone]
        
        # CNN Backbone
        base_model = timm.create_model(model_name, pretrained=False, features_only=True)
        self.cnn_backbone = base_model
        
        self.num_patches = feat_size * feat_size
        
        # Project CNN features to transformer embedding dimension
        self.patch_embed = nn.Conv2d(cnn_out_channels, embed_dim, kernel_size=1)
        
        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings for patches + class token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False  # Suppress warning with norm_first=True
        )
        
        # Layer norm before classification head
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head with intermediate layer
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, x):
        B = x.shape[0]
        
        # CNN feature extraction - get last feature map
        features = self.cnn_backbone(x)
        x = features[-1]
        
        # Project to embed_dim
        x = self.patch_embed(x)
        
        # Flatten spatial dims to sequence
        x = x.flatten(2).transpose(1, 2)
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Extract class token and normalize
        x = self.norm(x[:, 0])
        
        # Classification
        x = self.head(x)
        
        return x


def load_model(checkpoint_path: str, device: torch.device):
    """Load the trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    classes = ckpt["classes"]
    config = ckpt.get("model_config", {
        "num_classes": len(classes),
        "backbone": "resnet18",
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 6,
        "dropout": 0.1
    })

    # Create model with saved config
    model = CNNViTHybrid(
        num_classes=config["num_classes"],
        backbone=config.get("backbone", "resnet18"),
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    
    return model, classes


def get_transform():
    """Get the image transformation pipeline."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def predict_single(model, image_path: str, classes: list, device: torch.device, top_k: int = 3):
    """Run inference on a single image."""
    tf = get_transform()
    
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_idx = probs.topk(min(top_k, len(classes)))

    return {
        "prediction": classes[top_idx[0].item()],
        "confidence": float(top_probs[0]),
        "top_k": [(classes[idx.item()], float(prob)) for prob, idx in zip(top_probs, top_idx)]
    }


def predict_folder(model, folder_path: str, classes: list, device: torch.device):
    """Run inference on all images in a folder."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    folder = Path(folder_path)
    
    results = {}
    for img_path in folder.iterdir():
        if img_path.suffix.lower() in image_extensions:
            result = predict_single(model, str(img_path), classes, device, top_k=1)
            if result:
                results[img_path.name] = result
    
    return results


def benchmark_inference(model, device: torch.device, num_runs: int = 100):
    """Benchmark model inference speed."""
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark single image
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    single_ms = (end_time - start_time) / num_runs * 1000
    single_fps = 1000 / single_ms
    
    # Benchmark batch of 8 (simulating buffered processing)
    batch_input = torch.randn(8, 3, 224, 224).to(device)
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs // 4):
            _ = model(batch_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    batch_ms = (end_time - start_time) / (num_runs // 4) * 1000
    batch_fps = 8 * 1000 / batch_ms  # Images per second
    
    return {
        "single_ms": single_ms,
        "single_fps": single_fps,
        "batch_ms": batch_ms,
        "batch_fps": batch_fps,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    checkpoint_path = "cnn_vit_radar.pt"
    try:
        model, classes = load_model(checkpoint_path, device)
        print(f"Loaded model with {len(classes)} classes")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using train_vit.py")
        return
    
    # Handle --benchmark flag
    if "--benchmark" in sys.argv:
        print("\n" + "=" * 50)
        print("INFERENCE SPEED BENCHMARK")
        print("=" * 50)
        results = benchmark_inference(model, device)
        print(f"Single image:  {results['single_ms']:.2f} ms ({results['single_fps']:.1f} FPS)")
        print(f"Batch of 8:    {results['batch_ms']:.2f} ms ({results['batch_fps']:.1f} images/sec)")
        print()
        print("For real-time (30 FPS): need < 33ms per frame")
        print(f"Current: {'✓ PASS' if results['single_ms'] < 33 else '✗ FAIL'}")
        return
    
    input_path = sys.argv[1]
    top_k = 3
    
    # Parse optional arguments
    if "--top" in sys.argv:
        idx = sys.argv.index("--top")
        if idx + 1 < len(sys.argv):
            top_k = int(sys.argv[idx + 1])
    
    print("-" * 50)
    
    # Check if input is file or folder
    if os.path.isfile(input_path):
        result = predict_single(model, input_path, classes, device, top_k=top_k)
        if result:
            print(f"Image: {input_path}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"\nTop {len(result['top_k'])} predictions:")
            for cls, prob in result['top_k']:
                bar = "█" * int(prob * 20)
                print(f"  {cls:12s}: {prob:.4f} {bar}")
    
    elif os.path.isdir(input_path):
        results = predict_folder(model, input_path, classes, device)
        print(f"Processed {len(results)} images from {input_path}\n")
        
        # Group by prediction
        from collections import defaultdict
        by_class = defaultdict(list)
        for img_name, result in results.items():
            by_class[result['prediction']].append((img_name, result['confidence']))
        
        for cls in sorted(by_class.keys()):
            items = by_class[cls]
            print(f"{cls} ({len(items)} images):")
            for img_name, conf in sorted(items, key=lambda x: -x[1])[:5]:
                print(f"  {img_name}: {conf:.4f}")
            if len(items) > 5:
                print(f"  ... and {len(items) - 5} more")
            print()
    else:
        print(f"Error: Path not found: {input_path}")


if __name__ == "__main__":
    main()
