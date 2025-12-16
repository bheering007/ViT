#!/usr/bin/env python3
"""
Lightweight Edge Inference Script for Raspberry Pi / Jetson
Optimized for TI IWR6843ISK Range-Doppler radar classification

Requirements:
    pip install torch torchvision timm pillow numpy

Usage:
    python infer_edge.py <image_path>
    python infer_edge.py <image_path> --threshold 0.6
    python infer_edge.py --benchmark
    
For integration:
    from infer_edge import RadarClassifier
    classifier = RadarClassifier("cnn_vit_radar.pt")
    result = classifier.predict("image.png")
"""
import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Conditional imports for edge devices
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available, using built-in backbone")

try:
    from PIL import Image
    from torchvision import transforms
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/torchvision not available")


@dataclass
class PredictionResult:
    """Prediction result with confidence and reliability flag."""
    prediction: str
    confidence: float
    is_reliable: bool
    all_probs: List[Tuple[str, float]]


class CNNViTHybrid(nn.Module):
    """Hybrid CNN-ViT model for Range-Doppler classification."""
    
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
        
        if backbone not in self.BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        model_name, cnn_out_channels, feat_size = self.BACKBONES[backbone]
        
        if TIMM_AVAILABLE:
            base_model = timm.create_model(model_name, pretrained=False, features_only=True)
            self.cnn_backbone = base_model
        else:
            raise ImportError("timm is required for model loading")
        
        self.num_patches = feat_size * feat_size
        self.patch_embed = nn.Conv2d(cnn_out_channels, embed_dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
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
            enable_nested_tensor=False
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, x):
        B = x.shape[0]
        features = self.cnn_backbone(x)
        x = features[-1]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.transformer(x)
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x


class RadarClassifier:
    """
    High-level API for radar gesture classification.
    Optimized for edge deployment on Raspberry Pi / Jetson.
    
    Example:
        classifier = RadarClassifier("cnn_vit_radar.pt")
        result = classifier.predict("range_doppler.png")
        print(f"{result.prediction}: {result.confidence:.2%}")
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Load model from checkpoint.
        
        Args:
            model_path: Path to .pt checkpoint file
            device: "auto", "cpu", or "cuda"
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load checkpoint
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.classes = ckpt["classes"]
        config = ckpt.get("model_config", {})
        metadata = ckpt.get("metadata", {})
        
        self.confidence_threshold = metadata.get("confidence_threshold", 0.5)
        
        # Build model
        self.model = CNNViTHybrid(
            num_classes=config.get("num_classes", len(self.classes)),
            backbone=config.get("backbone", "resnet18"),
            embed_dim=config.get("embed_dim", 256),
            num_heads=config.get("num_heads", 4),
            num_layers=config.get("num_layers", 4),
            dropout=config.get("dropout", 0.1)
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()
        
        # Set to inference mode (disables dropout, etc.)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        # Warmup inference (important for consistent timing)
        self._warmup()
    
    def _warmup(self, n: int = 3):
        """Warmup inference for consistent timing."""
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            for _ in range(n):
                _ = self.model(dummy)
    
    def predict(self, image_input) -> PredictionResult:
        """
        Run inference on an image.
        
        Args:
            image_input: File path (str), PIL Image, or numpy array (H,W) or (H,W,C)
        
        Returns:
            PredictionResult with prediction, confidence, and reliability flag
        """
        # Handle different input types
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found: {image_input}")
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img = Image.fromarray(image_input).convert("RGB")
            else:
                img = Image.fromarray(image_input)
        elif hasattr(image_input, 'convert'):  # PIL Image
            img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")
        
        # Preprocess
        x = self.transform(img).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
        
        # Get results
        probs_np = probs.cpu().numpy()
        top_idx = int(probs_np.argmax())
        top_conf = float(probs_np[top_idx])
        
        all_probs = [(self.classes[i], float(probs_np[i])) for i in range(len(self.classes))]
        all_probs.sort(key=lambda x: -x[1])
        
        return PredictionResult(
            prediction=self.classes[top_idx],
            confidence=top_conf,
            is_reliable=top_conf >= self.confidence_threshold,
            all_probs=all_probs
        )
    
    def predict_batch(self, images: list) -> List[PredictionResult]:
        """Run inference on multiple images."""
        return [self.predict(img) for img in images]
    
    def benchmark(self, n_runs: int = 50) -> dict:
        """Benchmark inference speed."""
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy)
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(dummy)
            times.append(time.perf_counter() - start)
        
        times_ms = [t * 1000 for t in times]
        return {
            "mean_ms": np.mean(times_ms),
            "std_ms": np.std(times_ms),
            "min_ms": np.min(times_ms),
            "max_ms": np.max(times_ms),
            "fps": 1000 / np.mean(times_ms)
        }


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    
    # Find model file
    script_dir = Path(__file__).parent
    model_paths = [
        script_dir / "cnn_vit_radar.pt",
        script_dir.parent / "cnn_vit_radar.pt",
        Path("cnn_vit_radar.pt"),
    ]
    
    model_path = None
    for p in model_paths:
        if p.exists():
            model_path = str(p)
            break
    
    if model_path is None:
        print("Error: cnn_vit_radar.pt not found")
        print("Searched in:", [str(p) for p in model_paths])
        return 1
    
    # Parse args
    threshold = 0.5
    if "--threshold" in sys.argv:
        idx = sys.argv.index("--threshold")
        threshold = float(sys.argv[idx + 1])
    
    # Initialize classifier
    print(f"Loading model from: {model_path}")
    classifier = RadarClassifier(model_path)
    print(f"Device: {classifier.device}")
    print(f"Classes: {classifier.classes}")
    print(f"Confidence threshold: {threshold}")
    
    # Handle --benchmark
    if "--benchmark" in sys.argv:
        print("\nRunning benchmark...")
        results = classifier.benchmark()
        print(f"Inference time: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
        print(f"FPS: {results['fps']:.1f}")
        print(f"Min: {results['min_ms']:.2f} ms | Max: {results['max_ms']:.2f} ms")
        return 0
    
    # Run inference
    input_path = sys.argv[1]
    if input_path.startswith("--"):
        print("Error: No input file specified")
        return 1
    
    print("-" * 40)
    result = classifier.predict(input_path)
    
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2%}")
    
    if result.is_reliable:
        print(f"Status: ✓ RELIABLE")
    else:
        print(f"Status: ⚠ LOW CONFIDENCE")
    
    print("\nAll classes:")
    for cls, prob in result.all_probs[:5]:
        bar = "█" * int(prob * 20)
        print(f"  {cls:12s}: {prob:.2%} {bar}")
    
    return 0 if result.is_reliable else 2


if __name__ == "__main__":
    sys.exit(main())
