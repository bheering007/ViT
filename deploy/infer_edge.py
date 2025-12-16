#!/usr/bin/env python3
"""
Lightweight Edge Inference Script for Raspberry Pi / Jetson
Optimized for TI IWR6843ISK Range-Doppler radar classification
Supports both single-frame and sequence (streaming) models.

Requirements:
    pip install torch torchvision timm pillow numpy

Usage:
    python infer_edge.py <image_path>
    python infer_edge.py <image_path> --threshold 0.6
    python infer_edge.py --benchmark
    python infer_edge.py --stream <folder>    # Streaming mode
    
For integration:
    from infer_edge import RadarClassifier, StreamingClassifier
    
    # Single frame
    classifier = RadarClassifier("cnn_vit_radar.pt")
    result = classifier.predict("image.png")
    
    # Streaming
    streamer = StreamingClassifier("cnn_vit_lstm_radar.pt")
    while True:
        result = streamer.update(get_frame())
"""
import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque

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


@dataclass
class StreamPrediction:
    """Result from streaming classifier."""
    prediction: str
    confidence: float
    is_reliable: bool
    all_probs: List[Tuple[str, float]]
    frame_count: int
    latency_ms: float


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
    
    def forward_features(self, x):
        """Extract spatial features without classification."""
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
        return self.norm(x[:, 0])


class TemporalEncoder(nn.Module):
    """LSTM/GRU encoder for temporal modeling."""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, 
                 dropout=0.3, bidirectional=True, model_type='lstm'):
        super().__init__()
        
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        
        RNN = nn.LSTM if model_type == 'lstm' else nn.GRU
        self.rnn = RNN(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
    
    def forward(self, x):
        output, _ = self.rnn(x)
        if self.bidirectional:
            return torch.cat([output[:, -1, :self.hidden_dim], 
                            output[:, 0, self.hidden_dim:]], dim=1)
        return output[:, -1, :]


class CNNViTSequence(nn.Module):
    """Sequence model: CNN-ViT per frame + LSTM across frames."""
    
    def __init__(self, num_classes, config):
        super().__init__()
        
        self.spatial_encoder = CNNViTHybrid(
            num_classes=num_classes,
            backbone=config.get("backbone", "resnet18"),
            embed_dim=config.get("embed_dim", 256),
            num_heads=config.get("num_heads", 4),
            num_layers=config.get("num_layers", 4),
            dropout=config.get("dropout", 0.1)
        )
        
        self.temporal_encoder = TemporalEncoder(
            input_dim=config.get("embed_dim", 256),
            hidden_dim=config.get("temporal_hidden", 256),
            num_layers=config.get("temporal_layers", 2),
            dropout=config.get("temporal_dropout", 0.3),
            bidirectional=config.get("temporal_bidirectional", True),
            model_type=config.get("temporal_model", "lstm")
        )
        
        temporal_dim = self.temporal_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim // 2),
            nn.GELU(),
            nn.Dropout(config.get("classifier_dropout", 0.4)),
            nn.Linear(temporal_dim // 2, num_classes)
        )
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        spatial_features = self.spatial_encoder.forward_features(x)
        spatial_features = spatial_features.view(B, T, -1)
        temporal_features = self.temporal_encoder(spatial_features)
        return self.classifier(temporal_features)
    
    def forward_spatial_only(self, x):
        return self.spatial_encoder.forward_features(x)
    
    def forward_temporal_only(self, spatial_features):
        temporal_features = self.temporal_encoder(spatial_features)
        return self.classifier(temporal_features)


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


class StreamingClassifier:
    """
    Real-time streaming classifier with sliding window buffer.
    For sequence models (CNN-ViT-LSTM).
    
    Example:
        classifier = StreamingClassifier("cnn_vit_lstm_radar.pt")
        while True:
            frame = get_radar_frame()
            result = classifier.update(frame)
            if result:
                print(f"{result.prediction}: {result.confidence:.1%}")
    """
    
    def __init__(self, model_path: str, device: str = "auto", smoothing_window: int = 3):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.classes = ckpt["classes"]
        self.config = ckpt.get("model_config", {})
        self.sequence_length = self.config.get("sequence_length", 8)
        self.confidence_threshold = ckpt.get("metadata", {}).get("confidence_threshold", 0.5)
        
        mode = self.config.get("mode", "single")
        
        if mode == "sequence":
            self.model = CNNViTSequence(num_classes=len(self.classes), config=self.config)
        else:
            self.model = CNNViTHybrid(
                num_classes=len(self.classes),
                backbone=self.config.get("backbone", "resnet18"),
                embed_dim=self.config.get("embed_dim", 256),
                num_heads=self.config.get("num_heads", 4),
                num_layers=self.config.get("num_layers", 4),
                dropout=self.config.get("dropout", 0.1)
            )
            self.sequence_length = 1
        
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        self.feature_buffer = deque(maxlen=self.sequence_length)
        self.prediction_history = deque(maxlen=smoothing_window)
        self.frame_count = 0
        self.mode = mode
    
    def _preprocess(self, image_input) -> torch.Tensor:
        if isinstance(image_input, torch.Tensor):
            return image_input
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                image_input = np.stack([image_input] * 3, axis=-1)
            img = Image.fromarray(image_input.astype(np.uint8))
            return self.transform(img)
        elif isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert('RGB')
            return self.transform(img)
        elif hasattr(image_input, 'convert'):
            return self.transform(image_input)
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")
    
    @torch.no_grad()
    def update(self, frame) -> Optional[StreamPrediction]:
        """Add a new frame and get prediction if buffer is full."""
        start_time = time.perf_counter()
        
        tensor = self._preprocess(frame).unsqueeze(0).to(self.device)
        self.frame_count += 1
        
        if self.mode == "sequence":
            features = self.model.forward_spatial_only(tensor)
            self.feature_buffer.append(features)
            
            if len(self.feature_buffer) < self.sequence_length:
                return None
            
            feature_seq = torch.cat(list(self.feature_buffer), dim=0).unsqueeze(0)
            logits = self.model.forward_temporal_only(feature_seq)
        else:
            logits = self.model(tensor)
        
        probs = F.softmax(logits, dim=1)[0]
        
        self.prediction_history.append(probs.cpu())
        if len(self.prediction_history) > 1:
            avg_probs = torch.stack(list(self.prediction_history)).mean(dim=0)
        else:
            avg_probs = probs.cpu()
        
        confidence, pred_idx = avg_probs.max(0)
        confidence = confidence.item()
        pred_idx = pred_idx.item()
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        all_probs = [(self.classes[i], avg_probs[i].item()) for i in range(len(self.classes))]
        all_probs.sort(key=lambda x: x[1], reverse=True)
        
        return StreamPrediction(
            prediction=self.classes[pred_idx],
            confidence=confidence,
            is_reliable=confidence >= self.confidence_threshold,
            all_probs=all_probs,
            frame_count=self.frame_count,
            latency_ms=latency_ms
        )
    
    def reset(self):
        self.feature_buffer.clear()
        self.prediction_history.clear()
        self.frame_count = 0
    
    def benchmark(self, n_frames: int = 100) -> dict:
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        
        self.reset()
        for _ in range(self.sequence_length + 5):
            self.update(dummy[0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
        
        self.reset()
        latencies = []
        for _ in range(n_frames):
            result = self.update(dummy[0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
            if result:
                latencies.append(result.latency_ms)
        
        return {
            "mean_ms": np.mean(latencies) if latencies else 0,
            "std_ms": np.std(latencies) if latencies else 0,
            "fps": 1000 / np.mean(latencies) if latencies else 0,
            "sequence_length": self.sequence_length,
            "mode": self.mode
        }


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    
    # Find model file
    script_dir = Path(__file__).parent
    
    # Determine which model to look for
    if "--stream" in sys.argv or "--stream-benchmark" in sys.argv:
        model_names = ["cnn_vit_lstm_radar.pt"]
    else:
        model_names = ["cnn_vit_radar.pt", "cnn_vit_lstm_radar.pt"]
    
    model_path = None
    for name in model_names:
        for p in [script_dir / name, script_dir.parent / name, Path(name)]:
            if p.exists():
                model_path = str(p)
                break
        if model_path:
            break
    
    if model_path is None:
        print(f"Error: Model not found")
        print(f"Looked for: {model_names}")
        return 1
    
    # Parse args
    threshold = 0.5
    if "--threshold" in sys.argv:
        idx = sys.argv.index("--threshold")
        threshold = float(sys.argv[idx + 1])
    
    # Handle streaming benchmark
    if "--stream-benchmark" in sys.argv:
        print(f"Loading streaming model: {model_path}")
        classifier = StreamingClassifier(model_path)
        print(f"Device: {classifier.device}")
        print(f"Mode: {classifier.mode}")
        print(f"Sequence length: {classifier.sequence_length}")
        
        print("\nRunning streaming benchmark...")
        results = classifier.benchmark()
        print(f"Latency: {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
        print(f"FPS: {results['fps']:.1f}")
        print(f"30 FPS capable: {'✓ YES' if results['fps'] > 30 else '✗ NO'}")
        return 0
    
    # Handle streaming mode
    if "--stream" in sys.argv:
        idx = sys.argv.index("--stream")
        if idx + 1 >= len(sys.argv):
            print("Error: --stream requires a folder path")
            return 1
        
        stream_path = Path(sys.argv[idx + 1])
        fps = 30
        if "--fps" in sys.argv:
            fps_idx = sys.argv.index("--fps")
            fps = int(sys.argv[fps_idx + 1])
        
        print(f"Loading streaming model: {model_path}")
        classifier = StreamingClassifier(model_path)
        print(f"Device: {classifier.device}")
        print(f"Simulating {fps} FPS stream")
        print("-" * 40)
        
        frame_delay = 1.0 / fps
        frames = sorted(list(stream_path.glob("*.png")) + list(stream_path.glob("*.jpg")))
        
        try:
            for frame_path in frames:
                result = classifier.update(frame_path)
                if result:
                    status = "✓" if result.is_reliable else "?"
                    print(f"[{status}] {result.prediction:12s} {result.confidence:5.1%} ({result.latency_ms:.1f}ms)")
                time.sleep(frame_delay)
        except KeyboardInterrupt:
            print("\nStopped.")
        
        return 0
    
    # Standard single-frame inference
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
