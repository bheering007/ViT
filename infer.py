"""
CNN-ViT Hybrid Inference Script for Radar Gesture Classification
Optimized for TI IWR6843ISK Range-Doppler maps
PRODUCTION-GRADE with confidence thresholds and input validation

Supports both single-frame and sequence (streaming) inference.

Usage:
    python infer.py <image_path>              # Single image
    python infer.py <folder_path>             # All images in folder
    python infer.py <image_path> --top 5      # Show top 5 predictions
    python infer.py --benchmark               # Run inference speed benchmark
    python infer.py --info                    # Show model info and metadata
    
    # Streaming mode (for sequence model)
    python infer.py --stream <folder_path>    # Stream through frames
    python infer.py --stream-benchmark        # Benchmark streaming speed
    
Exit codes:
    0 - Success
    1 - Error (file not found, invalid input, etc.)
    2 - Low confidence (prediction below threshold)
"""
import sys
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from enum import IntEnum
from collections import deque


class ExitCode(IntEnum):
    SUCCESS = 0
    ERROR = 1
    LOW_CONFIDENCE = 2


@dataclass
class PredictionResult:
    """Structured prediction result."""
    prediction: str
    confidence: float
    top_k: List[Tuple[str, float]]
    is_reliable: bool  # True if confidence >= threshold
    raw_logits: Optional[np.ndarray] = None


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
    
    def forward_features(self, x):
        """Extract spatial features without classification (for sequence model)."""
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
    """LSTM/GRU encoder for temporal modeling across frames."""
    
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, 
                 dropout=0.3, bidirectional=True, model_type='lstm'):
        super().__init__()
        
        self.model_type = model_type
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        
        if model_type == 'lstm':
            self.rnn = nn.LSTM(
                input_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional
            )
        elif model_type == 'gru':
            self.rnn = nn.GRU(
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
        
        self.config = config
        self.embed_dim = config.get("embed_dim", 256)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        spatial_features = self.spatial_encoder.forward_features(x)
        spatial_features = spatial_features.view(B, T, -1)
        temporal_features = self.temporal_encoder(spatial_features)
        return self.classifier(temporal_features)
    
    def forward_spatial_only(self, x):
        """Process single frame, return spatial features."""
        return self.spatial_encoder.forward_features(x)
    
    def forward_temporal_only(self, spatial_features):
        """Process sequence of spatial features."""
        temporal_features = self.temporal_encoder(spatial_features)
        return self.classifier(temporal_features)


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, list, dict]:
    """Load the trained model from checkpoint.
    
    Returns:
        model: The loaded model (CNNViTHybrid or CNNViTSequence)
        classes: List of class names
        metadata: Training metadata (confidence threshold, etc.)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    classes = ckpt["classes"]
    config = ckpt.get("model_config", {
        "num_classes": len(classes),
        "backbone": "resnet18",
        "embed_dim": 384,
        "num_heads": 6,
        "num_layers": 6,
        "dropout": 0.1,
        "mode": "single"
    })
    
    metadata = ckpt.get("metadata", {
        "confidence_threshold": 0.5,
        "trained_at": "unknown",
        "config_hash": "unknown",
    })
    
    # Check if this is a sequence model
    mode = config.get("mode", "single")
    
    if mode == "sequence":
        model = CNNViTSequence(num_classes=config["num_classes"], config=config)
    else:
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
    
    # Add mode to metadata for easy access
    metadata["mode"] = mode
    
    return model, classes, metadata


def validate_input(image_path: str) -> Tuple[bool, str]:
    """Validate input image file.
    
    Returns:
        (is_valid, error_message)
    """
    if not os.path.exists(image_path):
        return False, f"File not found: {image_path}"
    
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    ext = Path(image_path).suffix.lower()
    if ext not in valid_extensions:
        return False, f"Invalid file type: {ext}. Supported: {valid_extensions}"
    
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as e:
        return False, f"Corrupted image file: {e}"
    
    return True, ""


def get_transform():
    """Get the image transformation pipeline."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def predict_single(
    model: nn.Module, 
    image_path: str, 
    classes: list, 
    device: torch.device, 
    confidence_threshold: float = 0.5,
    top_k: int = 3
) -> Optional[PredictionResult]:
    """Run inference on a single image with validation.
    
    Args:
        model: The trained model
        image_path: Path to the input image
        classes: List of class names
        device: torch device
        confidence_threshold: Minimum confidence for reliable prediction
        top_k: Number of top predictions to return
    
    Returns:
        PredictionResult or None if error
    """
    # Validate input
    is_valid, error_msg = validate_input(image_path)
    if not is_valid:
        print(f"Input validation failed: {error_msg}")
        return None
    
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

    top_confidence = float(top_probs[0])
    
    return PredictionResult(
        prediction=classes[top_idx[0].item()],
        confidence=top_confidence,
        top_k=[(classes[idx.item()], float(prob)) for prob, idx in zip(top_probs, top_idx)],
        is_reliable=top_confidence >= confidence_threshold,
        raw_logits=logits.cpu().numpy()
    )


def predict_folder(
    model: nn.Module, 
    folder_path: str, 
    classes: list, 
    device: torch.device,
    confidence_threshold: float = 0.5
) -> Dict[str, PredictionResult]:
    """Run inference on all images in a folder."""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'}
    folder = Path(folder_path)
    
    results = {}
    for img_path in folder.iterdir():
        if img_path.suffix.lower() in image_extensions:
            result = predict_single(
                model, str(img_path), classes, device, 
                confidence_threshold=confidence_threshold, top_k=1
            )
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


class StreamingClassifier:
    """
    Real-time streaming classifier with sliding window buffer.
    
    Maintains a buffer of recent frames and produces predictions
    once enough frames are accumulated.
    
    Example:
        classifier = StreamingClassifier("cnn_vit_lstm_radar.pt")
        while True:
            frame = get_radar_frame()  # Your radar driver
            result = classifier.update(frame)
            if result:
                print(f"{result.prediction}: {result.confidence:.1%}")
    """
    
    def __init__(self, model_path: str, device: str = "auto", smoothing_window: int = 3):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.classes = ckpt["classes"]
        self.config = ckpt.get("model_config", {})
        self.sequence_length = self.config.get("sequence_length", 8)
        self.confidence_threshold = ckpt.get("metadata", {}).get("confidence_threshold", 0.5)
        
        # Build model
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
            self.sequence_length = 1  # Single frame mode
        
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = get_transform()
        
        # Streaming buffers
        self.feature_buffer = deque(maxlen=self.sequence_length)
        self.prediction_history = deque(maxlen=smoothing_window)
        self.frame_count = 0
        self.mode = mode
    
    def _preprocess(self, image_input) -> torch.Tensor:
        """Convert various input types to tensor."""
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
        elif isinstance(image_input, Image.Image):
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
            # Extract spatial features
            features = self.model.forward_spatial_only(tensor)
            self.feature_buffer.append(features)
            
            if len(self.feature_buffer) < self.sequence_length:
                return None
            
            # Run temporal model
            feature_seq = torch.cat(list(self.feature_buffer), dim=0).unsqueeze(0)
            logits = self.model.forward_temporal_only(feature_seq)
        else:
            # Single frame mode
            logits = self.model(tensor)
        
        probs = F.softmax(logits, dim=1)[0]
        
        # Smooth predictions
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
        """Clear all buffers."""
        self.feature_buffer.clear()
        self.prediction_history.clear()
        self.frame_count = 0
    
    def benchmark(self, n_frames: int = 100) -> dict:
        """Benchmark streaming performance."""
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
            "mean_latency_ms": np.mean(latencies) if latencies else 0,
            "std_latency_ms": np.std(latencies) if latencies else 0,
            "fps": 1000 / np.mean(latencies) if latencies else 0,
            "sequence_length": self.sequence_length,
            "mode": self.mode,
            "device": str(self.device)
        }


def main() -> int:
    """Main entry point. Returns exit code."""
    if len(sys.argv) < 2:
        print(__doc__)
        return ExitCode.ERROR

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Determine which model to load
    if "--sequence" in sys.argv or "--stream" in sys.argv or "--stream-benchmark" in sys.argv:
        checkpoint_path = "cnn_vit_lstm_radar.pt"
    else:
        checkpoint_path = "cnn_vit_radar.pt"
    
    # Allow explicit model path override
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            checkpoint_path = sys.argv[idx + 1]
    
    # Handle streaming benchmark
    if "--stream-benchmark" in sys.argv:
        print(f"\nLoading streaming model: {checkpoint_path}")
        try:
            classifier = StreamingClassifier(checkpoint_path)
        except FileNotFoundError:
            print(f"Error: Model not found: {checkpoint_path}")
            print("Train sequence model first: set mode='sequence' in train_vit.py")
            return ExitCode.ERROR
        
        print(f"Mode: {classifier.mode}")
        print(f"Sequence length: {classifier.sequence_length}")
        print("\nRunning streaming benchmark...")
        results = classifier.benchmark()
        
        print(f"\n{'='*50}")
        print("STREAMING BENCHMARK RESULTS")
        print(f"{'='*50}")
        print(f"Device: {results['device']}")
        print(f"Mode: {results['mode']}")
        print(f"Sequence length: {results['sequence_length']} frames")
        print(f"Mean latency: {results['mean_latency_ms']:.2f} ms")
        print(f"Std latency: {results['std_latency_ms']:.2f} ms")
        print(f"Throughput: {results['fps']:.1f} FPS")
        print(f"\nFor 30 FPS input: {'✓ PASS' if results['fps'] > 30 else '✗ FAIL'}")
        return ExitCode.SUCCESS
    
    # Handle streaming mode
    if "--stream" in sys.argv:
        idx = sys.argv.index("--stream")
        if idx + 1 >= len(sys.argv):
            print("Error: --stream requires a folder path")
            return ExitCode.ERROR
        
        stream_path = Path(sys.argv[idx + 1])
        if not stream_path.exists():
            print(f"Error: Path not found: {stream_path}")
            return ExitCode.ERROR
        
        fps = 30
        if "--fps" in sys.argv:
            fps_idx = sys.argv.index("--fps")
            if fps_idx + 1 < len(sys.argv):
                fps = int(sys.argv[fps_idx + 1])
        
        print(f"\nLoading streaming model: {checkpoint_path}")
        try:
            classifier = StreamingClassifier(checkpoint_path)
        except FileNotFoundError:
            print(f"Error: Model not found: {checkpoint_path}")
            return ExitCode.ERROR
        
        print(f"Mode: {classifier.mode}")
        print(f"Sequence length: {classifier.sequence_length}")
        print(f"Simulating {fps} FPS stream from: {stream_path}")
        print("Press Ctrl+C to stop\n")
        
        frame_delay = 1.0 / fps
        
        try:
            # Check if it's a class folder structure or flat folder
            subdirs = [d for d in stream_path.iterdir() if d.is_dir()]
            
            if subdirs:
                # Process each class folder
                for class_dir in sorted(subdirs):
                    print(f"\n--- {class_dir.name} ---")
                    classifier.reset()
                    
                    frames = sorted(list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")))
                    for frame_path in frames:
                        result = classifier.update(frame_path)
                        
                        if result:
                            status = "✓" if result.is_reliable else "?"
                            print(f"  [{status}] {result.prediction:12s} {result.confidence:5.1%} "
                                  f"({result.latency_ms:.1f}ms)")
                        
                        time.sleep(frame_delay)
            else:
                # Flat folder
                frames = sorted(list(stream_path.glob("*.png")) + list(stream_path.glob("*.jpg")))
                for frame_path in frames:
                    result = classifier.update(frame_path)
                    
                    if result:
                        status = "✓" if result.is_reliable else "?"
                        print(f"[{status}] {result.prediction:12s} {result.confidence:5.1%} "
                              f"({result.latency_ms:.1f}ms) - {frame_path.name}")
                    
                    time.sleep(frame_delay)
        
        except KeyboardInterrupt:
            print("\nStopped.")
        
        return ExitCode.SUCCESS
    
    # Load model for standard inference
    try:
        model, classes, metadata = load_model(checkpoint_path, device)
        print(f"Loaded model: {checkpoint_path}")
        print(f"Mode: {metadata.get('mode', 'single')}")
        print(f"Classes: {len(classes)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using train_vit.py")
        return ExitCode.ERROR
    
    confidence_threshold = metadata.get("confidence_threshold", 0.5)
    
    # Handle --info flag
    if "--info" in sys.argv:
        print("\n" + "=" * 50)
        print("MODEL INFORMATION")
        print("=" * 50)
        print(f"Model: {checkpoint_path}")
        print(f"Mode: {metadata.get('mode', 'single')}")
        print(f"Classes: {classes}")
        print(f"Confidence threshold: {confidence_threshold}")
        print(f"Trained at: {metadata.get('trained_at', 'unknown')}")
        print(f"Config hash: {metadata.get('config_hash', 'unknown')}")
        print(f"PyTorch version: {metadata.get('pytorch_version', 'unknown')}")
        print(f"Best val accuracy: {metadata.get('best_val_accuracy', 'unknown')}")
        print(f"Train samples: {metadata.get('train_samples', 'unknown')}")
        print(f"Val samples: {metadata.get('val_samples', 'unknown')}")
        return ExitCode.SUCCESS
    
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
        return ExitCode.SUCCESS
    
    input_path = sys.argv[1]
    top_k = 3
    
    # Parse optional arguments
    if "--top" in sys.argv:
        idx = sys.argv.index("--top")
        if idx + 1 < len(sys.argv):
            top_k = int(sys.argv[idx + 1])
    
    print(f"Confidence threshold: {confidence_threshold}")
    print("-" * 50)
    
    # Check if input is file or folder
    if os.path.isfile(input_path):
        result = predict_single(
            model, input_path, classes, device, 
            confidence_threshold=confidence_threshold, top_k=top_k
        )
        if result is None:
            return ExitCode.ERROR
            
        print(f"Image: {input_path}")
        print(f"Prediction: {result.prediction}")
        print(f"Confidence: {result.confidence:.4f}")
        
        # Reliability indicator
        if result.is_reliable:
            print(f"Status: ✓ RELIABLE (above {confidence_threshold:.0%} threshold)")
        else:
            print(f"Status: ⚠ LOW CONFIDENCE (below {confidence_threshold:.0%} threshold)")
        
        print(f"\nTop {len(result.top_k)} predictions:")
        for cls, prob in result.top_k:
            bar = "█" * int(prob * 20)
            print(f"  {cls:12s}: {prob:.4f} {bar}")
        
        return ExitCode.SUCCESS if result.is_reliable else ExitCode.LOW_CONFIDENCE
    
    elif os.path.isdir(input_path):
        results = predict_folder(model, input_path, classes, device, confidence_threshold)
        print(f"Processed {len(results)} images from {input_path}")
        
        # Count reliable vs unreliable
        reliable = sum(1 for r in results.values() if r.is_reliable)
        unreliable = len(results) - reliable
        print(f"Reliable: {reliable} | Low confidence: {unreliable}\n")
        
        # Group by prediction
        from collections import defaultdict
        by_class = defaultdict(list)
        for img_name, result in results.items():
            by_class[result.prediction].append((img_name, result.confidence, result.is_reliable))
        
        for cls in sorted(by_class.keys()):
            items = by_class[cls]
            print(f"{cls} ({len(items)} images):")
            for img_name, conf, is_rel in sorted(items, key=lambda x: -x[1])[:5]:
                status = "✓" if is_rel else "⚠"
                print(f"  {status} {img_name}: {conf:.4f}")
            if len(items) > 5:
                print(f"  ... and {len(items) - 5} more")
            print()
        
        return ExitCode.SUCCESS
    
    else:
        print(f"Error: Path not found: {input_path}")
        return ExitCode.ERROR


if __name__ == "__main__":
    sys.exit(main())
