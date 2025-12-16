import os
import sys
import time
import hashlib
import json
import random
import logging
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import timm
import torch.multiprocessing as mp
from pathlib import Path
from collections import defaultdict, deque
from typing import Optional, List, Tuple
from PIL import Image

# ==================== LOGGING SETUP ====================
def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """Configure logging for training."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode='a'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)


# Model version - increment when architecture changes
MODEL_VERSION = "2.0.0"


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CNNViTHybrid(nn.Module):
    """
    Hybrid CNN-ViT model optimized for Range-Doppler radar classification.
    
    Architecture:
    - CNN backbone extracts local spatial features (range-doppler patterns)
    - Transformer encoder captures global temporal/spatial relationships
    - Designed for TI IWR6843ISK radar sensor data
    
    For real-time inference, use backbone='mobilenetv3' for ~3x speedup.
    For maximum accuracy, use backbone='resnet34' or 'efficientnet_b2'.
    """
    
    # Backbone configurations: (model_name, output_channels, feature_map_size)
    BACKBONES = {
        'mobilenetv3': ('mobilenetv3_small_100', 576, 7),   # Fastest - ~2ms inference
        'resnet18': ('resnet18', 512, 7),                    # Balanced - ~4ms inference
        'resnet34': ('resnet34', 512, 7),                    # More accurate - ~6ms inference
        'efficientnet_b0': ('efficientnet_b0', 1280, 7),     # Efficient - ~5ms inference
        'efficientnet_b2': ('efficientnet_b2', 1408, 7),     # High accuracy - ~8ms inference
    }
    
    def __init__(self, num_classes, embed_dim=384, num_heads=6, num_layers=6, 
                 dropout=0.1, pretrained_backbone=True, backbone='resnet34'):
        super().__init__()
        
        self.backbone_name = backbone
        if backbone not in self.BACKBONES:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from {list(self.BACKBONES.keys())}")
        
        model_name, cnn_out_channels, feat_size = self.BACKBONES[backbone]
        
        # CNN Backbone - remove classification head
        base_model = timm.create_model(model_name, pretrained=pretrained_backbone, features_only=True)
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
        
        # Classification head with intermediate layer for better features
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        B = x.shape[0]
        
        # CNN feature extraction - get last feature map
        features = self.cnn_backbone(x)
        x = features[-1]  # Last feature map
        
        # Project to embed_dim
        x = self.patch_embed(x)
        
        # Flatten spatial dims to sequence: (B, C, H, W) -> (B, H*W, C)
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
        return self.norm(x[:, 0])  # Return cls token features


class TemporalEncoder(nn.Module):
    """
    LSTM/GRU encoder for temporal modeling across frames.
    Used in sequence mode for real-time streaming.
    """
    
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
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
    
    def forward(self, x):
        """
        Args:
            x: [B, seq_len, input_dim] sequence of frame features
        Returns:
            [B, output_dim] sequence-level feature
        """
        output, _ = self.rnn(x)
        if self.bidirectional:
            forward_last = output[:, -1, :self.hidden_dim]
            backward_first = output[:, 0, self.hidden_dim:]
            return torch.cat([forward_last, backward_first], dim=1)
        return output[:, -1, :]


class CNNViTSequence(nn.Module):
    """
    Sequence model: CNN-ViT per frame + LSTM across frames.
    
    For real-time streaming inference with temporal consistency.
    Processes sequences of Range-Doppler frames.
    
    Architecture:
        Input: [B, seq_len, C, H, W]
        → CNNViTHybrid.forward_features (per frame): [B, seq_len, embed_dim]
        → TemporalEncoder (across frames): [B, temporal_dim]
        → Classifier: [B, num_classes]
    """
    
    def __init__(self, num_classes, config):
        super().__init__()
        
        # Spatial encoder (CNN-ViT for each frame)
        self.spatial_encoder = CNNViTHybrid(
            num_classes=num_classes,
            backbone=config["backbone"],
            embed_dim=config["embed_dim"],
            num_heads=config["num_heads"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            pretrained_backbone=True
        )
        
        # Temporal encoder (LSTM across frames)
        self.temporal_encoder = TemporalEncoder(
            input_dim=config["embed_dim"],
            hidden_dim=config.get("temporal_hidden", 256),
            num_layers=config.get("temporal_layers", 2),
            dropout=config.get("temporal_dropout", 0.3),
            bidirectional=config.get("temporal_bidirectional", True),
            model_type=config.get("temporal_model", "lstm")
        )
        
        # Classification head
        temporal_dim = self.temporal_encoder.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim // 2),
            nn.GELU(),
            nn.Dropout(config.get("classifier_dropout", 0.4)),
            nn.Linear(temporal_dim // 2, num_classes)
        )
        
        self.config = config
        self.embed_dim = config["embed_dim"]
    
    def forward(self, x):
        """
        Args:
            x: [B, seq_len, C, H, W] sequence of frames
        Returns:
            [B, num_classes] logits
        """
        B, T, C, H, W = x.shape
        
        # Process each frame through spatial encoder
        x = x.view(B * T, C, H, W)
        spatial_features = self.spatial_encoder.forward_features(x)
        spatial_features = spatial_features.view(B, T, -1)
        
        # Process sequence through temporal encoder
        temporal_features = self.temporal_encoder(spatial_features)
        
        return self.classifier(temporal_features)
    
    def forward_spatial_only(self, x):
        """Process single frame, return spatial features (for streaming)."""
        return self.spatial_encoder.forward_features(x)
    
    def forward_temporal_only(self, spatial_features):
        """Process sequence of spatial features (for streaming)."""
        temporal_features = self.temporal_encoder(spatial_features)
        return self.classifier(temporal_features)
    
    def freeze_backbone(self):
        """Freeze CNN backbone for initial training."""
        for param in self.spatial_encoder.cnn_backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze CNN backbone for fine-tuning."""
        for param in self.spatial_encoder.cnn_backbone.parameters():
            param.requires_grad = True


class RadarSequenceDataset(Dataset):
    """
    Dataset that creates sequences from radar frames.
    
    Supports TWO folder structures:
    
    1. HIERARCHICAL (each sample is a folder with frames):
        data_dir/split/class/sample_001/frames/frame_0001.png, frame_0002.png, ...
        data_dir/split/class/sample_002/frames/frame_0001.png, frame_0002.png, ...
    
    2. FLAT (all frames in class folder, create sliding windows):
        data_dir/split/class/frame_0001.png, frame_0002.png, ...
    
    Auto-detects structure based on folder contents.
    """
    
    def __init__(self, root_dir: str, split: str, sequence_length: int,
                 stride: int = 1, transform=None, temporal_augment: bool = False):
        self.root = Path(root_dir) / split
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.temporal_augment = temporal_augment
        
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.sequences = []
        self._build_sequences()
    
    def _detect_structure(self, class_dir: Path) -> str:
        """Detect if class uses hierarchical (sample folders) or flat structure."""
        # Check for direct image files
        direct_images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        
        # Check for subdirectories (sample folders)
        subdirs = [d for d in class_dir.iterdir() if d.is_dir()]
        
        if subdirs and not direct_images:
            return "hierarchical"
        elif direct_images and not subdirs:
            return "flat"
        elif subdirs:
            # Both exist - prefer hierarchical if subdirs have frames
            sample_dir = subdirs[0]
            # Check if frames are in subfolder or directly in sample dir
            frames_subdir = sample_dir / "frames"
            if frames_subdir.exists():
                return "hierarchical"
            sample_images = list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg"))
            if sample_images:
                return "hierarchical"
        return "flat"
    
    def _get_frames_from_sample(self, sample_dir: Path) -> list:
        """Get sorted frame paths from a sample directory."""
        # Check for frames subfolder first (MATLAB generator structure)
        frames_subdir = sample_dir / "frames"
        if frames_subdir.exists():
            search_dir = frames_subdir
        else:
            search_dir = sample_dir
        
        frames = sorted(list(search_dir.glob("*.png")) + list(search_dir.glob("*.jpg")))
        return frames
    
    def _build_sequences(self):
        """Build list of valid sequences from available frames."""
        for class_name in self.classes:
            class_dir = self.root / class_name
            structure = self._detect_structure(class_dir)
            
            if structure == "hierarchical":
                # Each subdirectory is a complete sequence/sample
                sample_dirs = sorted([d for d in class_dir.iterdir() if d.is_dir()])
                
                for sample_dir in sample_dirs:
                    frames = self._get_frames_from_sample(sample_dir)
                    
                    if len(frames) == 0:
                        continue
                    
                    if len(frames) < self.sequence_length:
                        # Pad by repeating frames
                        self.sequences.append({
                            'frames': frames,
                            'class': class_name,
                            'class_idx': self.class_to_idx[class_name],
                            'repeat': True,
                            'sample_id': sample_dir.name
                        })
                    elif len(frames) == self.sequence_length:
                        # Perfect match - one sequence per sample
                        self.sequences.append({
                            'frames': frames,
                            'class': class_name,
                            'class_idx': self.class_to_idx[class_name],
                            'repeat': False,
                            'sample_id': sample_dir.name
                        })
                    else:
                        # More frames than needed - create sliding windows
                        for i in range(0, len(frames) - self.sequence_length + 1, self.stride):
                            self.sequences.append({
                                'frames': frames[i:i + self.sequence_length],
                                'class': class_name,
                                'class_idx': self.class_to_idx[class_name],
                                'repeat': False,
                                'sample_id': f"{sample_dir.name}_win{i}"
                            })
            else:
                # Flat structure - all frames in class directory
                frames = sorted(list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg")))
                
                if len(frames) < self.sequence_length:
                    if len(frames) > 0:
                        self.sequences.append({
                            'frames': frames,
                            'class': class_name,
                            'class_idx': self.class_to_idx[class_name],
                            'repeat': True,
                            'sample_id': 'flat_0'
                        })
                else:
                    for i in range(0, len(frames) - self.sequence_length + 1, self.stride):
                        self.sequences.append({
                            'frames': frames[i:i + self.sequence_length],
                            'class': class_name,
                            'class_idx': self.class_to_idx[class_name],
                            'repeat': False,
                            'sample_id': f'flat_{i}'
                        })
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        frames = seq_info['frames'].copy()
        
        if seq_info['repeat']:
            while len(frames) < self.sequence_length:
                frames = frames + frames
            frames = frames[:self.sequence_length]
        
        images = []
        for frame_path in frames:
            img = Image.open(frame_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        sequence = torch.stack(images, dim=0)
        
        if self.temporal_augment and random.random() < 0.1:
            swap_idx = random.randint(0, self.sequence_length - 2)
            sequence[[swap_idx, swap_idx + 1]] = sequence[[swap_idx + 1, swap_idx]]
        
        return sequence, seq_info['class_idx']


# ==================== CONFIGURATION ====================
# Optimized for TI IWR6843ISK Range-Doppler maps
# Modify these values to tune the model
CONFIG = {
    # Reproducibility
    "seed": 42,                 # Random seed for reproducibility
    
    # Training mode: "single" for per-frame, "sequence" for temporal
    "mode": "sequence",           # Options: "single", "sequence"
    
    # Sequence settings (only used when mode="sequence")
    "sequence_length": 16,      # Number of frames per sequence (your data has 32)
    "sequence_stride": 8,       # Stride between sequences (creates ~3 samples per folder)
    "temporal_model": "lstm",   # Options: "lstm", "gru"
    "temporal_hidden": 256,     # Hidden dim for temporal encoder
    "temporal_layers": 2,       # Number of LSTM/GRU layers
    "temporal_dropout": 0.3,    # Dropout in temporal encoder
    "temporal_bidirectional": True,  # Use bidirectional LSTM
    "classifier_dropout": 0.4,  # Dropout before final classifier
    "backbone_frozen_epochs": 3, # Freeze CNN backbone for first N epochs
    
    # Data
    "data_dir": "splits",
    "image_size": 224,
    "batch_size": 32,           # Use 16 for sequence mode (more memory)
    "num_workers": 4,
    
    # Model architecture
    "backbone": "resnet18",     # Options: mobilenetv3, resnet18, resnet34, efficientnet_b0, efficientnet_b2
                                # Use mobilenetv3 for real-time (~2ms), resnet34/efficientnet for accuracy
    "embed_dim": 256,           # Reduced for small dataset (less overfitting)
    "num_heads": 4,             # Attention heads (must divide embed_dim evenly)
    "num_layers": 4,            # Reduced transformer layers (less overfitting)
    "dropout": 0.2,             # Moderate dropout
    
    # Training
    "epochs": 100,              # Max epochs (early stopping will likely trigger first)
    "learning_rate": 2e-4,      # Balanced LR
    "weight_decay": 0.05,       # Moderate L2 regularization
    "label_smoothing": 0.05,    # Back to normal - 0.2 was too aggressive
    "early_stopping_patience": 15, # More patience with regularization
    "warmup_epochs": 5,         # LR warmup epochs
    "gradient_clip": 1.0,       # Gradient clipping max norm
    "resume_checkpoint": None,  # Path to checkpoint to resume from (or None)
    
    # Range-Doppler specific augmentation
    # NOTE: Horizontal flip is DISABLED - flipping range is physically meaningless
    "horizontal_flip": False,   # DO NOT enable for range-doppler maps!
    "vertical_flip": False,     # Flipping doppler (velocity) is also meaningless
    "random_rotation": 0,       # Rotation doesn't make physical sense for radar
    "noise_factor": 0.05,       # More noise for robustness (small dataset needs more augmentation)
    "random_crop": True,        # Random crop helps with slight spatial variations
    
    # Production settings
    "confidence_threshold": 0.5, # Reject predictions below this confidence (medical safety)
    "compile_model": False,      # Use torch.compile() for faster inference (requires PyTorch 2.0+)
    
    # Output
    "save_path": "cnn_vit_radar.pt",       # Single-frame model
    "sequence_save_path": "cnn_vit_lstm_radar.pt",  # Sequence model
    
    # Production options
    "log_file": "training.log",  # Log file path (None to disable)
    "evaluate_test": True,       # Run final evaluation on test set
    "export_onnx": False,        # Export to ONNX after training
    "save_optimizer": True,      # Save optimizer state for resuming
}


def get_config_hash(config: dict) -> str:
    """Generate a hash of the config for tracking."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


class AddGaussianNoise:
    """Add Gaussian noise to tensor (simulates radar sensor noise)."""
    def __init__(self, mean=0., std=0.02):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        if self.std > 0:
            noise = torch.randn_like(tensor) * self.std + self.mean
            return tensor + noise
        return tensor


def parse_args():
    """Parse command line arguments for easy mode switching."""
    parser = argparse.ArgumentParser(description='Train CNN-ViT Radar Classifier')
    parser.add_argument('--mode', choices=['single', 'sequence'], 
                        help='Training mode (overrides CONFIG)')
    parser.add_argument('--resume', type=str, 
                        help='Path to checkpoint to resume training')
    parser.add_argument('--epochs', type=int, 
                        help='Number of epochs (overrides CONFIG)')
    parser.add_argument('--lr', type=float, 
                        help='Learning rate (overrides CONFIG)')
    parser.add_argument('--data', type=str, 
                        help='Data directory (overrides CONFIG)')
    parser.add_argument('--no-test', action='store_true',
                        help='Skip test set evaluation')
    parser.add_argument('--export-onnx', action='store_true',
                        help='Export model to ONNX after training')
    return parser.parse_args()


def export_onnx(model, save_path: str, config: dict, device: torch.device):
    """Export model to ONNX format for optimized inference."""
    model.eval()
    onnx_path = save_path.replace('.pt', '.onnx')
    
    if config["mode"] == "sequence":
        dummy = torch.randn(1, config["sequence_length"], 3, config["image_size"], config["image_size"]).to(device)
    else:
        dummy = torch.randn(1, 3, config["image_size"], config["image_size"]).to(device)
    
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=14
    )
    return onnx_path


def evaluate_test_set(model, data_path: Path, transform, config: dict, 
                      device: torch.device, classes: list, logger) -> dict:
    """Evaluate model on held-out test set."""
    test_path = data_path / "test"
    if not test_path.exists():
        logger.warning(f"Test directory not found: {test_path}")
        return {}
    
    if config["mode"] == "sequence":
        test_ds = RadarSequenceDataset(
            str(data_path), "test",
            sequence_length=config["sequence_length"],
            stride=config["sequence_length"],
            transform=transform,
            temporal_augment=False
        )
    else:
        test_ds = datasets.ImageFolder(str(test_path), transform=transform)
    
    if len(test_ds) == 0:
        logger.warning("Test dataset is empty!")
        return {}
    
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)
    
    model.eval()
    correct = total = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = correct / total
    
    # Per-class metrics
    num_classes = len(classes)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(all_labels, all_preds):
        confusion[true_label][pred_label] += 1
    
    return {
        "accuracy": accuracy,
        "total_samples": total,
        "confusion_matrix": confusion.tolist(),
        "predictions": all_preds,
        "labels": all_labels
    }


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Override CONFIG with command line arguments
    if args.mode:
        CONFIG["mode"] = args.mode
    if args.resume:
        CONFIG["resume_checkpoint"] = args.resume
    if args.epochs:
        CONFIG["epochs"] = args.epochs
    if args.lr:
        CONFIG["learning_rate"] = args.lr
    if args.data:
        CONFIG["data_dir"] = args.data
    if args.no_test:
        CONFIG["evaluate_test"] = False
    if args.export_onnx:
        CONFIG["export_onnx"] = True
    
    # Setup logging
    logger = setup_logging(CONFIG.get("log_file"))
    
    # Set seed for reproducibility
    set_seed(CONFIG["seed"])
    logger.info(f"{'='*60}")
    logger.info(f"CNN-ViT Radar Classifier v{MODEL_VERSION}")
    logger.info(f"{'='*60}")
    logger.info(f"Random seed: {CONFIG['seed']}")
    logger.info(f"Config hash: {get_config_hash(CONFIG)}")
    logger.info(f"Training mode: {CONFIG['mode'].upper()}")
    
    # Validate data directory exists
    data_path = Path(CONFIG["data_dir"])
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path.absolute()}")
    
    for split in ["train", "val"]:
        split_path = data_path / split
        if not split_path.exists():
            raise FileNotFoundError(f"Split directory not found: {split_path.absolute()}")
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, training will be slow!")
    
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Build transforms for Range-Doppler maps
    train_transforms = [
        transforms.Grayscale(num_output_channels=3),
    ]
    
    if CONFIG["random_crop"]:
        train_transforms.extend([
            transforms.Resize((int(CONFIG["image_size"] * 1.1), int(CONFIG["image_size"] * 1.1))),
            transforms.RandomCrop((CONFIG["image_size"], CONFIG["image_size"])),
        ])
    else:
        train_transforms.append(transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])))
    
    if CONFIG["horizontal_flip"]:
        train_transforms.append(transforms.RandomHorizontalFlip())
    if CONFIG["vertical_flip"]:
        train_transforms.append(transforms.RandomVerticalFlip())
    if CONFIG["random_rotation"] > 0:
        train_transforms.append(transforms.RandomRotation(CONFIG["random_rotation"]))
    
    train_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    if CONFIG["noise_factor"] > 0:
        train_transforms.append(AddGaussianNoise(std=CONFIG["noise_factor"]))
    
    train_tf = transforms.Compose(train_transforms)

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # ==================== MODE-SPECIFIC SETUP ====================
    if CONFIG["mode"] == "sequence":
        # Sequence mode: temporal training
        batch_size = min(CONFIG["batch_size"], 16)  # Reduce for memory
        
        train_ds = RadarSequenceDataset(
            CONFIG["data_dir"], "train",
            sequence_length=CONFIG["sequence_length"],
            stride=CONFIG["sequence_stride"],
            transform=train_tf,
            temporal_augment=True
        )
        val_ds = RadarSequenceDataset(
            CONFIG["data_dir"], "val",
            sequence_length=CONFIG["sequence_length"],
            stride=CONFIG["sequence_length"],  # No overlap for validation
            transform=eval_tf,
            temporal_augment=False
        )
        
        classes = train_ds.classes
        logger.info(f"Sequence length: {CONFIG['sequence_length']} frames")
        logger.info(f"Temporal model: {CONFIG['temporal_model'].upper()}")
        
        model = CNNViTSequence(
            num_classes=len(classes),
            config=CONFIG
        ).to(device)
        
        save_path = CONFIG["sequence_save_path"]
        
    else:
        # Single-frame mode (original)
        batch_size = CONFIG["batch_size"]
        
        train_ds = datasets.ImageFolder(str(data_path / "train"), transform=train_tf)
        val_ds = datasets.ImageFolder(str(data_path / "val"), transform=eval_tf)
        classes = train_ds.classes
        
        model = CNNViTHybrid(
            num_classes=len(classes),
            backbone=CONFIG["backbone"],
            embed_dim=CONFIG["embed_dim"],
            num_heads=CONFIG["num_heads"],
            num_layers=CONFIG["num_layers"],
            dropout=CONFIG["dropout"],
            pretrained_backbone=True
        ).to(device)
        
        save_path = CONFIG["save_path"]
    
    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty!")
    if len(val_ds) == 0:
        raise ValueError("Validation dataset is empty!")

    # DataLoaders
    use_persistent = CONFIG["num_workers"] > 0
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=CONFIG["num_workers"], 
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=CONFIG["num_workers"], 
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent
    )

    logger.info(f"Classes: {classes}")
    logger.info(f"Num classes: {len(classes)}")
    logger.info(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Backbone: {CONFIG['backbone']}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Benchmark inference speed
    model.eval()
    if CONFIG["mode"] == "sequence":
        dummy_input = torch.randn(1, CONFIG["sequence_length"], 3, CONFIG["image_size"], CONFIG["image_size"]).to(device)
    else:
        dummy_input = torch.randn(1, 3, CONFIG["image_size"], CONFIG["image_size"]).to(device)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    num_runs = 100
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    avg_inference_ms = (end_time - start_time) / num_runs * 1000
    fps = 1000 / avg_inference_ms
    logger.info(f"Inference speed: {avg_inference_ms:.2f} ms/{'sequence' if CONFIG['mode'] == 'sequence' else 'image'} ({fps:.1f} FPS)")
    
    if CONFIG["compile_model"] and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile()...")
        model = torch.compile(model)
        logger.info("Model compiled successfully")
    
    logger.info("-" * 60)

    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=CONFIG["learning_rate"], 
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < CONFIG["warmup_epochs"]:
            return (epoch + 1) / CONFIG["warmup_epochs"]
        else:
            progress = (epoch - CONFIG["warmup_epochs"]) / max(1, CONFIG["epochs"] - CONFIG["warmup_epochs"])
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if CONFIG["resume_checkpoint"] and os.path.exists(CONFIG["resume_checkpoint"]):
        logger.info(f"Resuming from checkpoint: {CONFIG['resume_checkpoint']}")
        ckpt = torch.load(CONFIG["resume_checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt.get("metadata", {}):
            start_epoch = ckpt["metadata"]["epoch"]
        if "best_val_accuracy" in ckpt.get("metadata", {}):
            best_acc = ckpt["metadata"]["best_val_accuracy"]
        logger.info(f"Resumed from epoch {start_epoch} with best acc {best_acc:.3f}")

    best_acc = 0.0
    best_preds = []
    best_labels = []
    epochs_without_improvement = 0
    
    for epoch in range(start_epoch, CONFIG["epochs"]):
        # Freeze/unfreeze backbone for sequence mode
        if CONFIG["mode"] == "sequence":
            if epoch < CONFIG["backbone_frozen_epochs"]:
                model.freeze_backbone()
            else:
                model.unfreeze_backbone()
        
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
            
            scaler.step(opt)
            scaler.update()
            
            train_loss_sum += loss.item() * y.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)
        
        scheduler.step()

        # Validation
        model.eval()
        correct = total = 0
        val_loss_sum = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits = model(x)
                    vloss = loss_fn(logits, y)

                val_loss_sum += vloss.item() * y.size(0)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        acc = correct / total
        val_loss = val_loss_sum / total
        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        current_lr = scheduler.get_last_lr()[0]
        
        logger.info(f"Epoch {epoch+1:2d}/{CONFIG['epochs']} | "
              f"LR {current_lr:.2e} | "
              f"train loss {train_loss:.4f} | train acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} | val acc {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            best_preds = all_preds
            best_labels = all_labels
            epochs_without_improvement = 0
            
            # Build save dict based on mode
            save_dict = {
                "model": model.state_dict(),
                "classes": classes,
                "model_config": {
                    "num_classes": len(classes),
                    "backbone": CONFIG["backbone"],
                    "embed_dim": CONFIG["embed_dim"],
                    "num_heads": CONFIG["num_heads"],
                    "num_layers": CONFIG["num_layers"],
                    "dropout": CONFIG["dropout"],
                    "mode": CONFIG["mode"],
                },
                "metadata": {
                    "model_version": MODEL_VERSION,
                    "trained_at": datetime.now().isoformat(),
                    "config_hash": get_config_hash(CONFIG),
                    "pytorch_version": torch.__version__,
                    "best_val_accuracy": acc,
                    "epoch": epoch + 1,
                    "train_samples": len(train_ds),
                    "val_samples": len(val_ds),
                    "confidence_threshold": CONFIG["confidence_threshold"],
                    "seed": CONFIG["seed"],
                }
            }
            
            # Save optimizer state for resuming
            if CONFIG["save_optimizer"]:
                save_dict["optimizer"] = opt.state_dict()
            
            # Add sequence-specific config
            if CONFIG["mode"] == "sequence":
                save_dict["model_config"].update({
                    "sequence_length": CONFIG["sequence_length"],
                    "temporal_model": CONFIG["temporal_model"],
                    "temporal_hidden": CONFIG["temporal_hidden"],
                    "temporal_layers": CONFIG["temporal_layers"],
                    "temporal_dropout": CONFIG["temporal_dropout"],
                    "temporal_bidirectional": CONFIG["temporal_bidirectional"],
                    "classifier_dropout": CONFIG["classifier_dropout"],
                })
            
            torch.save(save_dict, save_path)
            logger.info(f"  -> Saved new best model with acc {acc:.3f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= CONFIG["early_stopping_patience"]:
                logger.info(f"Early stopping triggered after {epoch+1} epochs (no improvement for {CONFIG['early_stopping_patience']} epochs)")
                break

    # ==================== FINAL RESULTS ====================
    logger.info("=" * 60)
    logger.info(f"TRAINING COMPLETE ({CONFIG['mode'].upper()} MODE)")
    logger.info("=" * 60)
    logger.info(f"Best validation accuracy: {best_acc:.3f}")
    logger.info(f"Model saved to: {save_path}")
    
    # Print confusion matrix and per-class metrics
    logger.info("=" * 60)
    logger.info("PER-CLASS METRICS (on best model's validation)")
    logger.info("=" * 60)
    
    num_classes = len(classes)
    
    # Build confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(best_labels, best_preds):
        confusion[true_label][pred_label] += 1
    
    # Calculate per-class metrics
    metrics_header = f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
    logger.info(metrics_header)
    logger.info("-" * 54)
    
    precisions, recalls, f1s, supports = [], [], [], []
    for i, cls_name in enumerate(classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        support = confusion[i, :].sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        
        logger.info(f"{cls_name:<12} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10}")
    
    # Macro averages
    logger.info("-" * 54)
    logger.info(f"{'Macro Avg':<12} {np.mean(precisions):>10.3f} {np.mean(recalls):>10.3f} {np.mean(f1s):>10.3f} {sum(supports):>10}")
    
    # Print confusion matrix
    logger.info("=" * 60)
    logger.info("CONFUSION MATRIX")
    logger.info("=" * 60)
    logger.info("(rows = actual, columns = predicted)")
    
    # Header
    header = "         " + " ".join([f"{c[:6]:>6}" for c in classes])
    logger.info(header)
    logger.info("-" * len(header))
    
    for i, cls_name in enumerate(classes):
        row = f"{cls_name[:8]:<8} " + " ".join([f"{confusion[i,j]:>6}" for j in range(num_classes)])
        logger.info(row)
    
    # Highlight most confused pairs
    logger.info("Most confused pairs (excluding correct):")
    confusions_list = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion[i, j] > 0:
                confusions_list.append((confusion[i, j], classes[i], classes[j]))
    
    confusions_list.sort(reverse=True)
    for count, true_cls, pred_cls in confusions_list[:5]:
        logger.info(f"  {true_cls} -> {pred_cls}: {count} times")

    # ==================== TEST SET EVALUATION ====================
    if CONFIG["evaluate_test"]:
        logger.info("=" * 60)
        logger.info("TEST SET EVALUATION")
        logger.info("=" * 60)
        
        # Load best model
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        
        test_results = evaluate_test_set(model, data_path, eval_tf, CONFIG, device, classes, logger)
        
        if test_results:
            logger.info(f"Test accuracy: {test_results['accuracy']:.3f}")
            logger.info(f"Test samples: {test_results['total_samples']}")
        else:
            logger.warning("Test set evaluation skipped (no test data found)")
    
    # ==================== ONNX EXPORT ====================
    if CONFIG["export_onnx"]:
        logger.info("=" * 60)
        logger.info("EXPORTING TO ONNX")
        logger.info("=" * 60)
        
        onnx_path = export_onnx(model, save_path, CONFIG, device)
        if onnx_path:
            logger.info(f"ONNX model saved to: {onnx_path}")
    
    logger.info("=" * 60)
    logger.info("ALL DONE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    mp.freeze_support()
    main()
