import os
import time
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import torch.multiprocessing as mp
from pathlib import Path
from collections import defaultdict


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


# ==================== CONFIGURATION ====================
# Optimized for TI IWR6843ISK Range-Doppler maps
# Modify these values to tune the model
CONFIG = {
    # Data
    "data_dir": "splits",
    "image_size": 224,
    "batch_size": 32,           # Increase to 64 if GPU memory allows
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
    "label_smoothing": 0.05,     # Back to normal - 0.2 was too aggressive
    "early_stopping_patience": 15, # More patience with regularization
    
    # Range-Doppler specific augmentation
    # NOTE: Horizontal flip is DISABLED - flipping range is physically meaningless
    "horizontal_flip": False,   # DO NOT enable for range-doppler maps!
    "vertical_flip": False,     # Flipping doppler (velocity) is also meaningless
    "random_rotation": 0,       # Rotation doesn't make physical sense for radar
    "noise_factor": 0.05,       # More noise for robustness (small dataset needs more augmentation)
    "random_crop": True,        # Random crop helps with slight spatial variations
    
    # Output
    "save_path": "cnn_vit_radar.pt",
}


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


def main():
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
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available, training will be slow!")
    
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Build transforms for Range-Doppler maps
    train_transforms = [
        transforms.Grayscale(num_output_channels=3),
    ]
    
    # Random crop with resize (spatial augmentation that makes physical sense)
    if CONFIG["random_crop"]:
        train_transforms.extend([
            transforms.Resize((int(CONFIG["image_size"] * 1.1), int(CONFIG["image_size"] * 1.1))),
            transforms.RandomCrop((CONFIG["image_size"], CONFIG["image_size"])),
        ])
    else:
        train_transforms.append(transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])))
    
    # Flip augmentations (disabled by default for range-doppler)
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
    
    # Add sensor noise simulation
    if CONFIG["noise_factor"] > 0:
        train_transforms.append(AddGaussianNoise(std=CONFIG["noise_factor"]))
    
    train_tf = transforms.Compose(train_transforms)

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_ds = datasets.ImageFolder(str(data_path / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(data_path / "val"), transform=eval_tf)
    
    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty!")
    if len(val_ds) == 0:
        raise ValueError("Validation dataset is empty!")

    # DataLoaders with persistent workers for efficiency
    use_persistent = CONFIG["num_workers"] > 0
    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=CONFIG["num_workers"], 
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent,
        drop_last=True  # Avoid batch size of 1 issues with BatchNorm
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        num_workers=CONFIG["num_workers"], 
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_persistent
    )

    print(f"Classes: {train_ds.classes}")
    print(f"Num classes: {len(train_ds.classes)}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Create hybrid CNN-ViT model
    model = CNNViTHybrid(
        num_classes=len(train_ds.classes),
        backbone=CONFIG["backbone"],
        embed_dim=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
        pretrained_backbone=True
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Backbone: {CONFIG['backbone']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Benchmark inference speed (important for real-time)
    model.eval()
    dummy_input = torch.randn(1, 3, CONFIG["image_size"], CONFIG["image_size"]).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
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
    print(f"Inference speed: {avg_inference_ms:.2f} ms/image ({fps:.1f} FPS)")
    print("-" * 60)

    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=CONFIG["learning_rate"], 
        weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CONFIG["epochs"])
    loss_fn = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    # Use new torch.amp API (not deprecated torch.cuda.amp)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == "cuda"))

    best_acc = 0.0
    best_preds = []
    best_labels = []
    epochs_without_improvement = 0
    
    for epoch in range(CONFIG["epochs"]):
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
            scaler.step(opt)
            scaler.update()
            
            train_loss_sum += loss.item() * x.size(0)
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

                val_loss_sum += vloss.item() * x.size(0)
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
        
        print(f"Epoch {epoch+1:2d}/{CONFIG['epochs']} | "
              f"LR {current_lr:.2e} | "
              f"train loss {train_loss:.4f} | train acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} | val acc {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            best_preds = all_preds
            best_labels = all_labels
            epochs_without_improvement = 0
            torch.save({
                "model": model.state_dict(),
                "classes": train_ds.classes,
                "model_config": {
                    "num_classes": len(train_ds.classes),
                    "backbone": CONFIG["backbone"],
                    "embed_dim": CONFIG["embed_dim"],
                    "num_heads": CONFIG["num_heads"],
                    "num_layers": CONFIG["num_layers"],
                    "dropout": CONFIG["dropout"]
                }
            }, CONFIG["save_path"])
            print(f"  -> Saved new best model with acc {acc:.3f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= CONFIG["early_stopping_patience"]:
                print(f"\nEarly stopping triggered after {epoch+1} epochs (no improvement for {CONFIG['early_stopping_patience']} epochs)")
                break

    # Print final results
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {best_acc:.3f}")
    print(f"Model saved to: {CONFIG['save_path']}")
    
    # Print confusion matrix and per-class metrics
    print("\n" + "=" * 60)
    print("PER-CLASS METRICS (on best model's validation)")
    print("=" * 60)
    
    classes = train_ds.classes
    num_classes = len(classes)
    
    # Build confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(best_labels, best_preds):
        confusion[true_label][pred_label] += 1
    
    # Calculate per-class metrics
    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 54)
    
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
        
        print(f"{cls_name:<12} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10}")
    
    # Macro averages
    print("-" * 54)
    print(f"{'Macro Avg':<12} {np.mean(precisions):>10.3f} {np.mean(recalls):>10.3f} {np.mean(f1s):>10.3f} {sum(supports):>10}")
    
    # Print confusion matrix
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print("(rows = actual, columns = predicted)\n")
    
    # Header
    header = "         " + " ".join([f"{c[:6]:>6}" for c in classes])
    print(header)
    print("-" * len(header))
    
    for i, cls_name in enumerate(classes):
        row = f"{cls_name[:8]:<8} " + " ".join([f"{confusion[i,j]:>6}" for j in range(num_classes)])
        print(row)
    
    # Highlight most confused pairs
    print("\nMost confused pairs (excluding correct):")
    confusions = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and confusion[i, j] > 0:
                confusions.append((confusion[i, j], classes[i], classes[j]))
    
    confusions.sort(reverse=True)
    for count, true_cls, pred_cls in confusions[:5]:
        print(f"  {true_cls} -> {pred_cls}: {count} times")


if __name__ == "__main__":
    mp.freeze_support()
    main()
