import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import torch.multiprocessing as mp

def main():
    device = "cuda" if torch.cuda.is_available() else print('error')

    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])

    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])

    train_ds = datasets.ImageFolder("splits/train", transform=train_tf)
    val_ds   = datasets.ImageFolder("splits/val", transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print("device:", device)
    print("classes:", train_ds.classes)
    print("train samples:", len(train_ds), "| val samples:", len(val_ds))

    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=True,
        num_classes=len(train_ds.classes)
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_acc = 0.0
    for epoch in range(20):
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        model.eval()
        correct = total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits = model(x)
                    vloss = loss_fn(logits, y)

                val_loss_sum += vloss.item() * x.size(0)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        val_loss = val_loss_sum / total
        print(f"Epoch {epoch+1} | val loss {val_loss:.4f} | val acc {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "classes": train_ds.classes}, "vit_radar.pt")

    print("Best val acc:", best_acc)

if __name__ == "__main__":
    mp.freeze_support()
    main()
