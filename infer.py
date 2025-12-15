import sys
import torch
import timm
from PIL import Image
from torchvision import transforms

def main():
    if len(sys.argv) < 2:
        print("Usage: py -3.12 infer_one.py path_to_image.png")
        return

    img_path = sys.argv[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load("vit_radar.pt", map_location=device)
    classes = ckpt["classes"]

    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])

    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top = probs.argmax().item()

    print("pred:", classes[top])
    print("confidence:", float(probs[top]))

if __name__ == "__main__":
    main()
