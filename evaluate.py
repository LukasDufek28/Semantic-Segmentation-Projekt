import torch
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import InriaDataset
from PIL import Image
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2
TILE_SIZE = 512
NUM_EXAMPLES = 100

# Cesty
IMG_DIR = "data/tiles/images"
MASK_DIR = "data/tiles/masks"
MODEL_PATH = "unet_resnet34_inria.pth"
OUT_DIR = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)

# Transformácia (rovnaká ako pri tréningu)
transform = transforms.Compose([
    transforms.Resize((TILE_SIZE, TILE_SIZE)),
    transforms.ToTensor()
])

# Dataset
dataset = InriaDataset(IMG_DIR, MASK_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Model
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# Farby pre vizualizáciu tried
colors = {
    0: (255, 255, 255),  # pozadie - biela
    1: (255, 0, 0),      # budova - červená
}

def decode_mask(mask):
    """Convert mask tensor [H,W] to RGB image"""
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for k, color in colors.items():
        rgb[mask == k] = color
    return rgb

# Vizualizácia
for idx, (img, true_mask) in enumerate(dataloader):
    if idx >= NUM_EXAMPLES:
        break

    img = img.to(DEVICE)
    with torch.no_grad():
        pred = model(img)
        pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

    img_np = img.squeeze().cpu().permute(1, 2, 0).numpy()
    gt_mask = true_mask.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_np)
    axs[0].set_title("Input Image")
    axs[1].imshow(decode_mask(gt_mask))
    axs[1].set_title("Ground Truth")
    axs[2].imshow(decode_mask(pred))
    axs[2].set_title("Prediction")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/result_{idx}.png")
    plt.close()