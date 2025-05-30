import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import segmentation_models_pytorch as smp
from dataset import InriaDataset  # vlastná trieda z dataset.py
from torchvision import transforms
import os
from tqdm import tqdm

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 2
    BATCH_SIZE = 4
    EPOCHS = 25
    TILE_SIZE = 512

    # Cesty
    IMG_DIR = "data/tiles/images"
    MASK_DIR = "data/tiles/masks"
    LOG_DIR = "runs/tl_segmentation"

    # Transformácie
    transform = transforms.Compose([
        transforms.Resize((TILE_SIZE, TILE_SIZE)),
        transforms.ToTensor()
    ])

    # Dataset + DataLoader
    train_dataset = InriaDataset(IMG_DIR, MASK_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Model: UNet + ResNet34 encoder
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES)
    model.to(DEVICE)

    # Zamraziť encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Loss + optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    # TensorBoard
    writer = SummaryWriter(LOG_DIR)

    # Tréningová slučka
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        iou_score = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for imgs, masks in loop:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = model(imgs)
            loss = loss_fn(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # výpočet IoU (len pre logging)
            preds_soft = torch.argmax(preds, dim=1)
            intersection = torch.logical_and(preds_soft == 1, masks == 1).sum().item()
            union = torch.logical_or(preds_soft == 1, masks == 1).sum().item()
            iou_score += (intersection / union) if union != 0 else 0

        avg_loss = epoch_loss / len(train_loader)
        avg_iou = iou_score / len(train_loader)
        print(f"Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f}")

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("IoU/train", avg_iou, epoch)

    # Uloženie modelu
    torch.save(model.state_dict(), "unet_resnet34_inria.pth")
    writer.close()

if __name__ == "__main__":
    main()