from pathlib import Path
from PIL import Image
import numpy as np
import os

def tile_image(img_path, mask_path, tile_size=512, stride=512, output_dir="data/tiles"):
    img = Image.open(img_path)
    mask = Image.open(mask_path).convert("L")

    img = np.array(img)
    mask = np.array(mask)

    img_name = Path(img_path).stem

    out_img_dir = Path(output_dir) / "images"
    out_mask_dir = Path(output_dir) / "masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    h, w = img.shape[:2]
    tile_id = 0
    for i in range(0, h - tile_size + 1, stride):
        for j in range(0, w - tile_size + 1, stride):
            img_tile = img[i:i+tile_size, j:j+tile_size]
            mask_tile = mask[i:i+tile_size, j:j+tile_size]

            img_out = Image.fromarray(img_tile)
            mask_out = Image.fromarray(mask_tile)

            img_out.save(out_img_dir / f"{img_name}_{tile_id}.png")
            mask_out.save(out_mask_dir / f"{img_name}_{tile_id}.png")
            tile_id += 1

for img_file in os.listdir("AerialImageDataset/train/images"):
    if img_file.endswith(".tif"):
        img_path = os.path.join("AerialImageDataset/train/images", img_file)
        mask_path = os.path.join("AerialImageDataset/train/gt", img_file)
        tile_image(img_path, mask_path, tile_size=512, stride=512, output_dir="data/tiles")