import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch

def find_image_mask_pairs(
    label_root: Path, task: str
) -> List[Tuple[Path, Path]]:
    """
    Gathers (image, mask) pairs for a specific task ("residue_background" or "sunlit_shaded")
    """
    pairs = []
    task_folder = label_root / task
    for field_dir in task_folder.iterdir():
        if not field_dir.is_dir():
            continue
        for image_dir in field_dir.iterdir():
            if not image_dir.is_dir():
                continue
            img_name = image_dir.name
            img_path = image_dir / f"{img_name}.jpg"
            if task == "residue_background":
                mask_path = image_dir / f"{img_name}_res.tif"
            elif task == "sunlit_shaded":
                mask_path = image_dir / f"{img_name}_sunshad.tif"
            else:
                continue
            if img_path.exists() and mask_path.exists():
                pairs.append((img_path, mask_path))
    return pairs

class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_mask_pairs: List[Tuple[Path, Path]],
        resize: Optional[Tuple[int, int]] = None,
        transform=None
    ):
        self.pairs = image_mask_pairs
        self.resize = resize
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.resize:
            image = image.resize(self.resize, Image.BILINEAR)
            mask = mask.resize(self.resize, Image.NEAREST)

        image = transforms.ToTensor()(image)  # Normalize to [0,1]
        mask = torch.from_numpy(np.array(mask)).long()  # shape: (H, W)

        return image, mask

def split_dataset(
    pairs: List[Tuple[Path, Path]],
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    random.seed(seed)
    random.shuffle(pairs)

    total = len(pairs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return pairs[:train_end], pairs[train_end:val_end], pairs[val_end:]

# Quick testing
if __name__ == "__main__":
    base_dir = Path("data/images_2048/label")
    task = "residue_background"  # or "sunlit_shaded"

    pairs = find_image_mask_pairs(base_dir, task)
    train_pairs, val_pairs, test_pairs = split_dataset(pairs)

    train_ds = SegmentationDataset(train_pairs, resize=(512, 512))
    val_ds = SegmentationDataset(val_pairs, resize=(512, 512))
    test_ds = SegmentationDataset(test_pairs, resize=(512, 512))

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    for images, masks in train_loader:
        print(images.shape, masks.shape)
        break
