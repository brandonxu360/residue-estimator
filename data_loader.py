import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch

def find_multitask_image_mask_pairs(label_root: Path) -> List[Tuple[Path, Path, Path]]:
    """
    Finds image paths with both residue and sunlit masks.
    Returns list of (image_path, residue_mask_path, sunlit_mask_path)
    """
    pairs = []
    residue_root = label_root / "residue_background"
    sunlit_root = label_root / "sunlit_shaded"

    for field_dir in residue_root.iterdir():
        if not field_dir.is_dir():
            continue
        sunlit_field_dir = sunlit_root / field_dir.name
        if not sunlit_field_dir.exists():
            continue

        for image_dir in field_dir.iterdir():
            if not image_dir.is_dir():
                continue
            img_name = image_dir.name
            img_path = image_dir / f"{img_name}.jpg"
            residue_mask = image_dir / f"{img_name}_res.tif"
            sunlit_image_dir = sunlit_field_dir / img_name
            sunlit_mask = sunlit_image_dir / f"{img_name}_sunshad.tif"

            if img_path.exists() and residue_mask.exists() and sunlit_mask.exists():
                pairs.append((img_path, residue_mask, sunlit_mask))
    return pairs

class MultiTaskSegmentationDataset(Dataset):
    def __init__(
        self,
        image_mask_triples: List[Tuple[Path, Path, Path]],
        resize: Optional[Tuple[int, int]] = None,
        transform=None
    ):
        self.triples = image_mask_triples
        self.resize = resize
        self.transform = transform

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        img_path, residue_path, sunlit_path = self.triples[idx]

        image = Image.open(img_path).convert("RGB")
        residue_mask = Image.open(residue_path).convert("L")
        sunlit_mask = Image.open(sunlit_path).convert("L")

        if self.resize:
            image = image.resize(self.resize, Image.BILINEAR)
            residue_mask = residue_mask.resize(self.resize, Image.NEAREST)
            sunlit_mask = sunlit_mask.resize(self.resize, Image.NEAREST)

        image = transforms.ToTensor()(image)
        residue_mask = torch.from_numpy(np.array(residue_mask)).long()
        sunlit_mask = torch.from_numpy(np.array(sunlit_mask)).long()

        return image, residue_mask, sunlit_mask

def split_dataset(
    items: List,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    random.seed(seed)
    random.shuffle(items)

    total = len(items)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return items[:train_end], items[train_end:val_end], items[val_end:]

# Quick testing
if __name__ == "__main__":
    base_dir = Path("data/images_2048/label")

    triples = find_multitask_image_mask_pairs(base_dir)
    train_set, val_set, test_set = split_dataset(triples)

    train_ds = MultiTaskSegmentationDataset(train_set, resize=(512, 512))
    val_ds = MultiTaskSegmentationDataset(val_set, resize=(512, 512))
    test_ds = MultiTaskSegmentationDataset(test_set, resize=(512, 512))

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    for images, residue_masks, sunlit_masks in train_loader:
        print(images.shape, residue_masks.shape, sunlit_masks.shape)
        break
