import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

def find_multitask_image_mask_pairs(label_root: Path) -> List[Tuple[Path, Path, Path]]:
    """
    Finds image paths with both residue and sunlit masks, accounting for split parts.
    Returns list of (image_part_path, residue_mask_part_path, sunlit_mask_part_path)
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
            sunlit_image_dir = sunlit_field_dir / image_dir.name
            if not sunlit_image_dir.exists():
                continue

            # Iterate over all image parts in the directory
            for image_part in image_dir.glob("*.jpg"):
                # Skip non-part files (if any)
                if "_part" not in image_part.stem:
                    continue
                
                # Extract base name and part number (e.g., "IMG_0629" and "01" from "IMG_0629_part01")
                stem_parts = image_part.stem.rsplit("_part", 1)
                if len(stem_parts) != 2:
                    continue  # Invalid filename format
                base_name, part_num = stem_parts

                # Construct mask filenames
                residue_mask_filename = f"{base_name}_res_part{part_num}.tif"
                residue_mask_path = image_dir / residue_mask_filename

                sunlit_mask_filename = f"{base_name}_sunshad_part{part_num}.tif"
                sunlit_mask_path = sunlit_image_dir / sunlit_mask_filename

                if residue_mask_path.exists() and sunlit_mask_path.exists():
                    pairs.append((image_part, residue_mask_path, sunlit_mask_path))
    
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

        # Convert to numpy and normalize to 0 or 1
        residue_mask = (np.array(residue_mask) > 127).astype(np.float32)
        sunlit_mask = (np.array(sunlit_mask) > 127).astype(np.float32)

        # Convert to tensors
        image = transforms.ToTensor()(image)
        residue_mask = torch.from_numpy(residue_mask).float()
        sunlit_mask = torch.from_numpy(sunlit_mask).float()

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

# Testing with new structure
if __name__ == "__main__":
    base_dir = Path("data/label")  # Updated path

    triples = find_multitask_image_mask_pairs(base_dir)
    train_set, val_set, test_set = split_dataset(triples)

    # Assuming images are already 512x512, resize can be None
    train_ds = MultiTaskSegmentationDataset(train_set, resize=None)
    val_ds = MultiTaskSegmentationDataset(val_set, resize=None)
    test_ds = MultiTaskSegmentationDataset(test_set, resize=None)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    for images, residue_masks, sunlit_masks in train_loader:
        print(images.shape, residue_masks.shape, sunlit_masks.shape)
        break