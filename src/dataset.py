from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TalcSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=512):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size


        self.image_paths = sorted(set(self.images_dir.glob("*.[jJ][pP][gG]")))

    def __len__(self):
        return len(self.image_paths)

    def _read_image(self, path):
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        mask_name = f"{image_path.stem}_mask.png"
        mask_path = self.masks_dir / mask_name

        image = self._read_image(image_path)
        mask = self._read_image(mask_path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))

        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            "filename": image_path.name,
        }