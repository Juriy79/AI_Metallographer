from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class OreClassificationDataset(Dataset):
    def __init__(self, root_dir, image_size=224):
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        self.classes = [
            "Рядовые руды",
            "Труднообогатимые руды",
            "Оталькованные руды",
        ]

        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = []

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for path in sorted(set(class_dir.glob("*.[jJ][pP][gG]"))):
                self.samples.append((path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def read_image(path):
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]

        image = self.read_image(image_path)
        if image is None:
            raise FileNotFoundError(image_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        return {
            "image": torch.tensor(image, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "filename": image_path.name,
        }