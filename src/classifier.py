from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class OreClassifier:
    def __init__(self, model_path, image_size=224):
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.classes = checkpoint["classes"]

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def read_image(path):
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image

    def predict(self, image_path):
        image_bgr = self.read_image(image_path)

        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (self.image_size, self.image_size))

        x = image_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs))

        return {
            "class_name": self.classes[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {
                self.classes[i]: float(probs[i])
                for i in range(len(self.classes))
            },
        }