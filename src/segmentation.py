from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


class TalcSegmenter:
    def __init__(self, model_path, image_size=512, threshold=0.5):
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def read_image(path):
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image

    @staticmethod
    def save_image(path, image):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        success, encoded = cv2.imencode(".png", image)
        if success:
            encoded.tofile(str(path))

    def predict_mask(self, image_bgr):
        original_h, original_w = image_bgr.shape[:2]

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.image_size, self.image_size))

        x = resized.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = torch.tensor(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

        mask_small = (prob > self.threshold).astype(np.uint8) * 255
        mask = cv2.resize(mask_small, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        return mask, prob

    def create_overlay(self, image_bgr, mask, alpha=0.45):
        overlay = image_bgr.copy()

        blue_layer = np.zeros_like(image_bgr)
        blue_layer[:, :, 0] = 255

        mask_bool = mask > 0
        overlay[mask_bool] = cv2.addWeighted(
            image_bgr[mask_bool],
            1 - alpha,
            blue_layer[mask_bool],
            alpha,
            0,
        )

        return overlay

    @staticmethod
    def calculate_talc_percent(mask):
        return float((mask > 0).sum() / mask.size * 100)

    def analyze(self, image_path):
        image_bgr = self.read_image(image_path)

        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        mask, prob = self.predict_mask(image_bgr)
        overlay = self.create_overlay(image_bgr, mask)
        talc_percent = self.calculate_talc_percent(mask)

        return {
            "image_bgr": image_bgr,
            "mask": mask,
            "overlay": overlay,
            "talc_percent": talc_percent,
            "probability_map": prob,
        }