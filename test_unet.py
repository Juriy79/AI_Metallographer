from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


MODEL_PATH = r"C:\Users\Юрий\PycharmProjects\AI_Metallographer\models\checkpoints\unet_talc.pth"

IMAGE_PATH = r"C:\Users\Юрий\Downloads\Задача 3. Скажи мне, кто твой шлиф\Задача 3. Скажи мне, кто твой шлиф\Фото руд по сортам. ч1\Оталькованные руды\Области оталькования\2550374-2 10х.JPG"

OUTPUT_PATH = r"C:\Users\Юрий\PycharmProjects\AI_Metallographer\outputs\predictions\pred_mask.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if success:
        encoded.tofile(str(path))


model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None,
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

image_bgr = read_image(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

resized = cv2.resize(image_rgb, (IMAGE_SIZE, IMAGE_SIZE))
x = resized.astype(np.float32) / 255.0
x = np.transpose(x, (2, 0, 1))
x = torch.tensor(x).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

mask = (prob > 0.5).astype(np.uint8) * 255

save_image(OUTPUT_PATH, mask)

print("Saved prediction:", OUTPUT_PATH)
print("Mask mean:", mask.mean())