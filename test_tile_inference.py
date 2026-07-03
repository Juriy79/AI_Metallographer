from pathlib import Path

from src.segmentation import TalcSegmenter
from src.tile_inference import predict_mask_tiled


BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "checkpoints" / "unet_talc.pth"

IMAGE_PATH = (
    r"C:\Users\Юрий\Downloads\Задача 3. Скажи мне, кто твой шлиф"
    r"\Задача 3. Скажи мне, кто твой шлиф"
    r"\Фото руд по сортам. ч1"
    r"\Оталькованные руды"
    r"\Области оталькования"
    r"\DSCN4718.JPG"
)

OUTPUT_DIR = BASE_DIR / "outputs" / "predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

segmenter = TalcSegmenter(MODEL_PATH)

# Используем функцию чтения, которая поддерживает русские пути
image = segmenter.read_image(IMAGE_PATH)

if image is None:
    raise RuntimeError(f"Не удалось открыть изображение:\n{IMAGE_PATH}")

mask, prob = predict_mask_tiled(
    segmenter,
    image,
    tile_size=512,
    overlap=64,
)

mask_path = OUTPUT_DIR / "tile_mask.png"

import cv2
cv2.imwrite(str(mask_path), mask)

print()
print("===================================")
print("Tile inference completed")
print("===================================")
print("Saved mask:", mask_path)
print(f"Image size: {image.shape[1]} x {image.shape[0]}")
print(f"Talc percent: {(mask > 0).mean() * 100:.2f}%")