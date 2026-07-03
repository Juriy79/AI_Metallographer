from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = r"C:\Users\Юрий\Downloads\Задача 3. Скажи мне, кто твой шлиф\Задача 3. Скажи мне, кто твой шлиф\Фото руд по сортам. ч1"

OUTPUT_DIR = r"C:\Users\Юрий\PycharmProjects\AI_Metallographer\data\masks\sulfides"


CLASSES = [
    "Рядовые руды",
    "Труднообогатимые руды",
    "Оталькованные руды",
]


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_png(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if success:
        encoded.tofile(str(path))


def create_sulfide_mask(image_bgr):
    """
    Быстрая pseudo-разметка сульфидов:
    светлые жёлтые/белые области считаем сульфидами.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # светло-жёлтые/золотистые области
    lower_yellow = np.array([15, 20, 120])
    upper_yellow = np.array([45, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # очень светлые области
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    bright_mask = cv2.inRange(gray, 150, 255)

    mask = cv2.bitwise_or(yellow_mask, bright_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def process():
    output_dir = Path(OUTPUT_DIR)
    total = 0

    for class_name in CLASSES:
        class_dir = Path(ROOT_DIR) / class_name

        image_files = sorted(set(class_dir.glob("*.[jJ][pP][gG]")))

        print(class_name, ":", len(image_files))

        for image_path in image_files:
            image = read_image(image_path)

            if image is None:
                print("Не удалось прочитать:", image_path)
                continue

            mask = create_sulfide_mask(image)

            safe_class = class_name.replace(" ", "_")
            mask_path = output_dir / safe_class / f"{image_path.stem}_sulfide_mask.png"

            save_png(mask_path, mask)
            total += 1

    print("Всего масок:", total)


if __name__ == "__main__":
    process()