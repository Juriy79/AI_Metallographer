from pathlib import Path

import cv2
import numpy as np


def create_talc_mask(image_path: Path, output_path: Path) -> None:

    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        print(f"Не удалось прочитать: {image_path}")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # диапазон синего цвета разметки
    lower_blue = np.array([90, 80, 50])
    upper_blue = np.array([140, 255, 255])

    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # утолщаем контур, чтобы он стал замкнутым
    kernel = np.ones((7, 7), np.uint8)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(
        blue_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    filled_mask = np.zeros_like(blue_mask)

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > 100:
            cv2.drawContours(
                filled_mask,
                [contour],
                -1,
                255,
                thickness=cv2.FILLED
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    success, encoded = cv2.imencode(".png", filled_mask)

    if success:
        encoded.tofile(str(output_path))
    else:
        print("Ошибка сохранения:", output_path)


    print(output_path)


def process_folder(input_dir: str, output_dir: str) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)


    image_files = sorted(set(input_path.glob("*.[jJ][pP][gG]")))

    print(f"Найдено изображений: {len(image_files)}")

    for image_file in image_files:

        safe_name = image_file.stem.replace("х", "x").replace("Х", "X")
        mask_file = output_path / f"{safe_name}_mask.png"
        create_talc_mask(image_file, mask_file)


if __name__ == "__main__":
    process_folder(
        input_dir = r"C:\Users\Юрий\Downloads\Задача 3. Скажи мне, кто твой шлиф\Задача 3. Скажи мне, кто твой шлиф\Фото руд по сортам. ч1\Оталькованные руды\Области оталькования",
        output_dir=r"C:\Users\Юрий\PycharmProjects\AI_Metallographer\data\masks\talc"
    )