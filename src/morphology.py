import cv2
import numpy as np
import pandas as pd


def analyze_mask_summary(mask, min_area=20):
    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    areas = []

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            areas.append(area)

    total_area_percent = binary.mean() * 100

    if len(areas) == 0:
        return pd.DataFrame({
            "Параметр": [
                "Количество объектов",
                "Доля площади, %",
                "Средняя площадь, px",
                "Максимальная площадь, px",
                "Минимальная площадь, px",
            ],
            "Значение": [0, round(total_area_percent, 4), 0, 0, 0]
        })

    return pd.DataFrame({
        "Параметр": [
            "Количество объектов",
            "Доля площади, %",
            "Средняя площадь, px",
            "Максимальная площадь, px",
            "Минимальная площадь, px",
        ],
        "Значение": [
            len(areas),
            round(total_area_percent, 4),
            round(float(np.mean(areas)), 4),
            int(np.max(areas)),
            int(np.min(areas)),
        ]
    })


def analyze_mask_objects(mask, min_area=20):
    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    objects = []

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        component = (labels == label_id).astype(np.uint8)

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)

        perimeter = cv2.arcLength(contour, True)
        equivalent_diameter = np.sqrt(4 * area / np.pi)

        circularity = 0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        orientation = 0
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            orientation = ellipse[2]

        cx, cy = centroids[label_id]

        objects.append({
            "ID объекта": label_id,
            "Площадь, px": area,
            "Периметр, px": round(perimeter, 2),
            "Эквивалентный диаметр, px": round(equivalent_diameter, 2),
            "Округлость": round(circularity, 3),
            "Aspect Ratio": round(aspect_ratio, 3),
            "Solidity": round(solidity, 3),
            "Ориентация, градусы": round(orientation, 2),
            "Центр X": round(cx, 1),
            "Центр Y": round(cy, 1),
        })

    return pd.DataFrame(objects)


def analyze_mask(mask, min_area=20):
    return analyze_mask_summary(mask, min_area)