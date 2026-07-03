import cv2
import numpy as np
import torch


def _make_positions(length, tile_size, stride):
    """
    Создаёт координаты тайлов так, чтобы последний тайл доходил до края изображения.
    """
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, stride))

    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)

    return positions


def predict_mask_tiled(
    segmenter,
    image_bgr,
    tile_size=1024,
    overlap=128,
    batch_size=4,
    progress_callback=None,
):
    """
    Быстрый тайловый инференс большого изображения.

    segmenter — объект TalcSegmenter или аналогичный сегментатор
    image_bgr — исходное изображение OpenCV в BGR
    tile_size — размер тайла, например 512 или 1024
    overlap — перекрытие тайлов
    batch_size — сколько тайлов обрабатывать за один проход модели
    progress_callback — функция для отображения прогресса в Streamlit

    Возвращает:
    mask — бинарная маска 0/255
    full_prob — карта вероятностей 0..1
    """

    h, w = image_bgr.shape[:2]
    stride = tile_size - overlap

    ys = _make_positions(h, tile_size, stride)
    xs = _make_positions(w, tile_size, stride)

    tiles_info = []

    for y1 in ys:
        for x1 in xs:
            y2 = min(y1 + tile_size, h)
            x2 = min(x1 + tile_size, w)
            tiles_info.append((x1, y1, x2, y2))

    total_tiles = len(tiles_info)

    full_prob = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    segmenter.model.eval()

    processed = 0

    for batch_start in range(0, total_tiles, batch_size):
        batch_tiles = tiles_info[batch_start: batch_start + batch_size]

        batch_tensors = []
        original_shapes = []

        for x1, y1, x2, y2 in batch_tiles:
            tile = image_bgr[y1:y2, x1:x2]

            original_h = y2 - y1
            original_w = x2 - x1
            original_shapes.append((original_h, original_w))

            pad_h = tile_size - tile.shape[0]
            pad_w = tile_size - tile.shape[1]

            if pad_h > 0 or pad_w > 0:
                tile = cv2.copyMakeBorder(
                    tile,
                    0,
                    pad_h,
                    0,
                    pad_w,
                    cv2.BORDER_REFLECT
                )

            tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            tile_rgb = cv2.resize(
                tile_rgb,
                (segmenter.image_size, segmenter.image_size),
                interpolation=cv2.INTER_AREA
            )

            tensor = tile_rgb.astype(np.float32) / 255.0
            tensor = np.transpose(tensor, (2, 0, 1))
            batch_tensors.append(tensor)

        x_tensor = torch.tensor(
            np.stack(batch_tensors, axis=0),
            dtype=torch.float32,
            device=segmenter.device
        )

        with torch.inference_mode():
            logits = segmenter.model(x_tensor)
            probs = torch.sigmoid(logits)[:, 0].detach().cpu().numpy()

        for i, (x1, y1, x2, y2) in enumerate(batch_tiles):
            original_h, original_w = original_shapes[i]

            prob = cv2.resize(
                probs[i],
                (tile_size, tile_size),
                interpolation=cv2.INTER_LINEAR
            )

            prob = prob[:original_h, :original_w]

            full_prob[y1:y2, x1:x2] += prob
            count_map[y1:y2, x1:x2] += 1

        processed += len(batch_tiles)

        if progress_callback is not None:
            progress_callback(processed, total_tiles)

    full_prob = full_prob / np.maximum(count_map, 1)
    mask = (full_prob > segmenter.threshold).astype(np.uint8) * 255

    return mask, full_prob