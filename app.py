import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Metallographer", layout="wide")

st.title("AI Metallographer")
st.write("Демо: анализ SEM/OM-изображения микроструктуры")

uploaded_file = st.file_uploader(
    "Загрузите SEM/OM изображение",
    type=["png", "jpg", "jpeg", "tif", "tiff"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    phase_light = np.sum(mask == 255)
    phase_dark = np.sum(mask == 0)
    total = mask.size

    light_percent = phase_light / total * 100
    dark_percent = phase_dark / total * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Исходное изображение")
        st.image(image_np, use_container_width=True)

    with col2:
        st.subheader("Серая версия")
        st.image(gray, use_container_width=True, clamp=True)

    with col3:
        st.subheader("Маска фаз")
        st.image(mask, use_container_width=True, clamp=True)

    st.subheader("Оценка фазовых долей")

    st.write(f"Светлая фаза: **{light_percent:.2f}%**")
    st.write(f"Тёмная фаза: **{dark_percent:.2f}%**")

    st.progress(int(light_percent))

    if light_percent < 10 or dark_percent < 10:
        st.warning("Одна из фаз занимает слишком малую площадь. Требуется экспертная проверка.")
    else:
        st.success("Анализ выполнен успешно.")
else:
    st.info("Загрузите изображение для анализа.")