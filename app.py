from pathlib import Path
import tempfile

import cv2
import streamlit as st

from src.segmentation import TalcSegmenter
from src.classifier import OreClassifier


BASE_DIR = Path(__file__).parent

TALC_MODEL_PATH = BASE_DIR / "models" / "checkpoints" / "unet_talc.pth"
SULFIDE_MODEL_PATH = BASE_DIR / "models" / "checkpoints" / "unet_sulfides.pth"
CLS_MODEL_PATH = BASE_DIR / "models" / "checkpoints" / "ore_classifier_resnet18.pth"


st.set_page_config(page_title="AI Metallographer", layout="wide")

st.title("AI Metallographer")
st.write("Автоматический анализ OM-изображений полированных шлифов")


@st.cache_resource
def load_talc_segmenter():
    return TalcSegmenter(TALC_MODEL_PATH)


@st.cache_resource
def load_sulfide_segmenter():
    return TalcSegmenter(SULFIDE_MODEL_PATH)


@st.cache_resource
def load_classifier():
    return OreClassifier(CLS_MODEL_PATH)


uploaded_file = st.file_uploader(
    "Загрузите изображение шлифа",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    st.success("Изображение загружено")

    if st.button("Запустить анализ"):
        with st.spinner("Модель анализирует изображение..."):
            talc_segmenter = load_talc_segmenter()
            sulfide_segmenter = load_sulfide_segmenter()
            classifier = load_classifier()

            talc_result = talc_segmenter.analyze(image_path)
            sulfide_result = sulfide_segmenter.analyze(image_path)
            cls_result = classifier.predict(image_path)

            image_rgb = cv2.cvtColor(talc_result["image_bgr"], cv2.COLOR_BGR2RGB)
            talc_overlay_rgb = cv2.cvtColor(talc_result["overlay"], cv2.COLOR_BGR2RGB)
            sulfide_overlay_rgb = cv2.cvtColor(sulfide_result["overlay"], cv2.COLOR_BGR2RGB)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Исходное")
            st.image(image_rgb, use_container_width=True)

        with col2:
            st.subheader("Тальк")
            st.image(talc_result["mask"], use_container_width=True)

        with col3:
            st.subheader("Сульфиды")
            st.image(sulfide_result["mask"], use_container_width=True)

        with col4:
            st.subheader("Overlay")
            st.image(sulfide_overlay_rgb, use_container_width=True)

        talc_percent = talc_result["talc_percent"]
        sulfide_percent = sulfide_result["talc_percent"]

        st.subheader("Количественный анализ")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Доля талька", f"{talc_percent:.2f}%")
        m2.metric("Доля сульфидов", f"{sulfide_percent:.2f}%")
        m3.metric("Класс руды", cls_result["class_name"])
        m4.metric("Уверенность", f"{cls_result['confidence'] * 100:.1f}%")

        st.subheader("Вероятности по классам")
        st.dataframe(
            {
                "Класс": list(cls_result["probabilities"].keys()),
                "Вероятность, %": [
                    round(v * 100, 2)
                    for v in cls_result["probabilities"].values()
                ],
            },
            use_container_width=True
        )

        st.subheader("Экспертная логика")

        if talc_percent > 10:
            final_class = "Оталькованная руда"
            explanation = (
                f"Доля талька составляет {talc_percent:.2f}%, "
                "что выше порога 10%. Образец классифицирован как оталькованная руда."
            )
        else:
            final_class = cls_result["class_name"]
            explanation = (
                f"Доля талька составляет {talc_percent:.2f}%, "
                "что не превышает порог 10%. Итоговый класс определён классификатором."
            )

        st.success(f"Итог: {final_class}")
        st.write(explanation)

else:
    st.info("Загрузите изображение для анализа.")