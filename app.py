from src.pdf_report import create_pdf_report
from pathlib import Path
import tempfile
import pandas as pd
from io import BytesIO

from src.morphology import analyze_mask_summary, analyze_mask_objects
from src.intergrowth import analyze_intergrowth

import cv2
import streamlit as st

from src.segmentation import TalcSegmenter
from src.classifier import OreClassifier
from src.tile_inference import predict_mask_tiled

def resize_for_display(image, max_side=2000):
    h, w = image.shape[:2]

    if max(h, w) <= max_side:
        return image

    scale = max_side / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def make_colored_overlay(image_bgr, talc_mask, sulfide_mask, alpha=0.4):
    """
    Цветной overlay:
    тальк — синий
    сульфиды — жёлтый
    пересечение — красный
    """

    image_bgr = image_bgr.astype("uint8")

    talc = talc_mask > 0
    sulfide = sulfide_mask > 0
    overlap = talc & sulfide

    color_layer = image_bgr.copy()

    # BGR-цвета, потому что OpenCV работает в BGR
    color_layer[talc] = (255, 0, 0)        # синий
    color_layer[sulfide] = (0, 255, 255)   # жёлтый
    color_layer[overlap] = (0, 0, 255)     # красный

    overlay = cv2.addWeighted(
        image_bgr,
        1 - alpha,
        color_layer,
        alpha,
        0
    )

    return overlay


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



            image_bgr = talc_segmenter.read_image(image_path)
            MAX_ANALYSIS_SIDE = 4000

            h, w = image_bgr.shape[:2]

            if max(h, w) > MAX_ANALYSIS_SIDE:
                scale = MAX_ANALYSIS_SIDE / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)

                image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

                st.warning(
                    f"Панорама уменьшена для анализа: {w}×{h} → {new_w}×{new_h}"
                )

            h, w = image_bgr.shape[:2]

            if max(h, w) > 1000:

                talc_mask, _ = predict_mask_tiled(
                    talc_segmenter,
                    image_bgr,
                    tile_size=1024,
                    overlap=128,
                    batch_size=4,
                )

                sulfide_mask, _ = predict_mask_tiled(
                    sulfide_segmenter,
                    image_bgr,
                    tile_size=1024,
                    overlap=128,
                    batch_size=4,

                )

                talc_percent = (talc_mask > 0).mean() * 100
                sulfide_percent = (sulfide_mask > 0).mean() * 100

                talc_overlay = image_bgr.copy()
                talc_overlay[talc_mask > 0] = (255, 0, 0)

                sulfide_overlay = image_bgr.copy()
                sulfide_overlay[sulfide_mask > 0] = (0, 255, 255)

                talc_result = {
                    "image_bgr": image_bgr,
                    "mask": talc_mask,
                    "overlay": talc_overlay,
                    "talc_percent": talc_percent,
                }

                sulfide_result = {
                    "image_bgr": image_bgr,
                    "mask": sulfide_mask,
                    "overlay": sulfide_overlay,
                    "talc_percent": sulfide_percent,
                }

            else:

                talc_result = talc_segmenter.analyze(image_path)
                sulfide_result = sulfide_segmenter.analyze(image_path)

            classifier = load_classifier()
            cls_result = classifier.predict(image_path)

            colored_overlay_bgr = make_colored_overlay(
                talc_result["image_bgr"],
                talc_result["mask"],
                sulfide_result["mask"],
                alpha=0.4
            )

            colored_overlay_rgb = cv2.cvtColor(
                colored_overlay_bgr,
                cv2.COLOR_BGR2RGB
            )

            image_rgb = cv2.cvtColor(
                talc_result["image_bgr"],
                cv2.COLOR_BGR2RGB,
            )

            talc_overlay_rgb = cv2.cvtColor(
                talc_result["overlay"],
                cv2.COLOR_BGR2RGB,
            )

            sulfide_overlay_rgb = cv2.cvtColor(
                sulfide_result["overlay"],
                cv2.COLOR_BGR2RGB,
            )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Исходное")

            st.image(resize_for_display(image_rgb), use_container_width=True)

        with col2:
            st.subheader("Тальк")

            st.image(resize_for_display(talc_result["mask"]), use_container_width=True)

        with col3:
            st.subheader("Сульфиды")

            st.image(resize_for_display(sulfide_result["mask"]), use_container_width=True)



        with col4:

            st.subheader("Карта распределения фаз")

            st.image(resize_for_display(colored_overlay_rgb), use_container_width=True)

        talc_percent = talc_result["talc_percent"]
        sulfide_percent = sulfide_result["talc_percent"]




        st.subheader("Количественный анализ")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Доля талька", f"{talc_percent:.2f}%")
        m2.metric("Доля сульфидов", f"{sulfide_percent:.2f}%")
        m3.metric("Класс руды", cls_result["class_name"])
        m4.metric("Уверенность", f"{cls_result['confidence'] * 100:.1f}%")



        # ===== Морфология =====


        st.subheader("Морфология — сводные показатели")

        talc_summary = analyze_mask_summary(talc_result["mask"])
        sulfide_summary = analyze_mask_summary(sulfide_result["mask"])

        col_a, col_b = st.columns(2)

        with col_a:
            st.write("Тальк")
            st.dataframe(talc_summary, use_container_width=True)

        with col_b:
            st.write("Сульфиды")
            st.dataframe(sulfide_summary, use_container_width=True)


        st.subheader("Морфометрия отдельных зерен")

        talc_objects = analyze_mask_objects(talc_result["mask"])
        sulfide_objects = analyze_mask_objects(sulfide_result["mask"])

        col_c, col_d = st.columns(2)

        with col_c:
            st.write("Тальк")
            st.dataframe(talc_objects, use_container_width=True)

        with col_d:
            st.write("Сульфиды")
            st.dataframe(sulfide_objects, use_container_width=True)

        # ===== Анализ характера срастаний =====

        intergrowth_result = analyze_intergrowth(
            talc_objects=talc_objects,
            sulfide_objects=sulfide_objects,
            talc_percent=talc_percent,
            sulfide_percent=sulfide_percent,
        )

        st.subheader("Анализ характера срастаний")

        i1, i2 = st.columns(2)

        with i1:
            st.metric("Тип срастаний", intergrowth_result["type"])

        with i2:
            st.metric("Уверенность", f"{intergrowth_result['confidence'] * 100:.1f}%")

        st.write(intergrowth_result["explanation"])

        with st.expander("Признаки, использованные для анализа срастаний"):
            st.dataframe(intergrowth_result["features"], use_container_width=True)







        # ===== Вероятности =====



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

        st.write("Визуализация вероятностей")

        for class_name, prob in cls_result["probabilities"].items():
            st.write(f"{class_name}: {prob * 100:.2f} %")
            st.progress(float(prob))



        #БЛОК ЭКСПЕРТНОЙ ЛОГИКИ

        st.subheader("Экспертное заключение")

        if talc_percent > 10:
            final_class = "Оталькованные руды"
            explanation = (
                f"Доля талька составляет {talc_percent:.2f} %, "
                "что выше установленного порогового значения 10 %. "
                "По экспертному правилу образец отнесён к оталькованным рудам."
            )
        else:
            final_class = cls_result["class_name"]
            explanation = (
                f"Доля талька составляет {talc_percent:.2f} %, "
                "что не превышает установленный порог 10 %. "
                f"Окончательный класс определён классификатором: {final_class}."
            )

        st.success(f"Итоговое заключение: {final_class}")
        st.write(explanation)

        # ===== Excel-отчёт =====

        report_data = {
            "Параметр": [
                "Доля талька, %",
                "Доля сульфидов, %",
                "Класс классификатора",
                "Уверенность, %",
                "Итоговый класс",
                "Тип срастаний",
                "Уверенность срастаний, %",
            ],
            "Значение": [
                round(talc_percent, 2),
                round(sulfide_percent, 2),
                cls_result["class_name"],
                round(cls_result["confidence"] * 100, 1),
                final_class,
                intergrowth_result["type"],
                round(intergrowth_result["confidence"] * 100, 1),
            ],
        }

        report_df = pd.DataFrame(report_data)

        prob_df = pd.DataFrame({
            "Класс": list(cls_result["probabilities"].keys()),
            "Вероятность, %": [
                round(v * 100, 2)
                for v in cls_result["probabilities"].values()
            ],
        })

        excel_buffer = BytesIO()

        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            report_df.to_excel(writer, sheet_name="Итог", index=False)
            prob_df.to_excel(writer, sheet_name="Вероятности", index=False)
            talc_summary.to_excel(writer, sheet_name="Тальк_сводка", index=False)
            sulfide_summary.to_excel(writer, sheet_name="Сульфиды_сводка", index=False)
            talc_objects.to_excel(writer, sheet_name="Тальк_объекты", index=False)
            sulfide_objects.to_excel(writer, sheet_name="Сульфиды_объекты", index=False)
            intergrowth_result["features"].to_excel(writer, sheet_name="Срастания", index=False)


        #СОХРАНЯЕМ ОЧЕТ НА СЕРВЕР



        from datetime import datetime

        # ---------- сохраняем отчет в папку reports ----------

        REPORTS_DIR = BASE_DIR / "reports"
        REPORTS_DIR.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report_name = f"report_{final_class}_{timestamp}.xlsx"

        report_path = REPORTS_DIR / report_name

        with open(report_path, "wb") as f:
            f.write(excel_buffer.getvalue())



        st.success(f"Отчет сохранен: {report_name}")

        st.subheader("Предпросмотр Excel-отчёта")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Итог",
            "Вероятности",
            "Тальк — сводка",
            "Сульфиды — сводка",
            "Тальк — объекты",
            "Сульфиды — объекты",
        ])

        with tab1:
            st.dataframe(report_df, use_container_width=True)

        with tab2:
            st.dataframe(prob_df, use_container_width=True)

        with tab3:
            st.dataframe(talc_summary, use_container_width=True)

        with tab4:
            st.dataframe(sulfide_summary, use_container_width=True)

        with tab5:
            st.dataframe(talc_objects, use_container_width=True)

        with tab6:
            st.dataframe(sulfide_objects, use_container_width=True)

        # КНОПКА СКАЧИВАНИЯ

        st.download_button(
            label="Скачать Excel-отчёт",
            data=excel_buffer.getvalue(),
            file_name="ai_metallographer_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # ===== PDF-отчёт =====

        PDF_DIR = BASE_DIR / "reports" / "pdf"
        PDF_DIR.mkdir(parents=True, exist_ok=True)

        pdf_name = report_name.replace(".xlsx", ".pdf")
        pdf_path = PDF_DIR / pdf_name



        create_pdf_report(
            output_path=pdf_path,
            image_rgb=image_rgb,
            talc_mask=talc_result["mask"],
            sulfide_mask=sulfide_result["mask"],
            overlay_rgb=colored_overlay_rgb,

            report_df=report_df,
            prob_df=prob_df,

            talc_summary=talc_summary,
            sulfide_summary=sulfide_summary,

            talc_objects=talc_objects,
            sulfide_objects=sulfide_objects,

            final_class=final_class,
            explanation=explanation,

            # -------- НОВОЕ --------

            intergrowth_result=intergrowth_result,
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Скачать PDF-отчёт",
                data=f.read(),
                file_name=pdf_name,
                mime="application/pdf"
            )

        st.success(f"PDF-отчёт сохранён: {pdf_name}")

else:
    st.info("Загрузите изображение для анализа.")



