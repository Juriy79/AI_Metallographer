from pathlib import Path
from io import BytesIO
from datetime import datetime

from PIL import Image

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def register_russian_font():
    font_path = Path("C:/Windows/Fonts/arial.ttf")

    if font_path.exists():
        pdfmetrics.registerFont(TTFont("Arial", str(font_path)))
        return "Arial"

    return "Helvetica"


def array_to_pdf_image(image_array, max_width=8 * cm, max_height=6 * cm):
    if image_array.ndim == 2:
        image = Image.fromarray(image_array).convert("RGB")
    else:
        image = Image.fromarray(image_array).convert("RGB")

    image.thumbnail((1600, 1200))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    img = RLImage(buffer)

    w, h = image.size
    ratio = min(max_width / w, max_height / h)

    img.drawWidth = w * ratio
    img.drawHeight = h * ratio

    return img


def dataframe_to_table(df, max_rows=20, font_name="Arial"):
    if df is None or len(df) == 0:
        data = [["Нет данных"]]
    else:
        df_view = df.head(max_rows).copy()
        data = [list(df_view.columns)] + df_view.astype(str).values.tolist()

    table = Table(data, repeatRows=1)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    return table


def create_pdf_report(
    output_path,
    image_rgb,
    talc_mask,
    sulfide_mask,
    overlay_rgb,
    report_df,
    prob_df,
    talc_summary,
    sulfide_summary,
    talc_objects,
    sulfide_objects,
    final_class,
    explanation,
    intergrowth_result=None,
):
    font_name = register_russian_font()

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="RussianTitle",
        fontName=font_name,
        fontSize=22,
        leading=26,
        spaceAfter=14,
    ))

    styles.add(ParagraphStyle(
        name="RussianHeading",
        fontName=font_name,
        fontSize=15,
        leading=18,
        spaceBefore=12,
        spaceAfter=8,
    ))

    styles.add(ParagraphStyle(
        name="RussianText",
        fontName=font_name,
        fontSize=10,
        leading=14,
    ))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )

    story = []

    # =========================
    # ТИТУЛ И ИТОГ
    # =========================

    story.append(Paragraph("AI Metallographer", styles["RussianTitle"]))
    story.append(Paragraph(
        "PDF-отчёт по анализу OM-изображения полированного шлифа",
        styles["RussianText"]
    ))
    story.append(Paragraph(
        f"Дата формирования: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}",
        styles["RussianText"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Итоговое заключение", styles["RussianHeading"]))
    story.append(Paragraph(f"<b>{final_class}</b>", styles["RussianText"]))
    story.append(Paragraph(explanation, styles["RussianText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Количественный анализ", styles["RussianHeading"]))
    story.append(dataframe_to_table(report_df, font_name=font_name))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Вероятности по классам", styles["RussianHeading"]))
    story.append(dataframe_to_table(prob_df, font_name=font_name))
    story.append(Spacer(1, 12))

    # =========================
    # АНАЛИЗ СРАСТАНИЙ
    # =========================

    if intergrowth_result is not None:
        story.append(PageBreak())
        story.append(Paragraph("Анализ характера срастаний", styles["RussianHeading"]))

        intergrowth_type = intergrowth_result.get("type", "Не определено")
        intergrowth_conf = intergrowth_result.get("confidence", 0) * 100
        intergrowth_explanation = intergrowth_result.get(
            "explanation",
            "Пояснение не сформировано."
        )

        story.append(Paragraph(
            f"<b>Тип срастаний:</b> {intergrowth_type}",
            styles["RussianText"]
        ))

        story.append(Paragraph(
            f"<b>Уверенность:</b> {intergrowth_conf:.1f} %",
            styles["RussianText"]
        ))

        story.append(Spacer(1, 8))

        story.append(Paragraph("<b>Основание решения:</b>", styles["RussianText"]))
        story.append(Paragraph(intergrowth_explanation, styles["RussianText"]))
        story.append(Spacer(1, 10))

        if "features" in intergrowth_result:
            story.append(Paragraph(
                "Морфометрические признаки, использованные для анализа",
                styles["RussianHeading"]
            ))
            story.append(dataframe_to_table(
                intergrowth_result["features"],
                max_rows=30,
                font_name=font_name
            ))

    # =========================
    # БОЛЬШАЯ КАРТА ФАЗ
    # =========================

    story.append(PageBreak())
    story.append(Paragraph("Карта распределения фаз", styles["RussianHeading"]))
    story.append(Paragraph(
        "Синим цветом выделены области талька, жёлтым — области сульфидной минерализации, "
        "красным — зоны пересечения масок. Остальная часть изображения сохраняет исходную "
        "цветовую информацию шлифа.",
        styles["RussianText"]
    ))
    story.append(Spacer(1, 10))

    story.append(array_to_pdf_image(
        overlay_rgb,
        max_width=17.5 * cm,
        max_height=20 * cm
    ))

    # =========================
    # ИЗОБРАЖЕНИЯ
    # =========================

    story.append(PageBreak())
    story.append(Paragraph("Изображения и выделенные области", styles["RussianHeading"]))

    images_table = Table([
        [
            array_to_pdf_image(image_rgb),
            array_to_pdf_image(talc_mask),
        ],
        [
            Paragraph("Исходное изображение", styles["RussianText"]),
            Paragraph("Выделенные области талька", styles["RussianText"]),
        ],
        [
            array_to_pdf_image(sulfide_mask),
            array_to_pdf_image(overlay_rgb),
        ],
        [
            Paragraph("Выделенные области сульфидов", styles["RussianText"]),
            Paragraph("Карта распределения фаз", styles["RussianText"]),
        ],
    ])

    images_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))

    story.append(images_table)

    # =========================
    # СВОДНАЯ МОРФОЛОГИЯ
    # =========================

    story.append(PageBreak())
    story.append(Paragraph("Морфология — сводные показатели", styles["RussianHeading"]))

    story.append(Paragraph("Тальк", styles["RussianText"]))
    story.append(dataframe_to_table(talc_summary, font_name=font_name))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Сульфиды", styles["RussianText"]))
    story.append(dataframe_to_table(sulfide_summary, font_name=font_name))
    story.append(Spacer(1, 10))

    # =========================
    # МОРФОМЕТРИЯ ОБЪЕКТОВ
    # =========================

    story.append(PageBreak())
    story.append(Paragraph("Морфометрия отдельных зерен", styles["RussianHeading"]))

    story.append(Paragraph("Тальк — первые 20 объектов", styles["RussianText"]))
    story.append(dataframe_to_table(talc_objects, max_rows=20, font_name=font_name))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Сульфиды — первые 20 объектов", styles["RussianText"]))
    story.append(dataframe_to_table(sulfide_objects, max_rows=20, font_name=font_name))

    doc.build(story)

    return output_path