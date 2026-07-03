import pandas as pd


def analyze_intergrowth(talc_objects, sulfide_objects, talc_percent, sulfide_percent):
    """
    Анализ характера срастаний по морфометрии выделенных объектов.

    Возвращает:
    - тип срастаний
    - уверенность
    - объяснение
    - таблицу признаков
    """

    def safe_mean(df, col):
        if df is None or len(df) == 0 or col not in df.columns:
            return 0
        return float(df[col].mean())

    def safe_count(df):
        if df is None:
            return 0
        return len(df)

    talc_count = safe_count(talc_objects)
    sulfide_count = safe_count(sulfide_objects)

    talc_area_mean = safe_mean(talc_objects, "Площадь, px")
    sulfide_area_mean = safe_mean(sulfide_objects, "Площадь, px")

    talc_diam_mean = safe_mean(talc_objects, "Эквивалентный диаметр, px")
    sulfide_diam_mean = safe_mean(sulfide_objects, "Эквивалентный диаметр, px")

    talc_aspect_mean = safe_mean(talc_objects, "Aspect Ratio")
    sulfide_aspect_mean = safe_mean(sulfide_objects, "Aspect Ratio")

    talc_circularity_mean = safe_mean(talc_objects, "Округлость")
    sulfide_circularity_mean = safe_mean(sulfide_objects, "Округлость")

    # ===== Простая экспертная логика =====
    score_thin = 0
    reasons = []

    # Много мелких объектов
    if talc_count + sulfide_count > 300:
        score_thin += 2
        reasons.append("выявлено большое количество мелких включений")

    # Малый средний диаметр
    mean_diameter = (talc_diam_mean + sulfide_diam_mean) / 2
    if mean_diameter > 0 and mean_diameter < 20:
        score_thin += 2
        reasons.append("средний эквивалентный диаметр объектов мал")

    # Малые площади объектов
    mean_area = (talc_area_mean + sulfide_area_mean) / 2
    if mean_area > 0 and mean_area < 300:
        score_thin += 2
        reasons.append("средняя площадь отдельных зерен мала")

    # Вытянутость
    mean_aspect = (talc_aspect_mean + sulfide_aspect_mean) / 2
    if mean_aspect > 1.8:
        score_thin += 1
        reasons.append("часть объектов имеет вытянутую форму")

    # Низкая округлость
    mean_circularity = (talc_circularity_mean + sulfide_circularity_mean) / 2
    if mean_circularity > 0 and mean_circularity < 0.55:
        score_thin += 1
        reasons.append("объекты имеют сложную или неокруглую форму")

    if score_thin >= 4:
        intergrowth_type = "Тонкие срастания"
        confidence = min(0.95, 0.55 + score_thin * 0.08)
    else:
        intergrowth_type = "Обычные срастания"
        confidence = max(0.55, 0.90 - score_thin * 0.08)

    if not reasons:
        reasons.append("морфометрические признаки не указывают на выраженные тонкие срастания")

    explanation = (
        f"Тип срастаний определён как «{intergrowth_type}». "
        f"Основание: {', '.join(reasons)}. "
        "Оценка выполнена по морфометрическим признакам выделенных областей талька и сульфидов."
    )

    features_df = pd.DataFrame({
        "Признак": [
            "Количество объектов талька",
            "Количество объектов сульфидов",
            "Средняя площадь талька, px",
            "Средняя площадь сульфидов, px",
            "Средний диаметр талька, px",
            "Средний диаметр сульфидов, px",
            "Средний Aspect Ratio талька",
            "Средний Aspect Ratio сульфидов",
            "Средняя округлость талька",
            "Средняя округлость сульфидов",
            "Доля талька, %",
            "Доля сульфидов, %",
            "Балл тонких срастаний",
        ],
        "Значение": [
            talc_count,
            sulfide_count,
            round(talc_area_mean, 2),
            round(sulfide_area_mean, 2),
            round(talc_diam_mean, 2),
            round(sulfide_diam_mean, 2),
            round(talc_aspect_mean, 3),
            round(sulfide_aspect_mean, 3),
            round(talc_circularity_mean, 3),
            round(sulfide_circularity_mean, 3),
            round(talc_percent, 2),
            round(sulfide_percent, 2),
            score_thin,
        ]
    })

    return {
        "type": intergrowth_type,
        "confidence": confidence,
        "explanation": explanation,
        "features": features_df,
    }