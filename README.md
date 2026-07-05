# AI Metallographer


 ## Features

- Semantic segmentation of talc and sulfides using U-Net
- Ore classification into three classes
- Morphometric analysis
- Large panorama processing using tiled inference
- Phase distribution map
- PDF report generation
- Excel report generation
- Intergrowth analysis (normal / fine)

   ## Возможности

- Семантическая сегментация талька и сульфидов с помощью U-Net
- Классификация руды по трем классам
- Морфометрический анализ
- Обработка больших панорам с помощью плиточного вывода
- Карта распределения фаз
- Создание отчета в формате PDF
- Создание отчета в формате Excel
- Анализ срастания (нормальное / тонкое)

Автоматический анализ OM-изображений полированных шлифов.

## Возможности

- сегментация талька
- сегментация сульфидов
- карта распределения фаз
- морфометрия
- анализ срастаний
- классификация руды
- Excel-отчёт
- PDF-отчёт
- анализ панорам
- тайловый инференс

## Стек

Python

PyTorch

Streamlit

OpenCV

ReportLab

OpenPyXL

## Tech Stack

- Python
- Streamlit
- OpenCV
- PyTorch
- U-Net
- segmentation-models-pytorch

## Quick Start

### Clone repository

```bash
git clone https://github.com/Juriy79/AI_Metallographer.git
cd AI_Metallographer
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run application

```bash
streamlit run app.py
```

### Trained models

The repository already contains all trained model checkpoints:

```
models/checkpoints/
├── unet_talc.pth
├── unet_sulfides.pth
└── ore_classifier_resnet18.pth
```

No additional model download is required.
