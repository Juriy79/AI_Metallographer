from src.segmentation import TalcSegmenter

MODEL_PATH = r"C:\Users\Юрий\PycharmProjects\AI_Metallographer\models\checkpoints\unet_talc.pth"
IMAGE_PATH = r"C:\Users\Юрий\Downloads\Задача 3. Скажи мне, кто твой шлиф\Задача 3. Скажи мне, кто твой шлиф\Фото руд по сортам. ч1\Оталькованные руды\Области оталькования\2550374-2 10х.JPG"

segmenter = TalcSegmenter(MODEL_PATH)
result = segmenter.analyze(IMAGE_PATH)

segmenter.save_image("outputs/predictions/talc_mask.png", result["mask"])
segmenter.save_image("outputs/predictions/talc_overlay.png", result["overlay"])

print(f"Доля оталькования: {result['talc_percent']:.2f}%")
print("Saved outputs/predictions/talc_mask.png")
print("Saved outputs/predictions/talc_overlay.png")