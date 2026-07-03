from src.classifier import OreClassifier

MODEL_PATH = r"C:\Users\Юрий\PycharmProjects\AI_Metallographer\models\checkpoints\ore_classifier_resnet18.pth"

IMAGE_PATH = r"C:\Users\Юрий\Downloads\Задача 3. Скажи мне, кто твой шлиф\Задача 3. Скажи мне, кто твой шлиф\Фото руд по сортам. ч1\Рядовые руды\2539589-1.JPG"

classifier = OreClassifier(MODEL_PATH)

result = classifier.predict(IMAGE_PATH)

print("Класс:", result["class_name"])
print("Уверенность:", round(result["confidence"], 3))
print("Вероятности:")

for class_name, prob in result["probabilities"].items():
    print(class_name, ":", round(prob, 3))