from src.sulfide_dataset import SulfideDataset

IMAGES_DIR = r"C:\Users\Юрий\Downloads\Задача 3. Скажи мне, кто твой шлиф\Задача 3. Скажи мне, кто твой шлиф\Фото руд по сортам. ч1"
MASKS_DIR = r"C:\Users\Юрий\PycharmProjects\AI_Metallographer\data\masks\sulfides"

dataset = SulfideDataset(IMAGES_DIR, MASKS_DIR, image_size=512)

print("Количество объектов:", len(dataset))

sample = dataset[0]

print("Image shape:", sample["image"].shape)
print("Mask shape:", sample["mask"].shape)
print("Filename:", sample["filename"])