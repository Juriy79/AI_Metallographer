from src.dataset import TalcSegmentationDataset

images_dir = r"C:\Users\Юрий\Downloads\Задача 3. Скажи мне, кто твой шлиф\Задача 3. Скажи мне, кто твой шлиф\Фото руд по сортам. ч1\Оталькованные руды\Области оталькования"
masks_dir = r"C:\Users\Юрий\PycharmProjects\AI_Metallographer\data\masks\talc"

dataset = TalcSegmentationDataset(images_dir, masks_dir, image_size=512)

print("Количество объектов:", len(dataset))

sample = dataset[0]

print("Image shape:", sample["image"].shape)
print("Mask shape:", sample["mask"].shape)
print("Filename:", sample["filename"])