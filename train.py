import torch
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp

from src.dataset import TalcSegmentationDataset

IMAGES_DIR = r"C:\Users\Юрий\Downloads\Задача 3. Скажи мне, кто твой шлиф\Задача 3. Скажи мне, кто твой шлиф\Фото руд по сортам. ч1\Оталькованные руды\Области оталькования"
MASKS_DIR = r"C:\Users\Юрий\PycharmProjects\AI_Metallographer\data\masks\talc"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 5
LR = 1e-4


def main():
    dataset = TalcSegmentationDataset(IMAGES_DIR, MASKS_DIR, image_size=IMAGE_SIZE)

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(DEVICE)

    loss_fn = smp.losses.DiceLoss(mode="binary")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Device:", DEVICE)
    print("Train:", len(train_ds), "Val:", len(val_ds))

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for batch in train_loader:
            images = batch["image"].to(DEVICE)
            masks = batch["mask"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                masks = batch["mask"].to(DEVICE)

                logits = model(images)
                loss = loss_fn(logits, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    import os

    print("Current directory:", os.getcwd())
    from pathlib import Path

    save_path = Path(__file__).parent / "models" / "checkpoints" / "unet_talc.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_path)

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()