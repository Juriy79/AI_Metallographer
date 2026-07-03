


from pathlib import Path
import argparse

import torch
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp

from src.dataset import TalcSegmentationDataset
from src.sulfide_dataset import SulfideDataset



BASE_DIR = Path(__file__).parent

IMAGES_DIRS = {
    "talc": r"C:\Users\Юрий\Downloads\Задача 3. Скажи мне, кто твой шлиф\Задача 3. Скажи мне, кто твой шлиф\Фото руд по сортам. ч1\Оталькованные руды\Области оталькования",
    "sulfides": r"C:\Users\Юрий\Downloads\Задача 3. Скажи мне, кто твой шлиф\Задача 3. Скажи мне, кто твой шлиф\Фото руд по сортам. ч1",
}

MASK_DIRS = {
    "talc": BASE_DIR / "data" / "masks" / "talc",
    "sulfides": BASE_DIR / "data" / "masks" / "sulfides",
}

MODEL_NAMES = {
    "talc": "unet_talc.pth",
    "sulfides": "unet_sulfides.pth",
}


def train_unet(task: str, epochs: int, batch_size: int, image_size: int, lr: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    images_dir = IMAGES_DIRS[task]
    masks_dir = MASK_DIRS[task]


    if task == "talc":
        dataset = TalcSegmentationDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            image_size=image_size,
        )
    else:
        dataset = SulfideDataset(
            images_dir=images_dir,
            masks_dir=masks_dir,
            image_size=image_size,
        )

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    loss_fn = smp.losses.DiceLoss(mode="binary")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("Task:", task)
    print("Device:", device)
    print("Total:", len(dataset))
    print("Train:", len(train_ds), "Val:", len(val_ds))

    best_val_loss = float("inf")
    save_path = BASE_DIR / "models" / "checkpoints" / MODEL_NAMES[task]
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

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
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                logits = model(images)
                loss = loss_fn(logits, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print("Saved best:", save_path)

    print("Best val_loss:", best_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["talc", "sulfides"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    train_unet(
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
    )