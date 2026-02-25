from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.src.dataset import BDDDetectionDataset
from model.src.model import get_model


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch))


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        for t in targets:
            if (t["labels"] >= 11).any():  # since NUM_CLASSES = 11
                print("BAD LABEL IN BATCH:", t["labels"])
                raise ValueError("Invalid label in batch")
            if (t["labels"] < 0).any():
                print("NEGATIVE LABEL:", t["labels"])
                raise ValueError("Negative label detected")

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(dataloader)


def main():
    project_root = Path(__file__).resolve().parents[2]

    images_dir = project_root / "data/raw/bdd100k/images/100k/train"
    labels_json = (
        project_root / "data/raw/bdd100k/labels/bdd100k_labels_images_train.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BDDDetectionDataset(
        images_dir=images_dir,
        labels_json=labels_json,
        max_samples=1000,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )

    model = get_model()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2, gamma=0.1  # decay every 2 epochs  # reduce LR by 10x
    )

    checkpoint_dir = project_root / "model/training/checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = 5

    print("Starting training on:", device)

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")

        epoch_loss = train_one_epoch(model, dataloader, optimizer, device)

        print(f"Epoch {epoch+1} completed. Avg Loss: {epoch_loss:.4f}")

        checkpoint_path = checkpoint_dir / f"fasterrcnn_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path.name}")

        scheduler.step()

        print("Current LR:", scheduler.get_last_lr()[0])


if __name__ == "__main__":
    main()
