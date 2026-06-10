from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import _path  # noqa: F401
from src.datasets import get_dataloaders
from src.models import get_resnet18


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    loaders = get_dataloaders(batch_size=128)

    model = get_resnet18(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_loss = total_loss / len(loaders["train"])

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Acc={train_acc:.4f}")

    checkpoint_path = Path("models/cifar10_resnet18.pth")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model to {checkpoint_path}")


if __name__ == "__main__":
    train()
