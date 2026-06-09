import torch
import torch.optim as optim
from tqdm import tqdm

from src.datasets import get_dataloaders
from src.vae import VAEClassifier, vae_loss


def evaluate_classifier(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, logits, _, _, _ = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    loaders = get_dataloaders(batch_size=128)

    model = VAEClassifier(latent_dim=128, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20

    for epoch in range(epochs):
        model.train()

        total_loss = 0
        total_recon = 0
        total_kl = 0
        total_cls = 0

        for images, labels in tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            recon, logits, mu, logvar, z = model(images)

            loss, recon_loss, kl_loss, cls_loss = vae_loss(
                recon, images, logits, labels, mu, logvar,
                beta=0.001,
                cls_weight=1.0
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_cls += cls_loss.item()

        val_acc = evaluate_classifier(model, loaders["val"], device)

        print(
            f"Epoch {epoch+1}: "
            f"Loss={total_loss/len(loaders['train']):.4f}, "
            f"Recon={total_recon/len(loaders['train']):.4f}, "
            f"KL={total_kl/len(loaders['train']):.4f}, "
            f"Cls={total_cls/len(loaders['train']):.4f}, "
            f"Val Acc={val_acc:.4f}"
        )

    torch.save(model.state_dict(), "models/vae_classifier.pth")
    print("Saved model to models/vae_classifier.pth")


if __name__ == "__main__":
    train()