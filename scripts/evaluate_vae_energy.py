import numpy as np
import torch
from tqdm import tqdm

import _path  # noqa: F401
from src.datasets import get_dataloaders
from src.vae import VAEClassifier
from src.ood_scores import energy_score, msp_score
from src.metrics import compute_ood_metrics


def collect_scores(model, loader, device):
    model.eval()
    energy_scores = []
    msp_scores = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            _, logits, _, _, _ = model(images)

            energy_scores.extend(energy_score(logits).cpu().numpy())
            msp_scores.extend(msp_score(logits).cpu().numpy())

    return np.array(msp_scores), np.array(energy_scores)


def evaluate_pair(id_scores, ood_scores):
    y_true = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])
    return compute_ood_metrics(y_true, scores)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    loaders = get_dataloaders(batch_size=128)

    model = VAEClassifier(latent_dim=128, num_classes=10).to(device)
    model.load_state_dict(torch.load("models/vae_classifier.pth", map_location=device))

    id_msp, id_energy = collect_scores(model, loaders["cifar10_test"], device)
    svhn_msp, svhn_energy = collect_scores(model, loaders["svhn"], device)
    cifar100_msp, cifar100_energy = collect_scores(model, loaders["cifar100"], device)

    print("\nVAE Classifier: CIFAR-10 vs SVHN")
    print("MSP:", evaluate_pair(id_msp, svhn_msp))
    print("Energy:", evaluate_pair(id_energy, svhn_energy))

    print("\nVAE Classifier: CIFAR-10 vs CIFAR-100")
    print("MSP:", evaluate_pair(id_msp, cifar100_msp))
    print("Energy:", evaluate_pair(id_energy, cifar100_energy))


if __name__ == "__main__":
    main()
