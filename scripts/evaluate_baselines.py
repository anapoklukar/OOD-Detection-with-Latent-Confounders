import numpy as np
import torch
from tqdm import tqdm

import _path  # noqa: F401
from src.datasets import get_dataloaders
from src.models import get_resnet18
from src.ood_scores import msp_score, energy_score
from src.metrics import compute_ood_metrics


def collect_scores(model, loader, device):
    model.eval()
    msp_scores = []
    energy_scores = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            logits = model(images)

            msp_scores.extend(msp_score(logits).cpu().numpy())
            energy_scores.extend(energy_score(logits).cpu().numpy())

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

    model = get_resnet18(num_classes=10).to(device)
    model.load_state_dict(
        torch.load("models/cifar10_resnet18.pth", map_location=device)
    )

    id_msp, id_energy = collect_scores(model, loaders["cifar10_test"], device)
    svhn_msp, svhn_energy = collect_scores(model, loaders["svhn"], device)
    cifar100_msp, cifar100_energy = collect_scores(model, loaders["cifar100"], device)

    print("\nCIFAR-10 vs SVHN")
    print("MSP:", evaluate_pair(id_msp, svhn_msp))
    print("Energy:", evaluate_pair(id_energy, svhn_energy))

    print("\nCIFAR-10 vs CIFAR-100")
    print("MSP:", evaluate_pair(id_msp, cifar100_msp))
    print("Energy:", evaluate_pair(id_energy, cifar100_energy))


if __name__ == "__main__":
    main()
