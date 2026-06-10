import numpy as np
import torch
from tqdm import tqdm

import _path  # noqa: F401
from src.datasets import get_dataloaders
from src.models import get_resnet18
from src.ood_scores import mahalanobis_score
from src.metrics import compute_ood_metrics


def extract_features(model, loader, device):
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            logits, features = model(images, return_features=True)

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_features, all_labels


def fit_mahalanobis(train_features, train_labels, num_classes=10, eps=1e-5):
    class_means = []

    for c in range(num_classes):
        class_features = train_features[train_labels == c]
        class_means.append(class_features.mean(axis=0))

    class_means = np.stack(class_means, axis=0)

    centered = []
    for c in range(num_classes):
        class_features = train_features[train_labels == c]
        centered.append(class_features - class_means[c])

    centered = np.concatenate(centered, axis=0)

    covariance = np.cov(centered, rowvar=False)
    covariance += eps * np.eye(covariance.shape[0])

    precision = np.linalg.inv(covariance)

    return class_means, precision


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

    print("Extracting train features...")
    train_features, train_labels = extract_features(model, loaders["train"], device)

    print("Fitting Mahalanobis statistics...")
    class_means, precision = fit_mahalanobis(train_features, train_labels)

    print("Extracting test/OOD features...")
    id_features, _ = extract_features(model, loaders["cifar10_test"], device)
    svhn_features, _ = extract_features(model, loaders["svhn"], device)
    cifar100_features, _ = extract_features(model, loaders["cifar100"], device)

    print("Computing Mahalanobis scores...")
    id_scores = mahalanobis_score(id_features, class_means, precision)
    svhn_scores = mahalanobis_score(svhn_features, class_means, precision)
    cifar100_scores = mahalanobis_score(cifar100_features, class_means, precision)

    print("\nRaw Mahalanobis")
    print("CIFAR-10 vs SVHN:", evaluate_pair(id_scores, svhn_scores))
    print("CIFAR-10 vs CIFAR-100:", evaluate_pair(id_scores, cifar100_scores))

    print("\nNegative Mahalanobis")
    print("CIFAR-10 vs SVHN:", evaluate_pair(-id_scores, -svhn_scores))
    print("CIFAR-10 vs CIFAR-100:", evaluate_pair(-id_scores, -cifar100_scores))

    print("\nCIFAR-10 vs SVHN")
    print("Mahalanobis:", evaluate_pair(id_scores, svhn_scores))

    print("\nCIFAR-10 vs CIFAR-100")
    print("Mahalanobis:", evaluate_pair(id_scores, cifar100_scores))


if __name__ == "__main__":
    main()
